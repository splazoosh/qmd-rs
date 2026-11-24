//! Real embedding backend via llama-cpp-2 (mean pooling, matching the
//! node-llama-cpp output the TS implementation stores, to a
//! cross-implementation cosine above 0.999).

use std::borrow::Cow;
use std::num::NonZeroU32;
use std::sync::{Mutex, OnceLock};

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::chunk::Tokenizer;
use crate::embed::Embedder;
use crate::error::{Error, Result};
use crate::models::GpuMode;

/// Embedding context size (EmbeddingGemma's window; texts are truncated to
/// fit, mirroring the TS context-size guard).
const EMBED_CONTEXT_SIZE: u32 = 2048;

pub(crate) fn backend() -> Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let _guard = INIT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    unsafe { llama_cpp_sys_2::llama_log_set(Some(keep_errors), std::ptr::null_mut()) };
    let init = LlamaBackend::init().map_err(|e| Error::Other(format!("llama init: {e}")))?;
    Ok(BACKEND.get_or_init(|| init))
}

/// Error lines llama.cpp and ggml have logged since [`take_log_errors`] last ran.
/// `new_context` reports only a null pointer; the reason (`cudaMalloc failed:
/// ...`) exists nowhere else.
static LOG_ERRORS: Mutex<String> = Mutex::new(String::new());

/// How much of a failure's log is kept.
const LOG_ERRORS_MAX_BYTES: usize = 1024;

/// The log callback llama.cpp and ggml call, from any thread.
///
/// # Safety
/// `text` must be null or a NUL-terminated string valid for this call.
unsafe extern "C" fn keep_errors(
    level: llama_cpp_sys_2::ggml_log_level,
    text: *const std::ffi::c_char,
    _user_data: *mut std::ffi::c_void,
) {
    if level != llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR || text.is_null() {
        return;
    }
    let line = unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy();
    let mut kept = LOG_ERRORS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if kept.len() + line.len() <= LOG_ERRORS_MAX_BYTES {
        if !kept.is_empty() {
            kept.push_str("; ");
        }
        kept.push_str(line.trim());
    }
}

/// Take the error lines logged since the last call.
pub(crate) fn take_log_errors() -> String {
    std::mem::take(
        &mut *LOG_ERRORS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

/// `error` with whatever llama.cpp logged while it happened.
pub(crate) fn with_log_errors(error: impl std::fmt::Display) -> String {
    let logged = take_log_errors();
    if logged.is_empty() {
        error.to_string()
    } else {
        format!("{error} ({logged})")
    }
}

/// `text` as llama.cpp's C strings can carry it: a NUL becomes a space, so a stray
/// one costs that text a character and not a failed batch.
pub(crate) fn without_nul(text: &str) -> Cow<'_, str> {
    if text.contains('\0') {
        Cow::Owned(text.replace('\0', " "))
    } else {
        Cow::Borrowed(text)
    }
}

/// Whether models loaded under `mode` run on a GPU: false for `off`, for a
/// build without a GPU backend, and for one that sees no usable device. Loads
/// no model.
///
/// # Errors
/// Fails if the llama backend cannot be initialised, or `mode` names a backend
/// this build finds no device of.
pub fn gpu_offload(mode: GpuMode) -> Result<bool> {
    model_params(mode)?;
    Ok(mode != GpuMode::Off && backend()?.supports_gpu_offload())
}

/// Model parameters that place a model where `mode` asks: every layer on the
/// GPU, or no device at all. `off` needs the empty device list as well as zero
/// layers, or a CUDA build still opens the device for its contexts.
///
/// # Errors
/// Fails if `mode` names a backend this build finds no device of.
pub(crate) fn model_params(mode: GpuMode) -> Result<LlamaModelParams> {
    backend()?;
    let params = LlamaModelParams::default();
    let devices = match mode {
        GpuMode::Auto => return Ok(params.with_n_gpu_layers(u32::MAX)),
        GpuMode::Off => Vec::new(),
        GpuMode::Cuda | GpuMode::Vulkan | GpuMode::Metal => {
            let found = llama_cpp_2::list_llama_ggml_backend_devices();
            let devices: Vec<usize> = found
                .iter()
                .filter(|d| backend_is(&d.backend, mode))
                .map(|d| d.index)
                .collect();
            if devices.is_empty() {
                let backends: Vec<&str> = found.iter().map(|d| d.backend.as_str()).collect();
                let backends = if backends.is_empty() {
                    "none".to_owned()
                } else {
                    backends.join(", ")
                };
                return Err(Error::Other(format!(
                    "GPU mode '{mode}' finds no {mode} device in this build (its devices: \
                     {backends}); use 'auto' or 'off'"
                )));
            }
            devices
        }
    };
    let layers = if mode == GpuMode::Off { 0 } else { u32::MAX };
    params
        .with_n_gpu_layers(layers)
        .with_devices(&devices)
        .map_err(|e| Error::Other(format!("GPU mode '{mode}': {e}")))
}

/// Whether a ggml backend registered as `name` is the one `mode` names.
fn backend_is(name: &str, mode: GpuMode) -> bool {
    let name = name.to_ascii_lowercase();
    match mode {
        GpuMode::Cuda => name == "cuda",
        GpuMode::Vulkan => name == "vulkan",
        GpuMode::Metal => name == "mtl" || name == "metal",
        GpuMode::Auto | GpuMode::Off => false,
    }
}

/// GGUF embedding model loaded through llama.cpp.
pub struct LlamaEmbedder {
    model: LlamaModel,
    model_uri: String,
}

impl LlamaEmbedder {
    /// Load the model for `model_uri` where `mode` asks, downloading it on
    /// first use.
    ///
    /// # Errors
    /// Fails on download or model-load errors, and on a `mode` this build has
    /// no device for.
    pub fn load(model_uri: &str, mode: GpuMode) -> Result<Self> {
        let path = crate::models::ensure_model(model_uri)?;
        let backend = backend()?;
        let params = model_params(mode)?;
        take_log_errors();
        let model = LlamaModel::load_from_file(backend, &path, &params).map_err(|e| {
            Error::Other(format!(
                "failed to load {}: {}",
                path.display(),
                with_log_errors(e)
            ))
        })?;
        Ok(Self {
            model,
            model_uri: model_uri.to_owned(),
        })
    }

    fn tokens(&self, text: &str) -> Result<Vec<LlamaToken>> {
        self.model
            .str_to_token(&without_nul(text), AddBos::Always)
            .map_err(|e| Error::Other(format!("tokenize failed: {e}")))
    }
}

impl Embedder for LlamaEmbedder {
    fn model_uri(&self) -> &str {
        &self.model_uri
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Option<Vec<f32>>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let backend = backend()?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(EMBED_CONTEXT_SIZE).expect("non-zero")))
            .with_n_batch(EMBED_CONTEXT_SIZE)
            .with_n_ubatch(EMBED_CONTEXT_SIZE)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean);
        take_log_errors();
        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| Error::Other(format!("embed context: {}", with_log_errors(e))))?;

        let max_tokens = EMBED_CONTEXT_SIZE as usize;
        let mut results = Vec::with_capacity(texts.len());
        let mut last_error = None;
        for text in texts {
            let mut tokens = self.tokens(text)?;
            tokens.truncate(max_tokens);
            if tokens.is_empty() {
                results.push(None);
                continue;
            }
            let embed = (|| -> Result<Vec<f32>> {
                let mut batch = LlamaBatch::new(max_tokens, 1);
                batch
                    .add_sequence(&tokens, 0, false)
                    .map_err(|e| Error::Other(format!("batch: {e}")))?;
                ctx.clear_kv_cache();
                ctx.decode(&mut batch)
                    .map_err(|e| Error::Other(format!("decode: {e}")))?;
                Ok(ctx
                    .embeddings_seq_ith(0)
                    .map_err(|e| Error::Other(format!("embeddings: {e}")))?
                    .to_vec())
            })();
            match embed {
                Ok(vector) => results.push(Some(vector)),
                Err(e) => {
                    results.push(None);
                    last_error = Some(e);
                }
            }
        }
        if let Some(e) = last_error
            && results.iter().all(Option::is_none)
        {
            return Err(Error::Other(with_log_errors(e)));
        }
        Ok(results)
    }
}

impl Tokenizer for LlamaEmbedder {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.tokens(text)?.len())
    }

    fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> Result<String> {
        let mut tokens = self.tokens(text)?;
        if tokens.len() <= max_tokens {
            return Ok(text.to_owned());
        }
        tokens.truncate(max_tokens.max(1));
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        for token in tokens {
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| Error::Other(format!("detokenize failed: {e}")))?;
            out.push_str(&piece);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backend_is_matched_by_the_name_ggml_registers() {
        assert!(backend_is("CUDA", GpuMode::Cuda));
        assert!(backend_is("MTL", GpuMode::Metal));
        assert!(backend_is("Vulkan", GpuMode::Vulkan));
        assert!(!backend_is("CUDA", GpuMode::Vulkan));
        assert!(!backend_is("CPU", GpuMode::Auto));
    }

    #[test]
    fn a_nul_becomes_a_space_and_clean_text_is_not_copied() {
        assert_eq!(without_nul("lea\0ders"), "lea ders");
        assert!(matches!(without_nul("leaders"), Cow::Borrowed(_)));
    }

    #[test]
    fn an_error_carries_what_llama_logged_and_only_errors_are_kept() {
        take_log_errors();
        let log = |level, text: &std::ffi::CStr| unsafe {
            keep_errors(level, text.as_ptr(), std::ptr::null_mut())
        };
        log(llama_cpp_sys_2::GGML_LOG_LEVEL_INFO, c"loading\n");
        log(
            llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR,
            c"cudaMalloc failed: out of memory\n",
        );
        log(llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR, c"no buffers\n");
        assert_eq!(
            with_log_errors("null reference"),
            "null reference (cudaMalloc failed: out of memory; no buffers)"
        );
        assert_eq!(with_log_errors("null reference"), "null reference");
    }
}
