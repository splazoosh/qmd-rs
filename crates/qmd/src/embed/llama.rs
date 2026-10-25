//! Real embedding backend via llama-cpp-2 (mean pooling, matching the
//! node-llama-cpp output the TS implementation stores, to a
//! cross-implementation cosine above 0.999).

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
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
    let init = LlamaBackend::init().map_err(|e| Error::Other(format!("llama init: {e}")))?;
    Ok(BACKEND.get_or_init(|| init))
}

/// Whether llama.cpp found a GPU to offload to: false for a build without the
/// `cuda` feature, and for a CUDA build that sees no usable device. Loads no
/// model.
///
/// # Errors
/// Fails if the llama backend cannot be initialised.
pub fn gpu_offload() -> Result<bool> {
    Ok(backend()?.supports_gpu_offload())
}

/// GGUF embedding model loaded through llama.cpp.
pub struct LlamaEmbedder {
    model: LlamaModel,
    model_uri: String,
}

impl LlamaEmbedder {
    /// Load the model for `model_uri`, downloading it on first use.
    ///
    /// GPU offload is requested for all layers; a CPU-only build simply
    /// ignores it and runs on the CPU.
    ///
    /// # Errors
    /// Fails on download or model-load errors.
    pub fn load(model_uri: &str) -> Result<Self> {
        let path = crate::models::ensure_model(model_uri)?;
        let backend = backend()?;
        let params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);
        let model = LlamaModel::load_from_file(backend, &path, &params)
            .map_err(|e| Error::Other(format!("failed to load {}: {e}", path.display())))?;
        Ok(Self {
            model,
            model_uri: model_uri.to_owned(),
        })
    }

    fn tokens(&self, text: &str) -> Result<Vec<LlamaToken>> {
        self.model
            .str_to_token(text, AddBos::Always)
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
        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| Error::Other(format!("embed context: {e}")))?;

        let max_tokens = EMBED_CONTEXT_SIZE as usize;
        let mut results = Vec::with_capacity(texts.len());
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
            results.push(embed.ok());
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
