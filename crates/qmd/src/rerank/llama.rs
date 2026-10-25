//! Cross-encoder reranker via llama-cpp-2: rank pooling, which for Qwen3
//! rerankers reads the last token's yes/no head and softmaxes it, so the
//! first output is P(yes) in `[0, 1]`.

use std::num::NonZeroU32;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::embed::llama_backend;
use crate::error::{Error, Result};
use crate::rerank::Reranker;

/// Context per scored pair; documents are truncated to fit beside the prompt.
const RERANK_CONTEXT_SIZE: u32 = 2048;

/// The Qwen3-Reranker prompt, used when the GGUF carries no
/// `tokenizer.chat_template.rerank` of its own.
const QWEN3_RERANK_TEMPLATE: &str = "<|im_start|>system\nJudge whether the Document meets \
     the requirements based on the Query and the Instruct provided. Note that the answer can \
     only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search \
     query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: \
     {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// GGUF reranker loaded through llama.cpp.
pub struct LlamaReranker {
    model: LlamaModel,
    model_uri: String,
    template: String,
}

impl LlamaReranker {
    /// Load the model for `model_uri`, downloading it on first use.
    ///
    /// # Errors
    /// Fails on download or model-load errors, and on a template without a
    /// `{document}` slot.
    pub fn load(model_uri: &str) -> Result<Self> {
        let path = crate::models::ensure_model(model_uri)?;
        let backend = llama_backend()?;
        let params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);
        let model = LlamaModel::load_from_file(backend, &path, &params)
            .map_err(|e| Error::Other(format!("failed to load {}: {e}", path.display())))?;
        let template = model
            .meta_val_str("tokenizer.chat_template.rerank")
            .ok()
            .filter(|t| t.contains("{query}") && t.contains("{document}"))
            .unwrap_or_else(|| QWEN3_RERANK_TEMPLATE.to_owned());
        Ok(Self {
            model,
            model_uri: model_uri.to_owned(),
            template,
        })
    }

    fn tokens(&self, text: &str, bos: AddBos) -> Result<Vec<LlamaToken>> {
        self.model
            .str_to_token(text, bos)
            .map_err(|e| Error::Other(format!("tokenize failed: {e}")))
    }

    /// Prompt tokens for one pair: the template around the document, with the
    /// document cut so the whole prompt fits the context.
    fn prompt(&self, query: &str, document: &str) -> Result<Vec<LlamaToken>> {
        let (head, tail) = fill_template(&self.template, query);
        Ok(fit_prompt(
            self.tokens(&head, AddBos::Always)?,
            self.tokens(document, AddBos::Never)?,
            self.tokens(&tail, AddBos::Never)?,
            RERANK_CONTEXT_SIZE as usize,
        ))
    }
}

/// `head`, `doc`, `tail` cut to `ctx` tokens: the document first, then the
/// head's end (an over-long query). The tail is never cut, because rank
/// pooling reads the score at the last token.
fn fit_prompt<T>(mut head: Vec<T>, mut doc: Vec<T>, tail: Vec<T>, ctx: usize) -> Vec<T> {
    head.truncate(ctx.saturating_sub(tail.len()));
    doc.truncate(ctx.saturating_sub(head.len() + tail.len()));
    head.extend(doc);
    head.extend(tail);
    head
}

impl Reranker for LlamaReranker {
    fn model_uri(&self) -> &str {
        &self.model_uri
    }

    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let backend = llama_backend()?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(
                NonZeroU32::new(RERANK_CONTEXT_SIZE).expect("non-zero"),
            ))
            .with_n_batch(RERANK_CONTEXT_SIZE)
            .with_n_ubatch(RERANK_CONTEXT_SIZE)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Rank);
        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| Error::Other(format!("rerank context: {e}")))?;

        let mut scores = Vec::with_capacity(documents.len());
        for document in documents {
            let tokens = self.prompt(query, document)?;
            let mut batch = LlamaBatch::new(RERANK_CONTEXT_SIZE as usize, 1);
            batch
                .add_sequence(&tokens, 0, false)
                .map_err(|e| Error::Other(format!("batch: {e}")))?;
            ctx.clear_kv_cache();
            ctx.decode(&mut batch)
                .map_err(|e| Error::Other(format!("rerank decode: {e}")))?;
            let out = ctx
                .embeddings_seq_ith(0)
                .map_err(|e| Error::Other(format!("rerank score: {e}")))?;
            let score = out
                .first()
                .copied()
                .ok_or_else(|| Error::Other("reranker returned no score".to_owned()))?;
            scores.push(f64::from(score));
        }
        Ok(scores)
    }
}

/// The template's text before and after its `{document}` slot, with the query
/// filled in. Split first: a query may itself contain `{document}`.
fn fill_template(template: &str, query: &str) -> (String, String) {
    let (head, tail) = template
        .split_once("{document}")
        .expect("template checked for {document} at load");
    (
        head.replace("{query}", query),
        tail.replace("{query}", query),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_query_naming_the_document_slot_stays_in_the_head() {
        let (head, tail) = fill_template("Q: {query}\nD: {document}<end>", "fill {document} here");
        assert_eq!(head, "Q: fill {document} here\nD: ");
        assert_eq!(tail, "<end>");
    }

    #[test]
    fn an_over_long_query_gives_up_its_end_never_the_tail() {
        assert_eq!(
            fit_prompt(vec![1, 2], vec![3, 4, 5], vec![9], 5),
            [1, 2, 3, 4, 9]
        );
        assert_eq!(
            fit_prompt(vec![1, 2, 3, 4, 5], vec![6], vec![8, 9], 4),
            [1, 2, 8, 9]
        );
    }
}
