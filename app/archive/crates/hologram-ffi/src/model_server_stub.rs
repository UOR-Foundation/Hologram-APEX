//! Model Server FFI Stubs
//!
//! These stubs are used when the `model-server` feature is not enabled.
//! They provide the same API but return appropriate error values.

/// Create a new inference engine (stub)
/// Returns 0 (error) when feature is not enabled
#[must_use]
pub fn inference_engine_create() -> u64 {
    tracing::warn!("Model server feature not enabled - inference_engine_create() returning 0");
    0
}

/// Destroy an inference engine (stub)
/// No-op when feature is not enabled
pub fn inference_engine_destroy(_engine_handle: u64) {
    tracing::warn!("Model server feature not enabled - inference_engine_destroy() is a no-op");
}

/// Load a BERT model (stub)
/// Returns 1 (error) when feature is not enabled
pub fn model_load_bert(_engine_handle: u64, _model_id: String, _model_path: String) -> u8 {
    tracing::warn!("Model server feature not enabled - model_load_bert() returning error");
    1
}

/// Load a GPT-2 model (stub)
/// Returns 1 (error) when feature is not enabled
pub fn model_load_gpt2(_engine_handle: u64, _model_id: String, _model_path: String) -> u8 {
    tracing::warn!("Model server feature not enabled - model_load_gpt2() returning error");
    1
}

/// Load models from configuration (stub)
/// Returns 1 (error) when feature is not enabled
pub fn model_load_from_config(_engine_handle: u64, _config_path: String) -> u8 {
    tracing::warn!("Model server feature not enabled - model_load_from_config() returning error");
    1
}

/// List loaded models (stub)
/// Returns empty JSON array when feature is not enabled
pub fn model_list_loaded(_engine_handle: u64) -> String {
    tracing::warn!("Model server feature not enabled - model_list_loaded() returning empty array");
    String::from("[]")
}

/// Unload a model (stub)
/// Returns 1 (error) when feature is not enabled
pub fn model_unload(_engine_handle: u64, _model_id: String) -> u8 {
    tracing::warn!("Model server feature not enabled - model_unload() returning error");
    1
}

/// Generate embedding (stub)
/// Returns empty string when feature is not enabled
pub fn inference_generate_embedding(_engine_handle: u64, _model_id: String, _text: String) -> String {
    tracing::warn!("Model server feature not enabled - inference_generate_embedding() returning empty string");
    String::new()
}

/// Generate completion (stub)
/// Returns empty string when feature is not enabled
pub fn inference_generate_completion(
    _engine_handle: u64,
    _model_id: String,
    _prompt: String,
    _max_tokens: u32,
    _temperature: f32,
) -> String {
    tracing::warn!("Model server feature not enabled - inference_generate_completion() returning empty string");
    String::new()
}

/// Count tokens (stub)
/// Returns 0 when feature is not enabled
pub fn inference_count_tokens(_engine_handle: u64, _model_id: String, _text: String) -> u32 {
    tracing::warn!("Model server feature not enabled - inference_count_tokens() returning 0");
    0
}
