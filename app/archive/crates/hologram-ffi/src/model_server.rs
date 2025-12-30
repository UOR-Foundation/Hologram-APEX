//! Model Server FFI
//!
//! FFI bindings for Hologram Models, providing AI model inference
//! capabilities including text embeddings and completion generation.

use crate::handles::INFERENCE_ENGINE_REGISTRY;
use hologram_models::ModelsConfig;
use parking_lot::Mutex;
use std::sync::Arc;

// Type alias for InferenceEngine from hologram-models
type InferenceEngine = hologram_models::InferenceEngine;

// =============================================================================
// Inference Engine Management
// =============================================================================

/// Create a new inference engine
/// Returns engine handle (u64) or 0 on error
#[must_use]
pub fn inference_engine_create() -> u64 {
    match InferenceEngine::new() {
        Ok(engine) => {
            let handle = crate::handles::next_handle();
            INFERENCE_ENGINE_REGISTRY
                .lock()
                .insert(handle, Arc::new(Mutex::new(engine)));
            tracing::info!("Created inference engine: handle={}", handle);
            handle
        }
        Err(e) => {
            tracing::error!("Failed to create inference engine: {}", e);
            0
        }
    }
}

/// Destroy an inference engine and free its resources
pub fn inference_engine_destroy(engine_handle: u64) {
    if INFERENCE_ENGINE_REGISTRY.lock().remove(&engine_handle).is_some() {
        tracing::info!("Destroyed inference engine: handle={}", engine_handle);
    } else {
        tracing::warn!(
            "Attempted to destroy non-existent inference engine: handle={}",
            engine_handle
        );
    }
}

// =============================================================================
// Model Management
// =============================================================================

/// Load a BERT model for embeddings
/// Returns 0 on success, 1 on error
pub fn model_load_bert(engine_handle: u64, model_id: String, model_path: String) -> u8 {
    let registry = INFERENCE_ENGINE_REGISTRY.lock();
    let engine = match registry.get(&engine_handle) {
        Some(e) => e.clone(),
        None => {
            tracing::error!("Inference engine not found: handle={}", engine_handle);
            return 1;
        }
    };
    drop(registry);

    let mut engine = engine.lock();
    match engine.load_bert_model(&model_id, &model_path) {
        Ok(()) => {
            tracing::info!(
                "Loaded BERT model '{}' from '{}' in engine {}",
                model_id,
                model_path,
                engine_handle
            );
            0
        }
        Err(e) => {
            tracing::error!("Failed to load BERT model '{}': {}", model_id, e);
            1
        }
    }
}

/// Load a GPT-2 model for text generation
/// Returns 0 on success, 1 on error
pub fn model_load_gpt2(engine_handle: u64, model_id: String, model_path: String) -> u8 {
    let registry = INFERENCE_ENGINE_REGISTRY.lock();
    let engine = match registry.get(&engine_handle) {
        Some(e) => e.clone(),
        None => {
            tracing::error!("Inference engine not found: handle={}", engine_handle);
            return 1;
        }
    };
    drop(registry);

    let mut engine = engine.lock();
    match engine.load_gpt2_model(&model_id, &model_path) {
        Ok(()) => {
            tracing::info!(
                "Loaded GPT-2 model '{}' from '{}' in engine {}",
                model_id,
                model_path,
                engine_handle
            );
            0
        }
        Err(e) => {
            tracing::error!("Failed to load GPT-2 model '{}': {}", model_id, e);
            1
        }
    }
}

/// Load models from configuration file
/// Returns 0 on success, 1 on error
pub fn model_load_from_config(engine_handle: u64, config_path: String) -> u8 {
    let registry = INFERENCE_ENGINE_REGISTRY.lock();
    let engine = match registry.get(&engine_handle) {
        Some(e) => e.clone(),
        None => {
            tracing::error!("Inference engine not found: handle={}", engine_handle);
            return 1;
        }
    };
    drop(registry);

    // Load config
    let config = match ModelsConfig::load(&config_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to load config from '{}': {}", config_path, e);
            return 1;
        }
    };

    let mut engine = engine.lock();
    match engine.load_from_config(&config) {
        Ok(()) => {
            tracing::info!(
                "Loaded models from config '{}' in engine {}",
                config_path,
                engine_handle
            );
            0
        }
        Err(e) => {
            tracing::error!("Failed to load models from config: {}", e);
            1
        }
    }
}

/// List all loaded models
/// Returns JSON array of model IDs, e.g. ["bert-base", "gpt2"]
pub fn model_list_loaded(engine_handle: u64) -> String {
    let registry = INFERENCE_ENGINE_REGISTRY.lock();
    let engine = match registry.get(&engine_handle) {
        Some(e) => e.clone(),
        None => {
            tracing::error!("Inference engine not found: handle={}", engine_handle);
            return String::from("[]");
        }
    };
    drop(registry);

    let engine = engine.lock();
    let models = engine.list_models();

    match serde_json::to_string(&models) {
        Ok(json) => json,
        Err(e) => {
            tracing::error!("Failed to serialize model list: {}", e);
            String::from("[]")
        }
    }
}

/// Unload a specific model
/// Returns 0 on success, 1 on error
pub fn model_unload(_engine_handle: u64, _model_id: String) -> u8 {
    // TODO: Implement model unloading in InferenceEngine
    tracing::warn!("Model unloading not yet implemented");
    1
}

// =============================================================================
// Inference Operations
// =============================================================================

/// Generate embedding for text
/// Returns JSON array of f32 embedding values, empty string on error
pub fn inference_generate_embedding(engine_handle: u64, model_id: String, text: String) -> String {
    let registry = INFERENCE_ENGINE_REGISTRY.lock();
    let engine = match registry.get(&engine_handle) {
        Some(e) => e.clone(),
        None => {
            tracing::error!("Inference engine not found: handle={}", engine_handle);
            return String::new();
        }
    };
    drop(registry);

    let engine = engine.lock();

    // Check if model exists
    if !engine.has_model(&model_id) {
        tracing::error!("Model not found: {}", model_id);
        return String::new();
    }

    // Generate embedding
    match engine.generate_embedding_with_model(&model_id, &text) {
        Ok(embedding) => match serde_json::to_string(&embedding) {
            Ok(json) => {
                tracing::debug!(
                    "Generated embedding with model '{}': {} dimensions",
                    model_id,
                    embedding.len()
                );
                json
            }
            Err(e) => {
                tracing::error!("Failed to serialize embedding: {}", e);
                String::new()
            }
        },
        Err(e) => {
            tracing::error!("Failed to generate embedding: {}", e);
            String::new()
        }
    }
}

/// Generate text completion
/// Returns generated completion text, empty string on error
pub fn inference_generate_completion(
    engine_handle: u64,
    model_id: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
) -> String {
    let registry = INFERENCE_ENGINE_REGISTRY.lock();
    let engine = match registry.get(&engine_handle) {
        Some(e) => e.clone(),
        None => {
            tracing::error!("Inference engine not found: handle={}", engine_handle);
            return String::new();
        }
    };
    drop(registry);

    let engine = engine.lock();

    // Check if model exists
    if !engine.has_model(&model_id) {
        tracing::error!("Model not found: {}", model_id);
        return String::new();
    }

    // Generate completion
    match engine.generate_completion_with_model(&model_id, &prompt, max_tokens, temperature) {
        Ok(completion) => {
            tracing::debug!(
                "Generated completion with model '{}': {} characters",
                model_id,
                completion.len()
            );
            completion
        }
        Err(e) => {
            tracing::error!("Failed to generate completion: {}", e);
            String::new()
        }
    }
}

/// Count tokens in text
/// Returns number of tokens (0 on error)
pub fn inference_count_tokens(engine_handle: u64, model_id: String, text: String) -> u32 {
    let registry = INFERENCE_ENGINE_REGISTRY.lock();
    let engine = match registry.get(&engine_handle) {
        Some(e) => e.clone(),
        None => {
            tracing::error!("Inference engine not found: handle={}", engine_handle);
            return 0;
        }
    };
    drop(registry);

    let engine = engine.lock();

    // Check if model exists
    if !engine.has_model(&model_id) {
        tracing::warn!("Model not found: {}, using character count fallback", model_id);
        return text.len() as u32;
    }

    // Count tokens
    match engine.count_tokens_with_model(&model_id, &text) {
        Ok(count) => count as u32,
        Err(e) => {
            tracing::error!("Failed to count tokens: {}", e);
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_engine_lifecycle() {
        let engine_handle = inference_engine_create();
        assert_ne!(engine_handle, 0);

        inference_engine_destroy(engine_handle);
    }

    #[test]
    fn test_model_list_loaded() {
        let engine_handle = inference_engine_create();
        let models = model_list_loaded(engine_handle);
        assert_eq!(models, "[]");

        inference_engine_destroy(engine_handle);
    }
}
