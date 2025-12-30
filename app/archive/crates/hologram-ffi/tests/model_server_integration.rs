//! Integration tests for Model Server FFI
//!
//! Tests the full stack: library functions → FFI → inference engine

#![cfg(feature = "model-server")]

use hologram_ffi::{
    clear_all_registries, inference_count_tokens, inference_engine_create, inference_engine_destroy,
    inference_generate_completion, inference_generate_embedding, model_list_loaded,
};

#[test]
fn test_inference_engine_lifecycle() {
    clear_all_registries();

    // Create engine
    let engine_id = inference_engine_create();
    assert_ne!(engine_id, 0, "Engine creation should return non-zero handle");

    // Destroy engine
    inference_engine_destroy(engine_id);

    // Try to use destroyed engine (should return empty/error)
    let models = model_list_loaded(engine_id);
    assert_eq!(models, "[]", "Destroyed engine should return empty list");
}

#[test]
fn test_model_list_empty() {
    clear_all_registries();

    let engine_id = inference_engine_create();
    assert_ne!(engine_id, 0);

    // No models loaded initially
    let models = model_list_loaded(engine_id);
    assert_eq!(models, "[]");

    inference_engine_destroy(engine_id);
}

#[test]
fn test_embedding_generation_fallback() {
    clear_all_registries();

    let engine_id = inference_engine_create();
    assert_ne!(engine_id, 0);

    // Try to generate embedding with non-existent model
    // Should return empty string (error)
    let embedding_json =
        inference_generate_embedding(engine_id, "non-existent-model".to_string(), "Hello, world!".to_string());

    assert!(
        embedding_json.is_empty(),
        "Non-existent model should return empty string"
    );

    inference_engine_destroy(engine_id);
}

#[test]
fn test_completion_generation_fallback() {
    clear_all_registries();

    let engine_id = inference_engine_create();
    assert_ne!(engine_id, 0);

    // Try to generate completion with non-existent model
    let completion = inference_generate_completion(
        engine_id,
        "non-existent-model".to_string(),
        "Once upon a time".to_string(),
        50,
        1.0,
    );

    assert!(completion.is_empty(), "Non-existent model should return empty string");

    inference_engine_destroy(engine_id);
}

#[test]
fn test_token_counting_fallback() {
    clear_all_registries();

    let engine_id = inference_engine_create();
    assert_ne!(engine_id, 0);

    // Count tokens with non-existent model (should use character count fallback)
    let text = "Hello, world!";
    let token_count = inference_count_tokens(engine_id, "non-existent-model".to_string(), text.to_string());

    // Fallback returns character count
    assert_eq!(token_count, text.len() as u32, "Fallback should return character count");

    inference_engine_destroy(engine_id);
}

#[test]
fn test_invalid_engine_handle() {
    clear_all_registries();

    // Try to use an invalid engine handle
    let models = model_list_loaded(99999);
    assert_eq!(models, "[]", "Invalid handle should return empty list");

    let embedding = inference_generate_embedding(99999, "model".to_string(), "text".to_string());
    assert!(embedding.is_empty(), "Invalid handle should return empty string");

    let completion = inference_generate_completion(99999, "model".to_string(), "prompt".to_string(), 10, 1.0);
    assert!(completion.is_empty(), "Invalid handle should return empty string");

    let tokens = inference_count_tokens(99999, "model".to_string(), "text".to_string());
    assert_eq!(tokens, 0, "Invalid handle should return 0 tokens");
}

#[test]
fn test_multiple_engines() {
    clear_all_registries();

    // Create multiple engines
    let engine1 = inference_engine_create();
    let engine2 = inference_engine_create();
    let engine3 = inference_engine_create();

    assert_ne!(engine1, 0);
    assert_ne!(engine2, 0);
    assert_ne!(engine3, 0);

    // All should have unique IDs
    assert_ne!(engine1, engine2);
    assert_ne!(engine2, engine3);
    assert_ne!(engine1, engine3);

    // Each should have empty model list
    assert_eq!(model_list_loaded(engine1), "[]");
    assert_eq!(model_list_loaded(engine2), "[]");
    assert_eq!(model_list_loaded(engine3), "[]");

    // Destroy all
    inference_engine_destroy(engine1);
    inference_engine_destroy(engine2);
    inference_engine_destroy(engine3);
}

#[test]
fn test_concurrent_access() {
    use std::thread;

    clear_all_registries();

    let engine_id = inference_engine_create();
    assert_ne!(engine_id, 0);

    // Spawn multiple threads trying to use the same engine
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let engine_id = engine_id;
            thread::spawn(move || {
                let text = format!("Thread {} message", i);
                let _tokens = inference_count_tokens(engine_id, "model".to_string(), text);
                let _models = model_list_loaded(engine_id);
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    inference_engine_destroy(engine_id);
}

#[test]
fn test_json_parsing() {
    clear_all_registries();

    let engine_id = inference_engine_create();
    assert_ne!(engine_id, 0);

    // Models list should be valid JSON
    let models_json = model_list_loaded(engine_id);
    let models: Result<Vec<String>, _> = serde_json::from_str(&models_json);
    assert!(models.is_ok(), "Models list should be valid JSON");
    assert_eq!(models.unwrap().len(), 0, "Should have no models");

    inference_engine_destroy(engine_id);
}
