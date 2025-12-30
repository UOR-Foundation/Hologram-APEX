/**
 * Type-safe schemas for SD Turbo ONNX models
 *
 * These schemas are validated against actual ONNX model metadata
 * and enforced at compile time to prevent shape mismatches.
 */

/**
 * SD Turbo Text Encoder (CLIP-L)
 * Model: stabilityai/sd-turbo text_encoder
 */
export interface TextEncoderSchema {
  inputs: {
    input_ids: {
      shape: [batch: 1, sequence: 77];
      dtype: 'int32';
    };
  };
  outputs: {
    last_hidden_state: {
      shape: [batch: 1, sequence: 77, hidden: 1024]; // CLIP-L uses 1024 dims
      dtype: 'float32';
    };
    pooler_output: {
      shape: [batch: 1, hidden: 1024];
      dtype: 'float32';
    };
  };
}

/**
 * SD Turbo U-Net
 * Model: stabilityai/sd-turbo unet
 */
export interface UNetSchema {
  inputs: {
    sample: {
      shape: [batch: 1, channels: 4, height: 64, width: 64];
      dtype: 'float32';
    };
    timestep: {
      shape: [batch: 1];
      dtype: 'float32';
    };
    encoder_hidden_states: {
      shape: [batch: 1, sequence: 77, hidden: 1024]; // Must match text encoder output
      dtype: 'float32';
    };
  };
  outputs: {
    out_sample: {
      shape: [batch: 1, channels: 4, height: 64, width: 64];
      dtype: 'float32';
    };
  };
}

/**
 * SD Turbo VAE Decoder
 * Model: stabilityai/sd-turbo vae_decoder
 */
export interface VAEDecoderSchema {
  inputs: {
    latent_sample: {
      shape: [batch: 1, channels: 4, height: 64, width: 64];
      dtype: 'float32';
    };
  };
  outputs: {
    sample: {
      shape: [batch: 1, channels: 3, height: 512, width: 512];
      dtype: 'float32';
    };
  };
}

/**
 * Tensor shape type for compile-time validation
 */
export type TensorShape = readonly number[];

/**
 * Calculate expected tensor size from shape
 */
export function calculateTensorSize(shape: TensorShape): number {
  return shape.reduce((acc, dim) => acc * dim, 1);
}

/**
 * Validate tensor data matches expected shape
 */
export function validateTensorShape(
  data: Float32Array | Int32Array,
  expectedShape: TensorShape,
  tensorName: string
): void {
  const expectedSize = calculateTensorSize(expectedShape);
  if (data.length !== expectedSize) {
    throw new Error(
      `Invalid ${tensorName} tensor shape: ` +
        `expected ${expectedSize} elements for shape [${expectedShape.join(', ')}], ` +
        `but got ${data.length} elements`
    );
  }
}

/**
 * SD Turbo model configuration constants
 */
export const SD_TURBO_CONFIG = {
  // Text encoder
  CLIP_SEQUENCE_LENGTH: 77,
  CLIP_HIDDEN_SIZE: 1024, // CLIP-L

  // Latent space
  LATENT_CHANNELS: 4,
  LATENT_HEIGHT: 64,
  LATENT_WIDTH: 64,

  // Output image
  IMAGE_CHANNELS: 3,
  IMAGE_HEIGHT: 512,
  IMAGE_WIDTH: 512,

  // Diffusion
  NUM_INFERENCE_STEPS: 1, // SD Turbo is single-step
  TIMESTEP_START: 999,
} as const;
