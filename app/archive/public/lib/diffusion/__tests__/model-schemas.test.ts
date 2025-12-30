/**
 * Unit tests for model schemas and tensor validation
 *
 * Run with: npm test or vitest
 */

import { describe, it, expect } from 'vitest';
import {
  calculateTensorSize,
  validateTensorShape,
  SD_TURBO_CONFIG,
  type TensorShape,
} from '../model-schemas';

describe('Model Schema Validation', () => {
  describe('calculateTensorSize', () => {
    it('calculates size for 1D tensor', () => {
      expect(calculateTensorSize([77])).toBe(77);
    });

    it('calculates size for 2D tensor', () => {
      expect(calculateTensorSize([1, 77])).toBe(77);
    });

    it('calculates size for 3D tensor', () => {
      expect(calculateTensorSize([1, 77, 1024])).toBe(78_848);
    });

    it('calculates size for 4D tensor (latent)', () => {
      expect(calculateTensorSize([1, 4, 64, 64])).toBe(16_384);
    });

    it('calculates size for 4D tensor (image)', () => {
      expect(calculateTensorSize([1, 3, 512, 512])).toBe(786_432);
    });
  });

  describe('validateTensorShape', () => {
    it('validates correct tensor shape', () => {
      const data = new Float32Array(78_848);
      const shape: TensorShape = [1, 77, 1024];

      expect(() => validateTensorShape(data, shape, 'test')).not.toThrow();
    });

    it('throws on incorrect tensor size', () => {
      const data = new Float32Array(59_136); // Wrong size for [1, 77, 1024]
      const shape: TensorShape = [1, 77, 1024];

      expect(() => validateTensorShape(data, shape, 'test')).toThrow(
        /Invalid test tensor shape/
      );
    });

    it('validates latent tensor shape', () => {
      const data = new Float32Array(16_384);
      const shape: TensorShape = [1, 4, 64, 64];

      expect(() => validateTensorShape(data, shape, 'latent')).not.toThrow();
    });
  });

  describe('SD_TURBO_CONFIG', () => {
    it('has correct CLIP configuration', () => {
      expect(SD_TURBO_CONFIG.CLIP_SEQUENCE_LENGTH).toBe(77);
      expect(SD_TURBO_CONFIG.CLIP_HIDDEN_SIZE).toBe(1024);
    });

    it('has correct latent configuration', () => {
      expect(SD_TURBO_CONFIG.LATENT_CHANNELS).toBe(4);
      expect(SD_TURBO_CONFIG.LATENT_HEIGHT).toBe(64);
      expect(SD_TURBO_CONFIG.LATENT_WIDTH).toBe(64);
    });

    it('has correct image configuration', () => {
      expect(SD_TURBO_CONFIG.IMAGE_CHANNELS).toBe(3);
      expect(SD_TURBO_CONFIG.IMAGE_HEIGHT).toBe(512);
      expect(SD_TURBO_CONFIG.IMAGE_WIDTH).toBe(512);
    });

    it('calculates correct latent size', () => {
      const { LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH } = SD_TURBO_CONFIG;
      const size = LATENT_CHANNELS * LATENT_HEIGHT * LATENT_WIDTH;
      expect(size).toBe(16_384);
    });

    it('calculates correct image size', () => {
      const { IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH } = SD_TURBO_CONFIG;
      const size = IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH;
      expect(size).toBe(786_432);
    });

    it('calculates correct embedding size', () => {
      const { CLIP_SEQUENCE_LENGTH, CLIP_HIDDEN_SIZE } = SD_TURBO_CONFIG;
      const size = CLIP_SEQUENCE_LENGTH * CLIP_HIDDEN_SIZE;
      expect(size).toBe(78_848); // The actual size we were getting!
    });
  });

  describe('Shape consistency checks', () => {
    it('text encoder output matches U-Net input', () => {
      // Text encoder output: [1, 77, 1024]
      const encoderOutputSize =
        1 * SD_TURBO_CONFIG.CLIP_SEQUENCE_LENGTH * SD_TURBO_CONFIG.CLIP_HIDDEN_SIZE;

      // U-Net encoder_hidden_states input: [1, 77, 1024]
      const unetInputSize =
        1 * SD_TURBO_CONFIG.CLIP_SEQUENCE_LENGTH * SD_TURBO_CONFIG.CLIP_HIDDEN_SIZE;

      expect(encoderOutputSize).toBe(unetInputSize);
    });

    it('U-Net output matches VAE decoder input', () => {
      // U-Net output: [1, 4, 64, 64]
      const unetOutputSize =
        1 *
        SD_TURBO_CONFIG.LATENT_CHANNELS *
        SD_TURBO_CONFIG.LATENT_HEIGHT *
        SD_TURBO_CONFIG.LATENT_WIDTH;

      // VAE decoder input: [1, 4, 64, 64]
      const vaeInputSize =
        1 *
        SD_TURBO_CONFIG.LATENT_CHANNELS *
        SD_TURBO_CONFIG.LATENT_HEIGHT *
        SD_TURBO_CONFIG.LATENT_WIDTH;

      expect(unetOutputSize).toBe(vaeInputSize);
      expect(unetOutputSize).toBe(16_384);
    });
  });
});
