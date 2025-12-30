/**
 * Hologram ONNX Runtime - Minimal TypeScript Wrapper
 *
 * Thin wrapper around hologram-onnx WASM module.
 * All ONNX logic is implemented in Rust - this just provides JS bindings.
 */

// @ts-ignore - WASM module generated at build time
import init, { init_onnx_webgpu, WasmOnnxModel, loadModelStreaming } from '@/lib/onnx/hologram_onnx';
import { loadModelStreamingCached, type ProgressCallback } from './cached-model-loader';

export interface OnnxTensor {
  data: Float32Array | BigInt64Array | Int32Array;
  shape: number[];
}

/**
 * Minimal ONNX Runtime wrapper
 * All logic is in Rust WASM - this just provides convenient JS API
 */
export class OnnxInference {
  private wasmInitialized = false;
  private model: WasmOnnxModel | null = null;

  constructor() {}

  /**
   * Initialize ONNX Runtime with WebGPU backend
   */
  async initialize(): Promise<void> {
    if (this.wasmInitialized) {
      return;
    }

    // Initialize WASM module
    await init();

    // Initialize WebGPU backend
    await init_onnx_webgpu();

    this.wasmInitialized = true;
  }

  /**
   * Load an ONNX model from URL (standard loading)
   *
   * Loads the entire model into WASM memory before parsing.
   * Use this for small models (<500MB).
   * For large models (>1GB), use loadModelStreaming() instead.
   */
  async loadModel(modelPath: string): Promise<void> {
    if (!this.wasmInitialized) {
      throw new Error('WASM not initialized. Call initialize() first.');
    }

    // Fetch model bytes
    const response = await fetch(modelPath);
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.statusText}`);
    }

    const modelBytes = new Uint8Array(await response.arrayBuffer());

    // Create and initialize ONNX model (all logic in Rust)
    this.model = await WasmOnnxModel.new(modelBytes);
    await this.model.initialize();
  }

  /**
   * Load a large ONNX model with streaming (for models >1GB)
   *
   * This method uses HTTP Range requests to stream large models
   * directly to WebGPU memory without loading the full model into
   * WASM memory. This enables loading models like SD Turbo U-Net
   * (1.7GB) that would otherwise crash.
   *
   * Requirements:
   * - Server must support HTTP Range headers
   * - Model must be accessible via URL (not a file path)
   *
   * @param modelUrl - Full URL to the ONNX model file
   */
  async loadModelStreaming(modelUrl: string): Promise<void> {
    if (!this.wasmInitialized) {
      throw new Error('WASM not initialized. Call initialize() first.');
    }

    console.log(`[OnnxInference] Loading model via streaming: ${modelUrl}`);

    // Call Rust streaming loader (handles everything)
    this.model = await loadModelStreaming(modelUrl);

    console.log('[OnnxInference] ✓ Model loaded via streaming');
  }

  /**
   * Load a large ONNX model with IndexedDB caching
   *
   * Same as loadModelStreaming() but caches models in IndexedDB for instant
   * loading on subsequent visits (40-125s -> ~1s).
   *
   * @param modelUrl - Full URL to the ONNX model file
   * @param cacheKey - Unique cache identifier (e.g., 'sd-turbo-unet')
   * @param version - Cache version for invalidation (default: '1.0')
   * @param onProgress - Optional progress callback
   */
  async loadModelStreamingCached(
    modelUrl: string,
    cacheKey: string,
    version: string = '1.0',
    onProgress?: ProgressCallback
  ): Promise<void> {
    if (!this.wasmInitialized) {
      throw new Error('WASM not initialized. Call initialize() first.');
    }

    console.log(`[OnnxInference] Loading model with caching: ${modelUrl}`);

    // Call cached streaming loader
    this.model = await loadModelStreamingCached(modelUrl, cacheKey, version, onProgress);

    console.log('[OnnxInference] ✓ Model loaded with caching');
  }

  /**
   * Check if ONNX inference is ready
   */
  isReady(): boolean {
    return this.wasmInitialized && this.model !== null && this.model.isReady();
  }

  /**
   * Run inference on the loaded model
   */
  async run(inputs: Record<string, OnnxTensor>): Promise<Record<string, OnnxTensor>> {
    if (!this.model) {
      throw new Error('No model loaded. Call loadModel() first.');
    }

    const outputs = await this.model.run(inputs);
    return outputs as Record<string, OnnxTensor>;
  }

  /**
   * Get model input names
   */
  getInputNames(): string[] {
    if (!this.model) return [];
    return this.model.getInputNames() as string[];
  }

  /**
   * Get model output names
   */
  getOutputNames(): string[] {
    if (!this.model) return [];
    return this.model.getOutputNames() as string[];
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.model = null;
    this.wasmInitialized = false;
  }
}

// Export singleton instance
export const onnxInference = new OnnxInference();
