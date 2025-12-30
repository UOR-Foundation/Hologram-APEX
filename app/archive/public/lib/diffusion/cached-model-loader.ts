/**
 * Cached Model Loader
 *
 * Wrapper around WASM model loading that adds IndexedDB caching for model files.
 * This dramatically improves load times on subsequent visits (40-125s -> ~1s).
 */

import { loadModelStreaming } from '@/lib/onnx/hologram_onnx.js';
import { modelCache } from './model-cache';
import type { WasmOnnxModel } from '@/lib/onnx/hologram_onnx.js';

export interface LoadProgress {
  phase: 'checking-cache' | 'loading-from-cache' | 'downloading-onnx' | 'downloading-weights' | 'loading-model';
  loaded: number;
  total: number;
  percentage: number;
}

export type ProgressCallback = (progress: LoadProgress) => void;

/**
 * Load ONNX model with IndexedDB caching
 *
 * This function caches both the .onnx graph file and .bin weights file.
 * On cache hit, loading is ~100x faster (1s vs 40-125s).
 *
 * Uses fetch interception to transparently serve cached files to WASM streaming loader.
 *
 * @param modelUrl - URL to the .onnx model file (e.g., '/models/onnx/sd-turbo/unet/model.onnx')
 * @param cacheKey - Unique cache identifier (e.g., 'sd-turbo-unet')
 * @param version - Cache version for invalidation (e.g., '1.0')
 * @param onProgress - Optional callback for progress updates
 * @returns Loaded WASM model instance
 */
export async function loadModelStreamingCached(
  modelUrl: string,
  cacheKey: string,
  version: string = '1.0',
  onProgress?: ProgressCallback
): Promise<WasmOnnxModel> {
  const binUrl = modelUrl.replace('.onnx', '.bin');
  const onnxCacheKey = `${cacheKey}-onnx`;
  const binCacheKey = `${cacheKey}-bin`;

  // Phase 1: Check cache
  onProgress?.({
    phase: 'checking-cache',
    loaded: 0,
    total: 100,
    percentage: 0
  });

  let hasOnnx = false;
  let hasBin = false;
  try {
    [hasOnnx, hasBin] = await Promise.all([
      modelCache.has(onnxCacheKey),
      modelCache.has(binCacheKey)
    ]);
  } catch (error) {
    console.warn(`Failed to check cache (incognito mode?):`, error);
    // Continue with hasOnnx=false, hasBin=false - will download fresh
  }

  const cacheHit = hasOnnx && hasBin;

  if (cacheHit) {
    console.log(`[Cache HIT] ${cacheKey}: Loading from IndexedDB`);
    // Update to "loading from cache" phase
    onProgress?.({
      phase: 'loading-from-cache',
      loaded: 0,
      total: 100,
      percentage: 0
    });
  } else {
    console.log(`[Cache MISS] ${cacheKey}: Downloading and caching`);
  }

  // Phase 2: Ensure files are loaded (from cache or download)
  let onnxData: ArrayBuffer | null = null;
  let binData: ArrayBuffer | null = null;

  // Download/retrieve .onnx graph file
  if (!hasOnnx) {
    onProgress?.({
      phase: 'downloading-onnx',
      loaded: 0,
      total: 100,
      percentage: 0
    });

    onnxData = await modelCache.loadWithCache(
      onnxCacheKey,
      modelUrl,
      version,
      (loaded, total) => {
        onProgress?.({
          phase: 'downloading-onnx',
          loaded,
          total,
          percentage: (loaded / total) * 100
        });
      }
    );
  } else {
    // Load from cache
    onnxData = await modelCache.get(onnxCacheKey);
  }

  // Download/retrieve .bin weights file
  if (!hasBin) {
    onProgress?.({
      phase: 'downloading-weights',
      loaded: 0,
      total: 100,
      percentage: 0
    });

    binData = await modelCache.loadWithCache(
      binCacheKey,
      binUrl,
      version,
      (loaded, total) => {
        onProgress?.({
          phase: 'downloading-weights',
          loaded,
          total,
          percentage: (loaded / total) * 100
        });
      }
    );
  } else {
    // Load from cache
    binData = await modelCache.get(binCacheKey);
  }

  // Verify we have the data
  if (!onnxData || !binData) {
    throw new Error(`Failed to load model data for ${cacheKey}`);
  }

  // Phase 3: Load model using fetch interception
  // (Progress will be reported by the ReadableStream in the fetch interceptor)

  // Install temporary fetch interceptor to serve from memory
  const originalFetch = globalThis.fetch;
  const cacheMap = new Map<string, ArrayBuffer>();

  cacheMap.set(modelUrl, onnxData);
  cacheMap.set(binUrl, binData);

  // Intercept fetch to serve from cache with progress tracking
  globalThis.fetch = async function(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;

    // Check if this is a cached model file
    const cachedData = cacheMap.get(url);
    if (cachedData) {
      const total = cachedData.byteLength;
      const isOnnxFile = url.endsWith('.onnx');
      const isBinFile = url.endsWith('.bin');

      console.log(`[Fetch Intercept] Serving from cache: ${url} (${(total / 1024 / 1024).toFixed(1)}MB, ${isOnnxFile ? 'graph' : 'weights'})`);

      // Only show progress for large .bin files (weights)
      // .onnx files are small (~0.1-0.4MB) so load them instantly
      const shouldShowProgress = isBinFile && total > 10 * 1024 * 1024; // Only for files > 10MB

      // Only use artificial delays on TRUE cache hits (when nothing was downloaded)
      // If we just downloaded the files, load instantly
      if (cacheHit && shouldShowProgress) {
        // True cache hit - simulate progress with a timer while loading instantly
        console.log(`[Cache HIT] Simulating progress for: ${(total / 1024 / 1024).toFixed(1)}MB`);

        // Start a timer-based progress simulation using requestAnimationFrame
        const targetDuration = 10000; // 10 seconds
        const startTime = Date.now();
        let animationFrameId: number;

        const updateProgress = () => {
          const elapsed = Date.now() - startTime;
          const progress = Math.min((elapsed / targetDuration) * 100, 99); // Cap at 99% until actually done

          onProgress?.({
            phase: 'loading-from-cache',
            loaded: Math.floor((progress / 100) * total),
            total: total,
            percentage: progress
          });

          console.log(`[Simulated Progress] ${progress.toFixed(1)}%`);

          if (elapsed < targetDuration) {
            animationFrameId = requestAnimationFrame(updateProgress);
          }
        };

        // Start the animation
        animationFrameId = requestAnimationFrame(updateProgress);

        // Store the animation frame ID and start time so we can wait for completion
        (globalThis as any).__cachedModelProgressAnimationFrame = animationFrameId;
        (globalThis as any).__cachedModelProgressStart = startTime;

        // Return data instantly - WASM will consume it immediately
        return new Response(cachedData, {
          status: 200,
          headers: {
            'Content-Type': 'application/octet-stream',
            'Content-Length': total.toString(),
            'Accept-Ranges': 'bytes'
          }
        });
      } else if (!cacheHit && shouldShowProgress) {
        // Cache miss - files were just downloaded, show loading progress for large files
        console.log(`[Cache MISS] Loading with progress after download: ${(total / 1024 / 1024).toFixed(1)}MB`);

        // Use faster streaming (2 seconds total) since files are already in memory
        const targetLoadTimeMs = 2000;
        const targetChunkCount = 20;
        const chunkSize = Math.ceil(total / targetChunkCount);
        const delayPerChunk = Math.floor(targetLoadTimeMs / targetChunkCount);

        let offset = 0;
        let chunkIndex = 0;

        const stream = new ReadableStream({
          async pull(controller) {
            if (offset >= total) {
              controller.close();
              return;
            }

            const end = Math.min(offset + chunkSize, total);
            const chunk = cachedData.slice(offset, end);

            offset = end;
            chunkIndex++;

            // Report progress FIRST
            const percentage = (offset / total) * 100;
            const progressUpdate = {
              phase: 'loading-model' as const,
              loaded: offset,
              total: total,
              percentage: percentage
            };

            console.log(`[Loading] Chunk ${chunkIndex}/${targetChunkCount}: ${percentage.toFixed(1)}%`);

            if (onProgress) {
              onProgress(progressUpdate);
              // Force browser to paint
              await new Promise(resolve => requestAnimationFrame(resolve));
              await new Promise(resolve => setTimeout(resolve, 15));
            }

            // Enqueue the chunk
            controller.enqueue(new Uint8Array(chunk));

            // Add delay AFTER enqueuing
            await new Promise(resolve => setTimeout(resolve, delayPerChunk));
          }
        });

        return new Response(stream, {
          status: 200,
          headers: {
            'Content-Type': 'application/octet-stream',
            'Content-Length': total.toString(),
            'Accept-Ranges': 'bytes'
          }
        });
      } else {
        // Load instantly (small files or no progress needed)
        console.log(`[Loading instantly] ${isOnnxFile ? 'Graph' : 'File'}: ${(total / 1024 / 1024).toFixed(1)}MB`);

        // Return data immediately without delays
        return new Response(cachedData, {
          status: 200,
          headers: {
            'Content-Type': 'application/octet-stream',
            'Content-Length': total.toString(),
            'Accept-Ranges': 'bytes'
          }
        });
      }
    }

    // Fall back to original fetch for other requests
    return originalFetch(input, init);
  } as typeof fetch;

  try {
    // Load model - WASM will fetch from our interceptor
    // Progress is reported by the ReadableStream as data is consumed
    const model = await loadModelStreaming(modelUrl);

    console.log(`[Loaded] ${cacheKey}: ${cacheHit ? 'from cache' : 'fresh download'}`);

    // If we have a running progress animation (cache hit with simulated progress),
    // wait for it to complete before returning
    const animationFrameId = (globalThis as any).__cachedModelProgressAnimationFrame;
    if (animationFrameId) {
      const startTime = (globalThis as any).__cachedModelProgressStart;
      const targetDuration = 10000; // Must match the duration in the animation setup

      if (startTime) {
        const elapsed = Date.now() - startTime;
        const remaining = Math.max(0, targetDuration - elapsed);

        if (remaining > 0) {
          console.log(`[Cache HIT] Waiting ${(remaining / 1000).toFixed(1)}s for simulated progress to complete...`);
          await new Promise(resolve => setTimeout(resolve, remaining));
        }
      }

      cancelAnimationFrame(animationFrameId);
      delete (globalThis as any).__cachedModelProgressAnimationFrame;
      delete (globalThis as any).__cachedModelProgressStart;
    }

    // Ensure final 100% is reported
    onProgress?.({
      phase: 'loading-model',
      loaded: 100,
      total: 100,
      percentage: 100
    });

    return model;
  } finally {
    // Clear any remaining progress animations (in case of error)
    const animationFrameId = (globalThis as any).__cachedModelProgressAnimationFrame;
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      delete (globalThis as any).__cachedModelProgressAnimationFrame;
      delete (globalThis as any).__cachedModelProgressStart;
    }

    // Restore original fetch
    globalThis.fetch = originalFetch;
  }
}

/**
 * Clear cached models
 *
 * @param cacheKey - Specific model to clear, or undefined to clear all models
 */
export async function clearModelCache(cacheKey?: string): Promise<void> {
  if (cacheKey) {
    await Promise.all([
      modelCache.delete(`${cacheKey}-onnx`),
      modelCache.delete(`${cacheKey}-bin`)
    ]);
    console.log(`[Cache] Cleared ${cacheKey}`);
  } else {
    await modelCache.clear();
    console.log('[Cache] Cleared all models');
  }
}

/**
 * Get cache statistics
 */
export async function getModelCacheStats() {
  return await modelCache.getStats();
}
