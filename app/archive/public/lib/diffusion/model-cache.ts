/**
 * IndexedDB Model Cache for Hologram
 *
 * Caches large model weights (safetensors) in browser storage for instant loading.
 * Based on the streaming & caching strategy from docs/wasm/MODEL_STREAMING_QUANTIZATION.md
 */

interface CachedModel {
  id: string;
  data: ArrayBuffer;
  timestamp: number;
  size: number;
  version: string;
}

interface CacheStats {
  totalSize: number;
  modelCount: number;
  oldestTimestamp: number;
  newestTimestamp: number;
}

export class ModelCache {
  private db: IDBDatabase | null = null;
  private readonly DB_NAME = 'hologram-model-cache';
  private readonly DB_VERSION = 1;
  private readonly STORE_NAME = 'models';

  /**
   * Initialize the IndexedDB database
   * Note: May fail in incognito mode or with restrictive storage settings
   */
  async init(): Promise<void> {
    if (this.db) return; // Already initialized

    return new Promise((resolve) => {
      try {
        const request = indexedDB.open(this.DB_NAME, this.DB_VERSION);

        request.onerror = () => {
          console.warn('Failed to open IndexedDB (incognito mode?):', request.error);
          // Don't reject - allow fallback to direct download
          resolve();
        };

        request.onsuccess = () => {
          this.db = request.result;
          console.log('IndexedDB initialized successfully');
          resolve();
        };

        request.onupgradeneeded = (event) => {
          const db = (event.target as IDBOpenDBRequest).result;

          // Create object store if it doesn't exist
          if (!db.objectStoreNames.contains(this.STORE_NAME)) {
            const store = db.createObjectStore(this.STORE_NAME, { keyPath: 'id' });

            // Create indexes for efficient queries
            store.createIndex('timestamp', 'timestamp', { unique: false });
            store.createIndex('size', 'size', { unique: false });
            store.createIndex('version', 'version', { unique: false });

            console.log('Created IndexedDB object store and indexes');
          }
        };
      } catch (error) {
        console.warn('IndexedDB not available:', error);
        resolve(); // Allow fallback to direct download
      }
    });
  }

  /**
   * Get a model from cache
   * @param key Unique identifier for the model (e.g., "sdxs-unet", "sdxs-vae")
   * @returns Model data as ArrayBuffer, or null if not cached or error
   */
  async get(key: string): Promise<ArrayBuffer | null> {
    if (!this.db) await this.init();
    if (!this.db) return null; // IndexedDB not available

    return new Promise((resolve) => {
      try {
        const transaction = this.db!.transaction([this.STORE_NAME], 'readonly');
        const store = transaction.objectStore(this.STORE_NAME);
        const request = store.get(key);

        request.onsuccess = () => {
          if (request.result) {
            const sizeMB = (request.result.size / 1024 / 1024).toFixed(2);
            console.log(`Cache HIT: ${key} (${sizeMB} MB)`);
            resolve(request.result.data);
          } else {
            console.log(`Cache MISS: ${key}`);
            resolve(null);
          }
        };

        request.onerror = () => {
          console.warn(`Cache read error for ${key} (quota exceeded?):`, request.error);
          // Don't reject - just return null to trigger download
          resolve(null);
        };
      } catch (error) {
        console.warn(`Cache read failed for ${key}:`, error);
        resolve(null);
      }
    });
  }

  /**
   * Store a model in cache
   * @param key Unique identifier for the model
   * @param data Model data as ArrayBuffer
   * @param version Optional version string for cache invalidation
   */
  async put(key: string, data: ArrayBuffer, version: string = '1.0'): Promise<void> {
    if (!this.db) await this.init();
    if (!this.db) {
      console.warn(`Cannot cache ${key}: IndexedDB not available (incognito mode?)`);
      return; // Silently fail - cache not available
    }

    return new Promise((resolve) => {
      try {
        const transaction = this.db!.transaction([this.STORE_NAME], 'readwrite');
        const store = transaction.objectStore(this.STORE_NAME);

        const cachedModel: CachedModel = {
          id: key,
          data,
          timestamp: Date.now(),
          size: data.byteLength,
          version,
        };

        const request = store.put(cachedModel);

        request.onsuccess = () => {
          const sizeMB = (data.byteLength / 1024 / 1024).toFixed(2);
          console.log(`Cached: ${key} (${sizeMB} MB, version: ${version})`);
          resolve();
        };

        request.onerror = () => {
          console.warn(`Cache write error for ${key}:`, request.error);
          resolve(); // Don't reject - just skip caching
        };
      } catch (error) {
        console.warn(`Cache write failed for ${key}:`, error);
        resolve(); // Don't reject - just skip caching
      }
    });
  }

  /**
   * Check if a model exists in cache
   */
  async has(key: string): Promise<boolean> {
    const data = await this.get(key);
    return data !== null;
  }

  /**
   * Delete a specific model from cache
   */
  async delete(key: string): Promise<void> {
    if (!this.db) await this.init();
    if (!this.db) {
      console.warn(`Cannot delete ${key}: IndexedDB not available`);
      return;
    }

    return new Promise((resolve) => {
      try {
        const transaction = this.db!.transaction([this.STORE_NAME], 'readwrite');
        const store = transaction.objectStore(this.STORE_NAME);
        const request = store.delete(key);

        request.onsuccess = () => {
          console.log(`Deleted from cache: ${key}`);
          resolve();
        };

        request.onerror = () => {
          console.warn(`Failed to delete ${key}:`, request.error);
          resolve();
        };
      } catch (error) {
        console.warn(`Failed to delete ${key}:`, error);
        resolve();
      }
    });
  }

  /**
   * Clear all cached models
   */
  async clear(): Promise<void> {
    if (!this.db) await this.init();
    if (!this.db) {
      console.warn('Cannot clear cache: IndexedDB not available');
      return;
    }

    return new Promise((resolve) => {
      try {
        const transaction = this.db!.transaction([this.STORE_NAME], 'readwrite');
        const store = transaction.objectStore(this.STORE_NAME);
        const request = store.clear();

        request.onsuccess = () => {
          console.log('Cache cleared');
          resolve();
        };

        request.onerror = () => {
          console.warn('Failed to clear cache:', request.error);
          resolve();
        };
      } catch (error) {
        console.warn('Failed to clear cache:', error);
        resolve();
      }
    });
  }

  /**
   * Get cache statistics
   */
  async getStats(): Promise<CacheStats> {
    if (!this.db) await this.init();

    try {
      const models = await this.listAll();

      if (models.length === 0) {
        return {
          totalSize: 0,
          modelCount: 0,
          oldestTimestamp: 0,
          newestTimestamp: 0,
        };
      }

      const totalSize = models.reduce((sum, m) => sum + m.size, 0);
      const timestamps = models.map(m => m.timestamp);

      return {
        totalSize,
        modelCount: models.length,
        oldestTimestamp: Math.min(...timestamps),
        newestTimestamp: Math.max(...timestamps),
      };
    } catch (error) {
      console.warn('Failed to get cache stats (storage quota exceeded?):', error);
      // Return empty stats
      return {
        totalSize: 0,
        modelCount: 0,
        oldestTimestamp: 0,
        newestTimestamp: 0,
      };
    }
  }

  /**
   * Evict oldest models until cache is under maxSizeMB
   * @param maxSizeMB Maximum cache size in megabytes
   */
  async evictOldest(maxSizeMB: number): Promise<void> {
    if (!this.db) await this.init();

    const models = await this.listAll();

    // Calculate total size
    const totalSize = models.reduce((sum, m) => sum + m.size, 0);
    const totalSizeMB = totalSize / 1024 / 1024;

    if (totalSizeMB <= maxSizeMB) {
      console.log(`Cache size (${totalSizeMB.toFixed(2)} MB) is under limit (${maxSizeMB} MB)`);
      return;
    }

    console.log(`Cache size (${totalSizeMB.toFixed(2)} MB) exceeds limit (${maxSizeMB} MB), evicting...`);

    // Sort by timestamp (oldest first)
    models.sort((a, b) => a.timestamp - b.timestamp);

    // Evict oldest until under limit
    let currentSize = totalSizeMB;
    for (const model of models) {
      if (currentSize <= maxSizeMB) break;

      await this.delete(model.id);
      currentSize -= model.size / 1024 / 1024;
      console.log(`Evicted: ${model.id} (${(model.size / 1024 / 1024).toFixed(2)} MB)`);
    }

    console.log(`Cache size after eviction: ${currentSize.toFixed(2)} MB`);
  }

  /**
   * List all cached models
   */
  private async listAll(): Promise<Array<{ id: string; timestamp: number; size: number; version: string }>> {
    if (!this.db) {
      return [];
    }

    return new Promise((resolve) => {
      try {
        const transaction = this.db!.transaction([this.STORE_NAME], 'readonly');
        const store = transaction.objectStore(this.STORE_NAME);
        const request = store.getAll();

        request.onsuccess = () => {
          const models = request.result.map((m: CachedModel) => ({
            id: m.id,
            timestamp: m.timestamp,
            size: m.size,
            version: m.version,
          }));
          resolve(models);
        };

        request.onerror = () => {
          console.warn('Failed to list cached models:', request.error);
          resolve([]);
        };
      } catch (error) {
        console.warn('Failed to list cached models:', error);
        resolve([]);
      }
    });
  }

  /**
   * Load model with automatic caching
   * Downloads from URL if not cached, then caches for future use
   *
   * @param key Unique identifier for caching
   * @param url URL to download from if not cached
   * @param version Model version for cache invalidation
   * @param onProgress Optional progress callback (loaded bytes, total bytes)
   */
  async loadWithCache(
    key: string,
    url: string,
    version: string = '1.0',
    onProgress?: (loaded: number, total: number) => void
  ): Promise<ArrayBuffer> {
    // Check cache first (may fail in incognito mode or with large files)
    try {
      const cached = await this.get(key);
      if (cached) {
        return cached;
      }
    } catch (error) {
      console.warn(`Cache read failed for ${key}, proceeding with download:`, error);
      // Continue with download
    }

    console.log(`Downloading ${key} from ${url}...`);

    // Download with progress tracking
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to download ${key}: ${response.statusText}`);
    }

    const contentLength = response.headers.get('Content-Length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;

    if (total > 0) {
      console.log(`[Download] ${key}: Content-Length = ${(total / 1024 / 1024).toFixed(1)}MB`);
    } else {
      console.warn(`[Download] ${key}: No Content-Length header - progress may be inaccurate`);
    }

    // Read with progress tracking
    const reader = response.body!.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    let lastProgressUpdate = 0;

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      chunks.push(value);
      loaded += value.length;

      // Throttle progress updates to avoid overwhelming React (max 10 updates/sec)
      const now = Date.now();
      if (onProgress && (now - lastProgressUpdate > 100 || done)) {
        // Always use actual total if available, otherwise loaded becomes the total
        onProgress(loaded, total > 0 ? total : loaded);
        lastProgressUpdate = now;

        if (total > 0) {
          console.log(`[Download Progress] ${key}: ${(loaded / 1024 / 1024).toFixed(1)}MB / ${(total / 1024 / 1024).toFixed(1)}MB (${((loaded / total) * 100).toFixed(1)}%)`);
        } else {
          console.log(`[Download Progress] ${key}: ${(loaded / 1024 / 1024).toFixed(1)}MB (unknown total)`);
        }
      }
    }

    console.log(`[Download Complete] ${key}: ${(loaded / 1024 / 1024).toFixed(1)}MB`);

    // Final progress update
    if (onProgress) {
      onProgress(loaded, loaded); // Use loaded as total to ensure 100%
    }

    // Concatenate chunks
    const data = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      data.set(chunk, offset);
      offset += chunk.length;
    }

    const arrayBuffer = data.buffer;

    // Try to cache for future use (may fail in incognito mode or with large files)
    try {
      await this.put(key, arrayBuffer, version);
    } catch (error) {
      console.warn(`Failed to cache ${key} (storage quota exceeded?):`, error);
      // Continue anyway - we have the data
    }

    return arrayBuffer;
  }
}

// Singleton instance
export const modelCache = new ModelCache();
