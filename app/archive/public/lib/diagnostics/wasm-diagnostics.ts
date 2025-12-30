/**
 * WASM Diagnostics Utility
 * 
 * Helps diagnose WASM loading issues in the browser
 */

export interface DiagnosticResult {
  test: string;
  passed: boolean;
  message: string;
  details?: any;
}

export class WasmDiagnostics {
  private results: DiagnosticResult[] = [];

  /**
   * Run all diagnostic tests
   */
  async runAll(): Promise<DiagnosticResult[]> {
    console.log('[Diagnostics] Running WASM diagnostics...');
    
    this.results = [];
    
    await this.testWebGPUAvailability();
    await this.testWasmFileAccessibility();
    await this.testJsModuleAccessibility();
    await this.testImportMetaUrl();
    await this.testFetchCapabilities();
    
    this.printSummary();
    
    return this.results;
  }

  /**
   * Test 1: WebGPU availability
   */
  private async testWebGPUAvailability() {
    const test = 'WebGPU Availability';
    
    if (!navigator.gpu) {
      this.addResult({
        test,
        passed: false,
        message: 'WebGPU not available (requires Chrome 113+ or Edge 113+)',
        details: { userAgent: navigator.userAgent }
      });
      return;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        this.addResult({
          test,
          passed: false,
          message: 'WebGPU adapter not available',
          details: { gpu: navigator.gpu }
        });
        return;
      }

      this.addResult({
        test,
        passed: true,
        message: 'WebGPU is available',
        details: { 
          adapterInfo: adapter.info,
          features: Array.from(adapter.features),
          limits: adapter.limits
        }
      });
    } catch (error) {
      this.addResult({
        test,
        passed: false,
        message: `WebGPU adapter request failed: ${error}`,
        details: { error }
      });
    }
  }

  /**
   * Test 2: WASM file HTTP accessibility
   */
  private async testWasmFileAccessibility() {
    const test = 'WASM File Accessibility';
    const wasmUrl = '/pkg/hologram_onnx_bg.wasm';

    try {
      const response = await fetch(wasmUrl, { method: 'HEAD' });
      
      if (response.ok) {
        const contentType = response.headers.get('content-type');
        const contentLength = response.headers.get('content-length');
        
        this.addResult({
          test,
          passed: true,
          message: `WASM file accessible at ${wasmUrl}`,
          details: {
            url: wasmUrl,
            status: response.status,
            contentType,
            contentLength: contentLength ? `${parseInt(contentLength) / 1024 / 1024}MB` : 'unknown'
          }
        });
      } else {
        this.addResult({
          test,
          passed: false,
          message: `WASM file returned HTTP ${response.status}`,
          details: {
            url: wasmUrl,
            status: response.status,
            statusText: response.statusText
          }
        });
      }
    } catch (error) {
      this.addResult({
        test,
        passed: false,
        message: `Failed to fetch WASM file: ${error}`,
        details: { url: wasmUrl, error }
      });
    }
  }

  /**
   * Test 3: JS module accessibility
   */
  private async testJsModuleAccessibility() {
    const test = 'JS Module Accessibility';
    const jsUrl = '/pkg/hologram_onnx.js';

    try {
      const response = await fetch(jsUrl, { method: 'HEAD' });
      
      if (response.ok) {
        const contentType = response.headers.get('content-type');
        
        this.addResult({
          test,
          passed: true,
          message: `JS module accessible at ${jsUrl}`,
          details: {
            url: jsUrl,
            status: response.status,
            contentType
          }
        });
      } else {
        this.addResult({
          test,
          passed: false,
          message: `JS module returned HTTP ${response.status}`,
          details: {
            url: jsUrl,
            status: response.status,
            statusText: response.statusText
          }
        });
      }
    } catch (error) {
      this.addResult({
        test,
        passed: false,
        message: `Failed to fetch JS module: ${error}`,
        details: { url: jsUrl, error }
      });
    }
  }

  /**
   * Test 4: import.meta.url resolution
   */
  private async testImportMetaUrl() {
    const test = 'import.meta.url Resolution';
    
    try {
      // Test if we can construct a URL relative to current location
      const baseUrl = window.location.href;
      const testUrl = new URL('/pkg/hologram_onnx.js', baseUrl);
      
      this.addResult({
        test,
        passed: true,
        message: 'URL resolution working',
        details: {
          baseUrl,
          resolvedUrl: testUrl.href
        }
      });
    } catch (error) {
      this.addResult({
        test,
        passed: false,
        message: `URL resolution failed: ${error}`,
        details: { error }
      });
    }
  }

  /**
   * Test 5: Fetch capabilities
   */
  private async testFetchCapabilities() {
    const test = 'Fetch Capabilities';
    
    try {
      // Test if fetch with range is supported
      const response = await fetch('/pkg/hologram_onnx_bg.wasm', {
        method: 'GET',
        headers: { 'Range': 'bytes=0-1023' }
      });
      
      const supportsRange = response.status === 206 || response.status === 200;
      const acceptsRange = response.headers.get('accept-ranges');
      
      this.addResult({
        test,
        passed: true,
        message: 'Fetch working',
        details: {
          supportsRange,
          acceptsRange,
          status: response.status
        }
      });
    } catch (error) {
      this.addResult({
        test,
        passed: false,
        message: `Fetch test failed: ${error}`,
        details: { error }
      });
    }
  }

  /**
   * Add a test result
   */
  private addResult(result: DiagnosticResult) {
    this.results.push(result);
    
    const icon = result.passed ? '✓' : '✗';
    const style = result.passed ? 'color: green' : 'color: red';
    
    console.log(`%c[Diagnostics] ${icon} ${result.test}: ${result.message}`, style);
    if (result.details) {
      console.log('[Diagnostics]   Details:', result.details);
    }
  }

  /**
   * Print summary
   */
  private printSummary() {
    const passed = this.results.filter(r => r.passed).length;
    const failed = this.results.filter(r => !r.passed).length;
    
    console.log(`\n[Diagnostics] Summary: ${passed} passed, ${failed} failed`);
    
    if (failed > 0) {
      console.log('[Diagnostics] Failed tests:');
      this.results
        .filter(r => !r.passed)
        .forEach(r => console.log(`  - ${r.test}: ${r.message}`));
    }
  }

  /**
   * Get results as a formatted string
   */
  getResultsString(): string {
    return this.results
      .map(r => `${r.passed ? '✓' : '✗'} ${r.test}: ${r.message}`)
      .join('\n');
  }
}

// Export singleton for convenience
export const diagnostics = new WasmDiagnostics();
