#!/usr/bin/env python3
"""
Patch generated wasm-bindgen JavaScript to fix Chrome 135+ WebGPU compatibility.
Chrome 135+ removed maxInterStageShaderComponents from WebGPU spec but wgpu 0.20 still includes it.
"""
import sys
import re

def patch_js(js_path):
    with open(js_path, 'r') as f:
        content = f.read()

    # Find and replace the requestDevice function
    pattern = r'(imports\.wbg\.__wbg_requestDevice_[a-f0-9]+ = function\(arg0, arg1\) \{)\s+(const ret = arg0\.requestDevice\(arg1\);)\s+(return ret;)\s+(\};)'

    replacement = r'''\1
        // PATCHED: Remove maxInterStageShaderComponents for Chrome 135+ compatibility
        if (arg1 && arg1.requiredLimits && arg1.requiredLimits.maxInterStageShaderComponents !== undefined) {
            delete arg1.requiredLimits.maxInterStageShaderComponents;
        }
        \2
        \3
    \4'''

    patched_content = re.sub(pattern, replacement, content)

    if patched_content == content:
        print("⚠️  Warning: Pattern not found, requestDevice may not have been patched")
        return False

    with open(js_path, 'w') as f:
        f.write(patched_content)

    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: patch-webgpu-limits.py <hologram_ai.js>")
        sys.exit(1)

    js_path = sys.argv[1]
    if patch_js(js_path):
        print(f"✓ Patched {js_path} for Chrome 135+ compatibility")
    else:
        print(f"✗ Failed to patch {js_path}")
        sys.exit(1)
