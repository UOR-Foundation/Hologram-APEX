# hologram-py

Python bindings for the Hologram Atlas virtual GPU runtime.

## Installation

```bash
pip install hologram
```

## Usage

```python
import hologram as hg
import numpy as np

# Create an executor
exec = hg.Executor()

# Create buffers
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

# Upload to Atlas
buf_a = exec.from_numpy(a)
buf_b = exec.from_numpy(b)

# Perform operations
buf_c = hg.ops.vector_add(buf_a, buf_b)

# Get results
c = buf_c.to_numpy()
print(c)  # [5.0, 7.0, 9.0]
```

## Features

- NumPy integration for easy data transfer
- CPU-based virtual GPU execution via Atlas runtime
- Full suite of operations: math, activations, reductions, linear algebra, loss functions
- Zero-copy buffer protocol support
