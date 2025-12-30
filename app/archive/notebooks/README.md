# Hologram-Core Performance Analysis Notebooks

This directory contains Jupyter notebooks for analyzing hologram-core benchmark results.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Benchmarks

From the workspace root:

```bash
cd /workspace
cargo bench --bench comprehensive_suite
```

This generates: `benchmarks/benchmark_results_{timestamp}.json`
And creates symlink: `benchmarks/benchmark_results_current.json`

### 3. Launch Jupyter Notebook

```bash
cd /workspace/notebooks
jupyter notebook performance_analysis.ipynb
```

### 4. Run All Cells

In Jupyter:
- Click **Cell** → **Run All**
- Or press **Shift+Enter** through each cell

## Notebooks

### `performance_analysis.ipynb`

Comprehensive visualization and analysis of benchmark results:

**Features:**
- Loads JSON benchmark results into pandas DataFrame
- 6 comprehensive visualization suites
- Statistical analysis by operation category
- Scaling analysis across problem sizes
- Addressing mode comparison (PhiCoordinate vs BufferOffset)
- Performance heatmaps
- Top performer identification
- Linear algebra operation analysis
- Exports summary CSV files

**Output Files:**
- `category_overview.png`
- `scaling_analysis.png`
- `addressing_mode_comparison.png`
- `performance_heatmaps.png`
- `top_performers.png`
- `linalg_analysis.png`
- `benchmark_summary.csv`
- `benchmark_full_data.csv`

## Requirements

- Python 3.10+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- jupyter >= 1.0.0

All dependencies listed in `requirements.txt`

## Customization

### Analyzing Multiple Benchmark Runs

Modify the data loading cell:

```python
# Load multiple files
import glob
files = sorted(glob.glob('/workspace/benchmark_results_*.json'), reverse=True)

# Compare first two runs
with open(files[0]) as f:
    latest = json.load(f)
with open(files[1]) as f:
    previous = json.load(f)

# Compare performance
```

### Custom Visualizations

Add new cells with your custom analysis:

```python
# Example: Filter for specific operations
math_ops = df[df['category'] == 'math']
math_ops.plot(x='size', y='throughput_mops', kind='line')
plt.title('Math Operations Throughput')
plt.show()
```

### Export Custom Reports

```python
# Generate markdown report
report = df.groupby('category').describe()
report.to_markdown('performance_report.md')
```

## Troubleshooting

### "No benchmark results found"

Ensure benchmarks have been run:
```bash
cd /workspace
cargo bench --bench comprehensive_suite
ls -l benchmarks/benchmark_results_*.json
```

### Import Errors

Reinstall dependencies:
```bash
pip install -r requirements.txt --upgrade
```

### Visualization Issues

Reset matplotlib backend:
```python
import matplotlib
matplotlib.use('Agg')  # For headless environments
# or
matplotlib.use('TkAgg')  # For GUI environments
```

### Kernel Issues

Restart the Jupyter kernel:
- Click **Kernel** → **Restart & Clear Output**
- Then **Cell** → **Run All**

## Performance Tips

### Large Datasets

For very large benchmark datasets:

```python
# Sample data
df_sample = df.sample(frac=0.1)  # Use 10% of data

# Or aggregate
df_agg = df.groupby(['operation', 'size']).mean()
```

### Memory Optimization

```python
# Use efficient data types
df['size'] = df['size'].astype('int32')
df['category'] = df['category'].astype('category')
```

## Additional Resources

- [Comprehensive Benchmark Suite Documentation](../docs/BENCHMARK_SUITE.md)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

## Contributing

When adding new visualizations:

1. Document the visualization purpose
2. Use consistent styling (seaborn themes)
3. Add axis labels and titles
4. Include save commands for publication-quality output
5. Update this README

## License

MIT OR Apache-2.0
