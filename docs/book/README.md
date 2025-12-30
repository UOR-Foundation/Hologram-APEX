# Hologram: The Physics of Information - Book

This directory contains the source for the comprehensive book about Hologram, exploring information's intrinsic mathematical structure and the computing platform that aligns with these natural properties.

## 📚 About the Book

The book is divided into three main parts:

1. **Foundations** - Discovering information's intrinsic structure
2. **Architecture** - How Hologram implements these principles
3. **Implications** - What this means for computing's future

Plus extensive appendices covering mathematical foundations, conservation laws, and implementation details.

## 🚀 Quick Start

### Prerequisites

- [mdBook](https://rust-lang.github.io/mdBook/) (v0.4.37+)
- Make (for build automation)
- Node.js (optional, for linting)

### Building the Book

```bash
# Install mdBook (if not already installed)
make install

# Build HTML version
make build

# Start development server with live reload
make serve

# Generate PDF (requires wkhtmltopdf)
make build-pdf
```

## 📂 Directory Structure

```
docs/book/
├── src/                    # Book source files (Markdown)
│   ├── SUMMARY.md         # Table of contents
│   ├── introduction.md    # Book introduction
│   ├── preface.md         # Preface
│   ├── part-1-foundations/  # Part I chapters
│   ├── part-2-architecture/ # Part II chapters
│   ├── part-3-implications/ # Part III chapters
│   ├── appendices/        # Appendices
│   └── conclusion.md      # Conclusion
├── theme/                 # Custom CSS and assets
├── build/                 # Generated output (git ignored)
├── book.toml             # mdBook configuration
├── Makefile              # Build automation
└── README.md             # This file
```

## 🛠️ Available Commands

### Building

- `make build` - Build HTML version
- `make build-pdf` - Generate PDF (requires wkhtmltopdf)
- `make serve` - Start dev server at http://localhost:3000
- `make watch` - Watch for changes and rebuild
- `make clean` - Remove build artifacts

### Quality Assurance

- `make lint` - Run all linters
- `make format` - Format markdown files
- `make check` - Run all checks
- `make test` - Test the build process

### Development

- `make stats` - Show book statistics
- `make release` - Create a release build
- `make deploy` - Deploy to GitHub Pages

### Shortcuts

- `make b` - Build
- `make s` - Serve
- `make c` - Clean
- `make l` - Lint
- `make f` - Format

## 📝 Writing Guidelines

### Markdown Style

- Use ATX-style headers (`#`, `##`, etc.)
- Keep lines under 120 characters (except code blocks and tables)
- Use fenced code blocks with language identifiers
- Add alt text to all images

### Mathematical Notation

The book uses MathJax for rendering mathematical formulas:

```markdown
Inline math: $E = mc^2$

Display math:
$$
S[ψ] = ∑_{i∈Λ} ∑_{a∈A} L_a(ψ_i, ∇ψ_i, constraints)
$$
```

### Special Terms

Domain-specific terms are defined in the glossary (Appendix A). First use should be **bold** with a brief explanation if the glossary reference isn't immediately available.

## 🔧 Configuration

### book.toml

Main configuration file for mdBook. Key settings:
- `title` - Book title
- `authors` - Author list
- `language` - Book language (en)
- `theme` - Custom theme directory
- `mathjax-support` - Enable math rendering

### Linting Configuration

- `.markdownlint.json` - Markdown linting rules
- `.prettierrc.json` - Formatting configuration
- `cspell.json` - Spell checking with custom dictionary

## 🚢 Deployment

### GitHub Pages

The book is automatically deployed to GitHub Pages on push to main:

1. Push changes to `main` branch
2. GitHub Actions builds the book
3. Deploys to `gh-pages` branch
4. Available at: https://[username].github.io/[repo]/book/

### Manual Deployment

```bash
# Build and deploy manually
make deploy
```

## 🧪 Testing

### Local Testing

```bash
# Run all tests
make test

# Check structure
make check-structure

# Check all chapters exist
make check-chapters

# Verify links
make lint-links
```

### CI/CD

GitHub Actions runs on every push:
- Builds HTML version
- Runs linters
- Generates PDF
- Deploys to GitHub Pages (main branch only)

## 📦 PDF Generation

### Requirements

- wkhtmltopdf (with patched Qt)

### Installation

```bash
# Ubuntu/Debian
sudo apt-get install wkhtmltopdf

# macOS
brew install --cask wkhtmltopdf

# Or download from: https://wkhtmltopdf.org/downloads.html
```

### Generating PDF

```bash
make build-pdf
# Output: build/hologram-book.pdf
```

## 🤝 Contributing

1. Make changes in `src/` directory
2. Run `make check` to verify
3. Run `make format` to format
4. Test with `make serve`
5. Submit pull request

## 📄 License

This book is part of the Hologram project by the UOR Foundation.

## 🔗 Links

- [Hologram Repository](https://github.com/UOR-Foundation/Hologram)
- [UOR Foundation](https://uor.foundation)
- [mdBook Documentation](https://rust-lang.github.io/mdBook/)

## ❓ Troubleshooting

### Build Fails

```bash
# Clean and rebuild
make clean
make build
```

### Missing Dependencies

```bash
# Install all dependencies
make install
```

### PDF Generation Issues

- Ensure wkhtmltopdf is installed
- Check for JavaScript errors in print.html
- Try increasing `--javascript-delay` in Makefile

### Live Reload Not Working

- Check if port 3000 is already in use
- Try `make clean` then `make serve`

## 📊 Book Statistics

Run `make stats` to see:
- Chapter count
- Word count
- Line count
- Character count

---

For more information about the Hologram project, see the main repository README.