# Docker Images Build Scripts and Examples

This directory contains alternative workflows, setup guides, and helper scripts for building and publishing Docker images.

## Files

### Alternative Workflows

#### [`publish-images-native-arm64.yml`](publish-images-native-arm64.yml)
Alternative GitHub Actions workflow that uses native ARM64 runners instead of QEMU emulation.

**Benefits:**
- 3-4x faster ARM64 builds (~8-12 min vs 30-60 min)
- No emulation overhead
- Parallel builds on different architectures

**Requirements:**
- ARM64 runner with label `arm64` (self-hosted or GitHub ARM64 runners)
- See setup guide below

**Usage:**
```bash
# Copy to workflows directory
cp scripts/publish-images-native-arm64.yml .github/workflows/

# Disable default workflow or rename it
mv .github/workflows/publish-images.yml .github/workflows/publish-images-qemu.yml.disabled
```

### Setup Guides

#### [`setup-oracle-arm64-runner.md`](setup-oracle-arm64-runner.md)
Complete guide for setting up a **free** Oracle Cloud ARM64 instance as a GitHub Actions self-hosted runner.

**What you get (free forever):**
- 4 ARM cores (Ampere A1)
- 24 GB RAM
- 200 GB storage
- 10 TB bandwidth/month

**Time to setup:** ~30 minutes

**Perfect for:**
- Fast native ARM64 builds
- Learning ARM64 development
- Cost-effective CI/CD

## Quick Start Guides

### Option 1: Keep Using QEMU (Current Setup)

**Pros:**
- ✅ Already configured
- ✅ Zero infrastructure
- ✅ Works everywhere
- ✅ Free

**Cons:**
- ⏱️ Slower ARM64 builds (30-60 min first time, 10-15 min cached)

**No changes needed** - just use the existing workflow!

### Option 2: Setup Oracle Cloud ARM64 Runner (Recommended)

**Pros:**
- ✅ Free forever
- ✅ 3-4x faster builds
- ✅ Native ARM64 performance
- ✅ Can use for other tasks

**Cons:**
- ⚙️ Requires setup (~30 min)
- 🔧 Need to maintain infrastructure

**Steps:**
1. Follow [`setup-oracle-arm64-runner.md`](setup-oracle-arm64-runner.md)
2. Copy [`publish-images-native-arm64.yml`](publish-images-native-arm64.yml) to `.github/workflows/`
3. Test with workflow dispatch
4. Enjoy faster builds!

### Option 3: Use GitHub ARM64 Runners

**Pros:**
- ✅ Fully managed
- ✅ Fast builds
- ✅ No maintenance

**Cons:**
- 💰 Requires GitHub Team/Enterprise ($$$)

**Steps:**
1. Enable ARM64 runners in your GitHub organization
2. Copy [`publish-images-native-arm64.yml`](publish-images-native-arm64.yml) to `.github/workflows/`
3. Change `runs-on: [self-hosted, arm64]` to `runs-on: ubuntu-latest-arm64`

### Option 4: Docker Build Cloud

**Pros:**
- ✅ Fully managed
- ✅ Fast builds
- ✅ Native builders for both architectures

**Cons:**
- 💰 Pay per build minute ($$)

**Steps:**
```bash
# One-time setup
docker login
docker buildx create --driver cloud yourorg/default

# Use with Makefile
make buildx-all DOCKER_USERNAME=yourorg --builder=cloud-yourorg-default
```

### Option 5: Build ARM64 Locally (Apple Silicon)

If you have an M1/M2/M3 Mac:

**Pros:**
- ✅ Free
- ✅ Native ARM64 build
- ✅ No cloud infrastructure

**Cons:**
- ⚙️ Manual process
- 💻 Requires Apple Silicon Mac

**Steps:**
```bash
# Build ARM64 natively
docker build --platform linux/arm64 \
  -t yourorg/hologram-devcontainer:arm64-v1.0.0 \
  -f docker-images/dev/Dockerfile \
  docker-images

# Push to registry
docker push yourorg/hologram-devcontainer:arm64-v1.0.0

# Let CI build AMD64 and combine manifests
# (or build AMD64 with emulation and combine locally)
```

## Performance Comparison

| Method | ARM64 Build Time | Cost | Setup Time |
|--------|------------------|------|------------|
| QEMU emulation (current) | 30-60 min (first)<br>10-15 min (cached) | Free | 0 min |
| Oracle Cloud ARM64 | 8-12 min | Free | 30 min |
| GitHub ARM64 runners | 8-12 min | $$$ | 5 min |
| Docker Build Cloud | 8-12 min | $$ | 10 min |
| Apple Silicon local | 10-15 min | Free* | 5 min |

*Free if you already have Apple Silicon Mac

## Recommendations

**For most users:**
- Start with QEMU (current setup)
- If builds are too slow, setup Oracle Cloud ARM64 runner (free!)

**For teams with budget:**
- Use GitHub ARM64 runners or Docker Build Cloud

**For Apple Silicon users:**
- Build ARM64 locally, let CI handle AMD64

## Additional Scripts

Add your own helper scripts here:

```bash
scripts/
├── publish-images-native-arm64.yml   # Alternative workflow
├── setup-oracle-arm64-runner.md      # Setup guide
├── build-local.sh                    # Your custom build script
├── cleanup.sh                        # Cleanup Docker images
└── README.md                         # This file
```

## Support

- Main documentation: [`../README.md`](../README.md)
- GitHub Actions workflows: [`../.github/workflows/`](../.github/workflows/)
- Issues: Report in main repository

## Contributing

Have a useful script or workflow? Add it here with documentation!
