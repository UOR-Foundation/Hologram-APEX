# Setup Oracle Cloud ARM64 Runner (Free)

Oracle Cloud offers a generous free tier that includes ARM64 compute instances - perfect for building multi-arch Docker images natively.

## What You Get (Free Forever)

- **4 ARM cores** (Ampere A1)
- **24 GB RAM**
- **200 GB boot volume**
- **10 TB outbound data transfer/month**
- No credit card expiration concerns

This is ideal for building ARM64 Docker images 3-4x faster than QEMU emulation.

## Step 1: Create Oracle Cloud Account

1. Go to https://www.oracle.com/cloud/free/
2. Sign up for a free account
3. Verify your email and phone number
4. Note: You need a credit card for verification, but won't be charged

## Step 2: Create ARM64 Instance

1. **Navigate to Compute > Instances > Create Instance**

2. **Instance Configuration:**
   - **Name**: `github-runner-arm64`
   - **Image**: Ubuntu 22.04 (Minimal)
   - **Shape**:
     - Click "Change Shape"
     - Select "Ampere"
     - Choose VM.Standard.A1.Flex
     - **OCPUs**: 4
     - **Memory**: 24 GB

3. **Networking:**
   - Use default VCN (Virtual Cloud Network)
   - Assign a public IPv4 address: ✓

4. **SSH Keys:**
   - Generate new key pair or use existing
   - **Download private key** (you'll need this!)

5. **Boot Volume:**
   - 200 GB (max free tier)

6. Click **Create**

Wait 2-3 minutes for instance to provision.

## Step 3: Configure Firewall

Oracle has both instance firewall AND security list rules:

1. **Security List (Cloud Level):**
   - Go to Networking > Virtual Cloud Networks
   - Click your VCN > Security Lists > Default Security List
   - Add Ingress Rule:
     - **Source CIDR**: `0.0.0.0/0`
     - **IP Protocol**: TCP
     - **Destination Port**: 22
     - Click "Add Ingress Rules"

2. **Instance Firewall (OS Level):**
   ```bash
   # SSH into instance first (see Step 4)
   sudo iptables -I INPUT -p tcp --dport 22 -j ACCEPT
   sudo netfilter-persistent save
   ```

## Step 4: Connect to Instance

```bash
# Use the private key you downloaded
chmod 400 ~/Downloads/oracle-arm64-key.pem

# Connect (replace with your instance's public IP)
ssh -i ~/Downloads/oracle-arm64-key.pem ubuntu@<INSTANCE_PUBLIC_IP>
```

## Step 5: Install Docker

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add ubuntu user to docker group
sudo usermod -aG docker ubuntu

# Start Docker
sudo systemctl enable docker
sudo systemctl start docker

# Verify
docker --version
docker run hello-world
```

## Step 6: Setup GitHub Actions Runner

### Create Runner on GitHub

1. Go to your repo: Settings > Actions > Runners > New self-hosted runner
2. Choose **Linux** and **ARM64**
3. Copy the download and configuration commands

### Install Runner on Oracle Instance

```bash
# Create a folder
mkdir actions-runner && cd actions-runner

# Download the latest runner (replace URL from GitHub instructions)
curl -o actions-runner-linux-arm64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-arm64-2.311.0.tar.gz

# Extract
tar xzf ./actions-runner-linux-arm64-2.311.0.tar.gz

# Create runner and start the configuration
./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO \
  --token YOUR_TOKEN_FROM_GITHUB \
  --labels arm64,self-hosted,linux \
  --name oracle-arm64-runner

# Install as a service
sudo ./svc.sh install
sudo ./svc.sh start

# Check status
sudo ./svc.sh status
```

## Step 7: Configure Runner as Service

```bash
# Make it start on boot
sudo systemctl enable actions.runner.YOUR_ORG-YOUR_REPO.oracle-arm64-runner

# Check it's running
sudo systemctl status actions.runner.YOUR_ORG-YOUR_REPO.oracle-arm64-runner
```

## Step 8: Update GitHub Workflow

Use the example workflow at `scripts/publish-images-native-arm64.yml`:

```yaml
# For jobs that should run on ARM64:
runs-on: [self-hosted, arm64]
```

## Step 9: Test the Runner

Create a simple test workflow:

```yaml
name: Test ARM64 Runner
on: workflow_dispatch

jobs:
  test:
    runs-on: [self-hosted, arm64]
    steps:
      - name: Check architecture
        run: |
          uname -m  # Should output: aarch64
          docker --version
          docker run --rm arm64v8/ubuntu uname -m
```

## Maintenance

### Update Runner

```bash
cd ~/actions-runner
sudo ./svc.sh stop
./config.sh remove
# Download latest version
curl -o actions-runner-linux-arm64-X.X.X.tar.gz -L <NEW_URL>
tar xzf ./actions-runner-linux-arm64-X.X.X.tar.gz
./config.sh --url ... --token ...
sudo ./svc.sh install
sudo ./svc.sh start
```

### Monitor Usage

```bash
# Check disk space
df -h

# Check memory
free -h

# Check Docker images
docker images
docker system df

# Clean up old images
docker system prune -a --volumes -f
```

### Security Best Practices

1. **Firewall**: Only allow GitHub Actions IPs (or use VPN)
   ```bash
   # GitHub Actions IP ranges (update periodically)
   sudo ufw allow from 140.82.112.0/20 to any port 22
   sudo ufw enable
   ```

2. **Auto-updates**:
   ```bash
   sudo apt-get install unattended-upgrades
   sudo dpkg-reconfigure -plow unattended-upgrades
   ```

3. **Monitoring**:
   ```bash
   # Install monitoring (optional)
   sudo apt-get install -y prometheus-node-exporter
   ```

## Troubleshooting

### Runner Not Connecting

```bash
# Check service status
sudo systemctl status actions.runner.*

# Check logs
sudo journalctl -u actions.runner.* -f

# Restart service
sudo ./svc.sh stop
sudo ./svc.sh start
```

### Out of Disk Space

```bash
# Clean Docker
docker system prune -a --volumes -f

# Clean runner work directories
cd ~/actions-runner/_work
sudo rm -rf */

# Check what's using space
du -sh * | sort -h
```

### Performance Issues

```bash
# Check CPU
top

# Check if swapping
free -h
vmstat 1

# Add swap if needed (optional)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Cost Comparison

| Method | Monthly Cost | Build Time (ARM64) |
|--------|--------------|-------------------|
| QEMU (current) | $0 | 30-60 min |
| **Oracle Cloud Free** | **$0** | **8-12 min** |
| GitHub ARM64 runners | ~$100-300 | 8-12 min |
| Docker Build Cloud | ~$30-100 | 8-12 min |
| AWS Graviton (t4g.xlarge) | ~$120 | 8-12 min |

## Expected Build Performance

**With Oracle Cloud ARM64 runner:**
- **First build**: ~10-15 minutes (no cache)
- **Subsequent builds**: ~8-12 minutes (with cache)
- **3-4x faster** than QEMU emulation
- **Native ARM64** - no emulation overhead

## Next Steps

1. ✅ Setup Oracle Cloud ARM64 instance
2. ✅ Install Docker and GitHub Actions runner
3. Copy `scripts/publish-images-native-arm64.yml` to `.github/workflows/`
4. Update workflow to use `[self-hosted, arm64]` runners
5. Test with a workflow dispatch
6. Enjoy faster ARM64 builds! 🚀

## Alternative: Multiple Runners

For even better performance, you can create multiple ARM64 instances:

- **Instance 1**: GitHub Actions runner (4 cores, 24GB)
- **Instance 2**: Development/testing (shared across team)
- **Instance 3**: Backup runner

Oracle Cloud free tier allows up to 4 ARM instances totaling 4 OCPUs and 24GB RAM.

## Resources

- [Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/)
- [GitHub Self-Hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)
- [Docker on ARM](https://docs.docker.com/desktop/install/linux-install/)
