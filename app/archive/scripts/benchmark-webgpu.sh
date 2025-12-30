#!/usr/bin/env bash
# WebGPU Performance Benchmark Script
# Runs CPU baseline and WebGPU benchmarks for performance comparison
#
# Usage:
#   ./benchmark-webgpu.sh              # Run and compare with default baseline
#   ./benchmark-webgpu.sh baseline.txt # Run and compare with custom baseline
#   ./benchmark-webgpu.sh --quick      # Quick test (only 2 benchmarks)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Configuration
QUICK_MODE=false
if [ "$1" = "--quick" ]; then
    QUICK_MODE=true
    BASELINE_FILE="benchmarks/webgpu-baseline.txt"
else
    BASELINE_FILE="${1:-benchmarks/webgpu-baseline.txt}"
fi

CURRENT_FILE="benchmarks/webgpu-current.txt"
REPORT_FILE="benchmarks/webgpu-report.md"
THRESHOLD_PERCENT=10  # Fail if >10% regression

echo ""
echo "======================================"
echo "  WebGPU Performance Benchmarks"
echo "======================================"
echo ""

# Create benchmarks directory
mkdir -p benchmarks

# Step 1: Run CPU baseline benchmarks
if [ "$QUICK_MODE" = true ]; then
    print_status "Running quick benchmark test (2 benchmarks)..."
    if timeout 60 cargo bench --bench webgpu_baseline "cpu_binary_ops/vector_add/100" > "$CURRENT_FILE" 2>&1; then
        print_success "Quick benchmark test complete"
    else
        print_error "Quick benchmark test failed or timed out"
        exit 1
    fi
else
    print_status "Running CPU baseline benchmarks..."
    print_warning "This may take 2-3 minutes..."
    if timeout 300 cargo bench --bench webgpu_baseline > "$CURRENT_FILE" 2>&1; then
        print_success "CPU baseline benchmarks complete"
    else
        print_error "CPU baseline benchmarks failed or timed out"
        exit 1
    fi
fi

# Step 2: Parse benchmark results
print_status "Parsing benchmark results..."

# Extract benchmark times from Criterion output (convert to nanoseconds)
# Criterion format: "time:   [lower median upper] unit"
parse_bench_time() {
    local file="$1"
    local bench_name="$2"

    # Find the exact benchmark section (not just any mention of the name)
    # and extract the first "time:" line after it
    local line=$(grep -A10 "^${bench_name}\$" "$file" | grep "time:" | head -1)
    if [ -z "$line" ]; then
        echo "0"
        return
    fi

    # Extract median (middle value in bracket)
    local value=$(echo "$line" | awk '{print $4}')
    local unit=$(echo "$line" | awk '{print $6}' | tr -d ']')

    # Remove brackets
    value=$(echo "$value" | sed 's/\[//g')

    # Convert to nanoseconds based on unit (using awk for portability)
    case "$unit" in
        ns)
            echo "$value"
            ;;
        µs|us)
            echo "$value" | awk '{printf "%.2f", $1 * 1000}'
            ;;
        ms)
            echo "$value" | awk '{printf "%.2f", $1 * 1000000}'
            ;;
        s)
            echo "$value" | awk '{printf "%.2f", $1 * 1000000000}'
            ;;
        *)
            # Default assume ns
            echo "$value"
            ;;
    esac
}

# Step 3: Compare with baseline (if exists)
if [ -f "$BASELINE_FILE" ]; then
    print_status "Comparing with baseline..."

    echo "# WebGPU Performance Report" > "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "Generated: $(date)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "| Operation | Baseline (ns) | Current (ns) | Change | Status |" >> "$REPORT_FILE"
    echo "|-----------|---------------|--------------|--------|--------|" >> "$REPORT_FILE"

    REGRESSIONS=0
    IMPROVEMENTS=0

    # Compare common benchmarks (using actual Criterion benchmark names)
    for bench in "cpu_binary_ops/vector_add/100" \
                 "cpu_binary_ops/vector_add/1000" \
                 "cpu_binary_ops/vector_add/10000" \
                 "cpu_unary_ops/sigmoid/100" \
                 "cpu_unary_ops/sigmoid/1000" \
                 "cpu_reduce_ops/sum/100" \
                 "cpu_reduce_ops/sum/1000"; do

        BASELINE_TIME=$(parse_bench_time "$BASELINE_FILE" "$bench" || echo "0")
        CURRENT_TIME=$(parse_bench_time "$CURRENT_FILE" "$bench" || echo "0")

        if [ "$BASELINE_TIME" != "0" ] && [ "$CURRENT_TIME" != "0" ]; then
            # Calculate percentage change using awk
            CHANGE=$(echo "$BASELINE_TIME $CURRENT_TIME" | awk '{printf "%.2f", (($2 - $1) / $1) * 100}')

            # Check if it's a regression (>10% slower) or improvement (>5% faster)
            IS_REGRESSION=$(echo "$CHANGE $THRESHOLD_PERCENT" | awk '{print ($1 > $2) ? 1 : 0}')
            IS_IMPROVEMENT=$(echo "$CHANGE" | awk '{print ($1 < -5) ? 1 : 0}')

            if [ "$IS_REGRESSION" = "1" ]; then
                STATUS="⚠️ REGRESSION"
                REGRESSIONS=$((REGRESSIONS + 1))
            elif [ "$IS_IMPROVEMENT" = "1" ]; then
                STATUS="✅ IMPROVEMENT"
                IMPROVEMENTS=$((IMPROVEMENTS + 1))
            else
                STATUS="✓ OK"
            fi

            echo "| $bench | $BASELINE_TIME | $CURRENT_TIME | ${CHANGE}% | $STATUS |" >> "$REPORT_FILE"
        fi
    done

    echo "" >> "$REPORT_FILE"
    echo "## Summary" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "- Regressions: $REGRESSIONS" >> "$REPORT_FILE"
    echo "- Improvements: $IMPROVEMENTS" >> "$REPORT_FILE"
    echo "- Threshold: ${THRESHOLD_PERCENT}%" >> "$REPORT_FILE"

    # Print report
    cat "$REPORT_FILE"

    if [ $REGRESSIONS -gt 0 ]; then
        print_error "$REGRESSIONS performance regressions detected!"
        print_error "Review $REPORT_FILE for details"
        exit 1
    else
        print_success "No performance regressions detected"
        if [ $IMPROVEMENTS -gt 0 ]; then
            print_success "$IMPROVEMENTS performance improvements!"
        fi
    fi
else
    print_warning "No baseline found at $BASELINE_FILE"
    print_status "Saving current results as baseline..."
    cp "$CURRENT_FILE" "$BASELINE_FILE"
    print_success "Baseline saved. Run again to compare."
fi

echo ""
print_success "Benchmark complete!"
echo ""
echo "Files:"
echo "  • Current results: $CURRENT_FILE"
echo "  • Baseline: $BASELINE_FILE"
echo "  • Report: $REPORT_FILE"
echo ""
