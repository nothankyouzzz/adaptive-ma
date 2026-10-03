#!/usr/bin/env bash
set -euo pipefail

# Script to verify release artifacts: sdist + wheel hygiene, denylist audit, and smoke test.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "=== 1. Creating temporary build workspace ==="
TMP_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

BUILD_DIR="$TMP_DIR/dist"
mkdir -p "$BUILD_DIR"

echo "=== 2. Building sdist and wheel ==="
if command -v uv >/dev/null 2>&1; then
    uv build --out-dir "$BUILD_DIR"
else
    python3 -m build --out-dir "$BUILD_DIR"
fi

WHL_FILE="$(ls "$BUILD_DIR"/*.whl | head -n 1)"
SDIST_FILE="$(ls "$BUILD_DIR"/*.tar.gz | head -n 1)"

if [[ ! -f "$WHL_FILE" ]]; then
    echo "ERROR: Wheel file not found in $BUILD_DIR" >&2
    exit 1
fi

if [[ ! -f "$SDIST_FILE" ]]; then
    echo "ERROR: Sdist file not found in $BUILD_DIR" >&2
    exit 1
fi

echo "Built wheel: $(basename "$WHL_FILE")"
echo "Built sdist: $(basename "$SDIST_FILE")"

echo "=== 3. Auditing artifact file lists against denylist ==="
DENYLIST=(
    "strategy"
    "execution"
    "AI_HANDOVER"
    "RESEARCH_REPORT"
    "artifacts"
    "bench/data"
    "bench"
    "data/cache"
    "\.venv"
    "\.tower"
    "\.csv$"
)

# Extract wheel entries
WHL_ENTRIES="$(python3 -c "import zipfile, sys; [print(n) for n in zipfile.ZipFile(sys.argv[1]).namelist()]" "$WHL_FILE")"

# Extract sdist entries
SDIST_ENTRIES="$(tar -tzf "$SDIST_FILE")"

FAILED=0
for pattern in "${DENYLIST[@]}"; do
    MATCHES_WHL="$(echo "$WHL_ENTRIES" | grep -E "$pattern" || true)"
    if [[ -n "$MATCHES_WHL" ]]; then
        echo "ERROR: Denylisted pattern '$pattern' found in wheel:" >&2
        echo "$MATCHES_WHL" >&2
        FAILED=1
    fi

    MATCHES_SDIST="$(echo "$SDIST_ENTRIES" | grep -E "$pattern" || true)"
    if [[ -n "$MATCHES_SDIST" ]]; then
        echo "ERROR: Denylisted pattern '$pattern' found in sdist:" >&2
        echo "$MATCHES_SDIST" >&2
        FAILED=1
    fi
done

if [[ "$FAILED" -ne 0 ]]; then
    echo "Release verification FAILED: denylisted files found in artifacts!" >&2
    exit 1
fi
echo "Denylist check: PASSED (no denylisted paths found in sdist or wheel)."

echo "=== 4. Auditing wheel file structure ==="
# Check that wheel only contains allowed top-level directories
NON_ALLOWED_WHL="$(echo "$WHL_ENTRIES" | grep -v -E '^(adaptive_ma/|adaptive_ma-[^/]+\.dist-info/)' || true)"
if [[ -n "$NON_ALLOWED_WHL" ]]; then
    echo "ERROR: Unexpected files in wheel outside allowed package/dist-info directories:" >&2
    echo "$NON_ALLOWED_WHL" >&2
    exit 1
fi

WHL_FILE_COUNT="$(echo "$WHL_ENTRIES" | wc -l)"
echo "Wheel file count: $WHL_FILE_COUNT files"
echo "Wheel contents:"
echo "$WHL_ENTRIES" | sed 's/^/  /'

# Verify key expected files in wheel
REQUIRED_MODULES=(
    "adaptive_ma/__init__.py"
    "adaptive_ma/core.py"
    "adaptive_ma/estimate.py"
    "adaptive_ma/instruments.py"
    "adaptive_ma/ma.py"
    "adaptive_ma/multiscale.py"
    "adaptive_ma/params.py"
    "adaptive_ma/volatility.py"
    "adaptive_ma/eval/__init__.py"
    "adaptive_ma/eval/baselines.py"
    "adaptive_ma/eval/data.py"
    "adaptive_ma/eval/dgp.py"
    "adaptive_ma/eval/metrics.py"
)

for mod in "${REQUIRED_MODULES[@]}"; do
    if ! echo "$WHL_ENTRIES" | grep -q "^$mod$"; then
        echo "ERROR: Required module '$mod' missing from wheel!" >&2
        exit 1
    fi
done
echo "Required modules check: PASSED (all 13 core and eval modules present)."

echo "=== 5. Creating throwaway virtualenv and installing wheel ==="
VENV_DIR="$TMP_DIR/venv"
if command -v uv >/dev/null 2>&1; then
    uv venv --seed "$VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install "$WHL_FILE"

echo "=== 6. Running smoke test from installed wheel ==="
# Change to $TMP_DIR so python cannot resolve adaptive_ma from local workspace directory
cd "$TMP_DIR"
"$VENV_DIR/bin/python" - << 'EOF'
import sys
import numpy as np

# 1. Import adaptive_ma and check location is inside site-packages
import adaptive_ma
location = adaptive_ma.__file__
print(f"Imported adaptive_ma from: {location}")
if "site-packages" not in location:
    sys.exit(f"ERROR: adaptive_ma was not loaded from site-packages! ({location})")

# 2. Check public API exports
from adaptive_ma import filter_price, KalmanMA, AdaptiveKalmanMA, FilterResult, StepResult

# 3. Import eval metrics
import adaptive_ma.eval.metrics as metrics

# 4. Generate synthetic price series and run filter_price (3-tuple API)
rng = np.random.default_rng(12345)
n_points = 100
series = np.cumsum(rng.normal(0, 0.05, n_points)) + 100.0

level, excitation, residual = filter_price(series, rho=0.8, alpha_eff=0.5)
assert level.shape == (n_points,), f"Unexpected level shape: {level.shape}"
assert excitation.shape == (n_points,), f"Unexpected excitation shape: {excitation.shape}"
assert residual.shape == (n_points,), f"Unexpected residual shape: {residual.shape}"
assert np.all(np.isfinite(level + excitation)), "Non-finite values in level + excitation"
assert np.all(np.isfinite(residual)), "Non-finite values in residual"

# 5. Test filter_price with return_full=True
full_res = filter_price(series, rho=0.8, alpha_eff=0.5, return_full=True)
assert isinstance(full_res, FilterResult), f"Expected FilterResult, got {type(full_res)}"
assert np.all(np.isfinite(full_res.residual)), "Non-finite values in FilterResult.residual"

# 6. Test KalmanMA and AdaptiveKalmanMA single steps
kma = KalmanMA(rho=0.8, alpha_eff=0.5)
step_kma = kma.step_full(0.1, series[0])
assert isinstance(step_kma, StepResult)
assert np.isfinite(step_kma.level)

akma = AdaptiveKalmanMA(rho=0.8, alpha_eff=0.5)
step_akma = akma.step_full(0.1, series[0])
assert isinstance(step_akma, StepResult)
assert np.isfinite(step_akma.level)

# 7. Test eval metrics calculation
smoothness = metrics.calc_smoothness(level + excitation)
assert np.isfinite(smoothness), f"Non-finite smoothness: {smoothness}"

print(f"filter_price output verification: length={n_points}")
print(f"  First 3 smoothed values: {(level + excitation)[:3]}")
print(f"  First 3 residual values: {residual[:3]}")
print(f"  Series smoothness: {smoothness:.6e}")
print("Installed wheel smoke test PASSED successfully!")
EOF

echo "=== Release verification COMPLETE: SUCCESS ==="
