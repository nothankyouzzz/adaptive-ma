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

echo "=== 3. Auditing git history for private-side leaks (HEAD) ==="
echo "Note: Excluding scripts/verify_release.sh from diff/content scan to avoid self-matching of denylist patterns."

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    python3 - << 'EOF'
import subprocess
import sys
import re

tree_denylist = re.compile(
    r"(whale|rider|strategy|execution|analyze_|ai_handover|research_report|artifacts|sharpe|sortino|drawdown|win[-_]?rate)",
    re.IGNORECASE
)

diff_denylist = re.compile(
    r"\b(whale|rider|ai_handover|research_report|artifacts|sharpe|sortino|drawdown|win[-_]?rate)\b|"
    r"\banalyze_|"
    r"\bstrategy\b|"
    r"\bexecution\b",
    re.IGNORECASE
)

# 1. Walk every commit reachable from HEAD and inspect tree file paths
head_commits = subprocess.check_output(["git", "rev-list", "HEAD"], text=True).split()
tree_errors = []

for commit in head_commits:
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], text=True).splitlines()
    for f in files:
        if f == "scripts/verify_release.sh":
            continue
        if tree_denylist.search(f):
            tree_errors.append((commit, f))

if tree_errors:
    print("ERROR: Denylisted file paths found in git history reachable from HEAD:", file=sys.stderr)
    for c, f in tree_errors:
        print(f"  commit {c[:10]}: {f}", file=sys.stderr)
    sys.exit(1)

print(f"Tree path scan: PASSED ({len(head_commits)} commits checked, zero denylisted filenames).")

# 2. Walk diffs on HEAD for added lines containing denylisted terms
log_proc = subprocess.Popen(
    ["git", "log", "-p", "-U0", "HEAD", "--", ".", ":(exclude)scripts/verify_release.sh"],
    stdout=subprocess.PIPE,
    text=True
)

diff_errors = []
current_commit = ""
current_file = ""

for line in log_proc.stdout:
    if line.startswith("commit "):
        current_commit = line.strip().split()[1]
    elif line.startswith("+++ b/"):
        current_file = line.strip()[6:]
    elif line.startswith("+") and not line.startswith("+++"):
        added_line = line[1:].strip()
        # Exemptions for benign public terms:
        # - CI workflow matrix strategy keyword
        if current_file.startswith(".github/workflows/") and re.match(r"^strategy:\s*$", added_line):
            continue
        # - Legitimate financial discussion of trading frictions ("live execution")
        if "live execution" in added_line.lower() and not re.search(
            r"\b(whale|rider|ai_handover|research_report|artifacts|sharpe|sortino|drawdown|win[-_]?rate)\b|\banalyze_",
            added_line,
            re.IGNORECASE
        ):
            continue
        # - Historical heading in initial multiscale commit
        if added_line == "## Advanced Multiscale & Strategy Modules":
            continue

        m = diff_denylist.search(added_line)
        if m:
            diff_errors.append((current_commit, current_file, m.group(0), added_line))

log_proc.wait()

if diff_errors:
    print("ERROR: Denylisted added lines found in git history reachable from HEAD:", file=sys.stderr)
    for c, f, token, text in diff_errors:
        print(f"  commit {c[:10]} in {f} (matched '{token}'): {text}", file=sys.stderr)
    sys.exit(1)

print("Diff additions scan: PASSED (zero denylisted additions in commits reachable from HEAD).")

# 3. Check other local refs and warn if they contain unpublishable historical leaks
all_refs = subprocess.check_output(
    ["git", "for-each-ref", "--format=%(refname)", "refs/heads", "refs/tags"],
    text=True
).splitlines()

head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

for ref in all_refs:
    try:
        ref_sha = subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    except subprocess.CalledProcessError:
        continue
    if ref_sha == head_sha:
        continue

    proc = subprocess.Popen(
        ["git", "log", "-p", "-U0", ref, "--not", "HEAD", "--", ".", ":(exclude)scripts/verify_release.sh"],
        stdout=subprocess.PIPE,
        text=True
    )
    hits = 0
    for line in proc.stdout:
        if line.startswith("+") and not line.startswith("+++"):
            added = line[1:].strip()
            if diff_denylist.search(added):
                hits += 1
    proc.wait()
    if hits > 0:
        print(f"WARNING: Local ref '{ref}' contains {hits} denylisted terms in unpushed history (e.g. backup/pre-scrub).")
        print(f"         Do NOT push '{ref}' to any public remote!")
EOF
else
    echo "Notice: Not a git repository; skipping git history audit."
fi

echo "=== 4. Auditing artifact file lists against denylist ==="
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

echo "=== 5. Auditing wheel file structure ==="
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

echo "=== 6. Creating throwaway virtualenv and installing wheel ==="
VENV_DIR="$TMP_DIR/venv"
if command -v uv >/dev/null 2>&1; then
    uv venv --seed "$VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install "$WHL_FILE"

echo "=== 7. Running smoke test from installed wheel ==="
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
