#!/bin/zsh
# Interleaved A/B wall-clock measurement: fixed (HEAD) vs baseline (7f783bc).
# Alternates BASE/FIX blocks so slow machine drift hits both equally.
set -e
cd /Users/marshecho/Codex/mirror-memory-spike
. .venv/bin/activate

TARGET="${1:-longrun}"   # longrun | full
ROUNDS=2

restore_fixed() {
  git checkout -q spike/v1-tenancy-perf -- src tests
  git status --porcelain src tests | head -3
}

use_base() {
  git checkout -q 7f783bc -- src tests
  mv tests/test_tenant_isolation.py /tmp/_iso_test.py
}

restore_fixed() {
  git checkout -q spike/v1-tenancy-perf -- src tests
  if [ -f /tmp/_iso_test.py ]; then mv /tmp/_iso_test.py tests/test_tenant_isolation.py; fi
}

if [ "$TARGET" = "longrun" ]; then
  ARGS=(tests/test_longrun_bench.py)
else
  ARGS=()
fi

for i in $(seq 1 $ROUNDS); do
  use_base
  T=$( { /usr/bin/time -p pytest "${ARGS[@]}" -q -p no:cacheprovider 2>&1 | grep -E "^real" | awk '{print $2}'; } )
  echo "BASE  round$i $TARGET: ${T}s"
  restore_fixed
  T=$( { /usr/bin/time -p pytest "${ARGS[@]}" -q -p no:cacheprovider 2>&1 | grep -E "^real" | awk '{print $2}'; } )
  echo "FIXED round$i $TARGET: ${T}s"
done
restore_fixed
echo "done"
