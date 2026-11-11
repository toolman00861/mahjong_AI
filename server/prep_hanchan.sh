#!/bin/bash
# Rebuild into separate locations; never remove the original data or raw archives.
set -euo pipefail
M=/root/gpufree-data/mj
source "$M/env.sh"
cd "$M"
export PREP_OUT="$M/data/games_hanchan"
export PREP_IDX="$M/data/index_hanchan"
export PREP_KIND=hanchan KEEP_HANCHAN=1 MIN_FREE_GB=12
export NPROC=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$PREP_IDX" "$PREP_OUT"
for year in 2022 2023 2024 2025; do
  echo "$(date -Is) START jade-$year"
  nice -n 10 python -u code/prep.py "$M/data/raw/majsoul-jade-$year.tar"
  echo "$(date -Is) VALIDATED jade-$year"
done
echo PREP_HANCHAN_DONE
# Publication to data/index is deliberately separate and follows validation review.
