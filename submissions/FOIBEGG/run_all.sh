#!/usr/bin/env bash
# run_all.sh — full benchmark suite
# Usage: bash run_all.sh
# All videos are written to /tmp first (AVFoundation workaround on macOS) then moved here.

set -euo pipefail
PYTHON="conda run -n robothon python3"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> validate_scene"
$PYTHON "$DIR/scripts/validate_scene.py"

echo "==> validate_registration"
$PYTHON "$DIR/scripts/validate_registration.py" "$DIR/registration.json"

echo "==> single-egg medium (10 eps)"
$PYTHON "$DIR/video/record_demo.py" \
    --episodes 10 --tier medium --seed 42 --dual-cam \
    --out /tmp/benchmark.mp4 --log "$DIR/results.csv"
mv /tmp/benchmark.mp4 "$DIR/benchmark.mp4"

echo "==> single-egg stress (10 eps)"
$PYTHON "$DIR/video/record_demo.py" \
    --episodes 10 --tier stress --seed 42 --dual-cam \
    --out /tmp/benchmark_stress.mp4 --log "$DIR/results_stress.csv"
mv /tmp/benchmark_stress.mp4 "$DIR/benchmark_stress.mp4"

echo "==> single-egg extreme (10 eps)"
$PYTHON "$DIR/video/record_demo.py" \
    --episodes 10 --tier extreme --seed 42 --dual-cam \
    --out /tmp/benchmark_extreme.mp4 --log "$DIR/results_extreme.csv"
mv /tmp/benchmark_extreme.mp4 "$DIR/benchmark_extreme.mp4"

echo "==> two-egg medium (10 eps)"
$PYTHON "$DIR/video/record_demo.py" \
    --two-egg --episodes 10 --tier medium --seed 42 --dual-cam \
    --out /tmp/benchmark_two_egg_medium.mp4 --log "$DIR/results_two_egg_medium.csv"
mv /tmp/benchmark_two_egg_medium.mp4 "$DIR/benchmark_two_egg_medium.mp4"

echo "==> two-egg extreme (10 eps)"
$PYTHON "$DIR/video/record_demo.py" \
    --two-egg --episodes 10 --tier extreme --seed 42 --dual-cam \
    --out /tmp/benchmark_two_egg_extreme.mp4 --log "$DIR/results_two_egg_extreme.csv"
mv /tmp/benchmark_two_egg_extreme.mp4 "$DIR/benchmark_two_egg_extreme.mp4"

echo ""
echo "Done. CSVs and videos written to $DIR"
echo "Summarise results:"
echo "  $PYTHON $DIR/scripts/summarize_results.py $DIR/results.csv"
echo "  $PYTHON $DIR/scripts/summarize_results.py $DIR/results_two_egg_medium.csv"
