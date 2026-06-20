#!/usr/bin/env bash
# run_all.sh — full benchmark suite
# Produces 3 submission videos + 5 CSVs (seed 42).
# Videos written to /tmp first (AVFoundation workaround on macOS) then moved here.

set -euo pipefail
PYTHON="conda run -n robothon python3"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> validate_scene"
$PYTHON "$DIR/scripts/validate_scene.py"

echo "==> single-egg medium (10 eps)  [video: benchmark.mp4]"
$PYTHON "$DIR/video/record_demo.py" \
    --episodes 10 --tier medium --seed 42 --dual-cam \
    --out /tmp/benchmark.mp4 --log "$DIR/results.csv"
mv /tmp/benchmark.mp4 "$DIR/benchmark.mp4"

echo "==> single-egg stress (10 eps)  [CSV only]"
$PYTHON "$DIR/video/record_demo.py" \
    --episodes 10 --tier stress --seed 42 \
    --out /tmp/_stress_scratch.mp4 --log "$DIR/results_stress.csv"
rm -f /tmp/_stress_scratch.mp4

echo "==> single-egg extreme (10 eps)  [CSV only]"
$PYTHON "$DIR/video/record_demo.py" \
    --episodes 10 --tier extreme --seed 42 \
    --out /tmp/_extreme_scratch.mp4 --log "$DIR/results_extreme.csv"
rm -f /tmp/_extreme_scratch.mp4

echo "==> two-egg medium (10 eps)  [CSV only — no video]"
$PYTHON "$DIR/video/record_demo.py" \
    --two-egg --episodes 10 --tier medium --seed 42 \
    --out /tmp/_te_medium_scratch.mp4 --log "$DIR/results_two_egg_medium.csv"
rm -f /tmp/_te_medium_scratch.mp4

echo "==> two-egg extreme (10 eps)  [video: benchmark_two_egg_extreme.mp4]"
$PYTHON "$DIR/video/record_demo.py" \
    --two-egg --episodes 10 --tier extreme --seed 42 --dual-cam \
    --out /tmp/benchmark_two_egg_extreme.mp4 --log "$DIR/results_two_egg_extreme.csv"
mv /tmp/benchmark_two_egg_extreme.mp4 "$DIR/benchmark_two_egg_extreme.mp4"

echo "==> highlight reel  [video: demo_highlight.mp4]"
$PYTHON "$DIR/video/record_highlight.py" --out /tmp/demo_highlight.mp4
mv /tmp/demo_highlight.mp4 "$DIR/demo_highlight.mp4"

echo ""
echo "Done. 3 videos + 5 CSVs written to $DIR"
echo "  benchmark.mp4                — single-egg medium 10/10"
echo "  benchmark_two_egg_extreme.mp4 — two-egg extreme  8/10"
echo "  demo_highlight.mp4           — 78s showcase across all tiers"
echo "Summarise results:"
echo "  $PYTHON $DIR/scripts/summarize_results.py $DIR/results.csv"
echo "  $PYTHON $DIR/scripts/summarize_results.py $DIR/results_two_egg_medium.csv"
