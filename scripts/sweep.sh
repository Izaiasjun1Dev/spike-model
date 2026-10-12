#!/usr/bin/env bash
# Runs the method × lookahead × compute matrix and collects RESULT lines into a CSV.
# Usage: scripts/sweep.sh <test-file> [preset] [tokens]
set -euo pipefail

file=${1:?usage: scripts/sweep.sh <test-file> [preset] [tokens]}
preset=${2:-qwen3-235b-a22b}
tokens=${3:-5}
bin=target/release/expert-bench
out=results/$(date +%Y%m%d-%H%M%S)-$preset.csv

cargo build --release --quiet
mkdir -p results
echo "method,expert_bytes,topk,layers,lookahead,threads,compute_us,gbps,tok_s" > "$out"

for method in pread metal mmap; do
  for lookahead in 0 1 2 4; do
    for compute_us in 0 2000; do
      line=$("$bin" run --path "$file" --method "$method" --preset "$preset" \
        --tokens "$tokens" --lookahead "$lookahead" --compute-us "$compute_us" 2>/dev/null | grep '^RESULT') || {
        echo "skip: $method lookahead=$lookahead compute_us=$compute_us failed" >&2
        continue
      }
      echo "$line"
      # "RESULT k=v k=v ..." -> "v,v,..."
      echo "$line" | sed -e 's/^RESULT //' -e 's/[a-z_]*=//g' -e 's/ /,/g' >> "$out"
    done
  done
done
echo "saved $out"
