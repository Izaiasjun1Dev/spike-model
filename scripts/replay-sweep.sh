#!/usr/bin/env bash
# Replays a routing trace across cache sizes, with and without pre-gate
# prefetch, and collects RESULT lines into a CSV.
# Usage: scripts/replay-sweep.sh <test-file> <trace-base> [tokens-per-seq] [compute-us] [seqs]
# Default seqs are the test split used by tools/routing-trace/sim.py.
set -euo pipefail

file=${1:?usage: scripts/replay-sweep.sh <test-file> <trace-base> [tokens] [compute-us] [seqs]}
trace=${2:?usage: scripts/replay-sweep.sh <test-file> <trace-base> [tokens] [compute-us] [seqs]}
tokens=${3:-200}
compute_us=${4:-270}
seqs=${5:-1,3,6,7,9,15,16,17,18,19,21}
bin=target/release/expert-bench
out=results/$(date +%Y%m%d-%H%M%S)-replay.csv

cargo build --release --quiet
mkdir -p results
echo "mode,cache_pct,prefetch,threads,compute_us,tokens,hit_pct,gb_per_token,gbps,stall_ms_per_token,ms_per_token,target_tok_s,wasted_prefetch,resident_pct,resident_end_pct,policy,tiered,low_reads,bypassed,buf_gib" > "$out"

for cache in 20 25 30; do
  for prefetch in "" "--prefetch"; do
    line=$("$bin" replay --path "$file" --trace "$trace" --cache-pct "$cache" $prefetch \
      --seqs "$seqs" --max-tokens "$tokens" --compute-us "$compute_us" 2>/dev/null | grep '^RESULT') || {
      echo "skip: cache=$cache $prefetch failed" >&2
      continue
    }
    echo "$line"
    echo "$line" | sed -e 's/^RESULT //' -e 's/[a-z_]*=//g' -e 's/ /,/g' >> "$out"
  done
done
echo "saved $out"
