#!/usr/bin/env bash
# Replays a routing trace at small RAM budgets (GB on the target model),
# comparing global LRU, per-layer LFU with bypass, and LFU with two-resolution
# experts (low-weight picks served by a low-precision copy).
# Usage: scripts/lowram-sweep.sh <test-file> <trace-base> [tokens] [compute-us] [budgets-gb] [seqs]
set -euo pipefail

file=${1:?usage: scripts/lowram-sweep.sh <test-file> <trace-base> [tokens] [compute-us] [budgets-gb] [seqs]}
trace=${2:?usage: scripts/lowram-sweep.sh <test-file> <trace-base> [tokens] [compute-us] [budgets-gb] [seqs]}
tokens=${3:-200}
compute_us=${4:-270}
budgets=${5:-8 16 24}
seqs=${6:-1,3,6,7,9,15,16,17,18,19,21}
bin=target/release/expert-bench
out=results/$(date +%Y%m%d-%H%M%S)-lowram.csv

cargo build --release --quiet
mkdir -p results
echo "cache_gb,mode,cache_pct,prefetch,threads,compute_us,tokens,hit_pct,gb_per_token,gbps,stall_ms_per_token,ms_per_token,target_tok_s,wasted_prefetch,resident_pct,resident_end_pct,policy,tiered,low_reads,bypassed,buf_gib" > "$out"

for gb in $budgets; do
  for mode in "--policy lru" "--policy lfu" "--policy lfu --tiered"; do
    line=$("$bin" replay --path "$file" --trace "$trace" --cache-gb "$gb" $mode \
      --seqs "$seqs" --max-tokens "$tokens" --compute-us "$compute_us" 2>/dev/null | grep '^RESULT') || {
      echo "skip: $gb GB $mode failed" >&2
      continue
    }
    echo "$gb GB $line"
    echo "$gb,$(echo "$line" | sed -e 's/^RESULT //' -e 's/[a-z_]*=//g' -e 's/ /,/g')" >> "$out"
  done
done
echo "saved $out"
