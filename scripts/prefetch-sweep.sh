#!/usr/bin/env bash
# Replays a trace with prefetch at different distances and widths: off,
# pre-gate one layer ahead, and tools/routing-trace/predictor.py's
# predictions 2-3 layers ahead (<trace>.pred<d>.bin). Collects RESULT lines.
# Usage: scripts/prefetch-sweep.sh <test-file> <trace-base> [cache-pcts] [tokens] [compute-us] [seqs]
set -euo pipefail

file=${1:?usage: scripts/prefetch-sweep.sh <test-file> <trace-base> [cache-pcts] [tokens] [compute-us] [seqs]}
trace=${2:?usage: scripts/prefetch-sweep.sh <test-file> <trace-base> [cache-pcts] [tokens] [compute-us] [seqs]}
caches=${3:-20,25}
tokens=${4:-200}
compute_us=${5:-270}
seqs=${6:-1,3,6,7,9,15,16,17,18,19,21}
bin=target/release/expert-bench
out=results/$(date +%Y%m%d-%H%M%S)-prefetch.csv

cargo build --release --quiet
mkdir -p results
echo "mode,cache_pct,prefetch,threads,compute_us,tokens,hit_pct,gb_per_token,gbps,stall_ms_per_token,ms_per_token,target_tok_s,wasted_prefetch,resident_pct,resident_end_pct,policy,tiered,low_reads,bypassed,buf_gib,skipped,prefetch_dist,prefetch_width" > "$out"

for cache in ${caches//,/ }; do
  for pf in "" "--prefetch" "--prefetch --prefetch-dist 2" "--prefetch --prefetch-dist 2 --prefetch-width 12" \
            "--prefetch --prefetch-dist 3" "--prefetch --prefetch-dist 3 --prefetch-width 12"; do
    line=$("$bin" replay --path "$file" --trace "$trace" --cache-pct "$cache" $pf \
      --seqs "$seqs" --max-tokens "$tokens" --compute-us "$compute_us" 2>/dev/null | grep '^RESULT') || {
      echo "skip: cache=$cache $pf failed" >&2
      continue
    }
    echo "$line"
    echo "$line" | sed -e 's/^RESULT //' -e 's/[a-z_]*=//g' -e 's/ /,/g' >> "$out"
  done
done
echo "saved $out"
