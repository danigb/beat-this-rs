#!/usr/bin/env bash
# Run examples/memory_probe over stages and durations, one process per cell, and print a markdown
# table with the median of --runs runs per cell.
#
#   scripts/memory-probe.sh [--minutes "5 10 20 60"]
#                           [--stages "models resample mel-input mel full"]
#                           [--runs 3] [--rate 48000] [--model PATH]
#
# Stages (see examples/memory_probe.rs): models resample resample-stream mel-input mel full
# full-owned stream.
#
# Columns: the probe's own ru_maxrss, `time`'s maximum resident set size, and (macOS only) the
# `peak memory footprint` from `/usr/bin/time -l`. On Linux `/usr/bin/time -v` is used and the
# footprint column is n/a. Cells run one at a time: parallel runs distort timing and RAM.
set -euo pipefail
cd "$(dirname "$0")/.."

minutes="5 10 20 60"
stages="models resample mel-input mel full"
runs=3
rate=48000
model=""
while [ $# -gt 0 ]; do
  case "$1" in
    --minutes) minutes="$2"; shift 2 ;;
    --stages) stages="$2"; shift 2 ;;
    --runs) runs="$2"; shift 2 ;;
    --rate) rate="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cargo build --release --example memory_probe >&2
bin=target/release/examples/memory_probe

if [ "$(uname -s)" = "Darwin" ]; then
  time_cmd=(/usr/bin/time -l)
  flavour=bsd
else
  time_cmd=(/usr/bin/time -v)
  flavour=gnu
fi

# Median of the numbers on stdin (one per line); empty input prints n/a.
median() {
  sort -n | awk '{ v[NR] = $1 } END { if (NR == 0) print "n/a"; else printf "%.1f\n", v[int((NR + 1) / 2)] }'
}

model_args=()
[ -n "$model" ] && model_args=(--model "$model")

host="$(uname -sm)"
if [ "$flavour" = bsd ]; then
  host="$host, $(sysctl -n machdep.cpu.brand_string), macOS $(sw_vers -productVersion), $(( $(sysctl -n hw.memsize) / 1073741824 )) GiB"
fi
echo "host: $host"
echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet HEAD 2>/dev/null || echo '+dirty')  rate: $rate  runs: $runs (median)"
echo
echo "| stage | min | probe rss MiB | time maxrss MiB | footprint MiB |"
echo "|---|---|---|---|---|"

for m in $minutes; do
  for s in $stages; do
    probe_vals=""; rss_vals=""; fp_vals=""; model_name="?"
    for _ in $(seq "$runs"); do
      err="$(mktemp)"
      line="$("${time_cmd[@]}" "$bin" --minutes "$m" --stage "$s" --rate "$rate" ${model_args[@]+"${model_args[@]}"} 2>"$err")" || {
        echo "probe failed: stage=$s minutes=$m" >&2; cat "$err" >&2; rm -f "$err"; exit 1; }
      probe_vals="$probe_vals$(echo "$line" | sed -n 's/.*peak_rss_mib=\([0-9.]*\).*/\1/p')"$'\n'
      model_name="$(echo "$line" | sed -n 's/.*model=\([^ ]*\).*/\1/p')"
      if [ "$flavour" = bsd ]; then
        rss_vals="$rss_vals$(awk '/maximum resident set size/ { printf "%.1f\n", $1 / 1048576 }' "$err")"$'\n'
        fp_vals="$fp_vals$(awk '/peak memory footprint/ { printf "%.1f\n", $1 / 1048576 }' "$err")"$'\n'
      else
        rss_vals="$rss_vals$(awk -F': ' '/Maximum resident set size/ { printf "%.1f\n", $2 / 1024 }' "$err")"$'\n'
      fi
      rm -f "$err"
    done
    p="$(printf '%s' "$probe_vals" | sed '/^$/d' | median)"
    r="$(printf '%s' "$rss_vals" | sed '/^$/d' | median)"
    f="$(printf '%s' "$fp_vals" | sed '/^$/d' | median)"
    echo "| $s | $m | $p | $r | $f |"
  done
done
echo
echo "beat model: $model_name"
