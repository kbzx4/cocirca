#!/usr/bin/env bash
# Download a large real recording for latency tests: PhysioNet "BIG IDEAs Lab Glycemic Variability and
# Wearable Device Data" v1.1.2, participant 001 (Empatica E4, ~9 days): ACC 32 Hz (879 MB), EDA, TEMP, HR, IBI.
# Open data (ODC-By 1.0): https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/
# Uses PhysioNet's AWS mirror in parallel 16 MB parts (one connection is slow), then checks SHA-256 sums.
#
#   bash desktop/scripts/fetch_big_ideas.sh [OUT_DIR]      (default: bench_data/big-ideas-001, git-ignored)
set -u
OUT="${1:-bench_data/big-ideas-001}"
BASE="https://physionet-open.s3.amazonaws.com/big-ideas-glycemic-wearable/1.1.2"
FILES="${FILES:-HR IBI TEMP EDA ACC}"   # e.g. FILES="HR TEMP" for a quick test
mkdir -p "$OUT"

fetch() { # URL FILE: parallel ranged download
  local url="$1" out="$2" ch=$((16 * 1024 * 1024)) len="" try
  for try in 1 2 3 4 5; do
    len=$(curl -sIL --max-time 60 "$url" | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{v=$2} END{print v}')
    [ -n "$len" ] && break
    sleep 3
  done
  [ -z "$len" ] && { echo "cannot reach $url" >&2; return 1; }
  local parts="$out.parts" n=$(( (len + ch - 1) / ch ))
  mkdir -p "$parts"
  seq 0 $((n - 1)) | xargs -P 16 -I{} bash -c '
    i={}; s=$((i * '$ch')); e=$((s + '$ch' - 1)); [ $e -ge '$len' ] && e=$(('$len' - 1))
    f="'$parts'/$(printf %05d $i)"; want=$((e - s + 1))
    for try in 1 2 3 4 5 6; do
      [ -f "$f" ] && [ "$(wc -c < "$f")" -eq $want ] && exit 0
      curl -sL --max-time 600 -r $s-$e -o "$f" "'$url'"
    done
    echo "part $i failed" >&2; exit 1' || return 1
  cat "$parts"/* > "$out" && rm -rf "$parts"
  echo "$(basename "$out"): $(wc -c < "$out") bytes"
}

curl -sL --max-time 60 "$BASE/SHA256SUMS.txt" | grep -E " 001/($(echo $FILES | tr ' ' '|'))_001\.csv$" | sed 's# 001/# #' > "$OUT/SHA256SUMS-001.txt"
for f in $FILES; do
  [ -f "$OUT/${f}_001.csv" ] || fetch "$BASE/001/${f}_001.csv" "$OUT/${f}_001.csv" || exit 1
done
cd "$OUT" && sha256sum -c SHA256SUMS-001.txt
