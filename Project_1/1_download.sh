#!/usr/bin/env bash
# Downloads the Project 1 datasets (ExtractBench + CUAD by default, about 60 MB) from their original hosts into data/raw/ (git-ignored).
# Usage: bash 1_download.sh [--accept-licenses] [--only extractbench|cuad|invoicebenchmark|drew_ipp] [--force]
#   Optional extra invoice sets: --only invoicebenchmark, --only drew_ipp (then python 2_prepare.py --sources extractbench cuad invoicebenchmark drew_ipp)
# Needs bash, curl, git, unzip, shasum.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="${RAW_DIR:-$HERE/data/raw}"
SELECTION="$HERE/selection.tsv"

ACCEPT=0
FORCE=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --accept-licenses) ACCEPT=1 ;;
    --force) FORCE=1 ;;
    --only) ONLY="${2:?--only needs a source name}"; shift ;;
    -h|--help) sed -n '2,4p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# ---- pinned revisions (fixed commits, so everyone gets the same data) ----
EB_REV="f9b910906cd7e73cc9eb5379a8d5d935c67b92c4"
EB_SHORT_SHA="3373bfdd44077f852b8f60e1d5b878a582c0c9517b22bcc5f4ec0d28f2ada36c"
CUAD_URL="https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
CUAD_SHA="f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a"
IB_REPO="https://huggingface.co/datasets/jngb-labs/InvoiceBenchmark"
IB_COMMIT="f0699d8c94468fb2fefaa22f669684b22e68c3d3"
DREW_REPO="https://huggingface.co/datasets/drew-ipp/invoice-extraction-benchmark"
DREW_COMMIT="446bab08662790cb39457792243bff5704373b95"

licence_notice() {
  cat <<'EOF'

Project 1 uses two public datasets (plus two optional invoice sets, see the end of this notice). You download them yourself from the original hosts.
Their licences, summarised (read the originals; this is not legal advice):

  ExtractBench   Apache-2.0. Source PDFs are real public records (SEC and regulatory filings,
                 government forms, court exhibits) and may contain third-party personal data.
                 https://huggingface.co/datasets/llamaindex/ExtractBench
                 https://raw.githubusercontent.com/run-llama/ExtractBench/main/LICENSE
  CUAD           CC BY 4.0 for the labels. Contracts are public SEC EDGAR filings; the creators
                 give no assurance about their licence status.
                 https://www.atticusprojectai.org/cuad
                 https://www.atticusprojectai.org/legal
  Optional extras (only if you download them with --only):
  InvoiceBenchmark
                 MIT (dataset card, by Jakob Neugebauer).
                 https://huggingface.co/datasets/jngb-labs/InvoiceBenchmark
  invoice-extraction-benchmark
                 Data CC BY 4.0, code MIT (by InvoiceParser Pro; a vendor, not independent).
                 https://huggingface.co/datasets/drew-ipp/invoice-extraction-benchmark

Conditions: keep the attribution (see README, Data section), use for the hackathon only,
do not commit the downloaded data, do not publish it outside the hackathon.
EOF
}

if [ "$ACCEPT" -ne 1 ]; then
  licence_notice
  if [ ! -t 0 ]; then
    echo; echo "Not interactive. Re-run with --accept-licenses once you have read the licences." >&2
    exit 1
  fi
  printf '\nType "yes" to accept these terms and download: '
  read -r answer
  [ "$answer" = "yes" ] || { echo "Aborted."; exit 1; }
fi

mkdir -p "$RAW"

want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
done_marker() { [ "$FORCE" -eq 0 ] && [ -f "$RAW/$1/.complete" ]; }

check_sha() { # file expected
  local got
  got="$(shasum -a 256 "$1" | cut -d' ' -f1)"
  if [ "$got" != "$2" ]; then
    echo "CHECKSUM MISMATCH for $1" >&2; echo "  expected $2" >&2; echo "  got      $got" >&2
    exit 1
  fi
}

fetch() { curl -fsSL --retry 3 --retry-delay 2 -o "$2" "$1"; }

# ---------------- ExtractBench (short split, PDFs <= 1 MB) ----------------
get_extractbench() {
  if done_marker extractbench; then echo "extractbench: already complete (use --force to redo)"; return; fi
  local dir="$RAW/extractbench" base="https://huggingface.co/datasets/llamaindex/ExtractBench/resolve/$EB_REV"
  rm -rf "$dir"; mkdir -p "$dir/docs/short"
  echo "extractbench: labels (short.jsonl, ~31 MB)"
  fetch "$base/short.jsonl" "$dir/short.jsonl"
  check_sha "$dir/short.jsonl" "$EB_SHORT_SHA"
  local n; n="$(grep -vc '^#' "$SELECTION")"
  echo "extractbench: $n selected PDFs (~44 MB)"
  grep -v '^#' "$SELECTION" | cut -f2 | sed 's/ /%20/g' \
    | xargs -P 8 -I{} sh -c 'f="$1"; curl -fsSL --retry 3 --retry-delay 2 -o "$2/$(printf "%s" "$f" | sed "s/%20/ /g")" "$3/$f" || { echo "FAILED: $f" >&2; exit 255; }' _ {} "$dir" "$base"
  local bad=0 id pdf bytes
  while IFS="$(printf '\t')" read -r id pdf bytes; do
    case "$id" in \#*) continue ;; esac
    if [ "$(wc -c < "$dir/$pdf" | tr -d ' ')" != "$bytes" ]; then echo "SIZE MISMATCH: $pdf" >&2; bad=1; fi
  done < "$SELECTION"
  [ "$bad" -eq 0 ] || exit 1
  touch "$dir/.complete"
}

# ---------------- CUAD ----------------
get_cuad() {
  if done_marker cuad; then echo "cuad: already complete (use --force to redo)"; return; fi
  local dir="$RAW/cuad"; rm -rf "$dir"; mkdir -p "$dir"
  echo "cuad: data.zip (~18 MB)"
  fetch "$CUAD_URL" "$dir/data.zip"
  check_sha "$dir/data.zip" "$CUAD_SHA"
  unzip -q -o -j "$dir/data.zip" CUADv1.json -d "$dir"   # one named member only
  rm -f "$dir/data.zip"
  touch "$dir/.complete"
}

# ---------------- HF git repos (pinned commit, .git removed afterwards) ----------------
# Images are Git LFS files: fetch the real content at the pinned commit and verify its sha256.
fetch_lfs_files() { # dir repo_url commit
  local dir="$1" repo="$2" commit="$3" f rel oid size got n=0
  while IFS= read -r f; do
    head -c 40 "$f" | grep -q '^version https://git-lfs' || continue
    rel="${f#$dir/}"; oid="$(sed -n 's/^oid sha256://p' "$f")"
    fetch "$repo/resolve/$commit/$(printf '%s' "$rel" | sed 's/ /%20/g')" "$f.part"
    got="$(shasum -a 256 "$f.part" | cut -d' ' -f1)"
    [ "$got" = "$oid" ] || { echo "LFS CHECKSUM MISMATCH: $rel" >&2; exit 1; }
    mv "$f.part" "$f"; n=$((n+1))
  done < <(find "$dir" -type f -size -300c)
  echo "  fetched $n image files (LFS)"
}

get_hf_repo() { # name url commit [drop-dir: a folder of LFS images we do not need]
  if done_marker "$1"; then echo "$1: already complete (use --force to redo)"; return; fi
  local dir="$RAW/$1"; rm -rf "$dir"
  echo "$1: git clone (pinned $3)"
  GIT_LFS_SKIP_SMUDGE=1 git clone --quiet "$2" "$dir"
  git -C "$dir" -c advice.detachedHead=false checkout --quiet "$3"
  rm -rf "$dir/.git" "$dir/__pycache__"
  [ -z "${4:-}" ] || rm -rf "$dir/$4"
  fetch_lfs_files "$dir" "$2" "$3"
  touch "$dir/.complete"
}

want extractbench    && get_extractbench
want cuad            && get_cuad
# The two invoice sets are optional extras: not downloaded unless asked for with --only.
want_opt() { [ "$ONLY" = "$1" ]; }
want_opt invoicebenchmark && get_hf_repo invoicebenchmark "$IB_REPO" "$IB_COMMIT" output/png   # we use its real PDFs, not the 32 MB of PNGs
want_opt drew_ipp        && get_hf_repo drew-ipp-invoices "$DREW_REPO" "$DREW_COMMIT"

echo
echo "Done. Downloaded data (untrusted, do not run code from inside it):"
du -sh "$RAW"/*/ 2>/dev/null || true
echo "Next: python 2_prepare.py"
