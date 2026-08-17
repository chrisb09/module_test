#!/usr/bin/env bash
set -u

# Our script dir:
SCRIPT_DIR=$(dirname "$0")

usage() {
  cat <<'EOF'
Usage: examine_md.sh [options] [file.md]

Summarize a SmartSim/PhyDLL test-matrix markdown file: success/failure
percentages and an automatic failure-pattern analysis.

Options:
  -n, --top N      Show the top N failure patterns (default: 10)
      --no-color   Disable ANSI colors
  -h, --help       Show this help

Arguments:
  file.md          Markdown matrix to analyze (default: test_matrix.md)
EOF
}

TOP=10
NOCOLOR=""
NAME="test_matrix.md"

while [ $# -gt 0 ]; do
  case "$1" in
    -n|--top)
      TOP="${2:-10}"
      shift 2
      ;;
    --top=*)
      TOP="${1#*=}"
      shift
      ;;
    -n[0-9]*)
      TOP="${1#-n}"
      shift
      ;;
    --no-color)
      NOCOLOR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      [ $# -gt 0 ] && NAME="$1" && shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      NAME="$1"
      shift
      ;;
  esac
done

FILENAME="${SCRIPT_DIR}/${NAME}"

if [ ! -f "$FILENAME" ]; then
  echo "File ${FILENAME} does not exist. Use $0 [path_to_file]"
  exit 1
fi

if [ -t 1 ] && [ -z "$NOCOLOR" ]; then
  RED=$(printf '\033[0;31m')
  GRN=$(printf '\033[0;32m')
  BLD=$(printf '\033[1m')
  DIM=$(printf '\033[2m')
  RST=$(printf '\033[0m')
else
  RED=""; GRN=""; BLD=""; DIM=""; RST=""
fi

awk -F'|' \
  -v red="$RED" -v grn="$GRN" -v bld="$BLD" -v dim="$DIM" -v rst="$RST" \
  -v top="$TOP" -v fname="$FILENAME" \
'
function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s; }
function pct(n, tot) { return (tot > 0) ? sprintf("%.1f", n * 100.0 / tot) : "0.0"; }

BEGIN {
  header_found = 0;
  success = 0; failure = 0;
  naxes = split("Provider,DL,Layout,ScoreP,API,Dev,Model,St/Cl/B", axes, ",");
  ncombos = split("Provider,API,St/Cl/B;Provider,API,Dev;Provider,Dev,Model;API,Dev;ScoreP,Dev;Model,St/Cl/B", combos, ";");
}

/^\|/ {
  if (!header_found) {
    if ($0 ~ /Provider/ && $0 ~ /Stat/) {
      for (i = 1; i <= NF; i++) {
        name = trim($i);
        if (name != "") col[name] = i;
      }
      header_found = 1;
    }
    next;
  }
  if ($0 ~ /^[ |:-]+$/) next;

  if (col["Stat"] == "") next;
  stat = trim($(col["Stat"]));
  if (stat ~ /✅/) { success++; st = 1; }
  else if (stat ~ /❌/) { failure++; st = 0; }
  else next;

  if (st == 0) {
    for (i = 1; i <= naxes; i++) {
      a = axes[i];
      ci = col[a];
      if (ci == "") continue;
      v = trim($(ci));
      if (v == "") v = "-";
      axis_fail[a SUBSEP v]++;
    }
    for (c = 1; c <= ncombos; c++) {
      n = split(combos[c], parts, ",");
      ok = 1; key = "";
      for (k = 1; k <= n; k++) {
        ci = col[parts[k]];
        if (ci == "") { ok = 0; break; }
        v = trim($(ci));
        if (v == "") v = "-";
        key = key ((k > 1) ? " / " : "") v;
      }
      if (ok) combo_count[key]++;
    }
  }
}

END {
  total = success + failure;

  print "";
  print bld "=====================================================================" rst;
  print bld "  Test Matrix Report: " fname rst;
  print bld "=====================================================================" rst;
  print "";
  printf "  %-10s %d\n", "Total:", total;
  printf "  %-10s %s%d%s  (%s%%)\n", "Success:", grn, success, rst, pct(success, total);
  printf "  %-10s %s%d%s  (%s%%)\n", "Failed:", red, failure, rst, pct(failure, total);

  frac = (total > 0) ? success / total : 0;
  f = int(frac * 40 + 0.5);
  filled = ""; empty = "";
  for (i = 1; i <= 40; i++) { if (i <= f) filled = filled "█"; else empty = empty "░"; }
  printf "  %s%s%s%s  %s%% pass\n", grn, filled, dim empty, rst, pct(success, total);
  print "";

  if (total == 0) {
    print "  No completed runs (✅/❌) found in this file.";
    exit 0;
  }

  print bld "Failure breakdown by axis" rst;
  print "";
  for (i = 1; i <= naxes; i++) {
    a = axes[i];
    if (col[a] == "") continue;
    delete list; delete sorted; m = 0; useful = 0;
    for (k in axis_fail) {
      split(k, kv, SUBSEP);
      if (kv[1] != a) continue;
      if (kv[2] != "-") useful = 1;
      list[++m] = sprintf("%09d\t%s", axis_fail[k], kv[2]);
    }
    if (m == 0 || !useful) continue;
    n = asort(list, sorted);
    printf "  %s%s%s\n", bld, a, rst;
    for (j = n; j >= 1; j--) {
      split(sorted[j], parts, "\t");
      c = parts[1] + 0; val = parts[2];
      printf "    %-16s %4d  (%s%% of failures)\n", val, c, pct(c, failure);
    }
    print "";
  }

  print bld "Top failing patterns" rst;
  print dim "  (combinations of Provider / API / Dev / Model / StClB / ScoreP)" rst;
  print "";
  if (failure > 0) {
    delete combolist; delete combosorted; m = 0;
    for (key in combo_count) combolist[++m] = sprintf("%09d\t%s", combo_count[key], key);
    n = asort(combolist, combosorted);
    shown = 0;
    for (j = n; j >= 1 && shown < top; j--) {
      split(combosorted[j], parts, "\t");
      c = parts[1] + 0; key = parts[2];
      printf "  %4d  %s(%s%%)%s  %s\n", c, red, pct(c, failure), rst, key;
      shown++;
    }
  } else {
    print "  (no failures)";
  }
  print "";
}
' "$FILENAME"
