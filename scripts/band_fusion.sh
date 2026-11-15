#!/usr/bin/env bash
# Report 2, section 8.f: full-frame vs bands vs fused median + Sobel pipeline.
#
#   bash scripts/band_fusion.sh build
#   bash scripts/band_fusion.sh time         VGA/HD, n = 3 and 5, bands of 16/32/64 rows
#   bash scripts/band_fusion.sh cachegrind   n = 5, B = 32 (laptop cache sizes)
#   bash scripts/band_fusion.sh perf         n = 5, B = 32
#   python3 scripts/summarize_band_fusion.py results_local/band_fusion
#
# Input images: $IMAGES/VGA.png and $IMAGES/HD.png. Results in $OUT/band_fusion.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGES=${IMAGES:-inputs}
OUT=${OUT:-results_local}/band_fusion
WARM=${WARM:-10}
RUNS=${RUNS:-110}
BIN=bin/band_fusion
export OMP_NUM_THREADS=1
mkdir -p "$OUT" bin

build() {
    pkg-config --exists opencv4 || { echo "OpenCV 4 not found (Fedora: sudo dnf install opencv-devel)"; exit 1; }
    local cg=""
    if [ -f /usr/include/valgrind/cachegrind.h ]; then
        cg="-DCACHEGRIND_SCOPE"
    else
        echo "no valgrind/cachegrind.h: building without the Cachegrind markers"
    fi
    g++ -std=c++17 -O2 -g -fopenmp -Wall -Wextra $cg -I src/include \
        src/benchmarks/band_fusion_bench.cpp src/optimizations/my_functions_optimized_4.cpp \
        -o "$BIN" $(pkg-config --cflags --libs opencv4) -lm
    echo "built $BIN"
}

image() {
    local f="$IMAGES/$1.png"
    [ -f "$f" ] || { echo "missing input image $f"; exit 1; }
    echo "$f"
}

# run <res> <n> <mode> <band height>
run() {
    local name="${1}_n${2}_${3}_B${4}"
    echo "== $name"
    "$BIN" "$3" "$2" "$(image $1)" "$WARM" "$RUNS" "$4" "$OUT/$name.csv" 2>&1 | tee "$OUT/$name.log"
}

cachegrind() {
    local name="${1}_n${2}_${3}_B${4}"
    echo "cachegrind $name"
    valgrind --tool=cachegrind --cache-sim=yes --branch-sim=no --instr-at-start=no \
        --I1=32768,8,64 --D1=32768,8,64 --LL=16777216,16,64 \
        --cachegrind-out-file="$OUT/cg_$name.out" \
        "$BIN" "$3" "$2" "$(image $1)" 1 2 "$4" "$OUT/cg_$name.csv" --no-validate \
        > "$OUT/cg_$name.log" 2>&1
    grep -E "D1  misses|LLd misses" "$OUT/cg_$name.log" || true
}

perf_run() {
    local name="${1}_n${2}_${3}_B${4}"
    echo "perf $name"
    perf stat -r 3 -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses \
        -o "$OUT/perf_$name.txt" \
        -- "$BIN" "$3" "$2" "$(image $1)" "$WARM" "$RUNS" "$4" "$OUT/perf_$name.csv" --no-validate \
        > "$OUT/perf_$name.stdout.txt" 2>&1
}

case "${1:-}" in
    build) build ;;
    time)
        [ -x "$BIN" ] || build
        for res in VGA HD; do
            for n in 3 5; do
                run "$res" "$n" baseline 32
                for B in 16 32 64; do run "$res" "$n" band "$B"; done
                run "$res" "$n" fused 32
            done
        done ;;
    cachegrind)
        command -v valgrind >/dev/null || { echo "valgrind not found (sudo dnf install valgrind valgrind-devel)"; exit 1; }
        [ -x "$BIN" ] || build
        for res in VGA HD; do
            for mode in baseline band fused; do cachegrind "$res" 5 "$mode" 32; done
        done ;;
    perf)
        command -v perf >/dev/null || { echo "perf not found (sudo dnf install perf)"; exit 1; }
        [ -x "$BIN" ] || build
        for res in VGA HD; do
            for mode in baseline band fused; do perf_run "$res" 5 "$mode" 32; done
        done ;;
    *)
        echo "usage: bash scripts/band_fusion.sh build|time|cachegrind|perf"
        exit 2 ;;
esac
