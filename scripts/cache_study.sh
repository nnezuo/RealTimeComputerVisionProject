#!/usr/bin/env bash
# Cache study (report 2): Sobel and median, timing + Cachegrind + perf.
#
#   bash scripts/cache_study.sh build        compile the 4 versions (normal and Cachegrind builds)
#   bash scripts/cache_study.sh time         Sobel (row step W and W+64) and median n=3, timing
#   bash scripts/cache_study.sh sweep        opt4 median for n = 3 ... 31
#   bash scripts/cache_study.sh cachegrind   simulated L1/LL misses (laptop cache sizes)
#   bash scripts/cache_study.sh perf         hardware counters (cycles, instructions, L1)
#
# Input images: $IMAGES/VGA.png (640x480) and $IMAGES/HD.png (1280x720), grey.
# Results go to $OUT (default results_local/, not tracked by git).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGES=${IMAGES:-inputs}
OUT=${OUT:-results_local}
WARM=${WARM:-10}
RUNS=${RUNS:-110}
BIN=bin/cache
export OMP_NUM_THREADS=1

CXXFLAGS="-std=c++17 -O2 -g -fopenmp -I src/include"
VERSIONS="column_at row_at row_ptr optimized4"

# -D flags and source file of each version
defs() {
    case "$1" in
        column_at) echo "-DCACHE_ORDER_COLUMN=1 -DCACHE_POINTER_ACCESS=0" ;;
        row_at)    echo "-DCACHE_ORDER_COLUMN=0 -DCACHE_POINTER_ACCESS=0" ;;
        row_ptr)   echo "-DCACHE_ORDER_COLUMN=0 -DCACHE_POINTER_ACCESS=1" ;;
        optimized4) echo "" ;;
    esac
}
impl() {
    if [ "$1" = optimized4 ]; then echo src/optimizations/my_functions_optimized_4.cpp
    else echo src/experiments/cache_study/cache_variants.cpp; fi
}

build() {
    pkg-config --exists opencv4 || { echo "OpenCV 4 not found (Fedora: sudo dnf install opencv-devel)"; exit 1; }
    mkdir -p "$BIN"
    for v in $VERSIONS; do
        g++ $CXXFLAGS $(defs $v) src/benchmarks/cache_probe.cpp $(impl $v) \
            -o "$BIN/$v" $(pkg-config --cflags --libs opencv4) -lm
        if [ -f /usr/include/valgrind/cachegrind.h ]; then
            g++ $CXXFLAGS -DCACHEGRIND_SCOPE $(defs $v) src/benchmarks/cache_probe.cpp $(impl $v) \
                -o "$BIN/${v}_cg" $(pkg-config --cflags --libs opencv4) -lm
        fi
        echo "built $v"
    done
    [ -f /usr/include/valgrind/cachegrind.h ] || echo "no valgrind/cachegrind.h: Cachegrind builds skipped (sudo dnf install valgrind-devel)"
}

image() {
    local f="$IMAGES/$1.png"
    [ -f "$f" ] || { echo "missing input image $f"; exit 1; }
    echo "$f"
}

# one timing run with validation: run <dir> <name> <version> <op> <n> <res> <pad>
run() {
    local dir="$OUT/$1" name=$2 v=$3 op=$4 n=$5 res=$6 pad=$7
    mkdir -p "$dir"
    "$BIN/$v" "$op" "$n" "$(image $res)" "$WARM" "$RUNS" "$pad" "$dir/$name.csv" | tee "$dir/$name.log"
}

# Cachegrind with the cache sizes of our laptop (Ryzen 7 5825U)
cachegrind() {
    local v=$1 op=$2 n=$3 res=$4 pad=$5 name="${4}_${2}_${1}_n${3}_pad${5}"
    local dir="$OUT/cache_misses"
    mkdir -p "$dir"
    echo "cachegrind $name"
    valgrind --tool=cachegrind --cache-sim=yes --branch-sim=no --instr-at-start=no \
        --I1=32768,8,64 --D1=32768,8,64 --LL=16777216,16,64 \
        --cachegrind-out-file="$dir/$name.out" \
        "$BIN/${v}_cg" "$op" "$n" "$(image $res)" 1 2 "$pad" "$dir/$name.csv" --no-validate \
        > "$dir/$name.log" 2>&1
    grep -E "D1  misses|LLd misses" "$dir/$name.log" || true
}

perf_run() {
    local v=$1 op=$2 n=$3 res=$4 pad=$5 name="${4}_${2}_${1}_n${3}_pad${5}"
    local dir="$OUT/cache_misses"
    mkdir -p "$dir"
    echo "perf $name"
    perf stat -r 3 -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses \
        -o "$dir/$name.perf.txt" \
        -- "$BIN/$v" "$op" "$n" "$(image $res)" "$WARM" "$RUNS" "$pad" "$dir/$name.perf_frames.csv" --no-validate \
        > "$dir/$name.perf_stdout.txt" 2>&1
}

# the list of cases used for Cachegrind and perf (same as in the report)
counter_cases() {
    local f=$1
    for res in VGA HD; do
        for v in $VERSIONS; do $f "$v" sobel 3 "$res" 0; done
        for v in column_at row_ptr optimized4; do $f "$v" sobel 3 "$res" 64; done
        for n in 3 5; do $f column_at median "$n" "$res" 0; done
        for n in 3 5 11 15 27 31; do $f optimized4 median "$n" "$res" 0; done
    done
}

case "${1:-}" in
    build) build ;;
    time)
        [ -x "$BIN/optimized4" ] || build
        for res in VGA HD; do
            for pad in 0 64; do
                for v in $VERSIONS; do run cache_study "${res}_sobel_${v}_pad${pad}" "$v" sobel 3 "$res" "$pad"; done
                for v in column_at optimized4; do run cache_study "${res}_median_n3_${v}_pad${pad}" "$v" median 3 "$res" "$pad"; done
            done
        done ;;
    sweep)
        [ -x "$BIN/optimized4" ] || build
        for res in VGA HD; do
            for n in 3 5 7 9 11 15 27 31; do run median_sweep "${res}_n${n}" optimized4 median "$n" "$res" 0; done
        done ;;
    cachegrind)
        command -v valgrind >/dev/null || { echo "valgrind not found (sudo dnf install valgrind valgrind-devel)"; exit 1; }
        [ -x "$BIN/optimized4_cg" ] || build
        counter_cases cachegrind ;;
    perf)
        command -v perf >/dev/null || { echo "perf not found (sudo dnf install perf)"; exit 1; }
        [ -x "$BIN/optimized4" ] || build
        counter_cases perf_run
        echo "if the counters show <not supported>, check /proc/sys/kernel/perf_event_paranoid" ;;
    *)
        echo "usage: bash scripts/cache_study.sh build|time|sweep|cachegrind|perf"
        exit 2 ;;
esac
