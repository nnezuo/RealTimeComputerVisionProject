# Median and Sobel filters on the Jetson Orin Nano CPU

Final project of the PERF course at ESIEE Paris (2026), by Ana Késia dos Reis Santos
and Júlia de Mello Teixeira.

We start from the C++/OpenCV code given in class (a median filter followed by a Sobel
edge detector) and try to make it fast enough for real time on the CPU of an NVIDIA
Jetson Orin Nano: 25 images per second, so 40 ms per image for the whole pipeline.
Each optimization step is kept as a separate file so it can be measured on its own.

We only had the Jetson during the lab sessions, so the original code and our versions
1 to 3 were measured on the board. The rest (version 4, the unrolling and compiler tests)
was measured on our Fedora laptop (Ryzen 7 5825U) and checked in an Ubuntu ARM64 container
running with QEMU. The QEMU times are much slower than the real board and are only used
to compare versions, not as Jetson times.

There are two reports in [`reports/`](reports/):
- [report 1](reports/Report1_DOS_REIS_SANTOS_Ana_Kesia_DE_MELLO_TEIXEIRA_Julia-1.pdf): the
  optimization of the two filters (estimates, measurements, unrolling, threads, compiler);
- [report 2](reports/Report2_DOS_REIS_SANTOS_Ana_Kesia_DE_MELLO_TEIXEIRA_Julia.pdf): the
  cache study (how the filters use the cache, predicted by hand and measured with
  Cachegrind and `perf`).

## What is in the repo

```text
src/
  baseline/        code given by the professor (not modified)
  include/         my_functions.hpp
  optimizations/   my_functions_optimized_1.cpp ... _4.cpp
  experiments/     sobel_unrolling, median_unrolling (d = 1, 2, 4, 8), cache_study
  benchmarks/      webcam_bench.cpp (main benchmark), cache_probe.cpp and
                   band_fusion_bench.cpp (cache study)
scripts/           cache_study.sh, band_fusion.sh, summarize_band_fusion.py (report 2)
results/           summary tables, figures and the text logs of our runs
reports/           the two reports (PDF)
```

## The versions

| Version | What changes |
|---|---|
| baseline | given code: median with `std::sort` (wrong rank and borders), Sobel with `sqrt`, column-by-column loops |
| opt1 | correct median (replicated border, rank n²/2), Sobel with `(\|gx\| + \|gy\|) / 2` |
| opt2 | row-by-row loops and row pointers instead of `at<>()` |
| opt3 | fixed sorting of the 3×3 window, Sobel unrolled by 4, OpenMP on the Sobel |
| opt4 | sliding histogram median (cost grows with n instead of n² log n) |

The two unrolling experiments use opt4 as a base and change only one filter: the Sobel
with `-DSOBEL_UNROLL_FACTOR=d`, or the 3×3 median (19 compare-exchange network) with
`-DMEDIAN_UNROLL_FACTOR=d`.

## Some results

On the Jetson (VGA, n = 3, `-O0`, mean of 100 frames):

| Version | Median (ms) | Sobel (ms) |
|---|---:|---:|
| baseline | 258.98 | 17.07 |
| opt1 | 529.47 | 18.11 |
| opt2 | 464.53 | 7.35 |
| opt3, 1 thread | 195.14 | 10.67 |
| opt3, 6 threads | 196.41 | 2.17 |

So on the board we stayed far from 40 ms: the median was the problem, and in opt3 it
is not parallel. Opt1 is slower than the baseline because it allocates a vector for
every pixel, and at `-O0` every small function is a real call.

On the laptop (median filter, `-O2`, median of 110 frames):

| Image | n | opt3 (ms) | opt4 (ms) |
|---|---:|---:|---:|
| VGA | 3 | 13.49 | 5.29 |
| VGA | 5 | 60.52 | 5.83 |
| HD | 3 | 42.92 | 14.54 |
| HD | 5 | 180.00 | 16.55 |

For the unrolling (laptop, one thread), the Sobel gets faster up to d = 8 (×1.67 at
`-O2` in VGA), but for the median network the best factor at `-O2` was d = 2
(2.00 ms down to 1.05 ms in VGA); d = 4 and d = 8 were slower again.

For the cache study (laptop, Cachegrind with the laptop caches), our count by hand was
very close to the simulation: 614 400 L1 misses per VGA frame predicted for the Sobel in
column order, 620 813 simulated, and 9 600 against 9 589 in row order. But the misses
explain little of the time: changing the loop order divides the misses by 65 and only
saves 11 to 20 % of the Sobel time, and a fused median + Sobel pipeline that removes
40 % of the misses was slightly slower because it runs more instructions.

More tables are in [`results/README.md`](results/README.md).

## Build and run

You need g++ with C++17, OpenCV 4 (with `pkg-config`) and OpenMP. On Fedora:

```bash
sudo dnf install gcc-c++ opencv-devel pkgconf-pkg-config
```

Every `my_functions_*.cpp` file defines `my_median` and `my_sobel`, so compile the
benchmark with exactly one of them. For example opt4:

```bash
g++ -std=c++17 -O2 -fopenmp -Isrc/include \
    src/benchmarks/webcam_bench.cpp src/optimizations/my_functions_optimized_4.cpp \
    -o bench_opt4 $(pkg-config --cflags --libs opencv4) -lm

# n = 3, webcam 0, VGA, 120 frames (the first 10 are not counted)
OMP_NUM_THREADS=4 ./bench_opt4 3 camera:0 640 480 120 out/opt4_vga --show
```

You can give a video file instead of `camera:0`. The program writes `frames.csv` with
the time of each frame and the number of pixels that differ from OpenCV, and prints
`VALIDATION PASS` if all frames match.

For the unrolling experiments, use one of the files in `src/experiments/` and add for
example `-DMEDIAN_UNROLL_FACTOR=2`.

### Cache study (report 2)

These scripts work on one grey frame instead of a video. Put a 640×480 and a 1280×720
grey PNG in `inputs/VGA.png` and `inputs/HD.png` (our webcam frames are not in the repo).
Results go to `results_local/`.

```bash
sudo dnf install valgrind valgrind-devel perf   # for the Cachegrind and perf parts

bash scripts/cache_study.sh build
bash scripts/cache_study.sh time         # Sobel versions, row step W and W+64, median n=3
bash scripts/cache_study.sh sweep        # opt4 median, n = 3 ... 31
bash scripts/cache_study.sh cachegrind   # simulated L1/LL misses
bash scripts/cache_study.sh perf         # hardware counters

bash scripts/band_fusion.sh time         # full frame vs bands vs fused pipeline
bash scripts/band_fusion.sh cachegrind
python3 scripts/summarize_band_fusion.py results_local/band_fusion
```

`IMAGES`, `OUT`, `WARM` and `RUNS` can be changed from the environment, for example
`WARM=1 RUNS=5 bash scripts/cache_study.sh time` for a quick test.

## Note

The files in `src/baseline/` were given by the professor for the course and are kept only
to compare with our versions.
