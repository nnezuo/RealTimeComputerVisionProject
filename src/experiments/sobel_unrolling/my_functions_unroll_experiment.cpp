// Unrolling experiment for the Sobel (report, section 7.1).
// Build with -DSOBEL_UNROLL_FACTOR=1, 2, 4 or 8. The median is the one of optimization 4
// and does not change between the builds.
#include "my_functions.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#ifndef SOBEL_UNROLL_FACTOR
#define SOBEL_UNROLL_FACTOR 1
#endif
static_assert(SOBEL_UNROLL_FACTOR == 1 || SOBEL_UNROLL_FACTOR == 2 ||
              SOBEL_UNROLL_FACTOR == 4 || SOBEL_UNROLL_FACTOR == 8,
              "Use SOBEL_UNROLL_FACTOR=1,2,4,8");

// For each block of d pixels we first load the d+2 columns of the 3 rows,
// then compute the d pixels. Neighbour pixels share the loaded values and
// the d pixels do not depend on each other.
void my_sobel(cv::Mat im_in, cv::Mat im_out) {
    if (im_in.type() != CV_8UC1 || im_out.type() != CV_8UC1 ||
        im_in.size() != im_out.size()) {
        throw std::invalid_argument("Sobel expects matching CV_8UC1 images");
    }
    const cv::Mat source = (im_in.u && im_out.u && im_in.u == im_out.u)
                           ? im_in.clone() : im_in;
    const int rows = source.rows;
    const int cols = source.cols;
    im_out.setTo(0);

    #pragma omp parallel for schedule(static)
    for (int j = 1; j < rows - 1; ++j) {
        const uchar* prev = source.ptr<uchar>(j - 1);
        const uchar* curr = source.ptr<uchar>(j);
        const uchar* next = source.ptr<uchar>(j + 1);
        uchar* out = im_out.ptr<uchar>(j);

        int i = 1;
        for (; i + SOBEL_UNROLL_FACTOR - 1 < cols - 1;
             i += SOBEL_UNROLL_FACTOR) {
            // load the d+2 columns once for the whole block
            uchar top[SOBEL_UNROLL_FACTOR + 2];
            uchar mid[SOBEL_UNROLL_FACTOR + 2];
            uchar bottom[SOBEL_UNROLL_FACTOR + 2];
            for (int t = 0; t < SOBEL_UNROLL_FACTOR + 2; ++t) {
                const int x = i - 1 + t;
                top[t] = prev[x];
                mid[t] = curr[x];
                bottom[t] = next[x];
            }

#define SOBEL_PIXEL(offset) do {                                   \
    constexpr int k = (offset) + 1;                                \
    const int gx = 2 * mid[k + 1] + top[k + 1] + bottom[k + 1]      \
                 - 2 * mid[k - 1] - bottom[k - 1] - top[k - 1];    \
    const int gy = 2 * top[k] + top[k + 1] + top[k - 1]             \
                 - 2 * bottom[k] - bottom[k - 1] - bottom[k + 1];  \
    const int g = (std::abs(gx) + std::abs(gy)) / 2;               \
    out[i + (offset)] = static_cast<uchar>(std::min(g, 255));       \
} while (0)

            SOBEL_PIXEL(0);
#if SOBEL_UNROLL_FACTOR >= 2
            SOBEL_PIXEL(1);
#endif
#if SOBEL_UNROLL_FACTOR >= 4
            SOBEL_PIXEL(2);
            SOBEL_PIXEL(3);
#endif
#if SOBEL_UNROLL_FACTOR >= 8
            SOBEL_PIXEL(4);
            SOBEL_PIXEL(5);
            SOBEL_PIXEL(6);
            SOBEL_PIXEL(7);
#endif
#undef SOBEL_PIXEL
        }

        // last columns when the width is not a multiple of d
        for (; i < cols - 1; ++i) {
            const int gx = 2 * curr[i + 1] + prev[i + 1] + next[i + 1]
                         - 2 * curr[i - 1] - next[i - 1] - prev[i - 1];
            const int gy = 2 * prev[i] + prev[i + 1] + prev[i - 1]
                         - 2 * next[i] - next[i - 1] - next[i + 1];
            const int g = (std::abs(gx) + std::abs(gy)) / 2;
            out[i] = static_cast<uchar>(std::min(g, 255));
        }
    }
}

// Sliding histogram median, same as optimization 4.
void my_median(cv::Mat im_in, cv::Mat im_out, int n) {
    if (n < 3 || n % 2 == 0 || n > 31) {
        throw std::invalid_argument("Median window n must be odd, 3..31");
    }
    if (im_in.empty() || im_in.type() != CV_8UC1 ||
        im_out.type() != CV_8UC1 || im_out.size() != im_in.size()) {
        throw std::invalid_argument("Median expects nonempty, matching CV_8UC1 images");
    }

    // copy the input if it is the same buffer as the output
    const cv::Mat src = (im_in.u && im_out.u && im_in.u == im_out.u)
                        ? im_in.clone() : im_in;
    const int rows = src.rows;
    const int cols = src.cols;
    const int half = n / 2;
    const int rank = (n * n) / 2;  // position of the median (starting at 0)

    std::array<int, 256> fine{};
    std::array<int, 16> coarse{};
    std::vector<const uchar*> window_rows(static_cast<std::size_t>(n));

    auto add_pixel = [&](uchar value) {
        ++fine[value];
        ++coarse[value >> 4];
    };
    auto remove_pixel = [&](uchar value) {
        --fine[value];
        --coarse[value >> 4];
    };
    auto read_median = [&]() -> uchar {
        int sum = 0;
        int group = 0;
        for (; group < 16; ++group) {
            if (sum + coarse[group] > rank) break;
            sum += coarse[group];
        }
        // the window always has n*n values, so group < 16 here
        for (int value = group * 16; value < group * 16 + 16; ++value) {
            sum += fine[value];
            if (sum > rank) return static_cast<uchar>(value);
        }
        throw std::logic_error("Invalid median histogram");
    };

    for (int r = 0; r < rows; ++r) {
        fine.fill(0);
        coarse.fill(0);
        for (int k = 0; k < n; ++k) {
            const int y = std::clamp(r - half + k, 0, rows - 1);
            window_rows[static_cast<std::size_t>(k)] = src.ptr<uchar>(y);
        }
        // first window of the row (c = 0)
        for (int k = 0; k < n; ++k) {
            const uchar* p = window_rows[static_cast<std::size_t>(k)];
            for (int dx = -half; dx <= half; ++dx) {
                const int x = std::clamp(dx, 0, cols - 1);
                add_pixel(p[x]);
            }
        }

        uchar* out = im_out.ptr<uchar>(r);
        for (int c = 0; c < cols; ++c) {
            out[c] = read_median();
            if (c + 1 == cols) break;
            const int remove_x = std::clamp(c - half, 0, cols - 1);
            const int add_x = std::clamp(c + half + 1, 0, cols - 1);
            if (remove_x == add_x) continue;
            for (int k = 0; k < n; ++k) {
                const uchar* p = window_rows[static_cast<std::size_t>(k)];
                remove_pixel(p[remove_x]);
                add_pixel(p[add_x]);
            }
        }
    }
}
