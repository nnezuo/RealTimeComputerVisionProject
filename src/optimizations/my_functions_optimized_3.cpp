// Optimization 3:
// - median 3x3: the 9 values are sorted with a fixed sequence of compare-exchanges
//   (bubble sort pattern, 36 comparisons), no more std::sort for n = 3
// - Sobel: unrolled by 4 and rows split between threads with OpenMP
#include "my_functions.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// sort the 9 values of the window, the median is p[4]
static inline uchar median9(const uchar* r0, const uchar* r1,
                            const uchar* r2, int c) {
    std::array<uchar, 9> p = {
        r0[c - 1], r0[c], r0[c + 1],
        r1[c - 1], r1[c], r1[c + 1],
        r2[c - 1], r2[c], r2[c + 1]
    };
    for (int pass = 0; pass < 8; ++pass) {
        for (int i = 0; i < 8 - pass; ++i) {
            if (p[i + 1] < p[i]) std::swap(p[i], p[i + 1]);
        }
    }
    return p[4];
}

void my_sobel(cv::Mat im_in, cv::Mat im_out) {
    if (im_in.empty() || im_in.type() != CV_8UC1 ||
        im_out.type() != CV_8UC1 || im_in.size() != im_out.size()) {
        throw std::invalid_argument("Sobel requires allocated CV_8UC1 input/output of equal size");
    }
    // the provided main can pass the same Mat as input and output, so we copy in that case
    cv::Mat source_copy;
    if (im_in.data == im_out.data) {
        source_copy = im_in.clone();
        im_in = source_copy;
    }

    const int rows = im_in.rows;
    const int cols = im_in.cols;

    #pragma omp parallel for schedule(static)
    for (int j = 1; j < rows - 1; ++j) {
        const uchar* prev = im_in.ptr<uchar>(j - 1);
        const uchar* curr = im_in.ptr<uchar>(j);
        const uchar* next = im_in.ptr<uchar>(j + 1);
        uchar* out = im_out.ptr<uchar>(j);

        auto compute_pixel = [&](int i) {
            const int north = prev[i], south = next[i];
            const int east = curr[i + 1], west = curr[i - 1];
            const int ne = prev[i + 1], nw = prev[i - 1];
            const int se = next[i + 1], sw = next[i - 1];
            const int gx = 2 * east + ne + se - 2 * west - sw - nw;
            const int gy = 2 * north + ne + nw - 2 * south - sw - se;
            const int g = (std::abs(gx) + std::abs(gy)) / 2;
            out[i] = static_cast<uchar>(std::min(g, 255));
        };

        // 4 pixels per iteration, then the last columns one by one
        int i = 1;
        for (; i + 3 < cols - 1; i += 4) {
            compute_pixel(i);
            compute_pixel(i + 1);
            compute_pixel(i + 2);
            compute_pixel(i + 3);
        }
        for (; i < cols - 1; ++i) compute_pixel(i);
    }
}

// n = 3: median9() on the interior pixels.
// other n, and the border pixels: copy the window (replicated border) and sort it.
void my_median(cv::Mat im_in, cv::Mat im_out, int n) {
    if (n < 3 || n % 2 == 0) {
        throw std::invalid_argument("Median kernel size must be odd and >= 3");
    }
    if (im_in.empty() || im_in.type() != CV_8UC1 ||
        im_out.type() != CV_8UC1 || im_in.size() != im_out.size()) {
        throw std::invalid_argument("Median requires allocated CV_8UC1 input/output of equal size");
    }
    // same problem as in my_sobel if input and output are the same Mat
    cv::Mat source_copy;
    if (im_in.data == im_out.data) {
        source_copy = im_in.clone();
        im_in = source_copy;
    }

    const int rows = im_in.rows;
    const int cols = im_in.cols;
    const int half = n / 2;
    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);

    // one buffer for the whole image (opt1/opt2 allocated a vector per pixel)
    std::vector<uchar> buffer(count);

    for (int r = 0; r < rows; ++r) {
        uchar* out = im_out.ptr<uchar>(r);
        const bool interior_row = (n == 3 && r > 0 && r < rows - 1);
        const uchar* r0 = interior_row ? im_in.ptr<uchar>(r - 1) : nullptr;
        const uchar* r1 = interior_row ? im_in.ptr<uchar>(r) : nullptr;
        const uchar* r2 = interior_row ? im_in.ptr<uchar>(r + 1) : nullptr;

        for (int c = 0; c < cols; ++c) {
            if (interior_row && c > 0 && c < cols - 1) {
                out[c] = median9(r0, r1, r2, c);
                continue;
            }

            size_t pos = 0;
            for (int rr = r - half; rr <= r + half; ++rr) {
                const int y = std::clamp(rr, 0, rows - 1);
                const uchar* src = im_in.ptr<uchar>(y);
                for (int cc = c - half; cc <= c + half; ++cc) {
                    const int x = std::clamp(cc, 0, cols - 1);
                    buffer[pos++] = src[x];
                }
            }
            std::sort(buffer.begin(), buffer.end());
            out[c] = buffer[count / 2];
        }
    }
}
