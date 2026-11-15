// Optimization 4: new median algorithm (sliding histogram).
// The Sobel is the same as in optimization 3.
#include "my_functions.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <vector>

void my_sobel(cv::Mat im_in, cv::Mat im_out) {
    if (im_in.type() != CV_8UC1 || im_out.type() != CV_8UC1 ||
        im_in.size() != im_out.size()) {
        throw std::invalid_argument("Sobel expects matching CV_8UC1 images");
    }
    // copy the input if it is the same buffer as the output (can happen with the provided main)
    const cv::Mat source = (im_in.u && im_out.u && im_in.u == im_out.u)
                           ? im_in.clone() : im_in;
    const int rows = source.rows;
    const int cols = source.cols;
    im_out.setTo(0);  // the 1-pixel border stays at 0

    #pragma omp parallel for schedule(static)
    for (int j = 1; j < rows - 1; ++j) {
        const uchar* prev = source.ptr<uchar>(j - 1);
        const uchar* curr = source.ptr<uchar>(j);
        const uchar* next = source.ptr<uchar>(j + 1);
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

// Sliding histogram median (idea from Huang, and Perreault & Hebert for the two levels).
// For each row we keep the histogram of the n x n window. Moving one pixel to the
// right removes one column of n pixels and adds the next one.
// 256 "fine" bins + 16 "coarse" bins (16 grey levels each), so reading the median
// needs at most 16 + 16 steps. Border: replicated, like cv::medianBlur.
void my_median(cv::Mat im_in, cv::Mat im_out, int n) {
    if (n < 3 || n % 2 == 0 || n > 31) {
        throw std::invalid_argument("Median window n must be odd, 3..31");
    }
    if (im_in.empty() || im_in.type() != CV_8UC1 ||
        im_out.type() != CV_8UC1 || im_out.size() != im_in.size()) {
        throw std::invalid_argument("Median expects nonempty, matching CV_8UC1 images");
    }

    // same input/output check as in my_sobel
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
            // slide: remove the column on the left, add the one on the right
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
