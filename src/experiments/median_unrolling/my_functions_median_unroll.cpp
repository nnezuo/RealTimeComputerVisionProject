// Unrolling experiment for the 3x3 median (report, section 7.1).
// Build with -DMEDIAN_UNROLL_FACTOR=1, 2, 4 or 8 = number of neighbour windows
// processed together in the inner loop. The Sobel is the one of optimization 4.
// For n != 3 the median falls back to the sliding histogram of optimization 4.
#include "my_functions.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// Sobel: same as optimization 4
void my_sobel(cv::Mat im_in, cv::Mat im_out) {
    if (im_in.type() != CV_8UC1 || im_out.type() != CV_8UC1 ||
        im_in.size() != im_out.size()) {
        throw std::invalid_argument("Sobel expects matching CV_8UC1 images");
    }
    // copy the input if it is the same buffer as the output
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

// The median of 9 values is found with a 19 compare-exchange network
// (Paeth, Graphics Gems 1990). We checked it on all 512 inputs made of 0 and 1.

#ifndef MEDIAN_UNROLL_FACTOR
#define MEDIAN_UNROLL_FACTOR 1
#endif
static_assert(MEDIAN_UNROLL_FACTOR == 1 || MEDIAN_UNROLL_FACTOR == 2 ||
              MEDIAN_UNROLL_FACTOR == 4 || MEDIAN_UNROLL_FACTOR == 8,
              "MEDIAN_UNROLL_FACTOR must be 1,2,4,8");

namespace {
template<int A, int B, int D>
inline void ce_for_lanes(std::array<std::array<uchar, 9>, D>& w) {
    // same compare-exchange on each of the d windows, one after the other
    auto ce = [&](int k) {
        uchar x = w[k][A], y = w[k][B];
        w[k][A] = std::min(x, y);
        w[k][B] = std::max(x, y);
    };
    ce(0);
    if constexpr (D >= 2) ce(1);
    if constexpr (D >= 4) { ce(2); ce(3); }
    if constexpr (D >= 8) { ce(4); ce(5); ce(6); ce(7); }
}

template<int D>
inline void median_network_9(std::array<std::array<uchar, 9>, D>& w) {
#define CE(a,b) ce_for_lanes<a,b,D>(w)
    CE(1,2); CE(4,5); CE(7,8);
    CE(0,1); CE(3,4); CE(6,7);
    CE(0,3); CE(1,2); CE(4,5);
    CE(7,8); CE(3,6); CE(4,7);
    CE(5,8); CE(1,4); CE(2,5);
    CE(4,7); CE(4,2); CE(6,4); CE(4,2);
#undef CE
    // after the network w[k][4] is the median (the other values are not fully sorted)
}

template<int D>
inline void load_window(std::array<std::array<uchar, 9>, D>& w,
                        const uchar* top, const uchar* mid,
                        const uchar* bot, int x) {
    auto load_one = [&](int lane) {
        int c = x + lane;
        w[lane] = {{top[c-1],top[c],top[c+1],
                    mid[c-1],mid[c],mid[c+1],
                    bot[c-1],bot[c],bot[c+1]}};
    };
    load_one(0);
    if constexpr (D >= 2) load_one(1);
    if constexpr (D >= 4) { load_one(2); load_one(3); }
    if constexpr (D >= 8) { load_one(4); load_one(5); load_one(6); load_one(7); }
}

template<int D>
inline void store_medians(uchar* out, int x,
                          const std::array<std::array<uchar,9>,D>& w) {
    out[x] = w[0][4];
    if constexpr(D >= 2) out[x+1] = w[1][4];
    if constexpr(D >= 4) { out[x+2] = w[2][4]; out[x+3] = w[3][4]; }
    if constexpr(D >= 8) {
        out[x+4] = w[4][4]; out[x+5] = w[5][4];
        out[x+6] = w[6][4]; out[x+7] = w[7][4];
    }
}

inline uchar median_edge(const cv::Mat& im, int row, int col) {
    std::array<std::array<uchar,9>,1> w{};
    int k=0;
    for(int yy=row-1; yy<=row+1; ++yy) {
        auto* ptr=im.ptr<uchar>(std::clamp(yy,0,im.rows-1));
        for(int xx=col-1; xx<=col+1; ++xx)
            w[0][k++]=ptr[std::clamp(xx,0,im.cols-1)];
    }
    median_network_9<1>(w);
    return w[0][4];
}
}  // namespace

static void median_histogram_generic(cv::Mat im_in, cv::Mat im_out, int n) {
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

// n = 3: d neighbour windows at a time, borders handled outside the main loop.
// other n: sliding histogram.
void my_median(cv::Mat im_in, cv::Mat im_out, int n) {
    if(n != 3) {
        median_histogram_generic(im_in, im_out, n);
        return;
    }
    if(im_in.empty() || im_in.type()!=CV_8UC1 || im_out.type()!=CV_8UC1 ||
       im_in.size()!=im_out.size())
        throw std::invalid_argument("Median expects matching CV_8UC1 images");

    const cv::Mat src = (im_in.u && im_out.u && im_in.u==im_out.u)
                        ? im_in.clone() : im_in;
    const int H=src.rows, W=src.cols;
    // border pixels (replicated border, like cv::medianBlur)
    for(int c=0;c<W;++c) {
        im_out.ptr<uchar>(0)[c]=median_edge(src,0,c);
        if(H>1) im_out.ptr<uchar>(H-1)[c]=median_edge(src,H-1,c);
    }
    if(H<=2) return;
    for(int r=1;r<H-1;++r) {
        const uchar* top=src.ptr<uchar>(r-1);
        const uchar* mid=src.ptr<uchar>(r);
        const uchar* bot=src.ptr<uchar>(r+1);
        uchar* out=im_out.ptr<uchar>(r);
        out[0]=median_edge(src,r,0);
        if(W>1) out[W-1]=median_edge(src,r,W-1);
        int c=1;
        for(; c+MEDIAN_UNROLL_FACTOR-1 < W-1; c+=MEDIAN_UNROLL_FACTOR) {
            std::array<std::array<uchar,9>,MEDIAN_UNROLL_FACTOR> windows{};
            load_window<MEDIAN_UNROLL_FACTOR>(windows,top,mid,bot,c);
            median_network_9<MEDIAN_UNROLL_FACTOR>(windows);
            store_medians<MEDIAN_UNROLL_FACTOR>(out,c,windows);
        }
        // last pixels of the row, one by one
        for(;c<W-1;++c) {
            std::array<std::array<uchar,9>,1> one{};
            load_window<1>(one,top,mid,bot,c);
            median_network_9<1>(one);
            out[c]=one[0][4];
        }
    }
}
