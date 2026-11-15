// Sobel variants for the cache experiments. They all give the same image,
// only the way the image is read changes:
//   CACHE_ORDER_COLUMN=1                       column by column with at<>()  (column_at)
//   CACHE_ORDER_COLUMN=0                       row by row with at<>()        (row_at)
//   CACHE_ORDER_COLUMN=0 CACHE_POINTER_ACCESS=1 row by row with ptr<>()       (row_ptr)
// The median is a simple sort of the window and is the same for every variant.
#include "my_functions.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#ifndef CACHE_ORDER_COLUMN
#define CACHE_ORDER_COLUMN 1
#endif
#ifndef CACHE_POINTER_ACCESS
#define CACHE_POINTER_ACCESS 0
#endif

static inline int sobel_l1(int n, int s, int e, int w, int ne, int nw, int se, int sw) {
    const int gx = 2 * e + ne + se - 2 * w - sw - nw;
    const int gy = 2 * n + ne + nw - 2 * s - sw - se;
    return std::min((std::abs(gx) + std::abs(gy)) / 2, 255);
}

void my_sobel(cv::Mat in, cv::Mat out) {
    if (in.empty() || in.type() != CV_8UC1 || out.type() != CV_8UC1 || in.size() != out.size())
        throw std::invalid_argument("Sobel: expected matching CV_8UC1 images");
    if (in.u && out.u && in.u == out.u) throw std::invalid_argument("Sobel: use separate buffers");
    out.setTo(0);
    const int H = in.rows, W = in.cols;
#if CACHE_POINTER_ACCESS
    // row by row, with row pointers
    for (int y = 1; y < H-1; ++y) {
        const uchar *top = in.ptr<uchar>(y-1), *mid = in.ptr<uchar>(y), *bot = in.ptr<uchar>(y+1);
        uchar *dest = out.ptr<uchar>(y);
        for (int x = 1; x < W-1; ++x)
            dest[x] = static_cast<uchar>(sobel_l1(top[x],bot[x],mid[x+1],mid[x-1],
                                                   top[x+1],top[x-1],bot[x+1],bot[x-1]));
    }
#elif CACHE_ORDER_COLUMN
    for (int x = 1; x < W-1; ++x)
        for (int y = 1; y < H-1; ++y)
            out.at<uchar>(y,x) = static_cast<uchar>(sobel_l1(
                in.at<uchar>(y-1,x),in.at<uchar>(y+1,x),in.at<uchar>(y,x+1),in.at<uchar>(y,x-1),
                in.at<uchar>(y-1,x+1),in.at<uchar>(y-1,x-1),in.at<uchar>(y+1,x+1),in.at<uchar>(y+1,x-1)));
#else
    for (int y = 1; y < H-1; ++y)
        for (int x = 1; x < W-1; ++x)
            out.at<uchar>(y,x) = static_cast<uchar>(sobel_l1(
                in.at<uchar>(y-1,x),in.at<uchar>(y+1,x),in.at<uchar>(y,x+1),in.at<uchar>(y,x-1),
                in.at<uchar>(y-1,x+1),in.at<uchar>(y-1,x-1),in.at<uchar>(y+1,x+1),in.at<uchar>(y+1,x-1)));
#endif
}

// median by sorting the window (replicated border), like optimization 1
void my_median(cv::Mat in, cv::Mat out, int n) {
    if (n < 3 || n > 31 || n % 2 == 0 || in.empty() || in.type() != CV_8UC1 ||
        out.type() != CV_8UC1 || in.size() != out.size())
        throw std::invalid_argument("Median: expected matching CV_8UC1 and odd kernel 3..31");
    if (in.u && out.u && in.u == out.u) throw std::invalid_argument("Median: use separate buffers");
    std::vector<uchar> window(static_cast<std::size_t>(n*n));
    const int half=n/2;
    for(int y=0;y<in.rows;++y) {
        uchar* dest=out.ptr<uchar>(y);
        for(int x=0;x<in.cols;++x) {
            int k=0;
            for(int dy=-half;dy<=half;++dy) {
                const uchar* row=in.ptr<uchar>(std::clamp(y+dy,0,in.rows-1));
                for(int dx=-half;dx<=half;++dx)
                    window[static_cast<std::size_t>(k++)]=row[std::clamp(x+dx,0,in.cols-1)];
            }
            std::sort(window.begin(),window.end());
            dest[x]=window[window.size()/2];
        }
    }
}
