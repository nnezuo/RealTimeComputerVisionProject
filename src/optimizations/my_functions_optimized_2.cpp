// Optimization 2: loops in row-major order and row pointers (ptr<uchar>)
// instead of at<>() for every pixel. Same results as optimization 1.
#include "my_functions.hpp"
#include <algorithm>
#include <vector>
#include <cmath>

void my_sobel(cv::Mat im_in, cv::Mat im_out) {
    int rows = im_in.rows;
    int cols = im_in.cols;

    // rows outside, columns inside: consecutive pixels are next to each other in memory
    for (int j = 1; j < rows - 1; j++) {
        const uchar* prev_row = im_in.ptr<uchar>(j - 1);
        const uchar* curr_row = im_in.ptr<uchar>(j);
        const uchar* next_row = im_in.ptr<uchar>(j + 1);
        uchar* out_row = im_out.ptr<uchar>(j);

        for (int i = 1; i < cols - 1; i++) {
            int n  = prev_row[i];
            int s  = next_row[i];
            int e  = curr_row[i + 1];
            int w  = curr_row[i - 1];
            int ne = prev_row[i + 1];
            int nw = prev_row[i - 1];
            int se = next_row[i + 1];
            int sw = next_row[i - 1];

            int gx = 2 * e + ne + se - 2 * w - sw - nw;
            int gy = 2 * n + ne + nw - 2 * s - sw - se;

            int g = (std::abs(gx) + std::abs(gy)) / 2;
            out_row[i] = static_cast<uchar>(std::min(g, 255));
        }
    }
}

void my_median(cv::Mat im_in, cv::Mat im_out, int n) {
    int rows = im_in.rows;
    int cols = im_in.cols;
    int half_n = n / 2;
    int total_elements = n * n;

    for (int r = 0; r < rows; r++) {
        uchar* out_row = im_out.ptr<uchar>(r);

        for (int c = 0; c < cols; c++) {
            std::vector<uchar> v;
            v.reserve(total_elements);

            for (int rr = r - half_n; rr <= r + half_n; rr++) {
                int clamped_r = std::clamp(rr, 0, rows - 1);
                const uchar* src_row = im_in.ptr<uchar>(clamped_r);

                for (int cc = c - half_n; cc <= c + half_n; cc++) {
                    int clamped_c = std::clamp(cc, 0, cols - 1);
                    v.push_back(src_row[clamped_c]);
                }
            }

            std::sort(v.begin(), v.end());
            out_row[c] = v[total_elements / 2];
        }
    }
}
