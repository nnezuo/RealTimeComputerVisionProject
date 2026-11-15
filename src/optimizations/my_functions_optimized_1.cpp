// Optimization 1: correctness fixes only.
// - median: replicated border and the right rank (v[n*n/2])
// - Sobel: (|gx| + |gy|) / 2 instead of sqrt, same norm as the OpenCV reference
// The Sobel loop order is still column by column, like the original code.
#include "my_functions.hpp"
#include <math.h>


void my_sobel(cv::Mat im_in, cv::Mat im_out) {
    int rows = im_in.rows;
    int cols = im_in.cols;

    // outer loop on columns, inner loop on rows (kept from the original version)
    for (int i = 1; i < cols - 1; i++) {
        for (int j = 1; j < rows - 1; j++) {
            int n  = im_in.at<uchar>(j - 1, i);
            int s  = im_in.at<uchar>(j + 1, i);
            int e  = im_in.at<uchar>(j, i + 1);
            int w  = im_in.at<uchar>(j, i - 1);
            int ne = im_in.at<uchar>(j - 1, i + 1);
            int nw = im_in.at<uchar>(j - 1, i - 1);
            int se = im_in.at<uchar>(j + 1, i + 1);
            int sw = im_in.at<uchar>(j + 1, i - 1);

            int gx = 2 * e + ne + se - 2 * w - sw - nw;
            int gy = 2 * n + ne + nw - 2 * s - sw - se;

            int g = (std::abs(gx) + std::abs(gy)) / 2;
            if (g > 255) g = 255;

            im_out.at<uchar>(j, i) = static_cast<uchar>(g);
        }
    }
}

void my_median(cv::Mat im_in, cv::Mat im_out, int n) {
    int rows = im_in.rows;
    int cols = im_in.cols;
    int half_n = n / 2;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // new vector for each pixel, so no old values are left from the previous window
            std::vector<uchar> v;
            v.reserve(n * n);

            for (int rr = r - half_n; rr <= r + half_n; rr++) {
                for (int cc = c - half_n; cc <= c + half_n; cc++) {
                    // replicate the border
                    int clamped_r = std::clamp(rr, 0, rows - 1);
                    int clamped_c = std::clamp(cc, 0, cols - 1);
                    v.push_back(im_in.at<uchar>(clamped_r, clamped_c));
                }
            }

            std::sort(v.begin(), v.end());
            im_out.at<uchar>(r, c) = v[v.size() / 2];  // index 4 for a 3x3 window
        }
    }
}
