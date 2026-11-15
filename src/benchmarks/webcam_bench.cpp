// Benchmark used for the measurements on the laptop and in the QEMU container.
// Reads a camera or a video file, times my_median and my_sobel separately
// (the first 10 frames are not counted) and compares every frame with OpenCV.
// Writes frames.csv and the first frame (input_gray.png, median.png, sobel.png).
//
// Usage: ./webcam_bench <odd n> <camera:INDEX|video_file> <width> <height> <frames> <output_dir> [--show]
// Example: ./webcam_bench 3 camera:0 640 480 120 results/native_n3
#include "my_functions.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using clock_type = std::chrono::steady_clock;

struct Stats { double median, minimum, maximum, mean; };

static Stats summarize(std::vector<double> v) {
    if (v.empty()) return {0, 0, 0, 0};
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    std::sort(v.begin(), v.end());
    std::size_t m = v.size() / 2;
    double med = v.size() % 2 ? v[m] : (v[m-1] + v[m])/2.0;
    return {med, v.front(), v.back(), sum / v.size()};
}

// reference Sobel with our norm: min((|gx| + |gy|) / 2, 255)
static void sobel_reference(const cv::Mat& input, cv::Mat& output) {
    output.setTo(0);
    for (int y = 1; y < input.rows-1; ++y) {
        const uchar* a = input.ptr<uchar>(y-1);
        const uchar* b = input.ptr<uchar>(y);
        const uchar* c = input.ptr<uchar>(y+1);
        uchar* o = output.ptr<uchar>(y);
        for (int x = 1; x < input.cols-1; ++x) {
            int gx = 2*b[x+1] + a[x+1] + c[x+1] - 2*b[x-1] - c[x-1] - a[x-1];
            int gy = 2*a[x] + a[x+1] + a[x-1] - 2*c[x] - c[x-1] - c[x+1];
            int g = (std::abs(gx) + std::abs(gy)) / 2;
            o[x] = static_cast<uchar>(std::min(g, 255));
        }
    }
}

// number of different pixels between a and b, and the largest difference
static int pixel_differences(const cv::Mat& a, const cv::Mat& b, double& maxdiff) {
    cv::Mat delta;
    cv::absdiff(a, b, delta);
    double minval;
    cv::minMaxLoc(delta, &minval, &maxdiff);
    return cv::countNonZero(delta);
}

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <odd_n>=3 <camera:INDEX|video_file> <width> <height> <frames>=110 <output_dir> [--show]\n";
        return 2;
    }
    try {
        const int n = std::stoi(argv[1]);
        const std::string source = argv[2];
        const int desired_width = std::stoi(argv[3]);
        const int desired_height = std::stoi(argv[4]);
        const int frame_count = std::stoi(argv[5]);
        const fs::path outdir = argv[6];
        const bool show = (argc == 8 && std::string(argv[7]) == "--show");
        if (n < 3 || n % 2 == 0 || n > 31 || desired_width <= 0 || desired_height <= 0 || frame_count <= 10)
            throw std::invalid_argument("Invalid n, size or frames. Choose odd n=3..31 and frames>10.");
        fs::create_directories(outdir);

        cv::VideoCapture cap;
        const bool camera = source.rfind("camera:", 0) == 0;
        if (camera) {
            int index = std::stoi(source.substr(7));
            cap.open(index, cv::CAP_V4L2);
            if (!cap.isOpened()) cap.open(index);
            if (cap.isOpened()) {
                cap.set(cv::CAP_PROP_FRAME_WIDTH, desired_width);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, desired_height);
            }
        } else cap.open(source);
        if (!cap.isOpened()) throw std::runtime_error("Could not open input " + source);

        std::ofstream csv(outdir / "frames.csv");
        if (!csv) throw std::runtime_error("Cannot create CSV");
        csv << "frame,median_ms,sobel_ms,median_different_pixels,median_max_abs_difference,sobel_different_pixels,sobel_max_abs_difference\n";
        csv << std::fixed << std::setprecision(6);
        cv::VideoWriter recorder;
        std::vector<double> med_times, sob_times;
        int bad_median_frames = 0, bad_sobel_frames = 0;
        int complete_frames = 0;
        const int warmup = 10;

        for (int k = 0; k < frame_count; ++k) {
            cv::Mat bgr;
            if (!cap.read(bgr) || bgr.empty()) {
                std::cerr << "Input ended after " << k << " frames.\n";
                break;
            }
            if (k == 0) {
                std::cout << "Actual frame size: " << bgr.cols << "x" << bgr.rows << " (requested "
                          << desired_width << "x" << desired_height << ")\n";
                if (bgr.cols != desired_width || bgr.rows != desired_height)
                    std::cout << "WARNING: requested resolution not provided by input.\n";
                if (camera) {
                    recorder.open((outdir / "webcam_capture.avi").string(),
                                  cv::VideoWriter::fourcc('M','J','P','G'), 25.0, bgr.size(), true);
                    if (!recorder.isOpened()) std::cerr << "Warning: could not record MJPG AVI.\n";
                }
            }
            if (recorder.isOpened()) recorder.write(bgr);
            cv::Mat gray;
            if (bgr.channels() == 3) cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
            else if (bgr.channels() == 4) cv::cvtColor(bgr, gray, cv::COLOR_BGRA2GRAY);
            else gray = bgr.clone();
            cv::Mat med(gray.size(), CV_8UC1, cv::Scalar(0));
            cv::Mat sob(gray.size(), CV_8UC1, cv::Scalar(0));
            // only the two filters are timed
            auto t0 = clock_type::now();
            my_median(gray, med, n);
            auto t1 = clock_type::now();
            my_sobel(med, sob);
            auto t2 = clock_type::now();
            const double tm = std::chrono::duration<double, std::milli>(t1-t0).count();
            const double ts = std::chrono::duration<double, std::milli>(t2-t1).count();

            cv::Mat ref_med, ref_sob(gray.size(), CV_8UC1, cv::Scalar(0));
            // check against OpenCV (medianBlur also replicates the border)
            cv::medianBlur(gray, ref_med, n);
            sobel_reference(ref_med, ref_sob);
            double max_m = 0, max_s = 0;
            int mism = pixel_differences(med, ref_med, max_m);
            int miss = pixel_differences(sob, ref_sob, max_s);
            if (mism) ++bad_median_frames;
            if (miss) ++bad_sobel_frames;
            if (k == 0) {
                cv::imwrite((outdir / "input_gray.png").string(), gray);
                cv::imwrite((outdir / "median.png").string(), med);
                cv::imwrite((outdir / "sobel.png").string(), sob);
            }
            if (show) {
                cv::imshow("webcam", bgr);
                cv::imshow("median", med);
                cv::imshow("sobel", sob);
                if (cv::waitKey(1) == 27) break;
            }
            if (k >= warmup) {
                csv << k << ',' << tm << ',' << ts << ',' << mism << ',' << max_m << ',' << miss << ',' << max_s << '\n';
                med_times.push_back(tm);
                sob_times.push_back(ts);
            }
            ++complete_frames;
        }
        auto m = summarize(med_times), s = summarize(sob_times);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Processed " << complete_frames << " frames; after warm-up: " << med_times.size() << "\n";
        std::cout << "Median: median=" << m.median << "ms min=" << m.minimum << "ms max=" << m.maximum << "ms mean=" << m.mean << "ms\n";
        std::cout << "Sobel:  median=" << s.median << "ms min=" << s.minimum << "ms max=" << s.maximum << "ms mean=" << s.mean << "ms\n";
        std::cout << "Frames with mismatches (all frames): Median=" << bad_median_frames << " Sobel=" << bad_sobel_frames << "\n";
        std::cout << "Files: " << outdir.string() << "/frames.csv; input_gray.png; median.png; sobel.png" << (camera ? "; webcam_capture.avi (if supported)" : "") << "\n";
        std::cout << (bad_median_frames == 0 && bad_sobel_frames == 0 ? "VALIDATION PASS\n" : "VALIDATION FAIL (inspect CSV and outputs)\n");
        return med_times.empty() ? 3 : 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
