// Small benchmark for the cache experiments.
// It reads one grey image, copies it into buffers whose rows can be padded
// (pad_bytes added to the row stride), runs one filter many times and saves the times.
//
// Usage: cache_probe sobel|median n input.png warmup measured pad_bytes output.csv [--no-validate]
//
// Built with -DCACHEGRIND_SCOPE, Cachegrind only counts the filter calls (warm-up + measured),
// not the image loading. Run it with valgrind --tool=cachegrind --instr-at-start=no.
#include "my_functions.hpp"
#ifdef CACHEGRIND_SCOPE
#include <valgrind/cachegrind.h>
#endif
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

namespace fs = std::filesystem;

static double now_ms() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1000000.0;
}

// Sobel with the same norm as ours: min((|gx| + |gy|) / 2, 255)
static void sobel_reference(const cv::Mat& in, cv::Mat& out) {
    out.setTo(0);
    for (int y = 1; y < in.rows - 1; ++y) {
        const uchar* a = in.ptr<uchar>(y - 1);
        const uchar* b = in.ptr<uchar>(y);
        const uchar* c = in.ptr<uchar>(y + 1);
        uchar* dst = out.ptr<uchar>(y);
        for (int x = 1; x < in.cols - 1; ++x) {
            int gx = 2 * b[x + 1] + a[x + 1] + c[x + 1] - 2 * b[x - 1] - c[x - 1] - a[x - 1];
            int gy = 2 * a[x] + a[x + 1] + a[x - 1] - 2 * c[x] - c[x - 1] - c[x + 1];
            dst[x] = static_cast<uchar>(std::min((std::abs(gx) + std::abs(gy)) / 2, 255));
        }
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 8 && argc != 9)
            throw std::invalid_argument("Usage: cache_probe sobel|median n input.png warmup measured pad_bytes output.csv [--no-validate]");

        std::string op = argv[1];
        int n = std::stoi(argv[2]);
        int warm = std::stoi(argv[4]);
        int measured = std::stoi(argv[5]);
        int pad = std::stoi(argv[6]);
        bool validate = (argc == 8);
        if (argc == 9 && std::string(argv[8]) != "--no-validate")
            throw std::invalid_argument("Only --no-validate is supported");
        if ((op != "sobel" && op != "median") || n < 3 || n > 31 || n % 2 == 0 ||
            warm < 0 || measured < 1 || pad < 0 || pad > 4096)
            throw std::invalid_argument("Bad arguments");

        cv::Mat original = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
        if (original.empty()) throw std::runtime_error("Cannot read input image");
        const int H = original.rows, W = original.cols;

        // our own buffers, so we choose the row stride (W + pad)
        const std::size_t stride = static_cast<std::size_t>(W) + static_cast<std::size_t>(pad);
        std::vector<uchar> storage_in(stride * H), storage_out(stride * H);
        cv::Mat in(H, W, CV_8UC1, storage_in.data(), stride);
        cv::Mat out(H, W, CV_8UC1, storage_out.data(), stride);
        for (int y = 0; y < H; ++y)
            std::memcpy(in.ptr<uchar>(y), original.ptr<uchar>(y), static_cast<std::size_t>(W));

        auto run = [&]() {
            if (op == "sobel") my_sobel(in, out);
            else my_median(in, out, n);
        };

        timespec resolution{};
        clock_getres(CLOCK_MONOTONIC, &resolution);
        std::cout << "mode=" << op << " n=" << n << " image=" << W << "x" << H
                  << " warm=" << warm << " measured=" << measured << " pad=" << pad << "\n";
        std::cout << "in.step=" << in.step << " out.step=" << out.step
                  << " in.base=" << static_cast<void*>(in.data) << " out.base=" << static_cast<void*>(out.data)
                  << " in.contiguous=" << in.isContinuous() << " out.contiguous=" << out.isContinuous() << "\n";
        std::cout << "CLOCK_MONOTONIC resolution=" << resolution.tv_sec << "s+" << resolution.tv_nsec << "ns\n";

#ifdef CACHEGRIND_SCOPE
        CACHEGRIND_START_INSTRUMENTATION;
#endif
        for (int i = 0; i < warm; ++i) run();

        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(measured));
        for (int i = 0; i < measured; ++i) {
            double t0 = now_ms();
            run();
            values.push_back(now_ms() - t0);
        }
#ifdef CACHEGRIND_SCOPE
        CACHEGRIND_STOP_INSTRUMENTATION;
#endif

        if (validate) {
            cv::Mat ref(H, W, CV_8UC1, cv::Scalar(0));
            if (op == "sobel") sobel_reference(in, ref);
            else cv::medianBlur(in, ref, n);
            cv::Mat diff;
            cv::absdiff(out, ref, diff);
            double min_diff = 0, max_diff = 0;
            cv::minMaxLoc(diff, &min_diff, &max_diff);
            int errors = cv::countNonZero(diff);
            std::cout << "Validation: mismatching_pixels=" << errors << " max_abs_difference=" << max_diff << "\n";
            if (errors) throw std::runtime_error("VALIDATION FAILED");
        }

        // sum of the output pixels, to compare runs quickly
        std::uint64_t checksum = 0;
        for (int y = 0; y < H; ++y) {
            const uchar* row = out.ptr<uchar>(y);
            for (int x = 0; x < W; ++x) checksum += row[x];
        }

        fs::path destination(argv[7]);
        if (!destination.parent_path().empty()) fs::create_directories(destination.parent_path());
        std::ofstream csv(destination);
        if (!csv) throw std::runtime_error("Cannot write CSV");
        csv << "iteration,filter_ms\n" << std::fixed << std::setprecision(6);
        for (std::size_t i = 0; i < values.size(); ++i) csv << i << ',' << values[i] << '\n';

        std::sort(values.begin(), values.end());
        std::cout << "VALIDATION " << (validate ? "PASS" : "SKIPPED") << " checksum=" << checksum
                  << " median_ms=" << values[values.size() / 2] << " min_ms=" << values.front()
                  << " max_ms=" << values.back() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
