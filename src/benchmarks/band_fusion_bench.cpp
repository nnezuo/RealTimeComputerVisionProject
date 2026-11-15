// Report 2, section 8.f: three ways to run median -> Sobel on one frame.
//   baseline: median on the whole image, then Sobel on the whole image (opt4 functions)
//   band:     the same two functions, on horizontal bands of B rows (plus a halo)
//   fused:    median rows computed one at a time into a ring of 3 rows, Sobel right after
// Link with src/optimizations/my_functions_optimized_4.cpp. Use one thread.
//
// Usage: band_fusion baseline|band|fused n input.png warmup measured band_height output.csv [--no-validate]
#include "my_functions.hpp"
#ifdef CACHEGRIND_SCOPE
#include <valgrind/cachegrind.h>
#endif
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>
#include <utility>

namespace fs = std::filesystem;

static double monotonic_ms() {
    timespec tp{};
    if (clock_gettime(CLOCK_MONOTONIC, &tp) != 0)
        throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC) failed");
    return 1000.0 * tp.tv_sec + tp.tv_nsec / 1.0e6;
}

// Same sliding histogram as opt4, but one row at a time, so the fused version
// never needs the full median image.
class MedianRowHistogram {
    int n_, half_, rank_;
    std::array<int, 256> fine_{};
    std::array<int, 16> coarse_{};
    std::vector<const uchar*> window_rows_;

    void add(uchar v) { ++fine_[v]; ++coarse_[v >> 4]; }
    void remove(uchar v) { --fine_[v]; --coarse_[v >> 4]; }
    uchar median_value() const {
        int sum = 0, group = 0;
        for (; group < 16; ++group) {
            if (sum + coarse_[group] > rank_) break;
            sum += coarse_[group];
        }
        if (group == 16) throw std::logic_error("median histogram invalid");
        for (int v = group * 16; v < group * 16 + 16; ++v) {
            sum += fine_[v];
            if (sum > rank_) return static_cast<uchar>(v);
        }
        throw std::logic_error("median histogram scan failed");
    }
public:
    explicit MedianRowHistogram(int n)
        : n_(n), half_(n / 2), rank_(n*n/2),
          window_rows_(static_cast<std::size_t>(n)) {}

    // median of row y of src, written in dst (replicated border)
    void compute_row(const cv::Mat& src, int y, uchar* dst) {
        const int H = src.rows, W = src.cols;
        fine_.fill(0);
        coarse_.fill(0);
        for (int k = 0; k < n_; ++k)
            window_rows_[static_cast<std::size_t>(k)] =
                src.ptr<uchar>(std::clamp(y - half_ + k, 0, H - 1));
        for (const uchar* row : window_rows_) {
            for (int dx = -half_; dx <= half_; ++dx)
                add(row[std::clamp(dx, 0, W - 1)]);
        }
        for (int x = 0; x < W; ++x) {
            dst[x] = median_value();
            if (x + 1 == W) break;
            const int remove_x = std::clamp(x - half_, 0, W - 1);
            const int add_x = std::clamp(x + half_ + 1, 0, W - 1);
            if (remove_x == add_x) continue;
            for (const uchar* row : window_rows_) {
                remove(row[remove_x]);
                add(row[add_x]);
            }
        }
    }
};

// one Sobel pixel, min((|gx| + |gy|) / 2, 255)
static inline uchar sobel_at(const uchar* prev, const uchar* curr,
                            const uchar* next, int x) {
    const int gx = 2*curr[x+1] + prev[x+1] + next[x+1]
                 - 2*curr[x-1] - prev[x-1] - next[x-1];
    const int gy = 2*prev[x] + prev[x+1] + prev[x-1]
                 - 2*next[x] - next[x+1] - next[x-1];
    return static_cast<uchar>(std::min((std::abs(gx)+std::abs(gy))/2, 255));
}

struct Pipeline {
    const cv::Mat& input;
    cv::Mat output;
    cv::Mat full_median;
    cv::Mat band_median, band_sobel;
    cv::Mat median_ring;  // 3 median rows, reused in turn
    MedianRowHistogram histogram;
    std::string mode;
    int n, band_height;

    Pipeline(const cv::Mat& src, std::string mode_, int window, int band)
        : input(src), output(src.size(),CV_8UC1,cv::Scalar(0)),
          histogram(window), mode(std::move(mode_)),
          n(window), band_height(band) {
        if (mode == "baseline") {
            full_median.create(src.size(), CV_8UC1);
        } else if (mode == "band") {
            // B output rows plus the rows needed above and below by the median and the Sobel
            const int scratch_h = std::min(src.rows, band_height + n + 2);
            band_median.create(scratch_h, src.cols, CV_8UC1);
            band_sobel.create(scratch_h, src.cols, CV_8UC1);
        } else {
            median_ring.create(3, src.cols, CV_8UC1);
        }
    }

    void baseline() {
        my_median(input,full_median,n);
        my_sobel(full_median,output);
    }

    // Each band is computed with a halo of n/2 + 1 rows on each side (n/2 for the
    // median, 1 for the Sobel), then the B useful rows are copied to the output.
    void bands() {
        output.setTo(0);
        if (input.rows < 3 || input.cols < 3) return;
        const int H=input.rows, W=input.cols;
        const int halo = n/2 + 1;
        for (int y0=1; y0 < H-1; y0 += band_height) {
            const int y1 = std::min(H-1, y0 + band_height);  // last row + 1
            const int r0 = std::max(0, y0 - halo);
            const int r1 = std::min(H, y1 + halo);
            const int count = r1 - r0;
            if (count > band_median.rows)
                throw std::logic_error("Band scratch height underestimated");
            const cv::Mat input_roi = input.rowRange(r0,r1);
            cv::Mat median_roi = band_median.rowRange(0,count);
            cv::Mat sobel_roi = band_sobel.rowRange(0,count);
            my_median(input_roi,median_roi,n);
            my_sobel(median_roi,sobel_roi);
            for (int y=y0;y<y1;++y)
                std::memcpy(output.ptr<uchar>(y),
                            sobel_roi.ptr<uchar>(y-r0),static_cast<std::size_t>(W));
        }
    }

    // Compute median row y+1, then the Sobel of row y from the 3 rows in the ring.
    void fused() {
        output.setTo(0);
        const int H=input.rows,W=input.cols;
        if (H<3 || W<3) return;
        histogram.compute_row(input,0,median_ring.ptr<uchar>(0));
        histogram.compute_row(input,1,median_ring.ptr<uchar>(1));
        for (int y=1; y<H-1; ++y) {
            histogram.compute_row(input,y+1,median_ring.ptr<uchar>((y+1)%3));
            const uchar* prev=median_ring.ptr<uchar>((y-1)%3);
            const uchar* curr=median_ring.ptr<uchar>(y%3);
            const uchar* next=median_ring.ptr<uchar>((y+1)%3);
            uchar* dst=output.ptr<uchar>(y);
            for (int x=1;x<W-1;++x)
                dst[x]=sobel_at(prev,curr,next,x);
        }
    }

    void run() {
        if(mode=="baseline") baseline();
        else if(mode=="band") bands();
        else fused();
    }
};

// Compare with OpenCV: medianBlur, then our Sobel on the OpenCV median.
// For the fused mode we also check the row-by-row median.
static void validate(Pipeline& p) {
    cv::Mat cv_median, reference_sobel(p.input.size(),CV_8UC1,cv::Scalar(0));
    cv::medianBlur(p.input,cv_median,p.n);
    cv::Mat our_median(p.input.size(),CV_8UC1);
    my_median(p.input,our_median,p.n);
    cv::Mat delta;
    cv::absdiff(cv_median,our_median,delta);
    const int median_errors=cv::countNonZero(delta);
    my_sobel(cv_median,reference_sobel);
    cv::absdiff(reference_sobel,p.output,delta);
    const int sobel_errors=cv::countNonZero(delta);
    double max_difference=0.0;
    cv::minMaxLoc(delta,nullptr,&max_difference);
    int fused_median_errors=0;
    if (p.mode=="fused") {
        cv::Mat med_check(p.input.size(),CV_8UC1);
        MedianRowHistogram worker(p.n);
        for(int y=0;y<p.input.rows;++y)
            worker.compute_row(p.input,y,med_check.ptr<uchar>(y));
        cv::absdiff(cv_median,med_check,delta);
        fused_median_errors=cv::countNonZero(delta);
    }
    std::cout << "Validation: median_diff=" << median_errors
              << " fused_median_diff=" << fused_median_errors
              << " sobel_diff=" << sobel_errors
              << " max_sobel_abs_diff=" << max_difference << '\n';
    if(median_errors || fused_median_errors || sobel_errors)
        throw std::runtime_error("VALIDATION FAIL");
}

int main(int argc,char** argv) {
    try {
        if(argc!=8 && argc!=9)
            throw std::invalid_argument("Usage: band_fusion baseline|band|fused n input.png warmup measured band_height output.csv [--no-validate]");
        const std::string mode=argv[1];
        const int n=std::stoi(argv[2]);
        const int warmup=std::stoi(argv[4]);
        const int measured=std::stoi(argv[5]);
        const int band=std::stoi(argv[6]);
        const bool do_validate=(argc==8);
        if(argc==9 && std::string(argv[8])!="--no-validate")
            throw std::invalid_argument("Unrecognized option");
        if((mode!="baseline" && mode!="band" && mode!="fused") ||
           n<3 || n>31 || n%2==0 || warmup<0 || measured<1 || band<1 || band>4096)
            throw std::invalid_argument("Invalid mode, kernel, counts or band height");

        cv::Mat src=cv::imread(argv[3],cv::IMREAD_GRAYSCALE);
        if(src.empty()) throw std::runtime_error("Could not load grayscale PNG");
        Pipeline p(src,mode,n,band);

        timespec res{};
        clock_getres(CLOCK_MONOTONIC,&res);
        std::cout<<"mode="<<mode<<" n="<<n<<" resolution="<<src.cols<<"x"<<src.rows
                 <<" warmup="<<warmup<<" measured="<<measured
                 <<" band_height="<<band<<" threads_hint=1\n";
        std::cout<<"CLOCK_MONOTONIC resolution="<<res.tv_sec<<" s + "<<res.tv_nsec<<" ns\n";
        std::cout<<"intermediate_bytes: full="<<p.full_median.total()
                 <<" band_median+sobel="<<(p.band_median.total()+p.band_sobel.total())
                 <<" fused_ring="<<p.median_ring.total()<<'\n';

        for(int i=0;i<warmup;++i) p.run();

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(measured));
        // here Cachegrind only counts the measured calls (not the warm-up)
#ifdef CACHEGRIND_SCOPE
        CACHEGRIND_START_INSTRUMENTATION;
#endif
        for(int i=0;i<measured;++i) {
            const double start=monotonic_ms();
            p.run();
            samples.push_back(monotonic_ms()-start);
        }
#ifdef CACHEGRIND_SCOPE
        CACHEGRIND_STOP_INSTRUMENTATION;
#endif
        if(do_validate) validate(p);

        std::uint64_t checksum=0;
        for(int y=0;y<p.output.rows;++y) {
            const uchar* row=p.output.ptr<uchar>(y);
            for(int x=0;x<p.output.cols;++x) checksum+=row[x];
        }

        const fs::path dest(argv[7]);
        if(!dest.parent_path().empty()) fs::create_directories(dest.parent_path());
        std::ofstream csv(dest);
        if(!csv) throw std::runtime_error("Could not open output CSV");
        csv<<"iteration,pipeline_ms\n"<<std::fixed<<std::setprecision(6);
        for(std::size_t i=0;i<samples.size();++i) csv<<i<<','<<samples[i]<<'\n';

        std::sort(samples.begin(),samples.end());
        const double med=samples[samples.size()/2];
        const double min=samples.front(), max=samples.back();
        const std::size_t q1=samples.size()/4, q3=(3*samples.size())/4;
        std::cout<<"VALIDATION "<<(do_validate?"PASS":"SKIPPED")
                 <<" checksum="<<checksum<<" median_ms="<<med
                 <<" min_ms="<<min<<" max_ms="<<max
                 <<" IQR_ms="<<samples[q3]-samples[q1]<<'\n';
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"ERROR: "<<e.what()<<'\n';
        return 1;
    }
}
