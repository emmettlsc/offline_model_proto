#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int INPUT_SIZE = 576;
constexpr int NUM_QUERIES = 300;

const std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
const std::array<float, 3> STD = {0.229f, 0.224f, 0.225f};

const char* COCO[] = {
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
};

struct Args {
    std::string video;
    std::string engine;
    std::string output = "rfdetr_out.mp4";
    float conf = 0.5f;
    int stride = 1;
    int max_frames = 0;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const std::string& name) {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << "\n"; std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (arg == "--video")       a.video = next(arg);
        else if (arg == "--engine")      a.engine = next(arg);
        else if (arg == "--output")      a.output = next(arg);
        else if (arg == "--conf")        a.conf = std::stof(next(arg));
        else if (arg == "--stride")      a.stride = std::stoi(next(arg));
        else if (arg == "--max-frames")  a.max_frames = std::stoi(next(arg));
        else if (arg == "-h" || arg == "--help") {
            std::cout << "usage: " << argv[0]
                      << " --video PATH --engine PATH [--output PATH] [--conf F] "
                         "[--stride N] [--max-frames N]\n";
            std::exit(0);
        } else { std::cerr << "unknown arg: " << arg << "\n"; std::exit(2); }
    }
    if (a.video.empty() || a.engine.empty()) {
        std::cerr << "--video and --engine are required\n";
        std::exit(2);
    }
    return a;
}

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity sev, const char* msg) noexcept override {
        if (sev <= Severity::kWARNING) std::cerr << "[trt] " << msg << "\n";
    }
};

void preprocess(const cv::Mat& frame, std::vector<float>& out) {
    cv::Mat resized, rgb, f32;
    cv::resize(frame, resized, cv::Size(INPUT_SIZE, INPUT_SIZE));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> ch(3);
    cv::split(f32, ch);
    for (int c = 0; c < 3; ++c) {
        ch[c] = (ch[c] - MEAN[c]) / STD[c];
        std::memcpy(out.data() + c * INPUT_SIZE * INPUT_SIZE,
                    ch[c].data, INPUT_SIZE * INPUT_SIZE * sizeof(float));
    }
}

std::string escape_label(const char* s) {
    std::string r;
    for (; *s; ++s) {
        if (*s == '"' || *s == '\\') r.push_back('\\');
        r.push_back(*s);
    }
    return r;
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(3); }
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
}

#define CK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::cerr << "cuda " << #call << ": " << cudaGetErrorString(e) << "\n"; \
        std::exit(10); \
    } \
} while (0)

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    cv::VideoCapture cap(args.video);
    if (!cap.isOpened()) { std::cerr << "cannot open " << args.video << "\n"; return 3; }
    double fps = cap.get(cv::CAP_PROP_FPS); if (fps <= 0) fps = 30.0;
    int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    Logger logger;
    auto buf = read_file(args.engine);
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    std::unique_ptr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(buf.data(), buf.size())};
    if (!engine) { std::cerr << "deserializeCudaEngine failed\n"; return 4; }
    std::unique_ptr<nvinfer1::IExecutionContext> ctx{engine->createExecutionContext()};

    void* d_input = nullptr;
    void* d_boxes = nullptr;
    void* d_scores = nullptr;
    void* d_labels = nullptr;
    size_t in_sz   = 3 * INPUT_SIZE * INPUT_SIZE * sizeof(float);
    size_t box_sz  = NUM_QUERIES * 4 * sizeof(float);
    size_t sc_sz   = NUM_QUERIES * sizeof(float);
    size_t lbl_sz  = NUM_QUERIES * sizeof(int64_t);
    CK(cudaMalloc(&d_input, in_sz));
    CK(cudaMalloc(&d_boxes, box_sz));
    CK(cudaMalloc(&d_scores, sc_sz));
    CK(cudaMalloc(&d_labels, lbl_sz));

    ctx->setTensorAddress("pixel_values", d_input);
    ctx->setTensorAddress("boxes", d_boxes);
    ctx->setTensorAddress("scores", d_scores);
    ctx->setTensorAddress("labels", d_labels);
    nvinfer1::Dims4 in_dims{1, 3, INPUT_SIZE, INPUT_SIZE};
    ctx->setInputShape("pixel_values", in_dims);

    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    std::cout << "engine: " << args.engine << "\n";

    cv::VideoWriter writer(args.output,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps / std::max(1, args.stride), cv::Size(W, H));
    if (!writer.isOpened()) { std::cerr << "cannot write " << args.output << "\n"; return 5; }

    std::ofstream json(args.output.substr(0, args.output.find_last_of('.')) + ".json");
    json << "{\"video\":\"" << args.video << "\",\"engine\":\"" << args.engine
         << "\",\"frames\":[";

    std::vector<float> h_input(3 * INPUT_SIZE * INPUT_SIZE);
    std::vector<float> h_boxes(NUM_QUERIES * 4);
    std::vector<float> h_scores(NUM_QUERIES);
    std::vector<int64_t> h_labels(NUM_QUERIES);

    cv::Mat frame;
    int i = 0, n = 0;
    auto t0 = std::chrono::steady_clock::now();
    bool first_frame_json = true;

    while (cap.read(frame)) {
        if (i % args.stride == 0) {
            preprocess(frame, h_input);
            CK(cudaMemcpyAsync(d_input, h_input.data(), in_sz,
                               cudaMemcpyHostToDevice, stream));
            if (!ctx->enqueueV3(stream)) {
                std::cerr << "enqueueV3 failed\n";
                return 6;
            }
            CK(cudaMemcpyAsync(h_boxes.data(),  d_boxes,  box_sz, cudaMemcpyDeviceToHost, stream));
            CK(cudaMemcpyAsync(h_scores.data(), d_scores, sc_sz,  cudaMemcpyDeviceToHost, stream));
            CK(cudaMemcpyAsync(h_labels.data(), d_labels, lbl_sz, cudaMemcpyDeviceToHost, stream));
            CK(cudaStreamSynchronize(stream));

            const float sx = (float)W / INPUT_SIZE, sy = (float)H / INPUT_SIZE;

            if (!first_frame_json) json << ",";
            first_frame_json = false;
            json << "{\"frame\":" << i << ",\"detections\":[";
            bool first_det = true;
            for (int q = 0; q < NUM_QUERIES; ++q) {
                if (h_scores[q] < args.conf) continue;
                float x1 = h_boxes[q * 4 + 0] * sx;
                float y1 = h_boxes[q * 4 + 1] * sy;
                float x2 = h_boxes[q * 4 + 2] * sx;
                float y2 = h_boxes[q * 4 + 3] * sy;
                int lbl = (int)h_labels[q];
                const char* lbl_str = (lbl >= 0 && lbl < (int)(sizeof(COCO) / sizeof(COCO[0])))
                                          ? COCO[lbl] : "?";

                cv::rectangle(frame, cv::Point((int)x1, (int)y1),
                              cv::Point((int)x2, (int)y2), cv::Scalar(0, 255, 255), 2);
                char buf[128];
                std::snprintf(buf, sizeof(buf), "%s:%.2f", lbl_str, h_scores[q]);
                cv::putText(frame, buf, cv::Point((int)x1, std::max(0, (int)y1 - 6)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

                if (!first_det) json << ",";
                first_det = false;
                json << "{\"box_xyxy\":[" << x1 << "," << y1 << "," << x2 << "," << y2
                     << "],\"score\":" << h_scores[q]
                     << ",\"label_id\":" << lbl
                     << ",\"label\":\"" << escape_label(lbl_str) << "\"}";
            }
            json << "]}";
            writer.write(frame);

            ++n;
            if (n % 20 == 0) std::cout << "  " << n << " frames\n";
            if (args.max_frames > 0 && n >= args.max_frames) break;
        }
        ++i;
    }

    json << "]}";
    json.close();
    writer.release();
    cap.release();

    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_labels);

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "wrote " << args.output << "  (" << n << " frames in "
              << sec << "s, " << (n / sec) << " fps)\n";
    return 0;
}
