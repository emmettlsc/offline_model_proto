#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int INPUT_SIZE = 640;
constexpr int NUM_QUERIES = 300;

const char* COCO[] = {
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
};

struct Args {
    std::string input;
    std::string model;
    std::string save;
    std::string stream_out;
    std::string trt_cache = "./trt_cache";
    bool no_print_json = false;
    bool display = false;
    float conf = 0.5f;
    int stride = 1;
    int max_frames = 0;
    bool cuda = false;
    bool trt = false;
    bool fp16 = true;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const std::string& name) {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << "\n"; std::exit(2); }
            return std::string(argv[++i]);
        };
        if (arg == "--input")              a.input = next(arg);
        else if (arg == "--model")         a.model = next(arg);
        else if (arg == "--save")          a.save = next(arg);
        else if (arg == "--stream-out")    a.stream_out = next(arg);
        else if (arg == "--no-print-json") a.no_print_json = true;
        else if (arg == "--display")       a.display = true;
        else if (arg == "--conf")          a.conf = std::stof(next(arg));
        else if (arg == "--stride")        a.stride = std::stoi(next(arg));
        else if (arg == "--max-frames")    a.max_frames = std::stoi(next(arg));
        else if (arg == "--cuda")          a.cuda = true;
        else if (arg == "--trt")           a.trt = true;
        else if (arg == "--no-fp16")       a.fp16 = false;
        else if (arg == "--trt-cache")     a.trt_cache = next(arg);
        else if (arg == "-h" || arg == "--help") {
            std::cout << "usage: " << argv[0] << " --input URL_OR_PATH --model PATH\n"
                      << "  [--save out.mp4] [--stream-out URL] [--no-print-json] [--display]\n"
                      << "  [--conf F] [--stride N] [--max-frames N]\n"
                      << "  [--cuda] [--trt] [--no-fp16] [--trt-cache DIR]\n";
            std::exit(0);
        }
        else { std::cerr << "unknown arg: " << arg << "\n"; std::exit(2); }
    }
    if (a.input.empty() || a.model.empty()) {
        std::cerr << "--input and --model are required\n";
        std::exit(2);
    }
    return a;
}

void preprocess(const cv::Mat& frame, std::vector<float>& out) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(INPUT_SIZE, INPUT_SIZE));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat f32;
    rgb.convertTo(f32, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> ch(3);
    cv::split(f32, ch);
    for (int c = 0; c < 3; ++c) {
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

void append_trt_ep(Ort::SessionOptions& opts, const Args& args) {
    OrtTensorRTProviderOptionsV2* trt = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trt));
    const char* fp16 = args.fp16 ? "1" : "0";
    std::vector<const char*> keys = {
        "trt_fp16_enable", "trt_engine_cache_enable", "trt_engine_cache_path"
    };
    std::vector<const char*> vals = {fp16, "1", args.trt_cache.c_str()};
    Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(
        trt, keys.data(), vals.data(), keys.size()));
    Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_TensorRT_V2(
        opts, trt));
    Ort::GetApi().ReleaseTensorRTProviderOptions(trt);
}

std::atomic<bool> stop_flag{false};
extern "C" void handle_sigint(int) { stop_flag.store(true); }

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::signal(SIGINT, handle_sigint);

    cv::VideoCapture cap(args.input);
    if (!cap.isOpened()) { std::cerr << "cannot open " << args.input << "\n"; return 3; }

    double fps = cap.get(cv::CAP_PROP_FPS); if (fps <= 0) fps = 30.0;
    int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cerr << "input: " << args.input << "  fps=" << fps << "  size=" << W << "x" << H << "\n";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rtdetr_stream");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    if (args.trt) {
        append_trt_ep(opts, args);
    }
    if (args.trt || args.cuda) {
        OrtCUDAProviderOptions cuda{};
        opts.AppendExecutionProvider_CUDA(cuda);
    }
    Ort::Session session(env, args.model.c_str(), opts);
    const char* dev = args.trt ? "trt" : (args.cuda ? "cuda" : "cpu");
    std::cerr << "model: " << args.model << "  ep: " << dev << "\n";

    cv::VideoWriter file_writer;
    if (!args.save.empty()) {
        file_writer.open(args.save, cv::VideoWriter::fourcc('m','p','4','v'),
                         fps / std::max(1, args.stride), cv::Size(W, H));
        if (!file_writer.isOpened()) { std::cerr << "cannot open --save " << args.save << "\n"; return 4; }
    }

    cv::VideoWriter stream_writer;
    if (!args.stream_out.empty()) {
        int fourcc = cv::VideoWriter::fourcc('H','2','6','4');
        stream_writer.open(args.stream_out, cv::CAP_FFMPEG, fourcc,
                           fps / std::max(1, args.stride), cv::Size(W, H));
        if (!stream_writer.isOpened()) {
            std::cerr << "cannot open --stream-out " << args.stream_out << "\n";
            return 5;
        }
    }

    std::vector<float> input(3 * INPUT_SIZE * INPUT_SIZE);
    std::array<int64_t, 4> input_shape{1, 3, INPUT_SIZE, INPUT_SIZE};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* in_names[] = {"pixel_values"};
    const char* out_names[] = {"boxes", "scores", "labels"};

    cv::Mat frame;
    int i = 0, n = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (!stop_flag.load() && cap.read(frame)) {
        if (frame.empty()) continue;
        if (i % args.stride == 0) {
            preprocess(frame, input);
            Ort::Value tensor = Ort::Value::CreateTensor<float>(
                mem, input.data(), input.size(), input_shape.data(), input_shape.size());
            auto outs = session.Run(Ort::RunOptions{}, in_names, &tensor, 1, out_names, 3);
            const float* boxes = outs[0].GetTensorData<float>();
            const float* scores = outs[1].GetTensorData<float>();
            const int64_t* labels = outs[2].GetTensorData<int64_t>();

            const float sx = (float)W / INPUT_SIZE, sy = (float)H / INPUT_SIZE;

            if (!args.no_print_json) std::cout << "{\"frame\":" << i << ",\"detections\":[";
            bool first = true;
            for (int q = 0; q < NUM_QUERIES; ++q) {
                if (scores[q] < args.conf) continue;
                float x1 = boxes[q * 4 + 0] * sx;
                float y1 = boxes[q * 4 + 1] * sy;
                float x2 = boxes[q * 4 + 2] * sx;
                float y2 = boxes[q * 4 + 3] * sy;
                int lbl = (int)labels[q];
                const char* lbl_str = (lbl >= 0 && lbl < (int)(sizeof(COCO) / sizeof(COCO[0])))
                                          ? COCO[lbl] : "?";

                cv::rectangle(frame, cv::Point((int)x1, (int)y1),
                              cv::Point((int)x2, (int)y2), cv::Scalar(0, 255, 255), 2);
                char buf[128];
                std::snprintf(buf, sizeof(buf), "%s:%.2f", lbl_str, scores[q]);
                cv::putText(frame, buf, cv::Point((int)x1, std::max(0, (int)y1 - 6)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

                if (!args.no_print_json) {
                    if (!first) std::cout << ",";
                    first = false;
                    std::cout << "{\"box_xyxy\":[" << x1 << "," << y1 << "," << x2 << "," << y2
                              << "],\"score\":" << scores[q]
                              << ",\"label_id\":" << lbl
                              << ",\"label\":\"" << escape_label(lbl_str) << "\"}";
                }
            }
            if (!args.no_print_json) {
                std::cout << "]}\n";
                std::cout.flush();
            }

            if (file_writer.isOpened())   file_writer.write(frame);
            if (stream_writer.isOpened()) stream_writer.write(frame);
            if (args.display) {
                cv::imshow("rtdetr_stream", frame);
                int k = cv::waitKey(1);
                if (k == 27 || k == 'q') break;
            }

            ++n;
            if (args.max_frames > 0 && n >= args.max_frames) break;
        }
        ++i;
    }

    if (file_writer.isOpened())   file_writer.release();
    if (stream_writer.isOpened()) stream_writer.release();
    cap.release();
    cv::destroyAllWindows();

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "processed " << n << " frames in " << sec << "s (" << (sec > 0 ? n / sec : 0)
              << " fps)" << (stop_flag.load() ? " [interrupted]" : "") << "\n";
    return 0;
}
