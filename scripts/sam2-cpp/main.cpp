#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int INPUT_SIZE = 1024;
constexpr int LOW_MASK_SIZE = 256;

const std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
const std::array<float, 3> STD = {0.229f, 0.224f, 0.225f};

struct Args {
    std::string video;
    std::string encoder;
    std::string decoder;
    std::string output = "sam2_out.mp4";
    std::string trt_cache = "./trt_cache";
    float bx1 = -1, by1 = -1, bx2 = -1, by2 = -1;
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
        if      (arg == "--video")    a.video = next(arg);
        else if (arg == "--encoder")  a.encoder = next(arg);
        else if (arg == "--decoder")  a.decoder = next(arg);
        else if (arg == "--output")   a.output = next(arg);
        else if (arg == "--box") {
            a.bx1 = std::stof(next(arg)); a.by1 = std::stof(next(arg));
            a.bx2 = std::stof(next(arg)); a.by2 = std::stof(next(arg));
        }
        else if (arg == "--stride")     a.stride = std::stoi(next(arg));
        else if (arg == "--max-frames") a.max_frames = std::stoi(next(arg));
        else if (arg == "--cuda")       a.cuda = true;
        else if (arg == "--trt")        a.trt = true;
        else if (arg == "--no-fp16")    a.fp16 = false;
        else if (arg == "--trt-cache")  a.trt_cache = next(arg);
        else if (arg == "-h" || arg == "--help") {
            std::cout << "usage: " << argv[0]
                      << " --video PATH --encoder PATH --decoder PATH [--output PATH]\n"
                      << "  [--box X1 Y1 X2 Y2] [--stride N] [--max-frames N]\n"
                      << "  [--cuda] [--trt] [--no-fp16] [--trt-cache DIR]\n";
            std::exit(0);
        }
        else { std::cerr << "unknown arg: " << arg << "\n"; std::exit(2); }
    }
    if (a.video.empty() || a.encoder.empty() || a.decoder.empty()) {
        std::cerr << "--video, --encoder, --decoder are required\n";
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
        ch[c] = (ch[c] - MEAN[c]) / STD[c];
        std::memcpy(out.data() + c * INPUT_SIZE * INPUT_SIZE,
                    ch[c].data, INPUT_SIZE * INPUT_SIZE * sizeof(float));
    }
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

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    cv::VideoCapture cap(args.video);
    if (!cap.isOpened()) { std::cerr << "cannot open " << args.video << "\n"; return 3; }
    double fps = cap.get(cv::CAP_PROP_FPS); if (fps <= 0) fps = 30.0;
    int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sam2_cpp");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    if (args.trt) {
        append_trt_ep(opts, args);
    }
    if (args.trt || args.cuda) {
        OrtCUDAProviderOptions cuda{};
        opts.AppendExecutionProvider_CUDA(cuda);
    }
    Ort::Session encoder(env, args.encoder.c_str(), opts);
    Ort::Session decoder(env, args.decoder.c_str(), opts);
    const char* dev = args.trt ? "trt" : (args.cuda ? "cuda" : "cpu");
    std::cout << "encoder: " << args.encoder << "\n";
    std::cout << "decoder: " << args.decoder << "\n";
    std::cout << "ep:      " << dev << "\n";

    cv::VideoWriter writer(args.output,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps / std::max(1, args.stride), cv::Size(W, H));
    if (!writer.isOpened()) { std::cerr << "cannot write " << args.output << "\n"; return 4; }

    std::ofstream json(args.output.substr(0, args.output.find_last_of('.')) + ".json");
    json << "{\"video\":\"" << args.video << "\",\"encoder\":\"" << args.encoder
         << "\",\"decoder\":\"" << args.decoder << "\",\"frames\":[";

    float bx1 = (args.bx1 < 0) ? 0.0f       : args.bx1;
    float by1 = (args.by1 < 0) ? 0.0f       : args.by1;
    float bx2 = (args.bx2 < 0) ? (float)W   : args.bx2;
    float by2 = (args.by2 < 0) ? (float)H   : args.by2;

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<float> input(3 * INPUT_SIZE * INPUT_SIZE);
    std::array<int64_t, 4> input_shape{1, 3, INPUT_SIZE, INPUT_SIZE};

    const char* enc_in_names[]  = {"pixel_values"};
    const char* enc_out_names[] = {"image_embeddings.0", "image_embeddings.1", "image_embeddings.2"};
    const char* dec_in_names[]  = {"input_points", "input_labels", "input_boxes",
                                   "image_embeddings.0", "image_embeddings.1", "image_embeddings.2"};
    const char* dec_out_names[] = {"iou_scores", "pred_masks", "object_score_logits"};

    cv::Mat frame;
    int i = 0, n = 0;
    auto t0 = std::chrono::steady_clock::now();
    bool first_frame = true;

    while (cap.read(frame)) {
        if (i % args.stride == 0) {
            preprocess(frame, input);

            Ort::Value image_tensor = Ort::Value::CreateTensor<float>(
                mem, input.data(), input.size(), input_shape.data(), input_shape.size());
            auto enc_outs = encoder.Run(Ort::RunOptions{},
                                         enc_in_names, &image_tensor, 1,
                                         enc_out_names, 3);

            float sx = (float)INPUT_SIZE / W, sy = (float)INPUT_SIZE / H;
            float box[4] = {bx1 * sx, by1 * sy, bx2 * sx, by2 * sy};

            std::array<int64_t, 4> pts_shape{1, 1, 0, 2};
            std::array<int64_t, 3> lbls_shape{1, 1, 0};
            std::array<int64_t, 3> box_shape{1, 1, 4};
            Ort::Value pts_t  = Ort::Value::CreateTensor<float>(
                allocator, pts_shape.data(), pts_shape.size());
            Ort::Value lbls_t = Ort::Value::CreateTensor<int64_t>(
                allocator, lbls_shape.data(), lbls_shape.size());
            Ort::Value box_t  = Ort::Value::CreateTensor<float>(
                mem, box, 4, box_shape.data(), box_shape.size());

            std::vector<Ort::Value> dec_inputs;
            dec_inputs.reserve(6);
            dec_inputs.push_back(std::move(pts_t));
            dec_inputs.push_back(std::move(lbls_t));
            dec_inputs.push_back(std::move(box_t));
            dec_inputs.push_back(std::move(enc_outs[0]));
            dec_inputs.push_back(std::move(enc_outs[1]));
            dec_inputs.push_back(std::move(enc_outs[2]));

            auto dec_outs = decoder.Run(Ort::RunOptions{},
                                         dec_in_names, dec_inputs.data(), 6,
                                         dec_out_names, 3);
            const float* iou   = dec_outs[0].GetTensorData<float>();
            const float* masks = dec_outs[1].GetTensorData<float>();

            int best = 0;
            for (int k = 1; k < 3; ++k) if (iou[k] > iou[best]) best = k;

            cv::Mat mask_low(LOW_MASK_SIZE, LOW_MASK_SIZE, CV_32F,
                             (void*)(masks + best * LOW_MASK_SIZE * LOW_MASK_SIZE));
            cv::Mat mask_orig;
            cv::resize(mask_low, mask_orig, cv::Size(W, H));
            cv::Mat mask_bin;
            cv::threshold(mask_orig, mask_bin, 0.0, 255.0, cv::THRESH_BINARY);
            mask_bin.convertTo(mask_bin, CV_8U);

            cv::Mat green(H, W, frame.type(), cv::Scalar(0, 255, 0));
            cv::Mat blended;
            cv::addWeighted(frame, 0.5, green, 0.5, 0, blended);
            blended.copyTo(frame, mask_bin);

            cv::rectangle(frame, cv::Point((int)bx1, (int)by1),
                          cv::Point((int)bx2, (int)by2), cv::Scalar(255, 0, 0), 1);
            writer.write(frame);

            int mask_area = cv::countNonZero(mask_bin);
            if (!first_frame) json << ",";
            first_frame = false;
            json << "{\"frame\":" << i
                 << ",\"box_xyxy\":[" << bx1 << "," << by1 << "," << bx2 << "," << by2 << "]"
                 << ",\"iou_score\":" << iou[best]
                 << ",\"mask_area_px\":" << mask_area << "}";

            ++n;
            if (n % 5 == 0) std::cout << "  " << n << " frames\n";
            if (args.max_frames > 0 && n >= args.max_frames) break;
        }
        ++i;
    }

    json << "]}";
    json.close();
    writer.release();
    cap.release();

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "wrote " << args.output << " (" << n << " frames in " << sec
              << "s, " << (sec > 0 ? n / sec : 0) << " fps)\n";
    return 0;
}
