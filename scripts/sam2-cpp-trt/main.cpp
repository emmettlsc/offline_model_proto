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

constexpr int INPUT_SIZE = 1024;
constexpr int LOW_MASK_SIZE = 256;

const std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
const std::array<float, 3> STD = {0.229f, 0.224f, 0.225f};

struct Args {
    std::string video;
    std::string encoder;
    std::string decoder;
    std::string output = "sam2_out.mp4";
    float bx1 = -1, by1 = -1, bx2 = -1, by2 = -1;
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
        else if (arg == "-h" || arg == "--help") {
            std::cout << "usage: " << argv[0]
                      << " --video PATH --encoder PATH --decoder PATH [--output PATH]\n"
                      << "  [--box X1 Y1 X2 Y2] [--stride N] [--max-frames N]\n";
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
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};

    auto enc_buf = read_file(args.encoder);
    std::unique_ptr<nvinfer1::ICudaEngine> enc_engine{
        runtime->deserializeCudaEngine(enc_buf.data(), enc_buf.size())};
    if (!enc_engine) { std::cerr << "encoder deserialize failed\n"; return 4; }
    std::unique_ptr<nvinfer1::IExecutionContext> enc_ctx{enc_engine->createExecutionContext()};

    auto dec_buf = read_file(args.decoder);
    std::unique_ptr<nvinfer1::ICudaEngine> dec_engine{
        runtime->deserializeCudaEngine(dec_buf.data(), dec_buf.size())};
    if (!dec_engine) { std::cerr << "decoder deserialize failed\n"; return 4; }
    std::unique_ptr<nvinfer1::IExecutionContext> dec_ctx{dec_engine->createExecutionContext()};

    void* d_pixel = nullptr;
    void* d_emb0 = nullptr;
    void* d_emb1 = nullptr;
    void* d_emb2 = nullptr;
    void* d_pts = nullptr;
    void* d_lbls = nullptr;
    void* d_box = nullptr;
    void* d_iou = nullptr;
    void* d_masks = nullptr;
    void* d_objlogits = nullptr;

    size_t pixel_sz = 3 * INPUT_SIZE * INPUT_SIZE * sizeof(float);
    size_t emb0_sz = 1 * 32 * 256 * 256 * sizeof(float);
    size_t emb1_sz = 1 * 64 * 128 * 128 * sizeof(float);
    size_t emb2_sz = 1 * 256 * 64 * 64 * sizeof(float);
    size_t box_sz = 4 * sizeof(float);
    size_t iou_sz = 3 * sizeof(float);
    size_t masks_sz = 3 * LOW_MASK_SIZE * LOW_MASK_SIZE * sizeof(float);
    size_t obj_sz = 1 * sizeof(float);

    CK(cudaMalloc(&d_pixel, pixel_sz));
    CK(cudaMalloc(&d_emb0, emb0_sz));
    CK(cudaMalloc(&d_emb1, emb1_sz));
    CK(cudaMalloc(&d_emb2, emb2_sz));
    CK(cudaMalloc(&d_pts, 1));
    CK(cudaMalloc(&d_lbls, 1));
    CK(cudaMalloc(&d_box, box_sz));
    CK(cudaMalloc(&d_iou, iou_sz));
    CK(cudaMalloc(&d_masks, masks_sz));
    CK(cudaMalloc(&d_objlogits, obj_sz));

    enc_ctx->setInputShape("pixel_values", nvinfer1::Dims4{1, 3, INPUT_SIZE, INPUT_SIZE});
    enc_ctx->setTensorAddress("pixel_values", d_pixel);
    enc_ctx->setTensorAddress("image_embeddings.0", d_emb0);
    enc_ctx->setTensorAddress("image_embeddings.1", d_emb1);
    enc_ctx->setTensorAddress("image_embeddings.2", d_emb2);

    nvinfer1::Dims pts_dims;  pts_dims.nbDims  = 4; pts_dims.d[0]  = 1; pts_dims.d[1]  = 1; pts_dims.d[2]  = 0; pts_dims.d[3]  = 2;
    nvinfer1::Dims lbls_dims; lbls_dims.nbDims = 3; lbls_dims.d[0] = 1; lbls_dims.d[1] = 1; lbls_dims.d[2] = 0;
    nvinfer1::Dims box_dims;  box_dims.nbDims  = 3; box_dims.d[0]  = 1; box_dims.d[1]  = 1; box_dims.d[2]  = 4;

    dec_ctx->setInputShape("input_points", pts_dims);
    dec_ctx->setInputShape("input_labels", lbls_dims);
    dec_ctx->setInputShape("input_boxes", box_dims);
    dec_ctx->setTensorAddress("input_points", d_pts);
    dec_ctx->setTensorAddress("input_labels", d_lbls);
    dec_ctx->setTensorAddress("input_boxes", d_box);
    dec_ctx->setTensorAddress("image_embeddings.0", d_emb0);
    dec_ctx->setTensorAddress("image_embeddings.1", d_emb1);
    dec_ctx->setTensorAddress("image_embeddings.2", d_emb2);
    dec_ctx->setTensorAddress("iou_scores", d_iou);
    dec_ctx->setTensorAddress("pred_masks", d_masks);
    dec_ctx->setTensorAddress("object_score_logits", d_objlogits);

    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    std::cout << "encoder: " << args.encoder << "\n";
    std::cout << "decoder: " << args.decoder << "\n";

    cv::VideoWriter writer(args.output,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps / std::max(1, args.stride), cv::Size(W, H));
    if (!writer.isOpened()) { std::cerr << "cannot write " << args.output << "\n"; return 5; }

    std::ofstream json(args.output.substr(0, args.output.find_last_of('.')) + ".json");
    json << "{\"video\":\"" << args.video << "\",\"encoder\":\"" << args.encoder
         << "\",\"decoder\":\"" << args.decoder << "\",\"frames\":[";

    float bx1 = (args.bx1 < 0) ? 0.0f       : args.bx1;
    float by1 = (args.by1 < 0) ? 0.0f       : args.by1;
    float bx2 = (args.bx2 < 0) ? (float)W   : args.bx2;
    float by2 = (args.by2 < 0) ? (float)H   : args.by2;

    std::vector<float> h_pixel(3 * INPUT_SIZE * INPUT_SIZE);
    std::vector<float> h_iou(3);
    std::vector<float> h_masks(3 * LOW_MASK_SIZE * LOW_MASK_SIZE);

    cv::Mat frame;
    int i = 0, n = 0;
    auto t0 = std::chrono::steady_clock::now();
    bool first_frame = true;

    while (cap.read(frame)) {
        if (i % args.stride == 0) {
            preprocess(frame, h_pixel);
            CK(cudaMemcpyAsync(d_pixel, h_pixel.data(), pixel_sz,
                               cudaMemcpyHostToDevice, stream));
            if (!enc_ctx->enqueueV3(stream)) { std::cerr << "encoder enqueue failed\n"; return 6; }

            float sx = (float)INPUT_SIZE / W, sy = (float)INPUT_SIZE / H;
            float box[4] = {bx1 * sx, by1 * sy, bx2 * sx, by2 * sy};
            CK(cudaMemcpyAsync(d_box, box, box_sz, cudaMemcpyHostToDevice, stream));

            if (!dec_ctx->enqueueV3(stream)) { std::cerr << "decoder enqueue failed\n"; return 6; }

            CK(cudaMemcpyAsync(h_iou.data(), d_iou, iou_sz, cudaMemcpyDeviceToHost, stream));
            CK(cudaMemcpyAsync(h_masks.data(), d_masks, masks_sz,
                               cudaMemcpyDeviceToHost, stream));
            CK(cudaStreamSynchronize(stream));

            int best = 0;
            for (int k = 1; k < 3; ++k) if (h_iou[k] > h_iou[best]) best = k;

            cv::Mat mask_low(LOW_MASK_SIZE, LOW_MASK_SIZE, CV_32F,
                             (void*)(h_masks.data() + best * LOW_MASK_SIZE * LOW_MASK_SIZE));
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
                 << ",\"iou_score\":" << h_iou[best]
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

    cudaStreamDestroy(stream);
    cudaFree(d_pixel);
    cudaFree(d_emb0); cudaFree(d_emb1); cudaFree(d_emb2);
    cudaFree(d_pts); cudaFree(d_lbls); cudaFree(d_box);
    cudaFree(d_iou); cudaFree(d_masks); cudaFree(d_objlogits);

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "wrote " << args.output << " (" << n << " frames in " << sec
              << "s, " << (sec > 0 ? n / sec : 0) << " fps)\n";
    return 0;
}
