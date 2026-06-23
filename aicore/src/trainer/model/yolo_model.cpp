// YOLOv8 模型实现 — 全部构建块：ConvBnSiLU, C2f, SPPF, Detect
#include "trainer/model/yolo_model.h"
#include <cmath>

namespace aicore {

// ============================================================
// ConvBnSiLU
// ============================================================
ConvBnSiLUImpl::ConvBnSiLUImpl(int inCh, int outCh, int k, int s, int p) {
    conv = torch::nn::Conv2d(register_module("conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(inCh, outCh, k).stride(s).padding(p).bias(false))));
    bn = torch::nn::BatchNorm2d(register_module("bn", torch::nn::BatchNorm2d(outCh)));
}

torch::Tensor ConvBnSiLUImpl::forward(torch::Tensor x) {
    return torch::silu(bn(conv(x)));
}

// ============================================================
// Bottleneck
// ============================================================
BottleneckImpl::BottleneckImpl(int inCh, int outCh, bool shortcut_)
    : add(shortcut_ && inCh == outCh) {
    int c = outCh;
    cv1 = ConvBnSiLU(register_module("cv1", ConvBnSiLU(inCh, c, 3, 1, 1)));
    cv2 = ConvBnSiLU(register_module("cv2", ConvBnSiLU(c, outCh, 3, 1, 1)));
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x) {
    auto y = cv2->forward(cv1->forward(x));
    return add ? x + y : y;
}

// ============================================================
// C2f
// ============================================================
C2fImpl::C2fImpl(int inCh, int outCh, int n, bool shortcut_, float e) {
    c = (int)(outCh * e); // hidden channels
    cv1 = torch::nn::Conv2d(register_module("cv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(inCh, 2 * c, 1).stride(1).padding(0))));
    cv2 = torch::nn::Conv2d(register_module("cv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions((2 + n) * c, outCh, 1).stride(1).padding(0))));
    for (int i = 0; i < n; ++i) {
        m->push_back(register_module(
            "m" + std::to_string(i),
            Bottleneck(c, c, shortcut_)));
    }
    add = shortcut_ && inCh == outCh;
}

torch::Tensor C2fImpl::forward(torch::Tensor x) {
    auto y = cv1->forward(x);
    auto chunks = y.chunk(2, 1); // split into two halves along channel dim
    // yList = [chunks[0], chunks[1], b1(chunks[1]), b2(b1(chunks[1])), ...]
    // Total channels: c + c + n*c = (2+n)*c — matches cv2 input
    auto yList = std::vector<torch::Tensor>{ chunks[0], chunks[1] };
    for (const auto& b : *m) {
        auto bottleneck = std::dynamic_pointer_cast<BottleneckImpl>(b);
        if (bottleneck) {
            yList.push_back(bottleneck->forward(yList.back()));
        }
    }
    auto out = torch::cat(yList, 1);
    return cv2->forward(out);
}

// ============================================================
// SPPF
// ============================================================
SPPFImpl::SPPFImpl(int inCh, int outCh, int k)
    : c(inCh) {
    cv1 = ConvBnSiLU(register_module("cv1", ConvBnSiLU(inCh, c, 1, 1, 0)));
    cv2 = ConvBnSiLU(register_module("cv2", ConvBnSiLU(c * 4, outCh, 1, 1, 0)));
    m1 = torch::nn::MaxPool2d(register_module("m1", torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(k).stride(1).padding(k / 2))));
    m2 = torch::nn::MaxPool2d(register_module("m2", torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(k).stride(1).padding(k / 2))));
    m3 = torch::nn::MaxPool2d(register_module("m3", torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(k).stride(1).padding(k / 2))));
}

torch::Tensor SPPFImpl::forward(torch::Tensor x) {
    x = cv1->forward(x);
    auto y1 = m1->forward(x);
    auto y2 = m2->forward(y1);
    auto y3 = m3->forward(y2);
    return cv2->forward(torch::cat({ x, y1, y2, y3 }, 1));
}

// ============================================================
// Detect
// ============================================================
DetectImpl::DetectImpl(int nc_, const std::vector<int>& ch)
    : nc(nc_), nl((int)ch.size()) {
    int regMax = 16;
    no = nc + 4 * regMax;
    proj = torch::arange(0, regMax, torch::kFloat);
    register_buffer("proj", proj);

    for (int i = 0; i < nl; ++i) {
        auto idx = std::to_string(i);
        cvReg.push_back(torch::nn::Conv2d(register_module(
            "cvReg" + idx, torch::nn::Conv2d(
                torch::nn::Conv2dOptions(ch[i], 4 * regMax, 1)))));
        cvCls.push_back(torch::nn::Conv2d(register_module(
            "cvCls" + idx, torch::nn::Conv2d(
                torch::nn::Conv2dOptions(ch[i], nc, 1)))));
    }
}

torch::Tensor DetectImpl::forward(const std::vector<torch::Tensor>& xs) {
    std::vector<torch::Tensor> regs, clss;
    for (int i = 0; i < nl; ++i) {
        regs.push_back(cvReg[i]->forward(xs[i]));
        clss.push_back(cvCls[i]->forward(xs[i]));
    }
    return torch::cat({ torch::cat(regs, 1), torch::cat(clss, 1) }, 1);
}

// ============================================================
// YOLOv8Model
// ============================================================
YOLOv8Model::YOLOv8Model(int nc) : nc_(nc) {}

Status YOLOv8Model::Build(const ModelConfig& config) {
    nc_ = config.numClasses;

    // 构建 YOLOv8-n 主干 + 颈网络
    auto m = torch::nn::Sequential();

    // --- Backbone ---
    // P1
    m->push_back("0", ConvBnSiLU(3, 16, 3, 2));      // 0:  320x320
    // P2
    m->push_back("1", ConvBnSiLU(16, 32, 3, 2));     // 1:  160x160
    m->push_back("2", C2f(32, 32, 1, true));          // 2:  160x160
    // P3
    m->push_back("3", ConvBnSiLU(32, 64, 3, 2));     // 3:  80x80
    m->push_back("4", C2f(64, 64, 2, true));          // 4:  80x80
    // P4
    m->push_back("5", ConvBnSiLU(64, 128, 3, 2));    // 5:  40x40
    m->push_back("6", C2f(128, 128, 2, true));        // 6:  40x40
    // P5
    m->push_back("7", ConvBnSiLU(128, 128, 3, 2));   // 7:  20x20
    m->push_back("8", C2f(128, 128, 1, true));        // 8:  20x20
    m->push_back("9", SPPF(128, 128));                // 9:  20x20

    // --- Neck ---
    // Upsample to P4
    m->push_back("10", torch::nn::Upsample(
        torch::nn::UpsampleOptions().scale_factor(std::vector<double>{2, 2}).mode(torch::kNearest)));
    // Concat with P4 (layer 6) — done via custom forward
    m->push_back("11", torch::nn::Identity());         // concat placeholder
    m->push_back("12", C2f(256, 64, 1, false));       // 12: concat [10,6] → C2f

    // Upsample to P3
    m->push_back("13", torch::nn::Upsample(
        torch::nn::UpsampleOptions().scale_factor(std::vector<double>{2, 2}).mode(torch::kNearest)));
    m->push_back("14", torch::nn::Identity());         // concat placeholder
    m->push_back("15", C2f(128, 32, 1, false));       // 15: concat [13,4] → C2f

    // Down to P4
    m->push_back("16", ConvBnSiLU(32, 64, 3, 2));     // 16
    m->push_back("17", torch::nn::Identity());         // concat placeholder
    m->push_back("18", C2f(128, 64, 1, false));       // 18: concat [16,12] → C2f

    // Down to P5
    m->push_back("19", ConvBnSiLU(64, 128, 3, 2));    // 19
    m->push_back("20", torch::nn::Identity());         // concat placeholder
    m->push_back("21", C2f(256, 128, 1, false));      // 21: concat [19,9] → C2f

    // Detect head: input channels from three scales (P3, P4, P5)
    std::vector<int> detectCh = {32, 64, 128};
    detect_ = Detect(register_module("22", Detect(nc_, detectCh)));

    // Store indices for concat nodes (tensor indices in sequential forward)
    // layer 4: P3 output (index in sequential: 4)
    // layer 6: P4 output (index: 6)
    // layer 9: P5 output (index: 9)
    // layer 12: neck P4 output (index: 12)
    // layer 15: neck P3 output (index: 15)
    // layer 18: neck P4 (index: 18)
    // layer 21: neck P5 (index: 21)

    model_ = torch::nn::Sequential(register_module("model", m));
    return Status{};
}

// 前向传播：带 concat 的 YOLOv8 自定义 forward
// 返回 detect 头处理后的三尺度输出 [B, no, H_i, W_i]
// no = 4*regMax + nc  (reg + cls)
std::vector<torch::Tensor> YOLOv8Model::Forward(torch::Tensor x) {
    if (!model_) return {};

    auto p3 = torch::Tensor();
    auto p4 = torch::Tensor();
    auto p5 = torch::Tensor();
    auto n4 = torch::Tensor();
    auto n3 = torch::Tensor();

    int idx = 0;
    for (auto& mod : *model_) {
        if (idx == 11) {
            x = torch::cat({ x, p4 }, 1);
        } else if (idx == 14) {
            x = torch::cat({ x, p3 }, 1);
        } else if (idx == 17) {
            x = torch::cat({ x, n4 }, 1);
        } else if (idx == 20) {
            x = torch::cat({ x, p5 }, 1);
        } else {
            x = *mod.any_forward(std::move(x)).template try_get<torch::Tensor>();
        }

        if (idx == 4) p3 = x;
        else if (idx == 6) p4 = x;
        else if (idx == 9) p5 = x;
        else if (idx == 12) n4 = x;
        else if (idx == 15) n3 = x;

        ++idx;
    }

    auto neckOuts = std::vector<torch::Tensor>{ n3, n4, x };
    if (!detect_) return neckOuts;

    std::vector<torch::Tensor> result;
    for (int i = 0; i < detect_->nl; ++i) {
        auto f = neckOuts[i];
        auto reg = detect_->cvReg[i]->forward(f);
        auto cls = detect_->cvCls[i]->forward(f);
        result.push_back(torch::cat({ reg, cls }, 1));
    }
    return result;
}

torch::Tensor YOLOv8Model::predict(torch::Tensor x) {
    auto features = Forward(x);
    if (features.empty()) return {};
    // Flatten spatial dims and concat all scales
    std::vector<torch::Tensor> flat;
    for (auto& f : features) {
        flat.push_back(f.flatten(2)); // [B, no, H*W]
    }
    return torch::cat(flat, 2); // [B, no, total_grid_points]
}

Status YOLOv8Model::Save(const std::string& path) {
    try {
        torch::save(model_, path);
        return Status{};
    } catch (const std::exception& e) {
        return Status{StatusCode::ErrorInternal, e.what()};
    }
}

Status YOLOv8Model::Load(const std::string& path) {
    try {
        torch::load(model_, path);
        return Status{};
    } catch (const std::exception& e) {
        return Status{StatusCode::ErrorInternal, e.what()};
    }
}

} // namespace aicore
