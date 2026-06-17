# ============================================================
# 文件: scripts/export_patchcore_backbone.py
# 用途: 导出支持多层特征提取的 TorchScript 模型供 PatchCore 使用
#
# 背景:
#   PatchCore 需要提取 CNN 中间层的局部特征（如 layer2, layer3 的输出），
#   而标准的 TorchScript 模型只返回最终输出。本脚本创建包装模型，
#   将指定中间层的特征图打包为元组输出，供 C++ LibTorchBackbone 消费。
#
# 用法:
#   python export_patchcore_backbone.py --model resnet50 \
#       --layers layer2,layer3 --input-size 224 \
#       --output patchcore_backbone.pt
#
# 输出:
#   导出的 TorchScript 模型接受 [1,3,H,W] 输入，返回 (feat_layer1, feat_layer2, ...)
#   其中每个 tensor 形状为 [1, C, H_feat, W_feat]
# ============================================================

import argparse
import torch
import torch.nn as nn
import torchvision.models as models


class PatchCoreBackboneWrapper(nn.Module):
    """包装模型，截取指定中间层的输出作为特征元组返回

    工作原理：
        在模型的指定子模块上注册 forward hook，将中间层的输出捕获到列表。
        前向传播时依次触发各 hook，收集特征图后打包为元组返回。

    支持模型:
        - resnet18 / resnet34 / resnet50 / resnet101
        - wide_resnet50_2（WideResNet50，PatchCore 论文默认 backbone）
        - 其他支持通过 children() 访问子模块的 CNN

    使用示例:
        wrapper = PatchCoreBackboneWrapper(
            base_model=resnet50(pretrained=True),
            layer_names=["layer2", "layer3"]
        )
        output = wrapper(torch.randn(1, 3, 224, 224))
        # output[0].shape = [1, 512, 28, 28]  (layer2)
        # output[1].shape = [1, 1024, 14, 14] (layer3)
    """

    def __init__(self, base_model: nn.Module, layer_names: list[str]):
        super().__init__()
        self.base_model = base_model
        self.layer_names = layer_names
        self._features: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        # 在指定层注册 forward hook
        # 模块名到实际子模块的映射，通过 named_modules() 遍历查找
        name_to_module = dict(base_model.named_modules())
        for name in layer_names:
            module = name_to_module.get(name)
            if module is None:
                available = [n for n, _ in base_model.named_modules()]
                raise ValueError(
                    f"Layer '{name}' not found in model. "
                    f"Available layers: {available}"
                )
            hook = module.register_forward_hook(self._make_hook())
            self._hooks.append(hook)

    def _make_hook(self):
        """创建 forward hook，将中间层输出追加到 _features 列表"""

        def hook(module, input, output):
            self._features.append(output)

        return hook

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """前向传播，返回指定中间层的特征图元组

        参数:
            x: 输入张量，形状 [N, C, H, W]，值范围 [0, 1]

        返回:
            tuple[torch.Tensor, ...]: 各中间层的输出特征图
        """
        self._features.clear()
        self.base_model(x)
        result = tuple(self._features)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="导出 PatchCore 多层特征提取 TorchScript 模型"
    )
    parser.add_argument(
        "--model",
        default="wide_resnet50_2",
        help="基础模型名称 (default: wide_resnet50_2)\n"
             "可选: resnet18, resnet34, resnet50, resnet101, wide_resnet50_2",
    )
    parser.add_argument(
        "--layers",
        default="layer2,layer3",
        help="需要提取特征的模块名，逗号分隔 (default: layer2,layer3)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="输入图像尺寸 (default: 224)",
    )
    parser.add_argument(
        "--output",
        default="patchcore_backbone.pt",
        help="输出 TorchScript 文件路径 (default: patchcore_backbone.pt)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="使用预训练权重 (default: True)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="导出设备 (default: cpu)",
    )
    args = parser.parse_args()

    # ---- 1. 加载预训练模型 ----
    print(f"Loading {args.model} (pretrained={args.pretrained})...")
    model_class = getattr(models, args.model, None)
    if model_class is None:
        print(f"Error: model '{args.model}' not found in torchvision.models")
        sys.exit(1)

    base_model = model_class(pretrained=args.pretrained)
    base_model.eval()

    # ---- 2. 解析层名并创建包装器 ----
    layer_names = [name.strip() for name in args.layers.split(",")]
    wrapper = PatchCoreBackboneWrapper(base_model, layer_names)
    wrapper.eval()

    # ---- 3. 脚本化 / 追踪导出 ----
    # 使用 script 而非 trace，因为 wrapper 包含控制流（list.clear 等）
    device = torch.device(args.device)
    wrapper.to(device)
    dummy = torch.randn(1, 3, args.input_size, args.input_size).to(device)

    print("Tracing with torch.jit.script...")
    scripted = torch.jit.script(wrapper, example_inputs=[dummy])

    # ---- 4. 验证输出结构 ----
    test_out = scripted(dummy)
    print(f"Output type: {type(test_out)}")
    if isinstance(test_out, (tuple, list)):
        print(f"Number of output tensors: {len(test_out)}")
        for i, t in enumerate(test_out):
            print(f"  [{i}] shape={list(t.shape)}, dtype={t.dtype}")
    elif isinstance(test_out, torch.Tensor):
        print(f"Single tensor shape: {list(test_out.shape)}")

    # ---- 5. 保存 ----
    scripted.save(args.output)
    print(f"TorchScript model saved to {args.output}")
    print("Done!")


if __name__ == "__main__":
    import sys
    main()
