# ============================================================
# 文件: scripts/export_onnx.py
# 用途: ONNX 导出 + YOLO 训练工具脚本
#   支持两种命令: export (PyTorch → ONNX) 和 train (YOLO 训练)
#   被 C++ 优化器通过 Python 嵌入调用
# ============================================================

import torch
import torch.nn as nn
import sys
import json

def export_onnx(config_json):
    """将 TorchScript 模型导出为 ONNX 格式

    参数:
        config_json: JSON 字符串，包含 model_path/onnx_path/input_shape/opset
    返回:
        JSON 字符串，包含导出状态和路径
    """
    cfg = json.loads(config_json)
    model_path = cfg["model_path"]
    onnx_path = cfg["onnx_path"]
    input_shape = cfg.get("input_shape", [1, 3, 640, 640])
    opset = cfg.get("opset", 17)

    model = torch.jit.load(model_path)
    model.eval()

    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    return json.dumps({"status": "ok", "onnx_path": onnx_path})

def train_yolo(config_json):
    """使用 Ultralytics YOLO 训练模型

    参数:
        config_json: JSON 字符串，包含训练参数
    返回:
        JSON 字符串，包含训练结果路径
    """
    cfg = json.loads(config_json)
    from ultralytics import YOLO
    model = YOLO(cfg.get("model", "yolov8n.pt"))
    results = model.train(
        data=cfg["data"],
        epochs=cfg.get("epochs", 100),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        device=cfg.get("device", 0),
        project=cfg.get("project", "runs/train"),
        name=cfg.get("name", "exp")
    )
    return json.dumps({"status": "ok", "results": str(results)})

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg = sys.argv[2] if len(sys.argv) > 2 else "{}"
    if cmd == "export":
        print(export_onnx(cfg))
    elif cmd == "train":
        print(train_yolo(cfg))
