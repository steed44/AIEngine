# ============================================================
# 文件: scripts/train_yolo.py
# 用途: YOLOv8 训练脚本 — 被 C++ 训练器通过 Python 嵌入调用
#   从 JSON 配置读取参数，训练 YOLO 模型并返回结果 JSON
# ============================================================

import json
import sys
from ultralytics import YOLO

def train(cfg_json):
    """训练 YOLO 模型

    参数:
        cfg_json: JSON 字符串，包含 data/epochs/imgsz/batch 等训练参数
    返回:
        JSON 字符串，包含 best.pt/last.pt 路径和 mAP 指标
    """
    cfg = json.loads(cfg_json)
    model = YOLO(cfg.get("model", "yolov8n.pt"))
    results = model.train(
        data=cfg["data"],
        epochs=cfg.get("epochs", 100),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        device=cfg.get("device", 0),
        project=cfg.get("project", "runs/train"),
        name=cfg.get("name", "exp"),
        pretrained=cfg.get("pretrained", True),
        optimizer=cfg.get("optimizer", "Adam"),
        lr0=cfg.get("lr0", 0.001),
        resume=cfg.get("resume", False)
    )
    return json.dumps({
        "status": "ok",
        "best": str(results.save_dir / "weights" / "best.pt"),
        "last": str(results.save_dir / "weights" / "last.pt"),
        "map50": results.results_dict.get("metrics/mAP50(B)", 0),
        "map5095": results.results_dict.get("metrics/mAP50-95(B)", 0)
    })

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "{}"
    print(train(cfg))
