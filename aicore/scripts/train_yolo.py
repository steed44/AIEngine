import json
import sys
from ultralytics import YOLO

def train(cfg_json):
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
