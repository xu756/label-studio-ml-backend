"""修改配置后重启 main.py。项目 ID 可从 Label Studio 项目 URL 中取得。"""

# 两个入口共用的服务监听配置。
SERVER_CONFIG = {"host": "0.0.0.0", "port": 9090}

SAM2_CONFIG = {
    "device": "cuda",
    "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "checkpoint": "sam2.1_hiera_large.pt",  # 相对于 checkpoints 目录
}

DEFAULT_CONFIG = {
    "model_type": "yoloe",  # yoloe 或 yolo
    "model_path": "yoloe-26s-seg.pt",
    "output_type": "rectangle",  # rectangle 或 polygon
    "control_name": None,  # 同类型有多个控件时，填写 XML 中的 name
    "conf": 0.25,
    "iou": 0.7,
    "imgsz": 640,
    "device": None,  # None 自动选择；也可以使用 "cpu"、"0"
    "text_prompts": True,  # YOLOE 文本提示；固定类别/无提示权重设为 False
    # 左侧用于 YOLOE set_classes，右侧用于 Label Studio 预测返回。
    "label_map": {
        "black iron frame": "iron-frame",
        "White box": "marker",
    },  # None 时读取项目 XML
}

# 按项目覆盖默认配置；未列出的项目使用 DEFAULT_CONFIG。
PROJECT_CONFIGS = {
    # "1": {"output_type": "polygon", "label_map": {"potato": "土豆"}},
    # "2": {"model_type": "yolo", "model_path": "weights/best.pt"},
}


def get_model_config(project_id):
    config = dict(DEFAULT_CONFIG)
    config.update(PROJECT_CONFIGS.get(str(project_id), {}))
    if config["model_type"] not in ("yolo", "yoloe"):
        raise ValueError("model_type 必须是 yolo 或 yoloe")
    if config["output_type"] not in ("rectangle", "polygon"):
        raise ValueError("output_type 必须是 rectangle 或 polygon")
    if not 0 <= config["conf"] <= 1 or not 0 <= config["iou"] <= 1:
        raise ValueError("conf 和 iou 必须在 0 到 1 之间")
    if config["imgsz"] <= 0:
        raise ValueError("imgsz 必须大于 0")
    return config
