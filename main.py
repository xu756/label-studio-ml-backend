import logging
from pathlib import Path
from threading import RLock

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from ultralytics import YOLO, YOLOE

from label_studio_ml.api import run_app
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from model_config import SERVER_CONFIG, get_model_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 每次请求创建封装实例；权重在进程内缓存。锁保护提示更新及完整推理，避免项目串类。
_MODEL_CACHE = {}
_MODEL_LOCK = RLock()


class YoloModel(LabelStudioMLBase):
    """YOLOE 文本提示 / 普通 YOLO 的矩形框与多边形预标注。"""

    def setup(self):
        self.config = get_model_config(self.project_id)
        self.set(
            "model_version",
            f"{self.config['model_type']}-{Path(self.config['model_path']).stem}-1.0",
        )

    def _get_model(self):
        key = (
            self.config["model_type"],
            self.config["model_path"],
            self.config["device"],
        )
        if key not in _MODEL_CACHE:
            factory = YOLOE if self.config["model_type"] == "yoloe" else YOLO
            logger.info("加载 %s 权重: %s", key[0], key[1])
            model = factory(key[1])
            if self.config["device"] is not None:
                model.to(self.config["device"])
            _MODEL_CACHE[key] = (model, None)
        return key, _MODEL_CACHE[key]

    def _get_control(self):
        tag = (
            "PolygonLabels"
            if self.config["output_type"] == "polygon"
            else "RectangleLabels"
        )
        candidates = [
            (name, info)
            for name, info in self.parsed_label_config.items()
            if info["type"] == tag
            and (not self.config["control_name"] or name == self.config["control_name"])
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"请配置一个 {tag} 控件；有多个时在 model_config.py 指定 control_name"
            )
        name, info = candidates[0]
        if len(info["inputs"]) != 1 or info["inputs"][0]["type"] != "Image":
            raise ValueError(f"{name} 必须关联一个 Image")
        return name, info["to_name"][0], info["inputs"][0]["value"]

    def predict(
        self, tasks: list[dict], context: dict | None = None, **kwargs
    ) -> ModelResponse:
        from_name, to_name, image_key = self._get_control()
        attrs = self.label_interface.get_control(from_name).labels_attrs
        explicit_map = self.config.get("label_map")
        if explicit_map is not None and (
            not isinstance(explicit_map, dict)
            or not explicit_map
            or any(
                not isinstance(k, str) or not k.strip() or v not in attrs
                for k, v in explicit_map.items()
            )
        ):
            raise ValueError(
                "label_map 必须是非空的 {YOLOE 类别: 当前控件的 Label value} 字典"
            )
        # predicted_values 支持英文提示映射到中文标签；默认用标签自身作为提示。
        prompts = []
        for label, attr in attrs.items():
            prompts.extend(
                p.strip()
                for p in (attr.attr.get("predicted_values") or label).split(",")
                if p.strip()
            )
        prompts = (
            list(explicit_map)
            if explicit_map is not None
            else list(dict.fromkeys(prompts))
        )
        if not prompts:
            raise ValueError("标注控件必须至少包含一个非空 Label")

        predictions = []
        with _MODEL_LOCK:
            key, (model, previous_prompts) = self._get_model()
            use_prompts = (
                self.config["model_type"] == "yoloe" and self.config["text_prompts"]
            )
            signature = tuple(prompts) if use_prompts else None
            # 同一权重若切换到固定类别模式，重新加载以恢复原始类别。
            if not use_prompts and previous_prompts is not None:
                del _MODEL_CACHE[key]
                key, (model, previous_prompts) = self._get_model()
            if use_prompts and signature != previous_prompts:
                model.set_classes(prompts, model.get_text_pe(prompts))
                _MODEL_CACHE[key] = (model, signature)
            names = (
                list(model.names.values())
                if isinstance(model.names, dict)
                else model.names
            )
            label_map = (
                explicit_map
                if explicit_map is not None
                else self.build_label_map(from_name, names)
            )
            label_map = {name.casefold(): label for name, label in label_map.items()}
            options = {k: self.config[k] for k in ("conf", "iou", "imgsz")}
            if self.config["device"] is not None:
                options["device"] = self.config["device"]
            for task in tasks:
                regions = []
                prediction = {
                    "result": regions,
                    "score": 0,
                    "model_version": self.get("model_version"),
                }
                predictions.append(prediction)
                image_url = task.get("data", {}).get(image_key)
                if not image_url:
                    logger.warning("任务 %s 缺少图像字段 %s", task.get("id"), image_key)
                    continue
                path = self.get_local_path(image_url, task_id=task.get("id"))
                with Image.open(path) as img:
                    width, height = img.size
                for result in model.predict(path, **options):
                    if result.boxes is None:
                        continue
                    if (
                        self.config["output_type"] == "polygon"
                        and len(result.boxes)
                        and result.masks is None
                    ):
                        raise ValueError(
                            "polygon 输出需要分割权重，当前模型未返回 masks"
                        )
                    for index, box in enumerate(result.boxes):
                        label = label_map.get(result.names[int(box.cls[0])].casefold())
                        if label is None:
                            continue
                        if self.config["output_type"] == "polygon":
                            points = result.masks.xyn[index]
                            if len(points) < 3:
                                continue
                            value = {
                                "polygonlabels": [label],
                                "points": (points.clip(0, 1) * 100).tolist(),
                            }
                            kind = "polygonlabels"
                        else:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, x2 = [max(0, min(width, x)) for x in (x1, x2)]
                            y1, y2 = [max(0, min(height, y)) for y in (y1, y2)]
                            value = {
                                "rectanglelabels": [label],
                                "x": x1 / width * 100,
                                "y": y1 / height * 100,
                                "width": (x2 - x1) / width * 100,
                                "height": (y2 - y1) / height * 100,
                                "rotation": 0,
                            }
                            kind = "rectanglelabels"
                        regions.append(
                            {
                                "from_name": from_name,
                                "to_name": to_name,
                                "type": kind,
                                "value": value,
                                "score": float(box.conf[0]),
                                "original_width": width,
                                "original_height": height,
                                "image_rotation": 0,
                            }
                        )
                prediction["score"] = sum(r["score"] for r in regions) / max(
                    len(regions), 1
                )
        return ModelResponse(
            predictions=predictions, model_version=self.get("model_version")
        )


if __name__ == "__main__":
    port = SERVER_CONFIG["port"]
    logger.info("启动预测服务，端口 %s；项目配置见 model_config.py", port)
    run_app(YoloModel, host=SERVER_CONFIG["host"], port=port)
