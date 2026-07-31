import logging
import os
import sys
from uuid import uuid4

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

# 加载当前目录下的 .env 环境变量文件
load_dotenv()

from label_studio_sdk.converter import brush

from label_studio_ml.api import run_app
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 SAM2 图像预测器
ROOT_DIR = os.getcwd()
sys.path.insert(0, ROOT_DIR)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    logger.error(
        "未找到 SAM2 包，请确保已正确安装 SAM2 及其依赖，并且 checkpoints 文件夹下存在权重文件。"
    )
    build_sam2 = None
    SAM2ImagePredictor = None

DEVICE = os.getenv("DEVICE", "cuda")
# 默认配置和权重文件
MODEL_CONFIG = os.getenv("MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "sam2.1_hiera_large.pt")

if DEVICE == "cuda" and torch.cuda.is_available():
    # 使用 bfloat16 优化精度和速度
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # 在 Ampere 系列 GPU 上开启 TensorFloat32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# 全局的 SAM2 预测器对象
predictor = None
if build_sam2 and SAM2ImagePredictor:
    sam2_checkpoint = str(os.path.join(ROOT_DIR, "checkpoints", MODEL_CHECKPOINT))
    if os.path.exists(sam2_checkpoint):
        logger.info(
            f"正在从 config {MODEL_CONFIG} 以及 checkpoint {sam2_checkpoint} 加载并初始化 SAM2 模型..."
        )
        sam2_model = build_sam2(MODEL_CONFIG, sam2_checkpoint, device=DEVICE)
        predictor = SAM2ImagePredictor(sam2_model)
    else:
        logger.warning(
            f"未在路径 '{sam2_checkpoint}' 找到 SAM2 权重文件。跳过模型初始化。"
        )


class SAM2ImageModel(LabelStudioMLBase):
    """
    SAM2 图像交互式标注 (BrushLabels) 机器学习后端。
    根据前台交互点击生成的点或框自动返回抠图掩码 (Mask)。
    """

    def get_results(self, masks, probs, width, height, from_name, to_name, label):
        """
        将 SAM2 预测返回的二进制 Mask 掩码序列化为 Label Studio 支持的 RLE 格式结果。
        """
        results = []
        total_prob = 0
        for mask, prob in zip(masks, probs):
            label_id = str(uuid4())[:4]
            mask = mask * 255  # 转换为 0-255 的灰度图格式
            rle = brush.mask2rle(mask)  # 将 Mask 编码为 RLE (Run-Length Encoding)
            total_prob += prob
            results.append(
                {
                    "id": label_id,
                    "from_name": from_name,
                    "to_name": to_name,
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "format": "rle",
                        "rle": rle,
                        "brushlabels": [label],
                    },
                    "score": prob,
                    "type": "brushlabels",
                    "readonly": False,
                }
            )

        return [
            {
                "result": results,
                "model_version": self.get("model_version"),
                "score": total_prob / max(len(results), 1),
            }
        ]

    def set_image(self, image_url, task_id):
        """设置当前处理的图像，供 SAM2 Image Predictor 提取 Image Embedding 缓存。"""
        if not predictor:
            raise RuntimeError("SAM2 预测器未正确初始化。")
        image_path = self.get_local_path(image_url, task_id=task_id)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        predictor.set_image(image)

    def _sam_predict(
        self, img_url, point_coords=None, point_labels=None, input_box=None, task=None
    ):
        """调用 SAM2 运行交互式预测。"""
        self.set_image(img_url, task.get("id"))
        point_coords = (
            np.array(point_coords, dtype=np.float32) if point_coords else None
        )
        point_labels = (
            np.array(point_labels, dtype=np.float32) if point_labels else None
        )
        input_box = np.array(input_box, dtype=np.float32) if input_box else None

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            multimask_output=True,
        )
        # 按得分从高到低排序，选择置信度最高的掩码结果
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        mask = masks[0, :, :].astype(np.uint8)
        prob = float(scores[0])
        return {"masks": [mask], "probs": [prob]}

    def predict(
        self, tasks: list[dict], context: dict | None = None, **kwargs
    ) -> ModelResponse:
        """
        根据用户在前台点击关键点 (Keypoints) 或画边界框 (Bounding Box) 交互信息计算抠图掩码。
        """
        if not predictor:
            logger.error("SAM2 预测器未加载，跳过预测。")
            return ModelResponse(predictions=[])

        # 获取 Labeling Config 中配置的 BrushLabels（控制标签）和 Image（数据标签）
        from_name, to_name, value = self.get_first_tag_occurence("BrushLabels", "Image")

        if not context or not context.get("result"):
            # 如果没有交互信息，说明前台还没有触发动作，直接返回空
            return ModelResponse(predictions=[])

        # 获取图像在前端界面上的宽度和高度
        image_width = context["result"][0]["original_width"]
        image_height = context["result"][0]["original_height"]

        # 搜集交互点的坐标、标签（正点、负点）或边界框位置
        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None
        for ctx in context["result"]:
            # 还原百分比坐标为图像真实绝对像素坐标
            x = ctx["value"]["x"] * image_width / 100
            y = ctx["value"]["y"] * image_height / 100
            ctx_type = ctx["type"]
            selected_label = ctx["value"][ctx_type][0]
            if ctx_type == "keypointlabels":
                point_labels.append(int(ctx.get("is_positive", 0)))
                point_coords.append([int(x), int(y)])
            elif ctx_type == "rectanglelabels":
                box_width = ctx["value"]["width"] * image_width / 100
                box_height = ctx["value"]["height"] * image_height / 100
                input_box = [int(x), int(y), int(box_width + x), int(box_height + y)]

        logger.info(
            f"接收到前台交互点坐标: {point_coords}, 点标签: {point_labels}, 边界框: {input_box}"
        )

        img_url = tasks[0]["data"][value]
        # 运行推理
        predictor_results = self._sam_predict(
            img_url=img_url,
            point_coords=point_coords or None,
            point_labels=point_labels or None,
            input_box=input_box,
            task=tasks[0],
        )

        # 序列化为 Label Studio 可读取的标注格式
        predictions = self.get_results(
            masks=predictor_results["masks"],
            probs=predictor_results["probs"],
            width=image_width,
            height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=selected_label,
        )

        return ModelResponse(predictions=predictions)


if __name__ == "__main__":
    # 从环境变量获取端口，默认 9090
    port = int(os.getenv("PORT", 9090))
    logger.info(f"正在启动 SAM2 抠图交互预标注服务，端口 {port}...")
    run_app(SAM2ImageModel, host="0.0.0.0", port=port)
