import logging
import os

from dotenv import load_dotenv
from PIL import Image

# 加载当前目录下的 .env 环境变量文件
load_dotenv()

from label_studio_ml.api import run_app
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("未找到 ultralytics 包，请使用以下命令安装: pip install ultralytics")
    YOLO = None


class YoloModel(LabelStudioMLBase):
    """
    用于 Label Studio 预标注的 YOLO 模型封装类。
    加载本地的 YOLO 模型权重文件（例如 best.pt）并预测边界框（RectangleLabels）。
    """

    def setup(self):
        """配置模型版本并加载 YOLO 权重文件。"""
        self.set("model_version", "yolo-best-1.0.0")

        # 从环境变量 MODEL_PATH 获取权重路径，默认为当前目录下的 best.pt
        model_path = os.getenv("MODEL_PATH", "best.pt")

        if YOLO is None:
            raise ImportError("请先安装 ultralytics 依赖包: pip install ultralytics")

        if not os.path.exists(model_path):
            logger.warning(
                f"模型文件 '{model_path}' 未找到！请将您训练好的 best.pt 放置在此目录下。"
            )
            self.model = None
        else:
            logger.info(f"正在从本地加载 YOLO 模型权重: {model_path}...")
            self.model = YOLO(model_path)

    def predict(
        self, tasks: list[dict], context: dict | None = None, **kwargs
    ) -> ModelResponse:
        """
        为输入的图像任务预测边界框。
        """
        if not self.model:
            logger.error("YOLO 模型未加载，跳过预测。")
            return ModelResponse(predictions=[])

        # 获取 Label Studio 标注配置中的 RectangleLabels（控制标签）和 Image（数据标签）
        try:
            from_name, to_name, value = self.label_interface.get_first_tag_occurence(
                "RectangleLabels", "Image"
            )
        except ValueError:
            logger.error(
                "Label Studio 项目的 Labeling Config 必须包含 <RectangleLabels> 和 <Image> 标签！"
            )
            return ModelResponse(predictions=[])

        # 获取 Label Studio 配置中定义的所有类别标签
        labels = self.label_interface.get_control(from_name).labels
        label_map = {l.lower(): l for l in labels}  # 用于不区分大小写的标签匹配

        predictions = []
        for task in tasks:
            image_url = task["data"].get(value)
            if not image_url:
                continue

            # 获取图像在本地的真实存储路径（Label Studio 传过来的是 URL，此函数会自动处理缓存和下载）
            local_image_path = self.get_local_path(image_url, task_id=task.get("id"))

            # 读取图像获取原始分辨率的宽和高
            try:
                img = Image.open(local_image_path)
                img_width, img_height = img.size
            except Exception as e:
                logger.error(f"无法打开图像文件 {local_image_path}: {e}")
                continue

            # 运行 YOLO 推理
            results = self.model(local_image_path)
            result_list = []

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # 获取绝对像素坐标 [xmin, ymin, xmax, ymax]
                        xyxy = box.xyxy[0].tolist()
                        xmin, ymin, xmax, ymax = xyxy

                        # 转换成 Label Studio 要求的百分比坐标 (0-100)
                        x = (xmin / img_width) * 100
                        y = (ymin / img_height) * 100
                        width = ((xmax - xmin) / img_width) * 100
                        height = ((ymax - ymin) / img_height) * 100

                        # 获取置信度分数
                        score = float(box.conf[0])

                        # 获取类别索引和类别名称
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]

                        # 将 YOLO 类别名映射到 Label Studio 配置的标签（支持不区分大小写匹配）
                        label = label_map.get(class_name.lower())
                        if not label:
                            logger.warning(
                                f"YOLO 预测类别 '{class_name}' 不在 Label Studio 的标签配置中，跳过该框。"
                            )
                            continue

                        # 构造 Label Studio 格式的标注框结果
                        result_list.append(
                            {
                                "from_name": from_name,
                                "to_name": to_name,
                                "type": "rectanglelabels",
                                "value": {
                                    "rectanglelabels": [label],
                                    "x": x,
                                    "y": y,
                                    "width": width,
                                    "height": height,
                                },
                                "score": score,
                            }
                        )

            # 计算平均置信度分数
            avg_score = sum(r["score"] for r in result_list) / max(len(result_list), 1)

            predictions.append(
                {
                    "result": result_list,
                    "score": avg_score,
                    "model_version": self.get("model_version"),
                }
            )

        return ModelResponse(
            predictions=predictions, model_version=self.get("model_version")
        )


if __name__ == "__main__":
    # 从环境变量获取端口，默认 9090
    port = int(os.getenv("PORT", 9090))
    logger.info(f"正在启动 YOLO 目标检测 (best.pt) ML 服务，端口 {port}...")
    run_app(YoloModel, host="0.0.0.0", port=port)
