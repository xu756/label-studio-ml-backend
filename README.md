# Label Studio ML Backend (基于 FastAPI)

这是一个经过精简重构的 Label Studio 机器学习后端（ML Backend）项目。原项目过于庞大且依赖 Docker，现在重构为了基于 **FastAPI + Uvicorn** 的轻量级架构。可以直接导入 `label_studio_ml` 作为标准 Python 包，并编写简单的入口脚本运行服务，不再需要复杂的 CLI 命令行工具。

---

## 依赖安装

项目使用 `uv` 管理依赖，也可以直接使用 `pip` 安装。

1. 安装基础依赖包（FastAPI、Uvicorn、Label Studio SDK 等）：
   ```bash
   uv sync
   # 或者使用 pip
   pip install -r requirements.txt
   ```

---

## 1. 启动 YOLOv8 目标检测预标注服务 (`main.py`)

如果您要使用自己训练的 YOLOv8 权重文件（例如 `best.pt`）进行图像检测预标注，请按照以下步骤配置。

### 准备工作
1. 安装 `ultralytics` 目标检测包：
   ```bash
   pip install ultralytics
   ```
2. 将您训练好的 `best.pt` 权重文件放置在项目根目录下。

### Label Studio 项目配置
在 Label Studio 项目设置的 **Labeling Interface**（标签页面配置）中，您的 XML 配置应类似于：
```xml
<View>
  <Image name="image" value="$image"/>
  <!-- 控制标签中定义的 labels 的值（value）需要和您的 YOLO 类别（如 car, person）一致，支持大小写自适应匹配 -->
  <RectangleLabels name="label" toName="image">
    <Label value="Car" background="blue"/>
    <Label value="Person" background="red"/>
  </RectangleLabels>
</View>
```

### 启动服务
运行服务：
```bash
uv run main.py
# 或者使用 Python 直接运行
python main.py
```
服务默认会在本机 `9090` 端口启动：`http://localhost:9090`。

---

## 2. 启动 SAM2 (Segment Anything 2) 交互式语义分割服务 (`main_sam2.py`)

如果您要使用 Meta 的 Segment Anything 2 模型在 Label Studio 中进行点/框交互式快速抠图语义分割，请按照以下步骤配置。

### 准备工作
1. 确保安装了 SAM2 库与 PyTorch GPU 环境。
2. 按照 SAM2 规范，在根目录下放置权重和配置文件：
   - 配置文件路径默认为：`configs/sam2.1/sam2.1_hiera_l.yaml`
   - 权重文件路径默认为：`checkpoints/sam2.1_hiera_large.pt`

### Label Studio 项目配置
在 Label Studio 中进行交互式语义分割，您的 XML 配置应当配置为 **BrushLabels** 以及对应的关键点交互：
```xml
<View>
  <Image name="image" value="$image"/>
  <!-- 交互式分割需要的标签，类型为 brushlabels 和 keypointlabels -->
  <BrushLabels name="tag" toName="image">
    <Label value="Object" background="green"/>
  </BrushLabels>
  <KeyPointLabels name="keypoint" toName="image" smart="true">
    <Label value="Object" background="green"/>
  </KeyPointLabels>
</View>
```

### 启动服务
运行服务：
```bash
uv run main_sam2.py
# 或者使用 Python 直接运行
python main_sam2.py
```
服务默认会在本机 `9090` 端口启动：`http://localhost:9090`。

---

## 如何在 Label Studio 中连接此机器学习后端？

1. 启动您选择的 Python 服务（如 `uv run main.py`，确保后台服务输出 `Uvicorn running on http://0.0.0.0:9090`）。
2. 打开 Label Studio 项目，进入 **Settings**（设置） -> **Machine Learning**（机器学习）。
3. 点击 **Add Model**（添加模型）按钮。
4. 输入模型参数：
   - **Title**: 模型名称（如 `YOLOv8-Predictor` 或 `SAM2-Backend`）
   - **URL**: 输入 `http://localhost:9090`（如果 Label Studio 和 ML 服务部署在不同机器，请输入对应 IP 端口）
   - **Interactive pre-annotations**: 开启（开启后点击图片或放置关键点时会自动触发预测）
5. 点击 **Validate and Save** 保存。

现在打开您的任务标注页面，即可享受极其轻量的实时 AI 自动预标注！
