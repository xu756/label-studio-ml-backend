# Label Studio ML Backend (基于 FastAPI)

这是一个经过精简重构的 Label Studio 机器学习后端（ML Backend）项目。原项目过于庞大且依赖 Docker，现在重构为了基于 **FastAPI + Uvicorn** 的轻量级架构。可以直接导入 `label_studio_ml` 作为标准 Python 包，并编写简单的入口脚本运行服务，不再需要复杂的 CLI 命令行工具。

---

## 依赖安装

项目使用 `uv` 管理依赖和 PyTorch CUDA 安装源。

1. 安装基础依赖包（FastAPI、Uvicorn、Label Studio SDK 等）：
   ```bash
   uv sync --locked
   ```

`pyproject.toml` 已固定 `torch==2.13.0+cu126` 和 `torchvision==0.28.0+cu126`，通过 `tool.uv.sources` 指定南京大学 CUDA 12.6 镜像；该镜像设置了 `explicit = true`，其余依赖使用清华 PyPI 镜像。安装时无需额外传入 `-f` 或 `--index-url`。索引配置方式见 [uv 官方文档](https://docs.astral.sh/uv/concepts/indexes/)。

转换功能使用 `label_studio_sdk.converter`，无需单独安装旧版 `label-studio-converter`，避免其固定的旧版 `requests` 与项目依赖冲突。

---

## 1. 启动 YOLOE 预测服务 (`main.py`)

```bash
uv sync
uv run main.py
```

默认使用 `yoloe-11s-seg.pt`，监听 `http://localhost:9090`。权重在第一次预测时加载；官方权重名由 Ultralytics 自动下载，也可以指定本地路径。首次使用文本提示还可能下载文本编码器资源，需要能访问下载源。服务启动或连接成功不代表权重已加载成功。

### 开发时直接改代码：YOLOE 类别 → Label Studio 标签

编辑 `model_config.py` 的 `DEFAULT_CONFIG`，然后重启服务。例如：

```python
"model_type": "yoloe",
"model_path": "yoloe-11s-seg.pt",
"label_map": {"potato": "土豆", "tomato": "番茄"},
"output_type": "rectangle",
"conf": 0.25,
```

映射左侧是 YOLOE 文本提示／类别名，右侧必须与 Label Studio 当前控件的 `<Label value="..."/>` 完全一致。例如识别到 `potato`，返回 `rectanglelabels: ["土豆"]`。你当前项目标签是英文 `potato` 时，直接用 `{"potato": "potato"}`。

对应的 Label Studio 配置：

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="土豆" background="blue"/>
    <Label value="番茄" background="red"/>
  </RectangleLabels>
</View>
```

`label_map=None` 时自动读取当前项目标签作为提示；也可以用 `<Label value="土豆" predicted_values="potato"/>` 在 XML 指定映射。文本提示接口参考 [Ultralytics YOLOE 官方文档](https://docs.ultralytics.com/models/yoloe/)。本服务实现文本提示预标注。

### 每个项目配置不同

同一服务连接多个项目时，在 `model_config.py` 的 `PROJECT_CONFIGS` 中用项目 ID 覆盖配置：

```python
PROJECT_CONFIGS = {
    "1": {"label_map": {"potato": "土豆"}, "output_type": "polygon"},
    "2": {"label_map": {"person": "行人"}, "conf": 0.4},
    "3": {"model_type": "yolo", "model_path": "weights/best.pt",
          "label_map": {"car": "汽车"}},
}
```

项目 ID 来自 Label Studio 项目 URL。未列出的项目使用默认配置。不同权重分别占用内存；相同权重在进程内复用，提示更新和推理串行执行，避免项目串类。修改代码或替换权重后重启服务。

- `output_type="rectangle"` 返回 `RectangleLabels`。
- `output_type="polygon"` 将分割轮廓转换为 `PolygonLabels`，需要分割权重及对应 XML 控件。当前示例 XML 同时有两种控件，此参数选择其中一种输出。
- 同类型有多个控件时，用 `control_name` 指定 XML 的 `name`。
- `device` 可设为 `"cpu"` 或 `"0"`，默认自动选择。
- 普通 YOLO 使用 `model_type="yolo"`。固定类别或无提示 YOLOE 权重使用 `text_prompts=False`，映射左侧必须使用权重自带类别。默认文本提示模式使用支持文本提示的 `.pt` 权重，不要使用 `*-seg-pf.pt`。

配置统一在 `model_config.py` 修改：`DEFAULT_CONFIG` 设置默认模型参数，`PROJECT_CONFIGS` 按项目覆盖，`SERVER_CONFIG` 设置监听地址和端口（默认 `0.0.0.0:9090`）。修改后重启入口脚本。

`.env` 只配置 Label Studio 连接信息：

```dotenv
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=你的API密钥
```

---

## 2. 启动 SAM2 (Segment Anything 2) 交互式语义分割服务 (`main_sam2.py`)

如果您要使用 Meta 的 Segment Anything 2 模型在 Label Studio 中进行点/框交互式快速抠图语义分割，请按照以下步骤配置。

### 准备工作
1. 确保安装了 SAM2 库与 PyTorch GPU 环境。模型参数在 `model_config.py` 的 `SAM2_CONFIG` 中修改，监听地址和端口使用 `SERVER_CONFIG`。
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
   - **Title**: 模型名称（如 `YOLOE-Predictor` 或 `SAM2-Backend`）
   - **URL**: 输入 `http://localhost:9090`（如果 Label Studio 和 ML 服务部署在不同机器，请输入对应 IP 端口）
   - **Interactive pre-annotations**: SAM2 点选交互时开启；YOLOE 使用普通预标注，不处理点击提示
5. 点击 **Validate and Save** 保存。

现在打开您的任务标注页面，即可享受极其轻量的实时 AI 自动预标注！
