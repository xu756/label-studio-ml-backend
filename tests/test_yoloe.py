"""Offline contract tests: no model weights or GPU required."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import main


def config(label="Potato", tag="RectangleLabels"):
    return f'''<View><Image name="photo" value="$photo"/>
    <{tag} name="objects" toName="photo">
      <Label value="{label}" predicted_values="potato"/>
    </{tag}></View>'''


class FakeModel:
    def __init__(self, path):
        self.names = {0: "potato"}
        self.prompts = []

    def get_text_pe(self, names):
        return names

    def set_classes(self, names, embeddings):
        self.prompts.append(names)
        self.names = dict(enumerate(names))

    def predict(self, source, **kwargs):
        box = SimpleNamespace(xyxy=np.array([[10, 5, 50, 25]]),
                              conf=[0.8], cls=[0])
        return [SimpleNamespace(boxes=[box], names=dict(self.names),
                                masks=SimpleNamespace(xyn=[np.array([[.1,.1],[.5,.1],[.5,.5]])]))]


class YoloETests(unittest.TestCase):
    def setUp(self):
        main._MODEL_CACHE.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "image.png")
        Image.new("RGB", (100, 50)).save(self.path)
        self.addCleanup(self.tmp.cleanup)
        self.factory = patch.object(main, "YOLOE", side_effect=FakeModel).start()
        self.addCleanup(patch.stopall)
        patch.object(main, "get_model_config", return_value={
            "model_type": "yoloe", "model_path": "test.pt", "output_type": "rectangle",
            "control_name": None, "conf": .25, "iou": .7, "imgsz": 640,
            "device": None, "text_prompts": True,
        }).start()
        patch.object(main.YoloModel, "get_local_path", return_value=self.path).start()

    def test_rectangles_and_task_alignment(self):
        model = main.YoloModel(project_id="test-yoloe", label_config=config())
        results = model.predict([{"data": {}}, {"data": {"photo": "image"}}]).model_dump()["predictions"]
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["result"], [])
        result = results[1]["result"][0]
        self.assertEqual(result["value"]["rectanglelabels"], ["Potato"])
        self.assertEqual(result["value"]["width"], 40)
        self.assertEqual(result["original_width"], 100)

    def test_project_labels_change_without_reloading_weights(self):
        first = main.YoloModel(project_id="test-a", label_config=config())
        second = main.YoloModel(project_id="test-b", label_config=config("土豆"))
        second.config = dict(second.config, label_map={"a potato": "土豆"})
        for model, expected in [(first, "Potato"), (second, "土豆"), (first, "Potato")]:
            result = model.predict([{"data": {"photo": "image"}}]).model_dump()
            self.assertEqual(result["predictions"][0]["result"][0]["value"]["rectanglelabels"], [expected])
        self.assertEqual(self.factory.call_count, 1)
        cached_model = next(iter(main._MODEL_CACHE.values()))[0]
        self.assertEqual(cached_model.prompts, [["potato"], ["a potato"], ["potato"]])

    def test_polygons(self):
        model = main.YoloModel(project_id="test-poly", label_config=config(tag="PolygonLabels"))
        model.config["output_type"] = "polygon"
        result = model.predict([{"data": {"photo": "image"}}]).model_dump()
        value = result["predictions"][0]["result"][0]["value"]
        self.assertEqual(value["polygonlabels"], ["Potato"])
        self.assertEqual(value["points"], [[10, 10], [50, 10], [50, 50]])

    def test_invalid_control_fails(self):
        model = main.YoloModel(project_id="test-invalid", label_config=config())
        model.config["control_name"] = "missing"
        with self.assertRaises(ValueError):
            model.predict([{"data": {"photo": "image"}}])

    def test_code_mapping_sets_prompts_and_returns_label_studio_label(self):
        model = main.YoloModel(project_id="test-map", label_config=config("土豆"))
        model.config["label_map"] = {"a potato": "土豆"}
        result = model.predict([{"data": {"photo": "image"}}]).model_dump()
        self.assertEqual(result["predictions"][0]["result"][0]["value"]["rectanglelabels"], ["土豆"])
        cached_model = next(iter(main._MODEL_CACHE.values()))[0]
        self.assertEqual(cached_model.prompts, [["a potato"]])

    def test_mapping_target_must_exist(self):
        model = main.YoloModel(project_id="test-map-invalid", label_config=config())
        model.config["label_map"] = {"potato": "不存在"}
        with self.assertRaises(ValueError):
            model.predict([{"data": {"photo": "image"}}])

    def test_predict_api_response(self):
        from label_studio_ml.api import init_app
        client = init_app(main.YoloModel).test_client()
        response = client.post("/predict", json={
            "project": "test-api", "label_config": config(),
            "tasks": [{"id": 1, "data": {"photo": "image"}}],
        })
        self.assertEqual(response.status_code, 200)
        region = response.json()["results"][0]["result"][0]
        self.assertEqual(region["value"]["rectanglelabels"], ["Potato"])


if __name__ == "__main__":
    unittest.main()
