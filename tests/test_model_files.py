import json
from pathlib import Path


def test_model_weights_exist():
    assert Path("asl_resnet18.pt").exists()


def test_classes_json_exists():
    assert Path("asl_classes.json").exists()


def test_classes_json_is_valid():
    with open("asl_classes.json") as f:
        classes = json.load(f)

    assert isinstance(classes, list)
    assert len(classes) > 0
    assert all(isinstance(c, str) for c in classes)