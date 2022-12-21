import os

import pytest
import requests
from PIL import Image

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import from_coco_or_voc


def download_sample_images():
    url = "https://raw.githubusercontent.com/open-mmlab/mmdetection/master/demo/demo.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    mmdet_image_name = "demo.jpg"
    image.save(mmdet_image_name)

    return mmdet_image_name


def download_sample_dataset():
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
    download_dir = "./tiny_motorbike_coco"

    load_zip.unzip(zip_file, unzip_dir=download_dir)
    data_dir = os.path.join(download_dir, "tiny_motorbike")

    return data_dir

# @pytest.mark.parametrize(
#     "checkpoint_name",
#     [
#         "yolov3_mobilenetv2_320_300e_coco",
#     ],
# )
def test_mmdet_object_detection_fit_ddp_spawn(checkpoint_name):
    data_dir = download_sample_dataset()
    mmdet_image_name = download_sample_images()

    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    # Init predictor
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 4,
            "env.per_gpu_batch_size": 2,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
    )

    # Fit
    predictor.fit(
        train_path,
        hyperparameters={
            "env.num_workers": 2,
            "optimization.learning_rate": 2e-4,
            "env.strategy": "ddp"
        },
        time_limit=30,
    )

    predictor.fit(
        train_path,
        hyperparameters={
            "env.num_workers": 2,
            "optimization.learning_rate": 2e-4,
            "env.strategy": "ddp"
        },
        time_limit=30,
    )
    pred = predictor.predict({"image": [mmdet_image_name] * 10})  # test batch inference


if __name__ == "__main__":
    model_spec = "yolov3_mobilenetv2_320_300e_coco"
    test_mmdet_object_detection_fit_ddp_spawn(model_spec)