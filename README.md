# LGFD: Language Guided Feature Disentanglement for Infrared Object Detection

## 1. Environment

The training platform is Linux with a single NVIDIA GeForce RTX 3090. The dependency file specifies PyTorch with CUDA 12.1 support; use a compatible NVIDIA driver.

Run the following commands from the project directory containing `train.py`:

```bash
conda create -n LGFD python=3.12 -y
conda activate LGFD
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## 2. Datasets

The infrared images used in this work are obtained from the following public
datasets:

- **M3FD:** [Download](https://github.com/JinyuanLiu-CV/TarDAL)
- **FLIR-aligned:** [Download](https://github.com/DocF/multispectral-object-detection)

The rule-based captions generated for our experiments are provided in this project:

| Dataset | Training captions | Validation captions |
|---|---|---|
| M3FD | [captions/M3FD/train.json](captions/M3FD/train.json) | [captions/M3FD/val.json](captions/M3FD/val.json) |
| FLIR-aligned | [captions/FLIR-aligned/train.json](captions/FLIR-aligned/train.json) | [captions/FLIR-aligned/val.json](captions/FLIR-aligned/val.json) |

Caption image names must match the image files in the corresponding dataset.

## 3. Model configuration

### Text encoder

The model uses `BertTokenizer` and `BertModel` in `clip/nets/clip.py`.
Download the text encoder into this project's `model_data` directory:

```python
from transformers import BertModel, BertTokenizer

path = "model_data/bert-base-chinese"
BertTokenizer.from_pretrained("google-bert/bert-base-chinese").save_pretrained(path)
BertModel.from_pretrained("google-bert/bert-base-chinese").save_pretrained(path)
```

Set `BERT_PATH` in `clip/nets/clip.py` to this project-local model directory:

```python
BERT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "model_data", "bert-base-chinese"
))
```

Keep `clip/model_data/bpe_simple_vocab_16e6.txt.gz` in place.

### Detector

Place the compatible YOLOv7 initialization checkpoint at `model_data/yolov7_weights.pth`. Set `model_data/voc_classes.txt` to the class names in the order used by the annotations and checkpoint, one name per line.

## 4. Training

Configure `voc_annotation.py`:

```python
annotation_mode = 2
classes_path = 'model_data/voc_classes.txt'
VOCdevkit_path = '/path/to/dataset/VOCdevkit'
```

Replace the dataset path and generate the training annotations:

```bash
python voc_annotation.py
```

This generates `2007_train.txt` and `2007_val.txt` using the existing split files.

For M3FD, configure `train.py` as follows:

```python
classes_path = 'model_data/voc_classes.txt'
model_path = 'model_data/yolov7_weights.pth'
train_annotation_path = '2007_train.txt'
val_annotation_path = '2007_val.txt'
train_json_path = 'captions/M3FD/train.json'
val_json_path = 'captions/M3FD/val.json'
```

For FLIR-aligned, use its class file and dataset paths, and set the caption paths to `captions/FLIR-aligned/train.json` and `captions/FLIR-aligned/val.json`.

Training settings are configured in `train.py`. Start training with:

```bash
python train.py
```

New training runs save checkpoints and logs to the `save_dir` configured in `train.py`, which defaults to `logs/`.

## 5. Evaluation

You can evaluate your own trained checkpoint or use our M3FD results in [`loss_2025_01_31_19_42_19_83.76/`](loss_2025_01_31_19_42_19_83.76/),
which contains our trained weights and training records. Set `model_path` in `YOLO._defaults` in `yolo.py` to the desired checkpoint. For our provided weights:

```python
"model_path": "loss_2025_01_31_19_42_19_83.76/ep240-loss0.186-val_loss0.284.pth",
"classes_path": "model_data/voc_classes.txt",
"phi": "l",
"input_shape": [640, 640],
"cuda": True,
```

Configure `get_map.py`:

```python
map_mode = 0
classes_path = 'model_data/voc_classes.txt'
VOCdevkit_path = '/path/to/dataset/VOCdevkit'
map_out_path = 'map_out_lgfd'
confidence = 0.001
nms_iou = 0.5
```

Run to generate predictions and ground truth for `VOC2007/ImageSets/Main/test.txt`:

```bash
python get_map.py
```

Then set `map_mode=4` in `get_map.py` and run the same command again to obtain COCO-style mAP, AP50, and AP75. Keep the same output directory for both steps.

If pycocotools raises `KeyError: 'info'`, add `results['info'] = {}` immediately after `results = {}` in `utils/utils_map.py::preprocess_gt`.
