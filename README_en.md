# LRS-DETR: Low-Altitude Remote Sensing Small Object Detection Based on Multi-Scale Semantic Fusion and Pruning-Distillation Two-Stage Compression

## 📋 Table of Contents

- [Project Introduction](#-project-introduction)
- [Environment Setup](#️-environment-setup)
- [Dataset Preparation](#-dataset-preparation)
- [Model Training](#-model-training)
- [Model Evaluation](#-model-evaluation)
- [Model Compression](#-model-compression)
- [Model Export](#-model-export)
- [Experimental Results](#-experimental-results)
- [Project Structure](#-project-structure)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

## 🚀 Project Introduction

LRS-DETR (Low-Altitude Remote Sensing DEtection TRansformer) is a real-time object detection model specifically designed for low-altitude remote sensing scenarios. This project addresses three major challenges in low-altitude remote sensing images: **significant target scale variations**, **difficult small object detection**, and **limited edge device deployment**, providing a complete solution.

<div align="center">
  <img src="assets/LRS-DETR 整体架构图.svg" width="800"/>
  <p>LRS-DETR Overall Architecture</p>
</div>

### 🎯 Key Features

- **High Accuracy**: LRS-DETR-B achieves 39.2% AP on VisDrone-DET dataset, improving 3.6 percentage points over baseline
- **Lightweight**: LRS-DETR-T has only 11.0M parameters and 48.3 GFLOPs computation, suitable for edge deployment
- **Small Object Friendly**: Specially optimized small object detection performance with significant AP_small improvement
- **Easy to Use**: Based on Ultralytics framework, providing complete training, evaluation, and inference interfaces

### 🔥 Main Innovations

- **Dynamic Feature Enhancement Network (DFENet)**: Multi-branch residual structure fused with self-attention mechanism to enhance feature representation
- **Scale-aware Enhanced Feature Pyramid (SEFP)**: Enhanced small object feature representation through Space-to-Depth Convolution (SPDConv) and Omni-Kernel Module (OKM)
- **Dual-Path Recalibration Attention (DPRA)**: Fusion of sparse and dense attention to improve target-background discrimination
- **Efficient Model Compression Scheme**: Combines Layer-wise Adaptive Magnitude Pruning (LAMP) with Multi-Level Supervised Self-Distillation (MLSSD) to reduce complexity while maintaining performance

## 🛠️ Environment Setup

### Basic Requirements
- Python >= 3.8
- PyTorch >= 2.2.2
- CUDA >= 11.7

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/yuchenwu73/LRS-DETR.git
cd LRS-DETR
```

2. Create virtual environment (recommended)
```bash
conda create -n lrs-detr python=3.9
conda activate lrs-detr
```

3. Install PyTorch
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

4. Install project dependencies
```bash
pip install -r requirements.txt
```
For complete dependency list, see [requirements.txt](requirements.txt)

5. Install Torch Pruning package
```bash
cd Torch-Pruning
pip uninstall torch_pruning
python setup.py build install
```

## 📊 Dataset Preparation

### VisDrone-DET Dataset
This project uses VisDrone-DET 2019 dataset for experimental validation, with data format in **YOLO format**.

> **Note**: To keep repository size reasonable, dataset files are not included in this repository. Please follow the steps below to download and prepare the dataset.

1. Download dataset
```bash
# Download VisDrone-DET dataset from official website
wget https://github.com/VisDrone/VisDrone-Dataset/raw/master/VisDrone2019-DET-train.zip
wget https://github.com/VisDrone/VisDrone-Dataset/raw/master/VisDrone2019-DET-val.zip
wget https://github.com/VisDrone/VisDrone-Dataset/raw/master/VisDrone2019-DET-test-dev.zip

# Extract datasets
unzip VisDrone2019-DET-train.zip
unzip VisDrone2019-DET-val.zip
unzip VisDrone2019-DET-test-dev.zip
```

2. Dataset structure
```
LRS-DETR/
├── dataset/
│   ├── images/
│   │   ├── train/      # 6471 training images
│   │   ├── val/        # 548 validation images
│   │   └── test/       # 3190 test images
│   ├── labels/
│   │   ├── train/      # YOLO format annotation files
│   │   ├── val/
│   │   └── test/
│   ├── data.yaml       # Dataset configuration file
│   ├── data.json       # COCO format annotations (for evaluation)
│   ├── split_data.py   # Dataset splitting script
│   ├── xml2txt.py      # XML to YOLO format conversion script
│   └── yolo2coco.py    # YOLO to COCO format conversion script
```

3. Dataset configuration file `data.yaml`
```yaml
# dataset path
path: /to/your/path # project root directory
train: ./dataset/images/train    # training set path (relative to path)
val: ./dataset/images/val        # validation set path
test: ./dataset/images/test      # test set path

# number of classes
nc: 10

# class names
names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
```

### Data Format Description
- **YOLO format**: Each image corresponds to a txt file, each line contains `class_id x_center y_center width height` (normalized coordinates)
- **COCO format conversion**: Conversion to COCO format needed for evaluation
```bash
python dataset/yolo2coco.py --image_path dataset/images/test --label_path dataset/labels/test
```

## 🎓 Model Training

### Training LRS-DETR-B (Base Version)

```python
import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# Specify GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create and train model
model = RTDETR('ultralytics/cfg/models/lrs-detr/lrs-detr-b.yaml')
model.train(
    data='dataset/data.yaml',
    epochs=300,
    imgsz=640,
    batch=8,
    workers=8,
    cache=False,
    project='runs/train',
    name='lrs-detr-b'
)
```

### Obtaining LRS-DETR-T (Lightweight Version)

LRS-DETR-T is obtained through two steps:
1. First train LRS-DETR-B base model
2. Obtain lightweight model through LAMP pruning + MLSSD distillation ([see Model Compression section](#-model-compression))

### Single GPU vs Multi-GPU Training

#### Single GPU Training

```python
# Specify GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model.train(
    data='dataset/data.yaml',
    device='0',        # Use GPU 0
    batch=8,          # Single card batch size
    epochs=300,
    project='runs/train',
    name='lrs-detr-b'
)
```

#### Multi-GPU Training

Command line launch:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node 4 \
    --master_port 29500 \
    train.py
```

Configure in train.py:
```python
model.train(
    data='dataset/data.yaml',
    device='0,1,2,3',     # Specify multiple cards
    batch=32,             # Total batch size (8 per card)
    epochs=300,
    project='runs/train',
    name='lrs-detr-b-multi'
)
```

### 💡 Notes

1. **Memory Optimization**: If encountering OOM, reduce `batch` or `imgsz`
2. **Multi-card Training**: Recommend using `setsid` to start, avoiding terminal disconnection issues
3. **Log Management**: Create `logs` directory to save training logs for troubleshooting
4. **Checkpoint Saving**: Training automatically saves `last.pt` and `best.pt`

## 📈 Model Evaluation

### Basic Evaluation
```python
from ultralytics import RTDETR

# Load trained model
model = RTDETR('runs/train/lrs-detr-b/weights/best.pt')

# Evaluate on test set
model.val(
    data='dataset/data.yaml',
    split='test',
    imgsz=640,
    batch=8,
    save_json=True,  # Save COCO format results
    project='runs/val',
    name='lrs-detr-b-test'
)
```

### Calculate COCO Metrics
```bash
# 1. First perform validation and save json results
python val.py

# 2. Convert data format (YOLO to COCO)
python dataset/yolo2coco.py --image_path dataset/images/test --label_path dataset/labels/test

# 3. Calculate COCO metrics and TIDE error analysis
python get_COCO_metrics.py --pred_json runs/val/exp/predictions.json --anno_json data_test.json
```

### Model Performance Analysis

#### View Model Information
```python
from ultralytics import RTDETR

model = RTDETR('ultralytics/cfg/models/lrs-detr/lrs-detr-b.yaml')
model.model.eval()
model.info(detailed=False)
model.profile(imgsz=[640, 640])

# Model information after fusion
print('after fuse:')
model.fuse()
model.info(detailed=False)
```

#### Calculate FPS and Latency
```bash
# Test inference speed
python get_FPS.py --weights runs/train/lrs-detr-b/weights/best.pt
```

#### Generate Heatmaps
```bash
# Visualize model attention regions
python heatmap.py --weights runs/train/lrs-detr-b/weights/best.pt --source dataset/images/test
```

## 🔧 Model Compression

### LAMP Pruning

```python
# Configure pruning parameters
param_dict = {
    # Basic parameters
    'model': 'runs/train/lrs-detr-b/weights/best.pt',
    'data': './dataset/data.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 8,
    'workers': 8,
    'cache': False,
    'device': '1',
    'project': 'runs/prune',
    'name': 'lamp_speedup1.35',

    # Pruning parameters
    'prune_method': 'lamp',  # Options: 'random', 'l1', 'lamp', 'slim', 'group_slim', 'group_norm', 'group_sl', 'growing_reg', 'group_hessian', 'group_taylor'
    'global_pruning': True,  # Whether to use global pruning
    'speed_up': 1.35,  # Pruning speedup ratio (pre-pruning FLOPs / post-pruning FLOPs)
    'reg': 0.0005,  # Regularization coefficient
    'sl_epochs': 500,  # Sparse learning iterations
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',  # Sparse learning hyperparameters
    'iterative_steps': 50  # Iterative pruning steps
}

# Execute pruning and fine-tuning
python pruning.py
```

### MLSSD Distillation

```python
# Configure distillation parameters
param_dict = {
    # Basic parameters
    'model': 'runs/prune/lamp_speedup1.35/weights/best.pt',  # Pruned model
    'data': 'dataset/data.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 8,
    'workers': 8,
    'device': '6',
    'project': 'runs/distill',
    'name': 'lamp1.35_logical_mlssd_0.0001',

    # Distillation parameters
    'prune_model': True,
    'teacher_weights': 'runs/train/lrs-detr-b/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/lrs-detr/lrs-detr-b.yaml',
    'kd_loss_type': 'feature',  # Options: 'logical', 'feature', 'all'
    'kd_loss_decay': 'constant',  # Options: 'cosine', 'linear', 'cosine_epoch', 'linear_epoch', 'constant'
    'kd_loss_epoch': 1.0,

    # Logical distillation parameters
    'logical_loss_type': 'mlssd',  # Use MLSSD, option: logical
    'logical_loss_ratio': 0.0001,

    # Feature distillation parameters
    'teacher_kd_layers': '21, 24, 27',
    'student_kd_layers': '21, 24, 27',
    'feature_loss_type': 'cwd',  # Options: mimic, mgd, cwd, chsim, sp
    'feature_loss_ratio': 0.05
}

# Execute distillation
python distill.py
```

### Pruning Effect Visualization
```bash
# Compare channel number changes before and after pruning
python plot_channel_image.py --base-weights base_weights.pt --prune-weights prune_weights.pt
```

## 📦 Model Export

```python
from ultralytics import RTDETR

# Load model
model = RTDETR('runs/train/lrs-detr-b/weights/best.pt')

# Export to ONNX format
model.export(format='onnx', simplify=True)

# Export to TensorRT format (requires TensorRT installation)
model.export(format='engine', half=True)

# Export to CoreML format (for iOS deployment)
model.export(format='coreml')
```

## 📊 Experimental Results

### Performance Comparison of Mainstream Object Detection Algorithms on VisDrone-DET Test Set

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Backbone</th>
      <th>Input Size</th>
      <th>#Params</th>
      <th>GFLOPs</th>
      <th><b>AP</b></th>
      <th><b>AP</b><sub>50</sub></th>
      <th><b>AP</b><sub>75</sub></th>
      <th><b>AP</b><sub>S</sub></th>
      <th><b>AP</b><sub>M</sub></th>
      <th><b>AP</b><sub>L</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="11" align="center"><b>One-stage Detector</b></td>
    </tr>
    <tr>
      <td>TOOD</td>
      <td>ResNet 50</td>
      <td>768×1344</td>
      <td>32.0M</td>
      <td>199.0</td>
      <td>20.5</td>
      <td>34.0</td>
      <td>21.7</td>
      <td>10.3</td>
      <td>31.7</td>
      <td>43.7</td>
    </tr>
    <tr>
      <td>YOLOv8-m</td>
      <td>CSPDarknet53</td>
      <td>640×640</td>
      <td>25.8M</td>
      <td>78.7</td>
      <td>19.1</td>
      <td>33.3</td>
      <td>19.4</td>
      <td>9.1</td>
      <td>29.6</td>
      <td>39.6</td>
    </tr>
    <tr>
      <td>YOLOv10-m</td>
      <td>CSPDarknet53</td>
      <td>640×640</td>
      <td>15.3M</td>
      <td>58.9</td>
      <td>19.3</td>
      <td>34.1</td>
      <td>19.4</td>
      <td>9.4</td>
      <td>29.6</td>
      <td>43.3</td>
    </tr>
    <tr>
      <td>YOLOv11-m</td>
      <td>CSPDarknet53</td>
      <td>640×640</td>
      <td>20.4M</td>
      <td>67.7</td>
      <td>19.8</td>
      <td>34.6</td>
      <td>20.2</td>
      <td>9.8</td>
      <td>30.2</td>
      <td>40.5</td>
    </tr>
    <tr>
      <td>YOLOv12-m</td>
      <td>CSPDarknet53</td>
      <td>640×640</td>
      <td>19.6M</td>
      <td>59.5</td>
      <td>19.5</td>
      <td>33.8</td>
      <td>19.8</td>
      <td>9.5</td>
      <td>30.2</td>
      <td>39.6</td>
    </tr>
    <tr>
      <td colspan="11" align="center"><b>Two-stage Detector</b></td>
    </tr>
    <tr>
      <td>Faster R-CNN</td>
      <td>ResNet 50</td>
      <td>768×1344</td>
      <td>41.4M</td>
      <td>208.0</td>
      <td>19.3</td>
      <td>32.9</td>
      <td>20.2</td>
      <td>9.5</td>
      <td>30.7</td>
      <td>43.6</td>
    </tr>
    <tr>
      <td>Cascade R-CNN</td>
      <td>ResNet 50</td>
      <td>768×1344</td>
      <td>69.3M</td>
      <td>236.0</td>
      <td>19.7</td>
      <td>32.7</td>
      <td>21.0</td>
      <td>9.9</td>
      <td>31.0</td>
      <td>40.6</td>
    </tr>
    <tr>
      <td>RetinaNet</td>
      <td>ResNet 50</td>
      <td>768×1344</td>
      <td>36.5M</td>
      <td>210.0</td>
      <td>16.5</td>
      <td>27.8</td>
      <td>17.2</td>
      <td>6.0</td>
      <td>27.5</td>
      <td>43.6</td>
    </tr>
    <tr>
      <td colspan="11" align="center"><b>Transformer-based Detector</b></td>
    </tr>
    <tr>
      <td>RT-DETR</td>
      <td>ResNet 18</td>
      <td>640×640</td>
      <td>19.9M</td>
      <td>57.0</td>
      <td>20.3</td>
      <td>35.6</td>
      <td>20.2</td>
      <td>11.2</td>
      <td>29.7</td>
      <td>35.9</td>
    </tr>
    <tr>
      <td>RT-DETR</td>
      <td>ResNet 50</td>
      <td>640×640</td>
      <td>41.8M</td>
      <td>129.6</td>
      <td>21.2</td>
      <td>37.1</td>
      <td>21.1</td>
      <td>11.9</td>
      <td>30.8</td>
      <td>43.3</td>
    </tr>
    <tr>
      <td>RT-DETR</td>
      <td>ResNet 101</td>
      <td>640×640</td>
      <td>41.8M</td>
      <td>129.6</td>
      <td>21.2</td>
      <td>37.1</td>
      <td>21.1</td>
      <td>11.9</td>
      <td>30.8</td>
      <td>43.3</td>
    </tr>
    <tr>
      <td>RT-DETR</td>
      <td>HGNetv2</td>
      <td>640×640</td>
      <td>65.5M</td>
      <td>222.5</td>
      <td>21.7</td>
      <td>37.0</td>
      <td>21.9</td>
      <td>12.3</td>
      <td>31.4</td>
      <td>41.2</td>
    </tr>
    <tr>
      <td><b>LRS-DETR-B</b></td>
      <td>DFENet</td>
      <td>640×640</td>
      <td>15.6M</td>
      <td>66.6</td>
      <td><b>22.7</b></td>
      <td><b>39.2</b></td>
      <td><b>22.9</b></td>
      <td><b>13.2</b></td>
      <td><b>32.5</b></td>
      <td><b>44.8</b></td>
    </tr>
    <tr>
      <td><b>LRS-DETR-T</b></td>
      <td>DFENet</td>
      <td>640×640</td>
      <td><b>11.0M</b></td>
      <td><b>48.3</b></td>
      <td>22.1</td>
      <td>38.6</td>
      <td>22.1</td>
      <td>12.9</td>
      <td>31.8</td>
      <td>40.1</td>
    </tr>
  </tbody>
</table>

## 📁 Project Structure

```
LRS-DETR/
├── assets/                         # Asset files
├── dataset/                        # Dataset directory (images/labels/data processing scripts)
├── ultralytics/                    # Model core code
│   ├── cfg/                        # Configuration files
│   ├── nn/                         # Neural network modules
│   │   ├── extra_modules/         # Innovation module implementations
│   ├── models/                     # Model implementations
│   │   └── rtdetr/                # RT-DETR related
│   ├── data/                       # Data processing
│   └── utils/                      # Utility functions
├── train.py                        # Training entry point
├── val.py                          # Validation script
├── detect.py                       # Detection script
├── distill.py                      # Distillation script
├── pruning.py                      # Pruning tool
├── get_FPS.py                      # Performance testing
├── heatmap.py                      # Heatmap generation
├── get_COCO_metrics.py             # COCO metrics calculation
├── plot_channel_image.py           # Pruning effect visualization
└── requirements.txt                # Dependency list
```

## 🙏 Acknowledgments

This project is based on the following open-source projects:
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO series implementation framework
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR) - Real-time DETR detector
- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) - Drone vision dataset
- [TIDE](https://github.com/dbolya/tide) - Object detection error analysis tool
- [Torch-Pruning](https://github.com/VainF/Torch-Pruning) - PyTorch pruning tool

## 📧 Contact

- Author: Yuchen Wu
- Email: ycwu73@gmail.com
- Institution: School of Computer Science and Engineering, Anhui University of Science and Technology

---

**Note**:
1. This project is for academic research use only. For commercial use, please contact the author for authorization
