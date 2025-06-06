# LRS-DETR: 基于多尺度语义融合与剪枝-蒸馏双阶段压缩的低空遥感小目标检测

## 📋 目录

- [项目简介](#-项目简介)
- [环境配置](#️-环境配置)
- [数据集准备](#-数据集准备)
- [模型训练](#-模型训练)
- [模型评估](#-模型评估)
- [模型压缩](#-模型压缩)
- [模型导出](#-模型导出)
- [实验结果](#-实验结果)
- [项目结构](#-项目结构)
- [致谢](#-致谢)
- [联系方式](#-联系方式)

## 🚀 项目简介

LRS-DETR（Low-Altitude Remote Sensing DEtection TRansformer）是一个专门针对低空遥感场景设计的实时目标检测模型。该项目针对低空遥感图像中的三大挑战：**目标尺度变化显著**、**小目标检测困难**、**边缘设备部署受限**，提出了一套完整的解决方案。

<div align="center">
  <img src="assets/LRS-DETR 整体架构图.svg" width="800"/>
  <p>LRS-DETR 整体架构图</p>
</div>

### 🎯 主要特点

- **高精度**：在VisDrone-DET数据集上，LRS-DETR-B达到39.2% AP，相比基线提升3.6个百分点
- **轻量化**：LRS-DETR-T参数量仅11.0M，计算量48.3 GFLOPs，适合边缘部署
- **小目标友好**：专门优化的小目标检测性能，$AP_{small}$提升显著
- **易于使用**：基于Ultralytics框架，提供完整的训练、评估、推理接口

### 🔥 主要创新

- **动态特征增强网络（DFENet）**：多分支残差结构与自注意力机制融合，提升特征表达能力
- **尺度感知增强型特征金字塔（SEFP）**：通过空间到深度卷积（SPDConv）和全内核模块（OKM）增强小目标特征表达
- **双路重校准注意力机制（DPRA）**：融合稀疏与密集注意力，提高目标与背景区分能力
- **高效模型压缩方案**：结合层自适应幅度剪枝（LAMP）与多层级监督自蒸馏（MLSSD），保持性能的同时降低复杂度


## 🛠️ 环境配置

### 基础环境要求
- Python >= 3.9
- PyTorch >= 2.2.2
- CUDA >= 12.1

### 安装步骤

1. 克隆项目仓库
```bash
git clone https://github.com/yuchenwu73/LRS-DETR.git
cd LRS-DETR
```

2. 创建虚拟环境（推荐）
```bash
conda create -n lrs-detr python=3.9
conda activate lrs-detr
```

3. 安装PyTorch
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

4. 安装项目依赖
```bash
pip install -r requirements.txt
```
完整依赖列表请查看 [requirements.txt](requirements.txt)

5. 安装Torch Pruning包
```bash
cd Torch-Pruning
pip uninstall torch_pruning
python setup.py build install
```

## 📊 数据集准备

### VisDrone-DET数据集
本项目使用VisDrone-DET 2019数据集进行实验验证，数据格式为**YOLO格式**。

> **注意**：为了保持仓库大小合理，数据集文件未包含在此仓库中。请按照以下步骤下载并准备数据集。

1. 下载数据集
```bash
# 1. 从官网下载VisDrone-DET数据集
wget https://github.com/VisDrone/VisDrone-Dataset/raw/master/VisDrone2019-DET-train.zip
wget https://github.com/VisDrone/VisDrone-Dataset/raw/master/VisDrone2019-DET-val.zip
wget https://github.com/VisDrone/VisDrone-Dataset/raw/master/VisDrone2019-DET-test-dev.zip

# 2. 解压数据集
unzip VisDrone2019-DET-train.zip
unzip VisDrone2019-DET-val.zip
unzip VisDrone2019-DET-test-dev.zip

# 3. 处理成YOLO格式（网络搜索，解决方案很多）
```

2. 数据集结构
```
LRS-DETR/
├── dataset/
│   ├── images/
│   │   ├── train/      # 6471张训练图像
│   │   ├── val/        # 548张验证图像
│   │   └── test/       # 3190张测试图像
│   ├── labels/
│   │   ├── train/      # YOLO格式标注文件
│   │   ├── val/
│   │   └── test/
│   ├── data.yaml       # 数据集配置文件
│   ├── data.json       # COCO格式标注（用于评估）
│   ├── split_data.py   # 数据集划分脚本
│   ├── xml2txt.py      # XML转YOLO格式脚本
│   └── yolo2coco.py    # YOLO转COCO格式脚本
```

3. 数据集配置文件 `data.yaml`
```yaml
# dataset path
path: /to/your/path # 项目根目录
train: ./dataset/images/train    # 训练集路径（相对于path）
val: ./dataset/images/val        # 验证集路径
test: ./dataset/images/test      # 测试集路径

# number of classes
nc: 10

# class names
names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
```

### 数据格式说明
- **YOLO格式**：每个图像对应一个txt文件，每行包含`class_id x_center y_center width height`（归一化坐标）
- **COCO格式转换**：评估时需要转换为COCO格式
```bash
python dataset/yolo2coco.py --image_path dataset/images/test --label_path dataset/labels/test
```


## 🎓 模型训练

### 训练 LRS-DETR-B（基础版本）（train.py）

```python
import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 创建并训练模型
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

### 获得 LRS-DETR-T（轻量化版本）（train.py）

LRS-DETR-T 通过两步获得：
1. 先训练 LRS-DETR-B 基础模型
2. 通过 LAMP剪枝 + MLSSD蒸馏 获得轻量化模型（[详见模型压缩部分](#-模型压缩)）

### 单GPU vs 多GPU训练

#### 单GPU训练

```python
# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model.train(
    data='dataset/data.yaml',
    device='0',        # 使用GPU 0
    batch=8,          # 单卡batch size
    epochs=300,
    project='runs/train',
    name='lrs-detr-b'
)
```

#### 多GPU训练

命令行启动：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node 4 \
    --master_port 29500 \
    train.py
```

在 train.py 中配置：
```python
model.train(
    data='dataset/data.yaml',
    device='0,1,2,3',     # 指定多卡
    batch=32,             # 总batch size（每卡8）
    epochs=300,
    project='runs/train',
    name='lrs-detr-b-multi'
)
```


### 💡 注意事项

1. **显存优化**：如遇到OOM，可减小 `batch` 或 `imgsz`
2. **多卡训练**：建议使用 `setsid` 启动，避免终端断开影响
3. **日志管理**：创建 `logs` 目录保存训练日志便于排查
4. **断点保存**：训练会自动保存 `last.pt` 和 `best.pt`


## 📈 模型评估

### 基础评估（val.py）
```python
from ultralytics import RTDETR

# 加载训练好的模型
model = RTDETR('runs/train/lrs-detr-b/weights/best.pt')

# 在测试集上评估
model.val(
    data='dataset/data.yaml',
    split='test',
    imgsz=640,
    batch=8,
    save_json=True,  # 保存COCO格式结果
    project='runs/val',
    name='lrs-detr-b-test'
)
```

### 计算COCO指标（get_COCO_metrics.py）
```bash
# 1. 首先进行验证并保存json结果
python val.py 

# 2. 转换数据格式（YOLO转COCO）
python dataset/yolo2coco.py --image_path dataset/images/test --label_path dataset/labels/test

# 3. 计算COCO指标和TIDE错误分析
python get_COCO_metrics.py --pred_json runs/val/exp/predictions.json --anno_json data_test.json
```

### 模型性能分析

#### 查看模型信息（main_profile.py）
```python
from ultralytics import RTDETR

model = RTDETR('ultralytics/cfg/models/lrs-detr/lrs-detr-b.yaml')
model.model.eval()
model.info(detailed=False)
model.profile(imgsz=[640, 640])

# 模型融合后的信息
print('after fuse:')
model.fuse()
model.info(detailed=False)
```

#### 计算FPS和延迟（get_FPS.py）
```bash
# 测试推理速度
python get_FPS.py --weights runs/train/lrs-detr-b/weights/best.pt 
```

#### 生成热力图（heatmap.py）
```bash
# 可视化模型关注区域
python heatmap.py --weights runs/train/lrs-detr-b/weights/best.pt --source dataset/images/test
```

## 🔧 模型压缩

### LAMP剪枝（pruning.py）

```python
# 配置剪枝参数
param_dict = {
    # 基础参数
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
    
    # 剪枝参数
    'prune_method': 'lamp',  # 可选 'random', 'l1', 'lamp', 'slim', 'group_slim', 'group_norm', 'group_sl', 'growing_reg', 'group_hessian', 'group_taylor'
    'global_pruning': True,  # 是否全局剪枝
    'speed_up': 1.35,  # 剪枝加速比（剪枝前计算量/剪枝后计算量）
    'reg': 0.0005,  # 正则化系数
    'sl_epochs': 500,  # 稀疏学习迭代次数
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',  # 稀疏学习超参数
    'iterative_steps': 50  # 迭代剪枝步数
}

# 执行剪枝和微调
python pruning.py
```

### MLSSD蒸馏（distill.py）

```python
# 配置蒸馏参数
param_dict = {
    # 基础参数
    'model': 'runs/prune/lamp_speedup1.35/weights/best.pt',  # 剪枝后的模型
    'data': 'dataset/data.yaml',
    'imgsz': 640,
    'epochs': 300,
    'batch': 8,
    'workers': 8,
    'device': '6',
    'project': 'runs/distill',
    'name': 'lamp1.35_logical_mlssd_0.0001',
    
    # 蒸馏参数
    'prune_model': True,
    'teacher_weights': 'runs/train/lrs-detr-b/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/lrs-detr/lrs-detr-b.yaml',
    'kd_loss_type': 'feature',  # 选择'logical', 'feature', 'all'
    'kd_loss_decay': 'constant',  # 选择 'cosine', 'linear', 'cosine_epoch', 'linear_epoch', 'constant'
    'kd_loss_epoch': 1.0,
    
    # 逻辑蒸馏参数
    'logical_loss_type': 'mlssd',  # 使用MLSSD，可选logical
    'logical_loss_ratio': 0.0001,
    
    # 特征蒸馏参数
    'teacher_kd_layers': '21, 24, 27',
    'student_kd_layers': '21, 24, 27',
    'feature_loss_type': 'cwd',  # 可选mimic、mgd、cwd、chsim、sp
    'feature_loss_ratio': 0.05
}

# 执行蒸馏
python distill.py
```

### 剪枝效果可视化（plot_channel_image.py） (可选)
```bash
# 对比剪枝前后的通道数变化
python plot_channel_image.py --base-weights base_weights.pt --prune-weights prune_weights.pt
```

## 📦 模型导出（export.py） (可选)

```python
from ultralytics import RTDETR

# 加载模型
model = RTDETR('runs/train/lrs-detr-b/weights/best.pt')

# 导出为ONNX格式
model.export(format='onnx', simplify=True)

# 导出为TensorRT格式（需要安装TensorRT）
model.export(format='engine', half=True)

# 导出为CoreML格式（用于iOS部署）
model.export(format='coreml')
```

## 📊 实验结果

### 主流目标检测算法在VisDrone-DET的Test集上的性能对比

<table>
  <thead>
    <tr>
      <th>模型</th>
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




## 📁 项目结构

```
LRS-DETR/
├── assets/                         # 资源文件
├── dataset/                        # 数据集目录(图像/标签/数据处理脚本) 
├── ultralytics/                    # 模型核心代码
│   ├── cfg/                        # 配置文件
│   ├── nn/                         # 神经网络模块
│   │   ├── extra_modules/         # 创新模块实现
│   ├── models/                     # 模型实现
│   │   └── rtdetr/                # RT-DETR相关
│   ├── data/                       # 数据处理
│   └── utils/                      # 工具函数
├── train.py                        # 训练入口
├── val.py                          # 验证脚本
├── detect.py                       # 检测脚本
├── distill.py                      # 蒸馏脚本
├── pruning.py                      # 剪枝工具
├── get_FPS.py                      # 性能测试
├── heatmap.py                      # 热力图生成
├── get_COCO_metrics.py             # 计算COCO指标
├── plot_channel_image.py           # 剪枝效果可视化
└── requirements.txt                # 依赖列表
```




## 🙏 致谢

本项目基于以下开源项目：
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO系列实现框架
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR) - 实时DETR检测器
- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) - 无人机视觉数据集
- [TIDE](https://github.com/dbolya/tide) - 目标检测错误分析工具
- [Torch-Pruning](https://github.com/VainF/Torch-Pruning) - PyTorch剪枝工具



## 📧 联系方式

- 作者：吴宇辰
- 邮箱：ycwu73@gmail.com
- 学校：安徽理工大学 计算机科学与工程学院

---

**注意**：
1. 本项目仅供学术研究使用，商业使用请联系作者获得授权
