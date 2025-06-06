import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.rtdetr.distill import RTDETRDistiller


if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/data2/wuyuchen/LRS-DETR/runs/train/lrs-detr-b2/weights/best.pt',
        'data':'dataset/data.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 8,
        'workers': 8,
        'cache': False,
        'device': '6', 
        'project':'runs/distill/',
        'name':'lamp1.35_logical_mlssd_0.0001', 
        
        # distill
        'prune_model': True,
        'teacher_weights': '/data2/wuyuchen/LRS-DETR/runs/train/lrs-detr-b2/weights/best.pt',
        'teacher_cfg': '/data2/wuyuchen/LRS-DETR/ultralytics/cfg/models/lrs-detr/lrs-detr-b.yaml',
        'kd_loss_type': 'feature', # 选择'logical', 'feature', 'all'
        'kd_loss_decay': 'constant',# 选择 'cosine', 'linear', 'cosine_epoch', 'linear_epoch', 'constant'
        'kd_loss_epoch': 1.0,
        
        'logical_loss_type': 'mlssd', # logical、mlssd
        'logical_loss_ratio': 0.0001, # 修改
        
        'teacher_kd_layers': '21, 24, 27',
        'student_kd_layers': '21, 24, 27',
        'feature_loss_type': 'cwd', # mimic、mgd、cwd、chsim、sp
        'feature_loss_ratio': 0.05   # 修改
    }
    
    model = RTDETRDistiller(overrides=param_dict)
    model.distill()
    
    
