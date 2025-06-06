import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.rtdetr.compress import RTDETRCompressor, RTDETRFinetune


def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = RTDETRCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = RTDETRFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/data2/wuyuchen/LRS-DETR-main/runs/train/exp/FINAL_rtdetr-SOEP-C2f-SHSA-CGLU-ASSA/weights/best.pt',
        'data':'./dataset/data.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 8,
        'workers': 8,
        'cache': False,
        'device': '1',
        'project':'runs/prune', # 修改
        'name':'lamp_speedup1.35', # 修改
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True, # 全局剪枝参数
        'speed_up': 1.35, # 剪枝前计算量/剪枝后计算量
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
        'iterative_steps': 50
    }
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)
    