import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import RTDETR

if __name__ == '__main__':
    # choose your yaml file
    model = RTDETR('choose your yaml file')
    model.model.eval()
    model.info(detailed=False)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()