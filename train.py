import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/lrs-detr/lrs-detr-b.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=8, 
                # device='5,6',
                # resume='', # last.pt path
                project='runs/train', # save results to project/name
                name='lrs-detr-b', 
                )
    
    
