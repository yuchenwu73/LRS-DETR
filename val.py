import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('choose your weight')
    model.val(data='dataset/data.yaml',
              split='test',
              imgsz=640,
              batch=8,
              save_json=True, # if you need to cal coco metrics
              project='runs/val',
              name='choose your name',
              )