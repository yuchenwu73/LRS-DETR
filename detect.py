import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


if __name__ == '__main__':
    model = RTDETR('choose your weight') # choose your model path
    model.predict(source='choose your image',
              conf=0.4,
              project='runs/detect',
              name='exp',
              save=True,
              # visualize=True # visualize model feature maps
              line_width=2, # bounding box line width      
              show_conf=False, # whether to display prediction confidence
              show_labels=False, # whether to display prediction labels
              # save_txt=True, # whether to save results as .txt file
              # save_crop=True, # whether to save cropped images with results
              )