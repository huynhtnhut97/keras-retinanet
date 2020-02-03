
##########################Load necessary modules################

# show images inline
#%matplotlib inline

# automatically reload modules when they have changed
#%load_ext autoreload
#%autoreload 2

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import argparse

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
    parser.add_argument("--imgpath", dest = 'image',help = 
                        "Path to image")
    parser.add_argument("--modelpath", dest = 'model',help = 
                        "Path to image")
    return parser.parse_args()
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

if __name__ == '__main__':
    args = arg_parse()
    img = args.image
    model_path = args.model
    ####################Load RetinaNet model##########################
    # adjust this to point to your downloaded/trained model
    # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
    #model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet101')

    # if the model is not converted to an inference model, use the line below
    # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
    #model = models.convert_model(model)

    #print(model.summary())

    # load label to names mapping for visualization purposes
    labels_to_names = {
    1:'pedestrian',
    2:'people',
    3:'bicycle',
    4:'car',
    5:'van',
    6:'truck',
    7:'tricycle',
    8:'awning-tricycle',
    9:'bus',
    10:'motor'}

    #######################Run detection on examples##########################
    images_path = "/mnt/data/nhuthuynh/sequences/"
    results_path = "./results"
    for folder in os.listdir(images_path):
        folder_path = os.path.join(images_path,folder)
        for image in os.listdir(folder_path):
            image_path = os.path.join(folder_path,image)
            print(image_path)
            # load image
            im = read_image_bgr(image_path)

            # copy to draw on
            draw = im.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            # preprocess image for network
            im = preprocess_image(im)
            im, scale = resize_image(im)

            # process image
            start = time.time()
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(im, axis=0))
            # correct for image scale
            boxes /= scale

            file_name = folder+'_'+os.path.splitext(image)[0]+'.txt' #detected result file
            file_path = os.path.join(results_path,file_name) # full file path of detected result file
            with open(file_path,'w') as file:
                for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    # scores are sorted so we can break
                    # if (int(label) != 1 and int(label)!=2):
                    #   continue
                    if score < 0.5:
                        break
                    
                    x = box[0]
                    y = box[1]
                    w = abs(box[2]-x)
                    h = abs(box[3]-y)
                    
                    file.write(str(labels_to_names[label])+','+str(int(x))+','+str(int(y))+','+str(int(w))+','+str(int(h)))
                    file.write('\n')
    
    