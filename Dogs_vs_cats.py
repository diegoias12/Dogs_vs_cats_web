import numpy as np
from cv2 import cv2

import tensorflow as tf
from tensorflow.keras import Sequential
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

import os

IMG_SIZE = 64

def read_image(name, margin=True, train=True):
    img_path = os.path.join('static', 'images', name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if margin:
        MARGIN = 5
        heigth = image.shape[0]
        width = image.shape[1]
        image = image[MARGIN:heigth-MARGIN, MARGIN:width-MARGIN]
    return image

def canny_box_plot(img):
    img_box = img.copy()
    edges = cv2.Canny(img, 100, 200)
    if edges.sum() == 0:
        return img
    height_sum = np.sum(edges, axis=1)
    width_sum = np.sum(edges, axis=0)
    row_min = np.where(height_sum > 0)[0][0]
    row_max = np.where(height_sum > 0)[0][-1:][0]
    col_min = np.where(width_sum > 0)[0][0]
    col_max = np.where(width_sum > 0)[0][-1:][0]
    img_box[row_min-10:row_min+10, col_min:col_max] = [0]
    img_box[row_max-10:row_max+10, col_min:col_max] = [0]
    img_box[row_min:row_max, col_min-10:col_min+10] = [0]
    img_box[row_min:row_max, col_max-10:col_max+10] = [0]
    return img[row_min:row_max, col_min:col_max]

def process_image(image):
    image = canny_box_plot(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return image

def make_prediction(filename):
    image = read_image(filename)
    image = process_image(image)
    model = load_model('static/models/model.h5')
    proba = model.predict(image)[0][0] 
    return round(proba * 100, 4)
