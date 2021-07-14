from tensorflow.keras.models import load_model
from mrcnn.config import Config
from mrcnn import model1 as modellib, utils
import time, os, pickle
import skimage.draw
import numpy as np
import cv2 as cv

DEFECT_DIR = './h5s/mask_rcnn_defect_0026.h5'
MANGO_DIR = './h5s/mask_rcnn_mango.h5'
SORTING_DIR = './h5s/model6.h5'
SVM_MODEL = './h5s/svm_model.pickle'
TEST_IMG = './sample/test_sample.jpg'


class MangoConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "mango"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1

    # Number of training steps per epoch
    # train張數//顯卡數量*IMAGES_PER_GPU
    STEPS_PER_EPOCH = 18

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    IMAGE_MAX_DIM = 1344
    LEARNING_RATE = 0.0001
    GPU_COUNT = 1


class DefectConfig(MangoConfig):
    NAME = "defect"


def detect_and_color_splash(model, image_path, save=False):
    # Run model detection and generate the color splash effect
    # print("Running on {}".format(image_path))
    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=0)[0]
    # Color splash
    splash = color_splash(image, r['masks'])

    if save:
        path = './mango_extracted'
        name = image_path.split('\\')[-1]
        file_name = path + "/" + name
        skimage.io.imsave(file_name, splash)
    return splash


def color_splash(image, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 0
    # Copy color pixels from the original color image where mask is set
    
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

# Preprocessing
class Preprocessing():
    def __init__(self, img_path):
        self.img_path = img_path

    def rotate(self, imgs):
        new_img_list = []
        for j in imgs:
            img = cv.imread(j)
            (h, w) = img.shape[:2]
            if h>w:
                center = (h//2, w//2)
                angle = 270
                M = cv.getRotationMatrix2D(center, angle, scale=1)
                img = cv.warpAffine(img, M, (h, w))
            new_img_list.append(img)
        return new_img_list


    def npstack(self, img_list):
        fix_imgs_size = []
        i, total = 0, len(img_list)
        for j in img_list:
            img_resize = cv.resize(j,(224,224))
            fix_imgs_size.append(img_resize)
            img_data = np.stack(fix_imgs_size, axis=0)
            i += 1
        return img_data

    def main(self):
        img_list = [self.img_path]
        x_input = self.rotate(img_list)
        x_input = self.npstack(x_input)
        return x_input


# Mango Area
config = MangoConfig()
d_config = DefectConfig()

model = modellib.MaskRCNN(mode="inference", config=config)
d_model = modellib.MaskRCNN(mode="inference", config=d_config)

model.load_weights(MANGO_DIR, by_name=True)
d_model.load_weights(DEFECT_DIR, by_name=True)

detect_and_color_splash(model, image_path=TEST_IMG)
s_model = open(SVM_MODEL, 'rb')
svm_model = pickle.load(s_model)
sorting_model = load_model(SORTING_DIR)

num2class = {0:'A', 1:'B', 2:'C'}

while True:
    TEST_IMG = input("請輸入欲分類芒果圖片:")
    t1 = time.time()
    image = detect_and_color_splash(model, image_path=TEST_IMG, save=True)

    # Defect Area
    results = d_model.detect([image], verbose=0)
    r = results[0]
    x = float(r['scores'][0])
    print("有檢測到瑕疵" if x else "\n")

    # Preprocessing
    path = './mango_extracted'
    name = TEST_IMG.split('\\')[-1]
    file_name = path + "/" + name
    pp = Preprocessing(file_name)
    r_img = pp.main()
   
    # Sorting Network
    r_img = r_img / 255.0
    feat_train_v = sorting_model.predict(r_img)
    feat_train_v = np.insert(feat_train_v, 3, x, axis = 1)
    predict_class = svm_model.predict(feat_train_v)
    predict_class = num2class[predict_class[0]]
    
    print("用了", round(time.time() - t1, 3), "秒, 等級", predict_class)