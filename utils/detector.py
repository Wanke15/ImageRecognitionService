import re
import base64
import uuid
import os
import cv2
from PIL import Image
import numpy as np

from yolo3.yolo import YOLO
from rcnn.model import FasterRCNN


class ImageConverter:
    def __init__(self, image_save_dir='raw_images'):
        self.image_save_dir = image_save_dir
        self.forehand = ''

    def str2image(self, img_str):
        splits = re.search(r'base64,(.*)', img_str.decode('utf8'))
        self.forehand = splits.group(0)
        imgstr = splits.group(1)
        output_name = os.path.join(self.image_save_dir, str(uuid.uuid1())+'.png')
        with open(output_name, 'wb') as output:
            output.write(base64.b64decode(imgstr))
        return output_name

    def image2str(self, image):
        with open(image, "rb") as imageFile:
            imgstr = base64.b64encode(imageFile.read()).decode('utf8')
            imgstr = "data:image/png;base64,"+imgstr
        return imgstr


class YoloObjectDetector(YOLO):
    def __init__(self, result_save_dir='detection_results', **kwargs):
        self.results_dir = result_save_dir
        self.image = None
        super().__init__(**kwargs)

        self.detect_single_image('raw_images/init_image.png')

    def detect(self, image_path):
        self.image = cv2.imread(image_path)
        result = cv2.rectangle(self.image, (90, 65), (195, 217), (0, 0, 255), 2)
        save_path = os.path.join(self.results_dir, image_path.split('\\')[-1])
        cv2.imwrite(save_path, result)
        return save_path

    def detect_image_array(self, image_array):
        image = Image.fromarray(image_array)
        image = self.detect_image(image)
        result = np.asarray(image)
        return result

    def detect_single_image(self, image_path):
        test_image = cv2.imread(image_path)
        result = self.detect_image_array(test_image)
        save_path = os.path.join(self.results_dir, image_path.split('\\')[-1])
        cv2.imwrite(save_path, result)
        return save_path


class RcnnObjectDetector(FasterRCNN):
    def __init__(self, result_save_dir='detection_results', **kwargs):
        self.results_dir = result_save_dir
        self.image = None
        super().__init__(**kwargs)

        self.detect_single_image('raw_images/init_image.png')

    def detect_single_image(self, image_path):
        save_path = os.path.join(self.results_dir, image_path.split('\\')[-1])
        self.detect(image_path, save_path)
        return save_path

