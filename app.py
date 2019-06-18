from flask import Flask, request
from utils.detector import ImageConverter, YoloObjectDetector, RcnnObjectDetector

converter = ImageConverter(image_save_dir='raw_images')
# yolo_configs = {
#             "model_path": 'models/yolo3/trained_weights_final.h5',
#             "anchors_path": 'models/yolo3/yolo_anchors.txt',
#             "classes_path": 'models/yolo3/classes.txt',
#             "score" : 0.2,
#             "iou" : 0.1,
#             "model_image_size" : (416, 416),
#             "gpu_num" : 0,
#         }
# detector = YoloObjectDetector(result_save_dir='detection_results', **yolo_configs)

faster_rcnn_configs = {
    "checkpoint_dir": "models/faster_rcnn",
    "labelmap_path": "models/faster_rcnn/labelmap.pbtxt",
    "class_num" : 4
}

detector = RcnnObjectDetector(result_save_dir='detection_results', **faster_rcnn_configs)

app = Flask(__name__)


@app.route('/detect', methods=['POST', 'GET'])
def detect():
    img_data = request.get_data()
    raw_img_path = converter.str2image(img_data)
    res_img_path = detector.detect_single_image(raw_img_path)
    test_res = converter.image2str(res_img_path)
    return test_res


# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=8000)
