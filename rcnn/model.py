import os
import cv2
import numpy as np
import tensorflow as tf

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

class FasterRCNN:
    def __init__(self, checkpoint_dir, labelmap_path, class_num):
        self.checkpoint_dir = checkpoint_dir
        self.labelmap_path = labelmap_path
        self.class_num = class_num

        self.sess = None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores =None
        self.detection_classes = None
        self.num_detections = None

        self.load()

    def load(self):
        print('Faster RCNN model initialization...')
        PATH_TO_CKPT = os.path.join(self.checkpoint_dir, 'frozen_inference_graph.pb')
        PATH_TO_LABELS = self.labelmap_path
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.class_num,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=detection_graph)

        # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def detect(self, image_path, save_path, thresh_hold=0.6):
        test_image = cv2.imread(image_path)
        image_expanded = np.expand_dims(test_image, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
            test_image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            # min_score_thresh=max(0.05, np.max(scores)-0.01))
            # min_score_thresh=np.max(scores) - 0.001)
            min_score_thresh=thresh_hold)
        cv2.imwrite(save_path, test_image)
        return save_path
