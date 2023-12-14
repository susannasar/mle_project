import numpy as np
import cv2
import io
import time

class Detector(object):
    def __init__(self):
        self.net = None
        self.classes = None
        self.conf_threshold = 0.5
        self.keep_aspect_ratio = False
        self.input_width = 640
        self.input_height = 640
        self.orig_width = -1
        self.orig_height = -1
        self.padding = 0

    def set_threshold(self, threshold):
        self.conf_threshold = threshold

    def set_classes(self, classes):
        self.classes = np.array(classes)

    def enable_keeping_aspect_ratio(self):
        self.keep_aspect_ratio = True
 
    def load_net(self, buffer):
        try:
            self.net = cv2.dnn.readNetFromONNX(buffer)
        except Exception as e:
            raise Exception('ERROR: Can not load model. ' + str(e))

    def preprocess(self, image):
        self.orig_height, self.orig_width, _ = image.shape
        if self.keep_aspect_ratio:
            processed_image = np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
            if self.orig_width > self.orig_height:
                scale = self.input_width / self.orig_width
                resized_img = cv2.resize(image, dsize=None, fx=scale, fy=scale)
                self.padding = int((self.input_height - resized_img.shape[0]) / 2)
                processed_image[self.padding : self.padding + resized_img.shape[0], :] = resized_img
                self.width_scale = self.orig_width / self.input_width
            else:
                scale = self.input_height / self.orig_height
                resized_img = cv2.resize(image, dsize=None, fx=scale, fy=scale)
                self.padding = int((self.input_width - resized_img.shape[1]) / 2)
                processed_image[:, self.padding : self.padding + resized_img.shape[1]] = resized_img
                self.height_scale = self.orig_height / self.input_height
        else:
            processed_image = cv2.resize(image, (self.input_width, self.input_height))
            self.width_scale = self.orig_width / self.input_width
            self.height_scale = self.orig_height / self.input_height
            

        processed_image = processed_image / 255.0
        processed_image = np.swapaxes(processed_image, 0, 2)
        processed_image = np.expand_dims(processed_image, axis=0)
        return processed_image

    def sclae_box(self, box, width_scale, height_scale):
        box[0] = round(box[0] * width_scale)
        box[2] = round(box[2] * width_scale)
        box[1] = round(box[1] * height_scale)
        box[3] = round(box[3] * height_scale)
        return box

    def rescale_box(self, box):
        if self.keep_aspect_ratio:
            if self.orig_width > self.orig_height:
                box[1] -= self.padding
                height_scale = self.orig_height / (self.input_height - 2 * self.padding)
                box = self.sclae_box(box, self.width_scale, height_scale)
            else:
                box[0] -= self.padding
                width_scale = self.orig_width / (self.input_width - 2 * self.padding)
                box = self.sclae_box(box, width_scale, self.height_scale)
        else:
            box = self.sclae_box(box, self.width_scale, self.height_scale)
        return box

    def predict(self, image):
        processed_image = self.preprocess(image)
        self.net.setInput(processed_image)
        model_output = self.net.forward()[0]

        scores = model_output[:, 4]
        boxes = model_output[:, :4]
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, 0.6)

        probs = model_output[:, 5:]
        class_indices = np.argmax(probs, axis=1)
        pred_classes = self.classes[class_indices]

        final_boxes = []
        final_scores = []
        final_classes = []
        for i in indices:
            box = boxes[i].astype(np.int32)
            box = [box[1], box[0], box[3], box[2]]
            final_boxes.append(self.rescale_box(box))
            final_scores.append(scores[i])
            final_classes.append(pred_classes[i])

        return [final_boxes, final_scores, final_classes]
