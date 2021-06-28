# -*- coding: utf-8 -*-

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F

class Detection:
    def __init__(self, model_path):
        caffemodel = os.path.join(model_path, "Widerface-RetinaFace.caffemodel")
        deploy = os.path.join(model_path, "deploy.prototxt")
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        filtered_bbox = [box for box in out if box[2] > self.detector_confidence]

        bbox = []
        for box in filtered_bbox:
            left, top, right, bottom = box[3]*width, box[4]*height, box[5]*width, box[6]*height
            bbox.append([int(left), int(top), int(right-left+1), int(bottom-top+1)])
        return bbox

class AntiSpoofPredict(Detection):
    def __init__(self, device_id, model_path):
        super(AntiSpoofPredict, self).__init__(model_path)
        self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")

