import numpy as np
from handGesture.utils import detector_utils as detector_utils
from handGesture.utils import pose_classification_utils as classifier
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from handGesture.utils.detector_utils import WebcamVideoStream
import datetime
import argparse
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import handGesture.gui

frame_processed = 0
score_thresh = 0.18


class HandGesture(object):

    def __init__(self, label_pose_path):
        self.nb_hand = 1
        self.num_workers = 4
        self.thr = 0.6
        self.poses = []
        _file = open(label_pose_path, "r")
        lines = _file.readlines()
        for line in lines:
            line = line.strip()
            if (line != ""):
                print(line)
                self.poses.append(line)

        self.detection_graph, self.sess = detector_utils.load_inference_graph()
        self.sess = tf.Session(graph=self.detection_graph)

        self.model, self.classification_graph, self.session = classifier.load_KerasGraph("/home/icub/Documents/Jonas/HandPose/cnn/models/hand_poses_wGarbage_10.h5")

    def run(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes, scores = detector_utils.detect_objects(
            frame, self.detection_graph, self.sess)

        # get region of interest
        res = detector_utils.get_box_image(self.nb_hand, self.thr,
                                           scores, boxes, frame.shape[1], frame.shape[0], frame)

        # draw bounding boxes
        detector_utils.draw_box_on_image(self.nb_hand, self.thr,
                                         scores, boxes, frame.shape[1], frame.shape[0], frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # classify hand pose
        if res is not None:
            class_res = classifier.classify(self.model, self.classification_graph, self.session, res)
            max_prediction = np.argmax(class_res)
            predicted_class = self.poses[max_prediction]
            return predicted_class, frame

        return [], frame

