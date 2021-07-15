import argparse

import cv2
import numpy as np
import torch

from humanpose.with_mobilenet import PoseEstimationWithMobileNet
from humanpose.modules.keypoints import extract_keypoints, group_keypoints
from humanpose.modules.load_state import load_state
from humanpose.modules.pose import Pose, track_poses
from humanpose.val import normalize, pad_width

from handGesture.utils import pose_classification_utils as classifier

from std_msgs.msg import String


class HumanPose(object):
    def __init__(self, checkpoint_path, input_size_model=256):
        self.model = PoseEstimationWithMobileNet()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        load_state(self.model, checkpoint)
        self.net_input_height_size = input_size_model

        self.model = self.model.eval()
        self.model = self.model.cuda()

        self.previous_poses = []
        self.model_hand, self.classification_graph, self.session = classifier.load_KerasGraph("/home/icub/Documents/Jonas/HandPose/cnn/models/hand_poses_wGarbage_10.h5")

        self.poses_class = []
        _file = open("/home/icub/catkin_build_ws/src/ROS_humanSensing/src/handGesture/poses.txt", "r")
        lines = _file.readlines()
        for line in lines:
            line = line.strip()
            if (line != ""):
                print(line)
                self.poses_class.append(line)

        self.saving_hand = False

    def infer_fast(self, img, stride=8, upsample_ratio=4,
                   pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
        height, width, _ = img.shape
        scale =  self.net_input_height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [ self.net_input_height_size, max(scaled_img.shape[1],  self.net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        tensor_img = tensor_img.cuda()

        stages_output = self.model(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad


    def run(self, img, track=True):


        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts
        delay = 1
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = self.infer_fast(img, stride=stride, upsample_ratio=upsample_ratio)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(self.previous_poses, current_poses)
            self.previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
            pose.compute_angle_joints()
            pose.compute_joints_velocity()
            left_roi_hand, right_roi_hand = pose.extract_hands(img,  self.saving_hand)

            if left_roi_hand is not None:
                class_res = classifier.classify(self.model_hand, self.classification_graph, self.session, left_roi_hand)
                max_prediction = np.argmax(class_res)
                predicted_class = self.poses_class[max_prediction]
                print(f"Left hand recognized gesture {predicted_class}")

            if right_roi_hand is not None:
                class_res = classifier.classify(self.model_hand, self.classification_graph, self.session, right_roi_hand)
                max_prediction = np.argmax(class_res)
                predicted_class = self.poses_class[max_prediction]
                print(f"Right hand recognized gesture {predicted_class}")


        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)



        return current_poses, img



