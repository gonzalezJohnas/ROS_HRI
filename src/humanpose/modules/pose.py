import math
import os.path

import cv2
import numpy as np
import csv
from humanpose.modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from humanpose.modules.one_euro_filter import OneEuroFilter
from datetime import datetime
from utils import check_img_limit

class Pose:
    num_kpts = 18
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [249, 1, 242]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]
        self.angle_joint = {"neck": ["neck", "nose"], "lshoulder": ["neck", "lshoulder"],
                            "rshoulder": ["neck", "rshoulder"],
                            "lelbow": ["lelbow", "lshoulder"], "relbow": ["relbow", "rshoulder"],
                            "lknee": ["lknee", "lhip"], "rknee": ["rknee", "rhip"], "rwrist": ["rwrist", "relbow"],
                            "lwrist": ["lwrist", "lelbow"]}

        self.joint_dictionary = {"nose": 0, "neck": 1, "lshoulder": 5, "rshoulder": 2, "lelbow": 6, "relbow": 3,
                                 "lwrist": 7,
                                 "rwrist": 4, "rhip": 8, "lhip": 11, "lknee": 12, "rknee": 9}

        self.previous_pose = None
        self.i = 0
        self.joint_angles = {}
        self.joint_velocity = {}

        self.saving_path = '/tmp/left_hand/'

        self.measure_time = None

    @staticmethod
    def get_bbox(keypoints):
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        return bbox

    def update_previous_pose(self, pose):
        self.previous_pose = pose

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img):

        self.measure_time = datetime.now()

        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 5, Pose.color, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 5, Pose.color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 5)




    def extract_hands(self, img, save=False):
        lshoulder_X, lshoulder_Y = self.keypoints[5]
        rshoulder_X, rshoulder_Y = self.keypoints[2]
        distance_forearm = math.sqrt(((lshoulder_X - rshoulder_X) ** 2) + ((lshoulder_Y - rshoulder_Y) ** 2))

        left_hand_coord, roi_left_hand = self._get_hand_bounding_box(self.keypoints[7], distance_forearm, self.joint_angles["lwrist"], img, save)
        right_hand_coord, roi_right_hand = self._get_hand_bounding_box(self.keypoints[4], distance_forearm, self.joint_angles["rwrist"], img, save)

        cv2.rectangle(img, (left_hand_coord[0], left_hand_coord[1]), (left_hand_coord[2], left_hand_coord[3]), (0, 0, 255), 2)
        cv2.rectangle(img, (right_hand_coord[0], right_hand_coord[1]), (right_hand_coord[2], right_hand_coord[3]), (0, 0, 255), 2)

        return roi_left_hand, roi_right_hand


    def _get_hand_bounding_box(self, wrist_keypoint, distance_forearm, wrist_angle, img, save):
        roi_hand = None
        start_point_x = start_point_y = end_point_x = end_point_y = -1

        if wrist_keypoint[0] != -1:

            wrist_X, wrist_Y = wrist_keypoint
            multiplier = -1 if wrist_angle > 0 else 1

            start_point_x = int(wrist_X - (distance_forearm / 2))
            start_point_y = int(wrist_Y)

            if -30 < wrist_angle < 30:
                start_point_y = int(wrist_Y + (distance_forearm / 2))

                end_point_x = int(start_point_x + (distance_forearm))
                end_point_y = int(start_point_y - (distance_forearm / 1.2))
            else:

                end_point_x = int(start_point_x + (distance_forearm))
                end_point_y = int(start_point_y - (multiplier * distance_forearm / 1.2))

            start_point_x, start_point_y, end_point_x, end_point_y = check_img_limit(start_point_x, start_point_y, end_point_x, end_point_y, img.shape)

            if multiplier > 0:
                roi_hand = img[end_point_y:end_point_y + abs(end_point_y - start_point_y),
                                start_point_x:start_point_x + abs(end_point_x - start_point_x)]
            else:
                roi_hand = img[start_point_y:start_point_y + abs(end_point_y - start_point_y),
                                start_point_x:start_point_x + abs(end_point_x - start_point_x)]

            if save:
                timestamp = datetime.timestamp(datetime.now())
                cv2.imwrite(os.path.join(self.saving_path, f"img_{timestamp}.png"), roi_hand)


        return [start_point_x, start_point_y, end_point_x, end_point_y], roi_hand




    def compute_angle_joints(self):
        self.joint_angles = {}

        for name_angle, parts_angle in self.angle_joint.items():
            root_joint = self.joint_dictionary[parts_angle[0]]
            target_joint = self.joint_dictionary[parts_angle[1]]
            self.joint_angles[name_angle] = getAngle(self.keypoints[root_joint][0],
                                               self.keypoints[root_joint][1],
                                               self.keypoints[target_joint][0],
                                               self.keypoints[target_joint][1])

        return self.joint_angles

    def compute_joints_velocity(self):
        current_tine = datetime.now()
        delta = (current_tine - self.measure_time).total_seconds()

        self.joint_velocity = {}
        row_x = []
        if self.previous_pose is None:
            return self.joint_velocity
        for name_angle, id_angle in self.joint_dictionary.items():
            current_joint_value = self.keypoints[id_angle]
            previous_joint_value = self.previous_pose.keypoints[id_angle]


            delta_pose = abs(current_joint_value - previous_joint_value)

            if 8 < delta_pose[0] < 250:
                speed = delta_pose[0]/delta
                speed /= 1000
                row_x.append(speed)
                self.joint_velocity[name_angle] = speed

            else:
                row_x.append(0)
                self.joint_velocity[name_angle] = 0


        row = [ current_tine.timestamp()] + row_x
        # with open('/home/icub/PycharmProjects/ROS_humanSensing/python_live_plot_data.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(row)
        return self.joint_velocity


def radiansToDegrees(radians):
    return (radians * 180 / np.pi)


def getAngle(x1, y1, x2, y2):
    angle = np.arctan2(y1 - y2, x1 - x2)
    return radiansToDegrees(angle)


def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def track_poses(previous_poses, current_poses, threshold=3, smooth=False):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.
    If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :param smooth: smooth pose keypoints between frames
    :return: None
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose in current_poses:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0

        if len(previous_poses) == 1:
            current_pose.update_id(previous_poses[0].id)
            current_pose.update_previous_pose(previous_poses[0])

        else:

            for id, previous_pose in enumerate(previous_poses):
                if not mask[id]:
                    continue
                iou = get_similarity(current_pose, previous_pose)
                if iou > best_matched_iou:
                    best_matched_iou = iou
                    best_matched_pose_id = previous_pose.id
                    best_matched_id = id

            if best_matched_iou >= threshold:
                mask[best_matched_id] = 0
                current_pose.update_previous_pose(previous_poses[0])

            else:  # pose not similar to any previous
                best_matched_pose_id = None
            current_pose.update_id(best_matched_pose_id)
            if smooth:
                for kpt_id in range(Pose.num_kpts):
                    if current_pose.keypoints[kpt_id, 0] == -1:
                        continue
                    # reuse filter if previous pose has valid filter
                    if (best_matched_pose_id is not None
                            and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
                        current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                    current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](
                        current_pose.keypoints[kpt_id, 0])
                    current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](
                        current_pose.keypoints[kpt_id, 1])
                current_pose.bbox = Pose.get_bbox(current_pose.keypoints)
