import time, os
import warnings
import numpy as np
import torch
import math
import torchvision
from torchvision import transforms
import cv2
from headpose.dectect import AntiSpoofPredict
from math import cos, sin
from headpose.pfld.pfld import PFLDInference, AuxiliaryNet
warnings.filterwarnings('ignore')

class Face:
    def __init__(self, yaw, pitch, roll, bbox):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.bbox = bbox


class HeadPose:
    def __init__(self, model_path, device_id):
        self.model_test = AntiSpoofPredict(device_id, model_path)
        checkpoint = torch.load(os.path.join(model_path, "checkpoint.pth.tar") , map_location=self.model_test.device)
        plfd_backbone = PFLDInference().to(self.model_test.device)
        plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
        plfd_backbone.eval()
        self.plfd_backbone = plfd_backbone.to(self.model_test.device)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def get_num(self, point_dict, name, axis):
        num = point_dict.get(f'{name}')[axis]
        num = float(num)
        return num


    def point_line(self, point,line):
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        x3 = point[0]
        y3 = point[1]

        k1 = (y2 - y1)*1.0 /(x2 -x1)
        b1 = y1 *1.0 - x1 *k1 *1.0
        k2 = -1.0/k1
        b2 = y3 *1.0 -x3 * k2 *1.0
        x = (b2 - b1) * 1.0 /(k1 - k2)
        y = k1 * x *1.0 +b1 *1.0
        return [x,y]

    def point_point(self, point_1,point_2):
        x1 = point_1[0]
        y1 = point_1[1]
        x2 = point_2[0]
        y2 = point_2[1]
        distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
        return distance

    def run(self, frame):
        height, width = frame.shape[:2]

        image_bboxes = self.model_test.get_bbox(frame)
        faces = []
        for image_bbox in image_bboxes:
            x1 = image_bbox[0]
            y1 = image_bbox[1]
            x2 = image_bbox[0] + image_bbox[2]
            y2 = image_bbox[1] + image_bbox[3]
            w = x2 - x1
            h = y2 - y1

            size = int(max([w, h]))
            cx = x1 + w / 2
            cy = y1 + h / 2
            x1 = cx - size / 2
            x2 = x1 + size
            y1 = cy - size / 2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            cropped = frame[int(y1):int(y2), int(x1):int(x2)]
            try:
                if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                    cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
            except Exception as e:
                raise Exception("Face not found")

            cropped = cv2.resize(cropped, (112, 112))

            input = cv2.resize(cropped, (112, 112))
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            input = self.transform(input).unsqueeze(0).to(self.model_test.device)
            _, landmarks = self.plfd_backbone(input)
            pre_landmark = landmarks[0]
            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
            point_dict = {}
            i = 0
            for (x, y) in pre_landmark.astype(np.float32):
                point_dict[f'{i}'] = [x, y]
                i += 1

            # yaw
            point1 = [self.get_num(point_dict, 1, 0), self.get_num(point_dict, 1, 1)]
            point31 = [self.get_num(point_dict, 31, 0), self.get_num(point_dict, 31, 1)]
            point51 = [self.get_num(point_dict, 51, 0), self.get_num(point_dict, 51, 1)]
            crossover51 = self.point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
            yaw_mean = self.point_point(point1, point31) / 2
            yaw_right = self.point_point(point1, crossover51)
            yaw = (yaw_mean - yaw_right) / yaw_mean
            yaw = int(yaw * 71.58 + 0.7037)

            # pitch
            pitch_dis = self.point_point(point51, crossover51)
            if point51[1] < crossover51[1]:
                pitch_dis = -pitch_dis
            pitch = int(1.497 * pitch_dis + 18.97)

            # roll
            roll_tan = abs(self.get_num(point_dict, 60, 1) - self.get_num(point_dict, 72, 1)) / abs(
                self.get_num(point_dict, 60, 0) - self.get_num(point_dict, 72, 0))
            roll = math.atan(roll_tan)
            roll = math.degrees(roll)
            if self.get_num(point_dict, 60, 1) > self.get_num(point_dict, 72, 1):
                roll = -roll
            roll = int(roll)
            cv2.putText(frame, f"Head_Yaw(degree): {yaw}", (30, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Head_Pitch(degree): {pitch}", (30, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Head_Roll(degree): {roll}", (30, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)
            frame = self.draw_axis(frame, yaw, pitch, roll, tdx=cx, tdy=cy)
            faces.append(Face(yaw, pitch, roll, image_bbox))
        return faces, frame

    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size=80):

            pitch = pitch * np.pi / 180
            yaw = -(yaw * np.pi / 180)
            roll = roll * np.pi / 180

            if tdx != None and tdy != None:
                tdx = tdx
                tdy = tdy
            else:
                height, width = img.shape[:2]
                tdx = width / 2
                tdy = height / 2

            # X-Axis pointing to right. drawn in red
            x1 = size * (cos(yaw) * cos(roll)) + tdx
            y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

            # Y-Axis | drawn in green
            #        v
            x2 = size * (-cos(yaw) * sin(roll)) + tdx
            y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

            # Z-Axis (out of the screen) drawn in blue
            x3 = size * (sin(yaw)) + tdx
            y3 = size * (-cos(yaw) * sin(pitch)) + tdy

            cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
            cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

            return img