import yaml
import rospy
import cv2


from ROS_humanSensing.srv import SaveHands
from ROS_humanSensing.msg import JointVelocity, JointAngles
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
import cv2
from humanpose import HumanPose
from headpose import HeadPose
import os
import json
import argparse

class ImageReceiver:

    def __init__(self, args ):
        self.run = False

        self.joint_pub_velocity = rospy.Publisher('joint_velocity', JointVelocity, queue_size=10)
        self.joint_pub_angles = rospy.Publisher('joint_angles', JointAngles, queue_size=10)
        self.image_pub = rospy.Publisher("hri_view", Image)

        self.service_saveHand = rospy.Service('save_hands', SaveHands, self.handle_hand_saving)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(args.camera_node_name, Image, self.callback)
        self.humanPose_estimator = HumanPose(args.model_humanpose_path, args.model_hand_path, args.label_hand_path)

        self.run = True


    def handle_hand_saving(self, req):
        self.humanPose_estimator.saving_hand = not self.humanPose_estimator.saving_hand

        return "start saving" if self.humanPose_estimator.saving_hand else "stop saving"

    def callback(self, data):
        if self.run:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                poses, cv_image, hand_gestures = self.humanPose_estimator.run(cv_image)
                cv2.imshow("Image window", cv_image)
                cv2.waitKey(1)

                self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

                for pose in poses:
                    joint_values = list(pose.joint_velocity.values())
                    angle_values = list(pose.joint_angles.values())
                    msg_joint_velocity = JointVelocity(*joint_values)
                    msg_joint_angles = JointAngles(*angle_values)

                    self.joint_pub_velocity.publish(msg_joint_velocity)
                    self.joint_pub_angles.publish(msg_joint_angles)

            except CvBridgeError as e:
                print(e)

        # (rows, cols, channels) = cv_image.shape
        # faces, cv_image = self.headPose_estimator.run(cv_image)
        # hand_gestures, cv_image = self.handGesture_Estimator.run(cv_image)

        # if hand_gestures == "Palm":
        #     cv2.putText(cv_image, f"Gesture: PALM", (30, 400), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)



def main():
    rospy.init_node('human_sensing', anonymous=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("camera_node_name", help="Name of the ROS node input camera",
                        type=str)
    parser.add_argument("model_humanpose_path", help="Model path for the human pose estimation",
                        type=str)
    parser.add_argument("model_hand_path", help="Model path for the hand classifier",
                        type=str)
    parser.add_argument("label_hand_path", help="Label file path for the hand classifier",
                        type=str)
    args = parser.parse_args()

    ic = ImageReceiver(args)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()