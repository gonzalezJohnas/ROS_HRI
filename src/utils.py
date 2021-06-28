from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
import cv2
from headpose.headPose import HeadPose
from humanpose.skeletonPose import HumanPose
from handGesture.handPose import HandGesture
import os

class image_converter:
    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_2", Image)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.headPose_estimator = HeadPose("/home/icub/PycharmProjects/ROS_humanSensing/src/headpose/checkpoint", 0)
        self.humanPose_estimator = HumanPose("/home/icub/PycharmProjects/ROS_humanSensing/src/humanpose/checkpoint/checkpoint_iter_370000.pth")
        self.handGesture_Estimator = HandGesture("/home/icub/PycharmProjects/ROS_humanSensing/src/handGesture/poses.txt")

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        (rows, cols, channels) = cv_image.shape
        faces, cv_image = self.headPose_estimator.getPose(cv_image)
        poses, cv_image = self.humanPose_estimator.getSkeleton(cv_image)
        hand_gestures, cv_image = self.handGesture_Estimator.run(cv_image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        if hand_gestures == "Palm":
            cv2.putText(cv_image, f"Gesture: PALM", (30, 400), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)


        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1)


        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

