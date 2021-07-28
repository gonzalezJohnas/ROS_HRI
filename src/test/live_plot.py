from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ROS_humanSensing.msg import JointVelocity
import argparse
from datetime import datetime
import rospy
import numpy as np

colors = ["red",  'blue', "yellow", "green"]


class JointVisualisation:

    def __init__(self, args):
        self.joint_sub = rospy.Subscriber("joint_velocity", JointVelocity, self.callback)
        self.legend = args.joints


        self.limit_data = 3
        self.x = np.empty(self.limit_data + 1, np.float)
        self.y = {joint_name: np.empty( self.limit_data + 1, np.float) for joint_name in self.legend}

        self.nb = 0

    def callback(self, data):
        now = datetime.now()
        self.x[self.nb] = datetime.timestamp(now)
        for i, joint_name in enumerate(self.legend):
            self.y[joint_name][self.nb] = (getattr(data, str(joint_name)))

        if self.nb == self.limit_data:
            plt.ion()

            self.nb = self.limit_data // 2
            for i, joint_name in enumerate(self.legend):
                plt.plot(self.x, self.y[joint_name], color=colors[i])
                self.y[joint_name][:self.nb] = self.y[joint_name][-self.nb:]

            self.x[:self.nb] = self.x[-self.nb:]

            plt.xlabel('Time')
            plt.ylabel('velocity')
            plt.ylim(-5, 5)
            plt.title('Joint velocity')
            plt.gcf().autofmt_xdate()

            plt.legend(self.legend, loc='upper left')

            plt.show()
            plt.pause(1e-12)

        else:
            self.nb += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints", nargs="+", default=["nose"], help="list of joints to visualise")
    args = parser.parse_args()

    rospy.init_node('joint_visualisation', anonymous=True)

    jointVis = JointVisualisation(args)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

