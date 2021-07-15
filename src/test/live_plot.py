
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_values = []
y_values = []

index = count()
joint_dictionary = {"nose": "red", "neck": 'blue'}

if __name__ == '__main__':
    legend = [joint_name for joint_name, col in joint_dictionary.items()]
    while True:
        plt.ion()
        data = pd.read_csv('/home/icub/PycharmProjects/ROS_humanSensing/python_live_plot_data.csv')

        for joint_name, col in joint_dictionary.items():
            x_values = data['time']
            y_values = data[joint_name]
            plt.plot(x_values, y_values, color=col)
        plt.xlabel('Time')
        plt.ylabel('velocity')
        plt.title('Neck nose joint velocity')
        plt.gcf().autofmt_xdate()

        plt.legend(legend, loc='upper left')

        plt.show()
        plt.pause(0.01)
