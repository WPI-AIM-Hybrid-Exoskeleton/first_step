import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer, GMMTrainer, TPGMMTrainer_old
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner, TPGMMRunner_old
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from scipy import signal
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib
from dtw import dtw
import numpy.polynomial.polynomial as poly

from GaitAnaylsisToolkit.Session import ViconGaitingTrial
from scipy.signal import find_peaks

matplotlib.rcParams['text.usetex'] = True


def get_angles(files):
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    fig, axs = plt.subplots(3)
    for file in files:
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        left_leg = trial.vicon.get_model_output().get_left_leg()
        right_leg = trial.vicon.get_model_output().get_right_leg()


        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)

        hip = np.deg2rad(right_leg.hip.angle.x)
        knee = np.deg2rad(right_leg.knee.angle.x)
        ankle = np.deg2rad(right_leg.ankle.angle.x)
        axs[0].plot(hip)
        axs[1].plot(knee)
        axs[2].plot(ankle)

        x1_heel = 437-354
        y1_heel = hip[x1_heel]

        x1_toe = 838-354
        y1_toe = hip[x1_toe]


        x2_heel = 926-354
        y2_heel = hip[x2_heel]

        x2_toe = 1226-354
        y2_toe = hip[x2_toe]


        x3_heel = 1323-354
        y3_heel = hip[x3_heel]

        x3_toe = 1687-354
        y3_toe = hip[x3_toe]

        axs[0].annotate("Heel down 1", xy=(x1_heel,y1_heel), xytext=(x1_heel-5, y1_heel-0.5),
                        arrowprops = dict(arrowstyle="->"))

        axs[0].annotate("Toe off 1", xy=(x1_toe, y1_toe), xytext=(x1_toe - 50, y1_toe + 0.5),
                        arrowprops=dict(arrowstyle="->"))


        axs[0].annotate("Heel down 2", xy=(x2_heel,y2_heel), xytext=(x2_heel-5, y2_heel-0.5),

                        arrowprops = dict(arrowstyle="->"))

        axs[0].annotate("Toe off 2", xy=(x2_toe, y2_toe), xytext=(x2_toe - 50, y2_toe + 0.5),
                        arrowprops=dict(arrowstyle="->"))


        axs[0].annotate("Heel down 3", xy=(x3_heel,y3_heel), xytext=(x3_heel-5, y3_heel-0.25),
                        arrowprops = dict(arrowstyle="->"))

        axs[0].annotate("Toe off 3", xy=(x3_toe, y3_toe), xytext=(x3_toe - 5, y3_toe - 0.35),
                        arrowprops=dict(arrowstyle="->"))

        fig.suptitle(r'Right Leg Joint Angles (rads) ', fontsize=16)


        axs[0].set_title("Hip", fontsize=20, rotation=0)
        axs[1].set_title("Knee", fontsize=20, rotation=0)
        axs[2].set_title("Ankle", fontsize=20, rotation=0)

        axs[0].set_ylabel(r'$\theta$', fontsize=20, rotation=0, labelpad=30)
        axs[1].set_ylabel(r'$\theta$', fontsize=20, rotation=0, labelpad=30)
        axs[2].set_ylabel(r'$\theta$', fontsize=20, rotation=0, labelpad=30)
        axs[2].set_xlabel("Frames (record at 100FPS)", fontsize=20, rotation=0, labelpad=25)


        #axs[0].plot(right_leg.hip.angle.x)
        #
        # axs[1].plot(left_leg.knee.angle.x)
        # axs[1].plot(right_leg.knee.angle.x)
        #
        # axs[2].plot(left_leg.ankle.angle.x)
        # axs[2].plot(right_leg.ankle.angle.x)

    plt.show()




def get_moments(files):
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    fig, axs = plt.subplots(3)
    for file in files:
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        left_leg = trial.vicon.get_model_output().get_left_leg()
        right_leg = trial.vicon.get_model_output().get_right_leg()



        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)

        axs[0].plot(np.array(right_leg.hip.moment.x)/1000)
        axs[1].plot(np.array(right_leg.knee.moment.x)/1000)
        axs[2].plot(np.array(right_leg.ankle.moment.x)/1000)
        print(len(right_leg.ankle.moment.x)/100)
        x1_heel = 437-354
        y1_heel = right_leg.hip.moment.x[x1_heel]/1000

        x1_toe = 838-354
        y1_toe = right_leg.hip.moment.x[x1_toe]/ 1000


        x2_heel = 926-354
        y2_heel = right_leg.hip.moment.x[x2_heel]/1000

        x2_toe = 1226-354
        y2_toe = right_leg.hip.moment.x[x2_toe]/ 1000


        x3_heel = 1323-354
        y3_heel = right_leg.hip.moment.x[x3_heel]/1000

        x3_toe = 1687-354
        y3_toe = right_leg.hip.moment.x[x3_toe]/ 1000

        axs[0].annotate("Heel down 1", xy=(x1_heel,y1_heel), xytext=(x1_heel-5, y1_heel-0.5),
                        arrowprops = dict(arrowstyle="->"))

        axs[0].annotate("Toe off 1", xy=(x1_toe, y1_toe), xytext=(x1_toe - 5, y1_toe - 0.5),
                        arrowprops=dict(arrowstyle="->"))


        axs[0].annotate("Heel down 2", xy=(x2_heel,y2_heel), xytext=(x2_heel-5, y2_heel-0.5),

                        arrowprops = dict(arrowstyle="->"))

        axs[0].annotate("Toe off 2", xy=(x2_toe, y2_toe), xytext=(x2_toe - 5, y2_toe - 0.5),
                        arrowprops=dict(arrowstyle="->"))


        axs[0].annotate("Heel down 3", xy=(x3_heel,y3_heel), xytext=(x3_heel-5, y3_heel-0.5),
                        arrowprops = dict(arrowstyle="->"))

        axs[0].annotate("Toe off 3", xy=(x3_toe, y3_toe), xytext=(x3_toe - 5, y3_toe - 0.5),
                        arrowprops=dict(arrowstyle="->"))

        fig.suptitle(r'Right Leg Joint Moments ($\frac{Nm}{Kg}$)', fontsize=16)


        axs[0].set_title("Hip", fontsize=20, rotation=0)
        axs[1].set_title("Knee", fontsize=20, rotation=0)
        axs[2].set_title("Ankle", fontsize=20, rotation=0)

        axs[0].set_ylabel(r'$\frac{Nm}{Kg}$', fontsize=20, rotation=0, labelpad=30)
        axs[1].set_ylabel(r'$\frac{Nm}{Kg}$', fontsize=20, rotation=0, labelpad=30)
        axs[2].set_ylabel(r'$\frac{Nm}{Kg}$', fontsize=20, rotation=0, labelpad=30)
        axs[2].set_xlabel("Frames (record at 100FPS)", fontsize=20, rotation=0, labelpad=25)


        #axs[0].plot(right_leg.hip.angle.x)
        #
        # axs[1].plot(left_leg.knee.angle.x)
        # axs[1].plot(right_leg.knee.angle.x)
        #
        # axs[2].plot(left_leg.ankle.angle.x)
        # axs[2].plot(right_leg.ankle.angle.x)

    plt.show()


if __name__ == '__main__':
    files = []
    for i in [6]:
        base = "/home/nathaniel/Documents/first_step/data/05_09_21_nathaniel_walking_0"
        files.append(base + str(i) + ".csv")

    get_moments(files)
    #get_angles(files)