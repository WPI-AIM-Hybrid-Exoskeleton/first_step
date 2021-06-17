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

def plot_model(angles):

    runner = TPGMMRunner.TPGMMRunner("first_step")
    path = runner.run()
    fig, axs = plt.subplots(3,2)



    for p in angles["hip_left"]:
        t = np.linspace(0, 1, len(p))
        axs[0, 0].plot(t, p)

    for p in angles["knee_left"]:
        t = np.linspace(0, 1, len(p))
        axs[1, 0].plot(t, p)

    for p in angles["ankle_left"]:
        t = np.linspace(0, 1, len(p))
        axs[2, 0].plot(t, p)


    for p in angles["hip_right"]:
        t = np.linspace(0, 1, len(p))
        axs[0, 1].plot(t, p)

    for p in angles["knee_right"]:
        t = np.linspace(0, 1, len(p))
        axs[1, 1].plot(t, p)

    for p in angles["ankle_right"]:
        t = np.linspace(0, 1, len(p))
        axs[2, 1].plot(t, p)



    t = np.linspace(0,1, len(path[:, 0]))
    axs[0,0].plot(t, path[:, 0], linewidth=4)
    axs[1,0].plot(t, path[:, 1], linewidth=4)
    axs[2,0].plot(t, path[:, 2], linewidth=4)

    axs[0, 1].plot(t, path[:, 3], linewidth=4)
    axs[1, 1].plot(t, path[:, 4], linewidth=4)
    axs[2, 1].plot(t, path[:, 5], linewidth=4)



    plt.show()



def get_joint_angles(files, indecies, sides):
    """
    :param files:
    :param indecies:
    :param sides:
    :param lables:
    :return:
    """
    angles = {}
    angles["hip"] = []
    angles["knee"] = []
    angles["ankle"] = []

    angles2 = {}
    angles2["Rhip"] = []
    angles2["Rknee"] = []
    angles2["Rankle"] = []
    angles2["Lhip"] = []
    angles2["Lknee"] = []
    angles2["Lankle"] = []

    samples = []
    for file, i, side in zip(files, indecies, sides):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        trial.create_index_seperators()
        body = trial.get_joint_trajectories()
        if side == "L":
            hip_angle = body.left.hip[i].angle.x.data
            knee_angle = body.left.knee[i].angle.x.data
            ankle_angle = body.left.ankle[i].angle.x.data
        else:
            hip_angle = body.right.hip[i].angle.x.data
            knee_angle = body.right.knee[i].angle.x.data
            ankle_angle = body.right.ankle[i].angle.x.data


        angles["hip"].append(hip_angle)
        angles["knee"].append(knee_angle)
        angles["ankle"].append(ankle_angle)
        samples.append(len(hip_angle))

    sample_size = min(samples)

    # for i in range(len(files)):
    #     angles2["hip"].append(signal.resample(angles["hip"][i], sample_size))
    #     angles2["knee"].append(signal.resample(angles["knee"][i], sample_size))
    #     angles2["ankle"].append(signal.resample(angles["ankle"][i], sample_size))

    for i in range(len(files)):
        angles2["Lhip"].append(np.deg2rad(angles["hip"][i]))
        angles2["Lknee"].append( np.deg2rad(angles["knee"][i]))
        angles2["Lankle"].append( -(np.deg2rad(angles["ankle"][i])) )

        angles2["Rhip"].append( np.deg2rad(np.flip(angles["hip"][i])))
        angles2["Rknee"].append( np.deg2rad(np.flip(angles["knee"][i])))
        angles2["Rankle"].append(np.deg2rad(np.flip(angles["ankle"][i])))

    return angles2



def extract_index(y):

    peaks, _ = find_peaks(y, height=50)

    end = peaks[0]

    while y[end] > 15:
        end += 1

    start = 0
    while y[start] < 15:
        start += 1
    return start, end


def compare_angle(angles):

    fig, axs = plt.subplots(3,2)

    for demo in angles["hip_left"]:
        axs[0,0].plot(demo)

    for demo in angles["knee_left"]:
        axs[1,0].plot(demo)

    for demo in angles["ankle_left"]:
        axs[2,0].plot(demo)

    for demo in angles["hip_right"]:
        axs[0, 1].plot(demo)

    for demo in angles["knee_right"]:
        axs[1, 1].plot(demo)

    for demo in angles["ankle_right"]:
        axs[2, 1].plot(demo)


    plt.show()

def extract_joints(files):
    angles = {}

    angles["hip_left"] = []
    angles["knee_left"] = []
    angles["ankle_left"] = []

    angles["hip_right"] = []
    angles["knee_right"] = []
    angles["ankle_right"] = []

    for file in files:
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        ltoe = trial.vicon.get_markers().get_marker("LTOE")
        rtoe = trial.vicon.get_markers().get_marker("RTOE")
        left_leg = trial.vicon.get_model_output().get_left_leg()
        right_leg = trial.vicon.get_model_output().get_right_leg()

        ltoe_z = []
        rtoe_z = []

        for pl, pr in zip(ltoe, rtoe):
            ltoe_z.append(pl.z)
            rtoe_z.append(pr.z)

        index = []
        y = np.array(rtoe_z - min( rtoe_z))
        start, end = extract_index(y)

        hip_left = np.array(left_leg.hip.angle.x[start:end])
        hip_right = np.array(right_leg.hip.angle.x[start:end])

        knee_left = np.array(left_leg.knee.angle.x[start:end])
        knee_right = np.array(right_leg.knee.angle.x[start:end])

        ankle_left = np.array(left_leg.ankle.angle.x[start:end])
        ankle_right = np.array(right_leg.ankle.angle.x[start:end])

        angles["hip_left"].append(np.deg2rad(hip_left))
        angles["knee_left"].append(np.deg2rad(knee_left))
        angles["ankle_left"].append(np.deg2rad(ankle_left))

        angles["hip_right"].append(np.deg2rad(hip_right))
        angles["knee_right"].append(np.deg2rad(knee_right))
        angles["ankle_right"].append(np.deg2rad(ankle_right))
    return angles


def train_model(angles):
    trainer = TPGMMTrainer.TPGMMTrainer(demo=[angles["hip_left"], angles["knee_left"], angles["ankle_left"], angles["hip_right"], angles["knee_right"], angles["ankle_right"]],
                                        file_name="first_step",
                                        n_rf=15,
                                        dt=0.01,
                                        reg=[1e-3],
                                        poly_degree=[7, 7, 7, 7, 7, 7])

    trainer.train()



def compare_legs(file):

    fig, axs = plt.subplots(3)
    for file in files:
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        ltoe = trial.vicon.get_markers().get_marker("LTOE")
        rtoe = trial.vicon.get_markers().get_marker("RTOE")
        left_leg = trial.vicon.get_model_output().get_left_leg()
        right_leg = trial.vicon.get_model_output().get_right_leg()



        axs[0].plot(left_leg.hip.angle.x)
        axs[0].plot(right_leg.hip.angle.x)

        axs[1].plot(left_leg.knee.angle.x)
        axs[1].plot(right_leg.knee.angle.x)

        axs[2].plot(left_leg.ankle.angle.x)
        axs[2].plot(right_leg.ankle.angle.x)

    plt.show()




if __name__ == '__main__':

    files = []
    for i in [0,2,3,4]:
        base = "/home/nathaniel/Documents/first_step/data/05_09_21_nathaniel_sit2stand_0"
        files.append(base + str(i) + ".csv")

    compare_legs(files)
