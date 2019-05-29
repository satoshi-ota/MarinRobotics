"""
GraphSLAM

UTokyo AILab
author: Ota Satoshi
"""

import numpy as np
import math
import copy
import itertools
import matplotlib.pyplot as plt


#  Simulation parameter
Qsim = np.diag([0.2, np.deg2rad(1.0)])**2
Rsim = np.diag([0.1, np.deg2rad(10.0)])**2

DT = 2.0  # time tick [s]
SIM_TIME = 100.0  # simulation time [s]
MAX_RANGE = 30.0  # maximum observation range
STATE_SIZE = 3  # State size [x,y,yaw]

# Covariance parameter of Graph Based SLAM
C_SIGMA1 = 0.1
C_SIGMA2 = 0.1
C_SIGMA3 = np.deg2rad(1.0)

MAX_ITR = 20  # Maximum iteration

show_graph_dtime = 20.0  # [s]
show_animation = True

class Robot():

    def __init__(self):

        # Trajectory
        self.xTrue = np.zeros((STATE_SIZE, 1))
        self.xDead = np.zeros((STATE_SIZE, 1))
        self.xSlam = np.zeros((STATE_SIZE, 1))

        # Trajectory history
        self.hxTrue = np.zeros((STATE_SIZE, 1))
        self.hxDead = np.zeros((STATE_SIZE, 1))

        # Observation history
        self.hz = np.zeros((3, 1))

        # Step Count
        self.step_cnt = 0

    def count_up(self):
        selt.step_cnt += 1

    def observation(self, u, LANDMARK):

        self.xTrue = motion_model(self.xTrue, u)

        # add noise to gps x-y
        z = np.zeros((0, 4))

        for i in range(len(LANDMARK[:, 0])):

            dx = LANDMARK[i, 0] - self.xTrue[0, 0]
            dy = LANDMARK[i, 1] - self.xTrue[1, 0]
            d = math.sqrt(dx**2 + dy**2)
            angle = pi_2_pi(math.atan2(dy, dx)) - xTrue[2, 0]
            if d <= MAX_RANGE:
                dn = d + np.random.randn() * Qsim[0, 0]  # add noise
                angle_with_noise = angle + np.random.randn() * Qsim[1, 1]

                zi = np.array([dn, angle, i])
                z = np.vstack((z, zi))

        # add noise to input
        u_v = u[0, 0] + np.random.randn() * Rsim[0, 0]
        u_w = u[1, 0] + np.random.randn() * Rsim[1, 1]
        u_with_noise = np.array([[u_v, u_w]]).T

        self.xDead = motion_model(self.xDead, u_with_noise)

        return z, u_with_noise

    def graph_slam(self):
        self.count_up()
        H, xi = self.edge_init()
        H, xi = self.edge_linearize()
        H, xi = self.edge_reduce()
        H, xi = self.edge_solve()

    def edge_init(self):

        full_size = self.step_cnt + len(LANDMARK[:, 0])
        H = np.zeros((full_size * 3, full_size * 3))    # Full scale Infomation matrix
        xi = np.zeros((full_size * 3, 1))               # Full scale Infomation vector

        return H, xi

    def edge_linearize(self):

    def add_initial_attitude(self, H, xi):

        omega_0 = np.diag([np.inf, np.inf, np.inf])


    def compute_jacobian(self):

    def add_edge(self):

    def edge_reduce(self):

    def edge_solve(self):

class Edge():

    def __init__(self):
        self.omega = np.zeros((3, 3))  # Information matrix
        self.xi = np.zeros((3, 1))     # Infomation vector


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v, yawrate]]).T
    return u

def motion_model(x, u):

    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = F @ x + B @ u

    return x


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    print("Running!!")

    time = 0.0

    # RFID positions [x, y, yaw]
    LANDMARK = np.array([[10.0, -2.0],
                        [15.0, 10.0],
                        [3.0, 15.0],
                        [-5.0, 20.0],
                        [-5.0, 5.0]
                        ])

    # State Vector [x y yaw v]'
    xTrue = np.zeros((STATE_SIZE, 1))
    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxTrue = []
    hxDR = []
    hz = []
    dtime = 0.0
    init = False
    while SIM_TIME >= time:

        if not init:
            hxTrue = xTrue
            hxDR = xTrue
            init = True
        else:
            hxDR = np.hstack((hxDR, xDR))
            hxTrue = np.hstack((hxTrue, xTrue))

        time += DT
        dtime += DT
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID)

        hz.append(z)

        if dtime >= show_graph_dtime:
            x_opt = graph_based_slam(hxDR, hz)
            dtime = 0.0

            if show_animation:  # pragma: no cover
                plt.cla()

                plt.plot(RFID[:, 0], RFID[:, 1], "*k")

                plt.plot(hxTrue[0, :].flatten(),
                         hxTrue[1, :].flatten(), "-b")
                plt.plot(hxDR[0, :].flatten(),
                         hxDR[1, :].flatten(), "-k")
                plt.plot(x_opt[0, :].flatten(),
                         x_opt[1, :].flatten(), "-r")
                plt.axis("equal")
                plt.grid(True)
                plt.title("Time" + str(time)[0:5])
                plt.pause(1.0)


if __name__ == '__main__':
    main()
