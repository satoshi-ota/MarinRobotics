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
Q = np.diag([0.2, np.deg2rad(1.0)])**2
M = np.diag([0.1, np.deg2rad(10.0)])**2

DT = 2.0  # time tick [s]
SIM_TIME = 100.0  # simulation time [s]
MAX_RANGE = 30.0  # maximum observation range
STATE_SIZE = 3  # State size [x,y,yaw]

show_graph_dtime = 20.0  # [s]
show_animation = True

first_obs = True
first_control = True

class Robot():

    def __init__(self, Q, M):

        # Trajectory
        self.xTrue = np.zeros((STATE_SIZE, 1))
        self.xDead = np.zeros((STATE_SIZE, 1))
        self.xSlam = np.zeros((STATE_SIZE, 1))

        # Trajectory history
        self.hxTrue = self.xTrue
        self.hxDead = self.xDead
        self.hxSlam = self.xSlam

        # Observation
        self.z = np.zeros((3, 1))

        # Observation history
        self.hz = np.zeros((3, 1))

        # Control history
        self.hu = np.zeros((2, 1))

        # Step Count
        self.step_cnt = 0

        # SLAM Covariance
        self.Q = Q
        self.M = M


    def count_up(self):
        selt.step_cnt += 1

    def action(self, u):

        self.xTrue = motion_model(self.xTrue, u)
        self.hxTrue = np.hstack((self.hxTrue, self.xTrue))

        # add noise to input
        u_v = u[0, 0] + np.random.randn() * self.M[0, 0]
        u_w = u[1, 0] + np.random.randn() * self.M[1, 1]
        u_with_noise = np.array([[u_v, u_w]]).T

        # add control history
        if first_control:

            first_control = False
            self.hu = u

        else:
            self.hz = np.hstack((self.hz, u))

        self.xDead = motion_model(self.xDead, u_with_noise)
        self.hxDead = np.hstack((self.hxDead, self.xDead))


    def observation(self, LANDMARK):

        for i in range(len(LANDMARK[:, 0])):

            dx = LANDMARK[i, 0] - self.xTrue[0, 0]
            dy = LANDMARK[i, 1] - self.xTrue[1, 0]
            d = math.sqrt(dx**2 + dy**2)
            angle = pi_2_pi(math.atan2(dy, dx)) - self.xTrue[2, 0]
            if d <= MAX_RANGE:
                dn = d + np.random.randn() * self.Q[0, 0]  # add noise
                angle_with_noise = angle + np.random.randn() * self.Q[1, 1]

                zi = np.array([dn, angle, i])
                self.z = np.vstack((z, zi))

        if first_obs:

            first_obs = False
            self.hz = self.z

        else:
            self.hz = np.hstack((self.hz, self.z))


    def graph_slam(self):       # return Sig, xEst
        self.count_up()
        InfoM, InfoV = self.edge_init()
        InfoM, InfoV = self.edge_linearize()
        InfoM, InfoV = self.edge_reduce()
        Sig, xEst = self.edge_solve()

        return Sig, xEst

    def edge_init(self):

        full_size = self.step_cnt + len(LANDMARK[:, 0])
        InfoM = np.zeros((full_size * 3, full_size * 3))    # Full scale Infomation matrix
        InfoV = np.zeros((full_size * 3, 1))               # Full scale Infomation vector

        return InfoM, InfoV

    def edge_linearize(self, InfoM, InfoV):

        # Add Infomation matrix[t=0]
        H[0:3, 0:3] = np.diag([np.inf, np.inf, np.inf])

        # Add Edge for all controls
        for t in range(len(self.hxDead[0, :])):

            if t != 0:
                xn = self.hxDead[:, t]
                xp = self.hxDead[:, t-1]
                ui = self.hu[:, t-1]

                R = self.compute_R(xn)
                G = self.compute_jacob_G(xn, ui)

                H = self.add_edge_control(InfoM, InfoV, R, G, xn, xp, t-1)



        # Add Edge for all observations

    def compute_R(self, xD):

        V = np.array([[DT * math.cos(xD[2, 0]), 0.0],
                      [DT * math.sin(xD[2, 0]), 0.0],
                      [0.0, DT]])

        return V @ self.M @ V.T

    def compute_jacob_G(self, xD, ui):

        G = np.array([[1.0, 0.0, - DT * ui[0, 0] * math.sin(xD[2, 0])],
                      [0.0, 1.0,   DT * ui[1, 0] * math.cos(xD[2, 0])],
                      [0.0, 0.0, 1.0]])

        return G

    def compute_jacob_H(self):

        return H

    def add_edge_control(self, InfoM, InfoV, R, G, xn, xp, i):

        I = np.eye(STATE_SIZE)
        GI = np.hstack((-G, I))

        RInv = np.linalg.inv(R)

        Om = GI.T @ RInv @ GI

        # Add Edge
        InfoM[i:3, i:3] = Om[0:3, 0:3]
        InfoM[i:3, i+3:3] = Om[0:3, 3:3]
        InfoM[i+3:3, i:3] = Om[3:3, 0:3]
        InfoM[i+3:3, i+3:3] = Om[3:3, 3:3]

        xi = GI.T @ RInv @ (xn - G @ xp)

        InfoV[i:3, 1] = xi[0:3, 1]
        InfoV[i+3:3, 1] = xi[3:3, 1]

        return InfoM, InfoV

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

    # Creat robot instance
    rov = Robot()

    # RFID positions [x, y, yaw]
    LANDMARK = np.array([[10.0, -2.0],
                        [15.0, 10.0],
                        [3.0, 15.0],
                        [-5.0, 20.0],
                        [-5.0, 5.0]
                        ])

    dtime = 0.0
    init = False
    while SIM_TIME >= time:

        u = calc_input()
        rov.action(u)
        rov.observation(LANDMARK)
        rov.graph_slam()

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
