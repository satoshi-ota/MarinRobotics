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
MAX_RANGE = 300.0  # maximum observation range
STATE_SIZE = 3  # State size [x,y,yaw]
OBS_SIZE = 3    # Observation size [d, phi, c]

show_graph_dtime = 20.0  # [s]
show_animation = True

first_obs = True
first_control = True

class Robot():

    def __init__(self, Q, M, LANDMARK):

        # Trajectory
        # self.x = [[ x[0]],
        #           [ y[0]],
        #           [th[0]]]
        self.xTrue = np.zeros((STATE_SIZE, 1))
        self.xDead = np.zeros((STATE_SIZE, 1))
        self.xSlam = np.zeros((STATE_SIZE, 1))

        # Trajectory history
        # self.hx = [[ x[0],  x[1], ... , x[t]],
        #            [ y[0],  y[1], ... , y[t]],
        #            [th[0], th[1], ... , th[t]]
        self.hxTrue = self.xTrue
        self.hxDead = self.xDead
        self.hxSlam = self.xSlam

        # Observation
        # self.z = [[d0, phi0, c0],
        #           [d1, phi1, c1],
        #           ...
        #           [dN, phiN, cN]]
        self.z = np.zeros((0, OBS_SIZE))

        # Observation history
        # self.hz = [[d0[0], phi0[0], c0[0], d0[1], phi0[0], c0[0], ... ,d0[t], phi0[t], c0[t]],
        #            [d1[0], phi1[0], c1[0], d1[1], phi1[1], c1[1], ... ,d1[t], phi1[t], c1[t]],
        #            ...
        #            [dN[0], phiN[0], cN[0], dN[1], phiN[1], cN[1], ... ,dN[t], phiN[t], cN[t]]]
        self.hz = np.zeros((len(LANDMARK[:, 0]), 0))

        # Control history
        self.hu = np.zeros((2, 0))

        # Step Count
        self.step_cnt = 0

        # SLAM Covariance
        self.Q = Q
        self.M = M

        # Map
        # self.m = [[m0_x, m0_y],
        #           [m1_x, m1_y],
        #           ...
        #           [mN_x, mN_y]]
        self.m =  np.zeros((len(LANDMARK[:, 0]), 2))

        # Map history
        # self.hm = [[m0_x[0], m0_y[0], m0_x[1], m0_y[1], ... ,m0_x[t], m0_y[t]],
        #            [m1_x[0], m1_y[0], m1_x[1], m1_y[1], ... ,m1_x[t], m1_y[t]],
        #            ...
        #            [mN_x[0], mN_y[0], mN_x[1], mN_y[1], ... ,mN_x[t], mN_y[t]]]
        self.hm = np.zeros((len(LANDMARK[:, 0]), 0))

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
        self.hu = np.hstack((self.hu, u))

        # dead reckoning
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

                # add all observations
                self.z = np.vstack((z, zi))

                mi_x = self.xDead[0, 0] + d * math.cos(angle_with_noise)
                mi_y = self.xDead[1, 0] + d * math.sin(angle_with_noise)
                mi = np.array([m_x, m_y])

                # add all features
                sekf.m = np.vstack((m, mi))

        # add observation history
        self.hz = np.hstack((self.hz, self.z))

        # add map history
        self.hm = np.hstack((self.hm, self.m))

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

                InfoM, InfoV = self.add_edge_control(InfoM, InfoV, R, G, xn, xp, t-1)

        # Add Edge for all observations
        for t in range(len(self.hz[0, :]) / 3):

            # Features[t]
            zt = self.hz[:, t:(t+1)*3]

            # Map[t-1]
            mt = self.hm[:, (t-1)*2:t*2]

            for i in range(len(self.z[:, 0])):

                # i th feature[t-1]
                mi = mt[t, :]

                xn = self.hxDead[:, t]

                dx = mt[0, 0] - xn[0, 0]
                dy = mt[0, 1] - xn[1, 0]

                H = self.compute_jacob_H(dx, dy)

                # i th observation[t]
                zi = zt[i, :]

                # compute zHat
                d = math.sqrt(dx**2 + dy**2)
                angle = pi_2_pi(math.atan2(dy, dx)) - xn[2, 0]

                zHat = np.array([d, angle, i])

                InfoM, InfoV = self.add_edge_observe(InfoM, InfoV, H, zi, zHat, t, i)

        return InfoM, InfoV


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

    def compute_jacob_H(self, dx, dy):

        q = dx**2 + dy**2
        d = math.sqrt(q)

        H = np.array([[dx/d, -dy/d,  0.0, -dx/d,  dy/d, 0.0],
                      [dy/q,  dx/q, -1.0, -dy/q, -dx/q, 0.0],
                      [0.0,    0.0,  0.0,   0.0,   0.0, 1.0]])

        return H

    def add_edge_control(self, InfoM, InfoV, R, G, xn, xp, i):

        I = np.eye(STATE_SIZE)
        GI = np.hstack((-G, I))

        RInv = np.linalg.inv(R)

        Om = GI.T @ RInv @ GI

        # Add Edge
        InfoM[i*3:(i+1)*3, i*3:(i+1)*3] = Om[0:3, 0:3]
        InfoM[i*3:(i+1)*3, (i+1)*3:(i+2)*3] = Om[0:3, 3:6]
        InfoM[(i+1)*3:(i+2)*3, i*3:(i+1)*3] = Om[3:6, 0:3]
        InfoM[(i+1)*3:(i+2)*3, (i+1)*3:(i+2)*3] = Om[3:6, 3:6]

        xi = GI.T @ RInv @ (xn - G @ xp)

        InfoV[i:(i+1)*3, 1] = xi[0:3, 1]
        InfoV[(i+1)*3:(i+2)*3, 1] = xi[3:6, 1]

        return InfoM, InfoV

    def add_edge_observe(self, InfoM, InfoV, H, zi, zHat, t, i):

        QInv = np.linalg.inv(Q)
        Om = H.T @ QInv @ H

        xlen = len(self.hxDead[0, :])

        # Add Edge
        InfoM[t*3:(t+1)*3, t*3:(t+1)*3] = Om[0:3, 0:3]
        InfoM[t*3:(t+1)*3, xlen*3+i*3:xlen*3+(i+1)*3] = Om[0:3, 3:6]
        InfoM[xlen*3+i*3:xlen*3+(i+1)*3, t*3:(t+1)*3] = Om[3:6, 0:3]
        InfoM[xlen*3+i*3:xlen*3+(i+1)*3, xlen*3+i*3:xlen*3*(i+1)*3] = Om[3:6, 3:6]

        # Creat complex state vector
        yt = np.array([self.hxDead[:, t], self.hm[i, t]])

        xi = H.T @ QInv @ (zi.T - zHat.T + H @ y.T)

        InfoV[] = xi[]
        InfoV[] = xi[]

        return InfoM, InfoV

    def edge_reduce(self):

    def edge_solve(self):

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
