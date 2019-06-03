"""
GraphSLAM

UTokyo AILab
author: Ota Satoshi
"""

import numpy as np
import math
import matplotlib.pyplot as plt

#  Simulation parameter
Q = np.diag([0.3, np.deg2rad(1.0), 1.0])**2
M = np.diag([0.3, np.deg2rad(5.0)])**2

DT = 2.0  # time tick [s]
SIM_TIME = 100.0  # simulation time [s]
MAX_STEP = 100
MAX_RANGE = 300.0  # maximum observation range
STATE_SIZE = 3  # State size [x,y,yaw]
OBS_SIZE = 3    # Observation size [d, phi, c]
MAP_SIZE = 3    # Map size [mx, my, s]

show_animation = True

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
        self.hz = np.zeros((len(LANDMARK[:, 0]), 3))

        # Control history
        self.hu = np.zeros((2, 1))

        # Step Count
        self.step_cnt = 1

        # SLAM Covariance
        self.Q = Q # Observation
        self.M = M # Motion

        # Map
        # self.m = [[m0_x, m0_y, s0],
        #           [m1_x, m1_y, s1],
        #           ...
        #           [mN_x, mN_y, sN]]
        self.m =  np.zeros((0, MAP_SIZE))

        # Map history
        # self.hm = [[m0_x[0], m0_y[0], s0[0], m0_x[1], m0_y[1], s0[1], ... ,m0_x[t], m0_y[t], s0[t]],
        #            [m1_x[0], m1_y[0], s1[0], m1_x[1], m1_y[1], s1[1], ... ,m1_x[t], m1_y[t], s1[t]],
        #            ...
        #            [mN_x[0], mN_y[0], sN[0], mN_x[1], mN_y[1], sN[2], ... ,mN_x[t], mN_y[t], sN[t]]]
        self.hm = np.zeros((len(LANDMARK[:, 0]), 3))

        self.lm_num = len(LANDMARK[:, 0])

    def count_up(self):
        self.step_cnt += 1

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

        self.z = np.zeros((0, OBS_SIZE))
        self.m = np.zeros((0, MAP_SIZE))

        for i in range(len(LANDMARK[:, 0])):

            dx = LANDMARK[i, 0] - self.xTrue[0, 0]
            dy = LANDMARK[i, 1] - self.xTrue[1, 0]
            d = math.sqrt(dx**2 + dy**2)
            angle = pi_2_pi(math.atan2(dy, dx)) - self.xTrue[2, 0]

            if d <= MAX_RANGE:

                dn = d + np.random.randn() * self.Q[0, 0]  # add noise
                angle_with_noise = angle + np.random.randn() * self.Q[1, 1]
                zi = np.array([dn, angle_with_noise, i])
                # add all observations
                self.z = np.vstack((self.z, zi))

                mi_x = self.xDead[0, 0] + d * math.cos(self.xDead[2, 0] + angle_with_noise)
                mi_y = self.xDead[1, 0] + d * math.sin(self.xDead[2, 0] + angle_with_noise)
                mi = np.array([mi_x, mi_y, i])
                # add all features
                self.m = np.vstack((self.m, mi))

        # add observation history
        self.hz = np.hstack((self.hz, self.z))
        # add map history
        #self.hm = np.hstack((self.hm, self.m))
        self.hm = np.hstack((self.hm, LANDMARK))

    def graph_slam(self):
        self.count_up()
        InfoM, InfoV = self.edge_init()
        InfoM, InfoV = self.edge_linearize(InfoM, InfoV)
        IMTil, IVTil = self.edge_reduce(InfoM, InfoV)
        xCov, xAve = self.edge_solve(IMTil, IVTil)
        #print(xAve)
        self.extract_pos(xAve)

        return xCov, xAve

    def edge_init(self):

        full_size = self.step_cnt * 3 + self.lm_num * 2
        InfoM = np.zeros((full_size, full_size))    # Full scale Infomation matrix
        InfoV = np.zeros((full_size, 1))                # Full scale Infomation vector

        return InfoM, InfoV

    def edge_linearize(self, InfoM, InfoV):

        # Add Infomation matrix[t=0]
        InfoM[0:3, 0:3] = np.diag([1e+3, 1e+3, 1e+3])

        for t in range(len(self.hxDead[0, :])):
            #print(self.hu)
            if t == 0:
                continue
            #print(self.hu)
            # t = 1, 2, ... ,t
            xNow = self.hxDead[:, t].reshape((3, 1))        # x[t]
            xPast = self.hxDead[:, t-1].reshape((3, 1))     # x[t-1]
            uNow = self.hu[:, t].reshape((2, 1))            # u[t]
            uPast = self.hu[:, t-1].reshape((2, 1))         # u[t-1]

            #plt.plot(xNow[0, 0], xNow[1, 0], "xr")
            #print(xNow)
            R = np.diag([0.1, 0.1, np.deg2rad(1.0)])
            #R = self.compute_R(xPast, uNow)
            G = self.compute_jacob_G(xPast, uNow)
            # Add Edge for all controls
            InfoM, InfoV = self.add_edge_control(InfoM, InfoV, R, G, xNow, xPast, t)
            # Observation[t]
            zNow = self.hz[:, t*3:(t+1)*3]
            # Map[t-1]
            mPast = self.hm[:, (t-1)*3:t*3]

            for i in range(len(self.z[:, 0])):
                #print(self.hz, zt)
                # i th feature[t-1]
                miPast = mPast[i, :].reshape((3, 1))
                #print(miPast)
                dx = miPast[0, 0] - xNow[0, 0]
                dy = miPast[1, 0] - xNow[1, 0]

                H = self.compute_jacob_H(dx, dy)

                # i th observation[t]
                ziNow = zNow[i, :].reshape((3, 1))
                # compute zHat
                d = math.sqrt(dx**2 + dy**2)
                angle = pi_2_pi(math.atan2(dy, dx)) - xNow[2, 0]
                zHat = np.array([d, angle, i]).reshape((3, 1))
                #print(ziNow-zHat)
                # Add Edge for all observation
                InfoM, InfoV = self.add_edge_observe(InfoM, InfoV, H, ziNow, zHat, t, i)
        #print(InfoV)
        return InfoM, InfoV


    def compute_R(self, x, u):

        if u[1, 0] == 0:

            V = np.array([[DT * math.cos(x[2, 0]), 0.0],
                          [DT * math.sin(x[2, 0]), 0.0],
                          [0.0, DT]])
        else:

            r = u[0, 0] / u[1, 0]
            w = u[0, 0] / u[1, 0]**2

            s0 = math.sin(x[2, 0])
            s1 = math.sin(x[2, 0] + DT * u[1, 0])
            c0 = math.cos(x[2, 0])
            c1 = math.cos(x[2, 0] + DT * u[1, 0])

            V = np.array([[-(s0 - s1) / u[1, 0],  w * (s0 - s1) + DT * r * c1],
                          [ (c0 - c1) / u[1, 0], -w * (c0 - c1) + DT * r * s1],
                          [0.0, DT]])

        return V @ self.M @ V.T

    def compute_jacob_G(self, x, u):

        if u[1, 0] == 0:

            G = np.array([[1.0, 0.0, -DT * u[0, 0] * math.sin(x[2, 0])],
                          [0.0, 1.0,  DT * u[0, 0] * math.sin(x[2, 0])],
                          [0.0, 0.0, 1.0]])

        else:

            r = u[0, 0] / u[1, 0]

            G = np.array([[1.0, 0.0, -r * math.cos(x[2, 0]) + r * math.cos(x[2, 0] + DT * u[1, 0])],
                          [0.0, 1.0, -r * math.sin(x[2, 0]) + r * math.sin(x[2, 0] + DT * u[1, 0])],
                          [0.0, 0.0, 1.0]])

        return G

    def compute_jacob_H(self, dx, dy):

        q = dx**2 + dy**2
        d = math.sqrt(q)

        H = np.array([[dx/d, -dy/d,  0.0, -dx/d,  dy/d, 0.0],
                      [dy/q,  dx/q, -1.0, -dy/q, -dx/q, 0.0],
                      [0.0,    0.0,  0.0,   0.0,   0.0, 1.0]])

        return H

    def add_edge_control(self, InfoM, InfoV, R, G, xNow, xPast, t):

        I = np.eye(STATE_SIZE)
        GI = np.hstack((-G, I))

        RInv = np.linalg.inv(R)

        Om = GI.T @ RInv @ GI

        xId1 = (t-1)*3
        xId2 = t*3

        # Add Edge
        InfoM[xId1:xId1+3, xId1:xId1+3] += Om[0:3, 0:3]
        InfoM[xId1:xId1+3, xId2:xId2+3] += Om[0:3, 3:6]
        InfoM[xId2:xId2+3, xId1:xId1+3] += Om[3:6, 0:3]
        InfoM[xId2:xId2+3, xId2:xId2+3] += Om[3:6, 3:6]

        #print(xNow - G @ xPast)
        xi = GI.T @ RInv @ (xNow - G @ xPast)
        #xi = GI.T @ RInv @ np.zeros((3, 1))
        #print(np.linalg.cond(R))
        InfoV[xId1:xId1+3, 0] += xi[0:3, 0]
        InfoV[xId2:xId2+3, 0] += xi[3:6, 0]
        #print(InfoV)
        return InfoM, InfoV

    def add_edge_observe(self, InfoM, InfoV, H, ziNow, zHat, t, i):

        QInv = np.linalg.inv(Q)
        Om = H.T @ QInv @ H

        xlen = len(self.hxDead[0, :])
        xId = t*3
        mId = xlen*3 + i*2
        #print(Om)
        # Add Edge
        InfoM[xId:xId+3, xId:xId+3] += Om[0:3, 0:3]
        InfoM[xId:xId+3, mId:mId+2] += Om[0:3, 3:5]
        InfoM[mId:mId+2, xId:xId+3] += Om[3:5, 0:3]
        InfoM[mId:mId+2, mId:mId+2] += Om[3:5, 3:5]
        #print(self.hm[i, (t-1)*3:t*3])
        # Creat complex state vector
        yt = np.zeros((6, 1))
        yt[0, 0] = self.hxDead[0, t]
        yt[1, 0] = self.hxDead[1, t]
        yt[2, 0] = self.hxDead[2, t]
        yt[3:6, 0] = self.hm[i, (t-1)*3:t*3]
        #yt[3, 0] = self.hm[i, t]
        #print(ziNow - zHat + H @ yt)
        xi = H.T @ QInv @ (ziNow - zHat + H @ yt)
        #xi = H.T @ QInv @ np.zeros((3, 1))
        #print(Q @ QInv)

        InfoV[xId:xId+3, 0] += xi[0:3, 0]
        InfoV[mId:mId+2, 0] += xi[3:5, 0]
        #print(InfoV)
        return InfoM, InfoV

    def edge_reduce(self, InfoM, InfoV):

        xlen = len(self.hxDead[0, :])
        mlen = len(self.z[:, 0])
        xId = xlen*3
        mId = mlen*2
        #print(mlen)
        IMxx = InfoM[0:xId, 0:xId]
        IMxm = InfoM[0:xId, xId:xId+mId]
        IMmx = InfoM[xId:xId+mId, 0:xId]
        IMmm = InfoM[xId:xId+mId, xId:xId+mId]

        IMmmInv = np.linalg.inv(IMmm)

        IMTil = IMxx - IMxm @ IMmmInv @ IMmx

        IVx = InfoV[0:xId, 0]
        IVm = InfoV[xId:xId+mId, 0]

        IVTil = IVx - IMxm @ IMmmInv @ IVm
        #print(np.linalg.cond(IMmm))
        return IMTil, IVTil

    def edge_solve(self, IMTil, IVTil):

        xCov = np.linalg.inv(IMTil)
        xAve = xCov @ IVTil
        #print(np.linalg.cond(IMTil))
        return xCov, xAve

    def extract_pos(self, xAve):

        self.hxSlam = np.zeros((STATE_SIZE, 1))

        for t in range(int(len(xAve) / 3)):

            self.xSlam = xAve[t*3:(t+1)*3].reshape((3, 1))

            self.hxSlam = np.hstack((self.hxSlam, self.xSlam ))
            #print(self.xSlam )

def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v, yawrate]]).T
    return u

def motion_model(x, u):

    if u[1, 0] == 0:

        dx = np.array([[DT * u[0, 0] * math.cos(x[2, 0])],
                       [DT * u[0, 0] * math.sin(x[2, 0])],
                       [0.0]])
    else:

        r = u[0, 0] / u[1, 0]

        dx = np.array([[-r * math.sin(x[2, 0]) + r * math.sin(x[2, 0] + DT * u[1, 0])],
                       [ r * math.cos(x[2, 0]) - r * math.cos(x[2, 0] + DT * u[1, 0])],
                       [DT * u[1, 0]]])

    x = x + dx

    return x

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    print("Running!!")

    time = 0.0
    step = 0

    # RFID positions [x, y, yaw]
    LANDMARK = np.array([[10.0, -2.0, 0],
                        [15.0, 10.0, 1],
                        [3.0, 15.0, 2],
                        [-5.0, 20.0, 3],
                        [-5.0, 5.0, 4]])

    # Creat robot instance
    rov = Robot(Q, M, LANDMARK)

    while MAX_STEP >= step:
        step += 1

        #print(rov.step_cnt)

        u = calc_input()
        rov.action(u)
        rov.observation(LANDMARK)
        _, xAve = rov.graph_slam()

        if show_animation:  # pragma: no cover
            plt.cla()

            plt.plot(LANDMARK[:, 0], LANDMARK[:, 1], "*k")

            for i in range(len(rov.hm[:, 0])):
                plt.plot(rov.hm[i, 3].flatten(), rov.hm[i, 4].flatten(), "xg")
                plt.plot(rov.m[i, 0], rov.m[i, 1], "*y")

            plt.plot(rov.hxTrue[0, :].flatten(),
                     rov.hxTrue[1, :].flatten(), "-b")
            plt.plot(rov.hxDead[0, :].flatten(),
                     rov.hxDead[1, :].flatten(), "-k")
            #plt.plot(xAve[0::3].flatten(),
            #         xAve[1::3].flatten(), "-r")
            plt.plot(rov.xSlam[0], rov.xSlam[1], "xk")
            plt.axis("equal")
            plt.grid(True)
            #plt.title("Time" + str(time)[0:5])
            plt.pause(0.5)

if __name__ == '__main__':
    main()
