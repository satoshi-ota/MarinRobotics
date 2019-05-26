"""
FastSLAM1.0 with EKF
author: Ota Satoshi
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import time as TIME

# Fast SLAM covariance
Q1 = np.diag([0.3, np.deg2rad(2.0)])**2
R1 = np.diag([0.1, np.deg2rad(6.0)])**2
OFFSET_YAWRATE_NOISE1 = 0.01

#  Simulation parameter
Q2 = np.diag([0.3, np.deg2rad(2.0)])**2
R2 = np.diag([0.1, np.deg2rad(6.0)])**2
OFFSET_YAWRATE_NOISE2 = 0.01

DT = 0.1	# time delta
MAX_STEP = 5000 	# maximum step
SIM_TIME = 200.0	# simulation time
MAX_RANGE = 20.0	# maximum observation range
STATE_SIZE = 3 # Robot state(x, y, yaw)
LM_SIZE = 2 # Land mark(x, y)
PARTICLE_NUM = 100 # Nuber of particles
NTH = PARTICLE_NUM / 5.0  # Number of particle for re-sampling

show_animation = True

class Robot:

    def __init__(self, xIni, Q, R, yawNoise):

        # Creat particle
        self.particles = [Particle() for i in range(PARTICLE_NUM)]
        # Initialize particle position
        for i in range(PARTICLE_NUM):
            self.particles[i].x = xIni[0, 0]
            self.particles[i].y = xIni[1, 0]
            self.particles[i].yaw = xIni[2, 0]

        self.xSlam = np.zeros((STATE_SIZE, 1)) # Estimate with fast_slam1.0
        self.xDead = np.zeros((STATE_SIZE, 1)) # Estimate with deadreconing
        self.xTrue = np.zeros((STATE_SIZE, 1)) # True position

        # Initialize position
        self.xSlam = xIni # Estimate with fast_slam1.0
        self.xDead = xIni # Estimate with deadreconing
        self.xTrue = xIni # True position

        # History
        self.hxSlam = self.xSlam
        self.hxDead = self.xDead
        self.hxTrue = self.xTrue

        # Set covariance
        self.Q = Q
        self.R = R

        # Initialize observation
        self.z = np.zeros((3, 0))

        # Yawrate noise
        self.yawNoise = yawNoise

    def store_history(self):

        # Store data history
        x_state = self.xSlam[0: STATE_SIZE]
        self.hxSlam = np.hstack((self.hxSlam, x_state))
        self.hxDead = np.hstack((self.hxDead, self.xDead))
        self.hxTrue = np.hstack((self.hxTrue, self.xTrue))

    def action(self, u):

        # Compute True positions
        self.xTrue = motion_model(self.xTrue, u)

        # Add noise to input
        uReal_v = u[0, 0] + np.random.randn() * self.R[0, 0]
        uReal_w = u[1, 0] + np.random.randn() * self.R[1, 1] + self.yawNoise
        uReal = np.array([uReal_v, uReal_w]).reshape(2, 1)

        # Compute deadreconing
        self.xDead = motion_model(self.xDead, uReal)

    def observation(self, AxTrue):

        # Compute another robot positions
        dx = AxTrue[0, 0] - self.xTrue[0, 0]
        dy = AxTrue[1, 0] - self.xTrue[1, 0]

        d = math.sqrt(dx**2 + dy**2) # Distanse between particle and another robot
        angle = pi_2_pi(math.atan2(dy, dx) - self.xTrue[2, 0]) # Angle for landmark

        if d <= MAX_RANGE:
            # Add noise to observation
            dReal = d + np.random.randn() * self.Q[0, 0]
            angleReal = angle + np.random.randn() * self.Q[1, 1]
            self.z = np.array([dReal, angleReal]).reshape(2, 1) # Observe another robot

        # DEBUG
        #r = self.z[0]
        #b = self.z[1]
        #s = math.sin(pi_2_pi(self.xTrue[2, 0] + b))
        #c = math.cos(pi_2_pi(self.xTrue[2, 0] + b))
        #plt.plot(self.xTrue[0, 0] + r * c, self.xTrue[1, 0] + r * s, "ob")

    def fast_slam(self, u):
        self.predict_particles(u)      # Estimate particles position from input
        self.update_with_observation() # Update with observation
        self.resampling()              # Resampling with weight

    def predict_particles(self, u):

        # Compute particles positions
        for i in range(PARTICLE_NUM):
            xTemp = np.zeros((STATE_SIZE, 1))

            xTemp[0, 0] = self.particles[i].x
            xTemp[1, 0] = self.particles[i].y
            xTemp[2, 0] = self.particles[i].yaw

            uReal = u + (np.random.randn(1, 2) @ self.R).T # Add noise
            xTemp = motion_model(xTemp, uReal) # Compute particle position from motion model

            self.particles[i].x = xTemp[0, 0]
            self.particles[i].y = xTemp[1, 0]
            self.particles[i].yaw = xTemp[2, 0]

    def update_with_observation(self):

        # Update with observation
        for i in range(PARTICLE_NUM):
            # New landmark
            if self.particles[i].lm[0, 2] == False:
                self.particles[i] = self.add_new_lm(self.particles[i])
            # Known landmark
            else:
                self.particles[i] = self.update_landmark(self.particles[i])

    def add_new_lm(self, particle):

        # Extract observation data
        r = self.z[0]
        b = self.z[1]

        # Compute landmark position
        s = math.sin(pi_2_pi(particle.yaw + b))
        c = math.cos(pi_2_pi(particle.yaw + b))

        # Update landmark state
        particle.lm[0, 0] = particle.x + r * c
        particle.lm[0, 1] = particle.y + r * s
        particle.lm[0, 2] = True

        # DEBUG
        #print("Find another Robot")
        #plt.plot(particle.lm[0, 0], particle.lm[0, 1], "oy")

        # Compute Jacobian
        dx = particle.lm[0, 0] - particle.x
        dy = particle.lm[0, 1] - particle.y
        q = dx**2 + dy**2
        d = math.sqrt(q)

        H = np.array([[ dx / d, dy / d],
                      [-dy / q, dx / q]])

        # Initialize covariance
        HInv = np.linalg.inv(H)
        particle.lmP = HInv @ self.Q @ HInv.T

        return particle

    def update_landmark(self, particle):

        # DEBUG
        r = self.z[0]
        b = self.z[1]
        s = math.sin(pi_2_pi(self.xTrue[2, 0] + b))
        c = math.cos(pi_2_pi(self.xTrue[2, 0] + b))
        plt.plot(self.xTrue[0, 0] + r * c, self.xTrue[1, 0] + r * s, "ob")

        # Compute Delta
        lmx = particle.lm[0, 0:2].reshape(2, 1) # (lm_x[t-1], lm_y[t-1])
        lmP = particle.lmP # covariance[t-1]

        # DEBUG
        #plt.plot(lmx[0, 0], lmx[1, 0], "oy")

        dx = lmx[0, 0] - particle.x
        dy = lmx[1, 0] - particle.y

        # Update landmark
        dz = self.creat_inovation_vector(particle, dx, dy)
        H = self.compute_jacobian(dx, dy)

        # Update Observation Covariance
        Qt = H @ lmP @ H.T + self.Q
        QtInv = np.linalg.inv(Qt)

        # Update weight
        w = self.compute_weight(dz, Qt, QtInv)
        particle.w *= w

        # Update EKF
        lmx, lmP = self.update_EKF(lmx, lmP, dz, H, QtInv)

        particle.lm[0, 0:2] = lmx.T
        particle.lmP = lmP

        # DEBUG
        #plt.plot(lmx[0, 0], lmx[1, 0], "oy")

        return particle

    def creat_inovation_vector(self, particle, dx, dy):

        q = dx**2 + dy**2
        d = math.sqrt(q)
        # Compute Inovation Vector
        zHat = np.array([d, math.atan2(dy, dx) - particle.yaw]).reshape(2, 1)
        zHat[1, 0] = pi_2_pi(zHat[1, 0])
        dz = self.z[0:2].reshape(2, 1) - zHat
        dz[1, 0] = pi_2_pi(dz[1, 0])

        return dz

    def compute_jacobian(self, dx, dy):

        q = dx**2 + dy**2
        d = math.sqrt(q)

        H = np.array([[ dx / d, dy / d],
                      [-dy / q, dx / q]])

        return H

    def compute_weight(self, dz, Qt, QtInv):

        # Compute particle weight
        num = math.exp(-0.5 * dz.T @ QtInv @ dz)
        den = 2 * math.pi * math.sqrt(np.linalg.det(Qt))
        w = num / den

        return w

    def update_EKF(self, lmx, lmP, dz, H, QtInv):

        # Compute Kalman Gain
        Kt = lmP @ H.T @ QtInv

        # Update
        lmx = lmx + Kt @ dz
        lmP = (np.eye(LM_SIZE) - (Kt @ H)) @ lmP

        return lmx, lmP

    def resampling(self):

        self.particles = self.normalize_weight()

        pw = []
        for i in range(PARTICLE_NUM):
            pw.append(self.particles[i].w)

        pw = np.array(pw)

        Neff = 1.0 / (pw @ pw.T)  # Effective particle number
        # print(Neff)

        if Neff < NTH:  # resampling
            print("resampling")
            wcum = np.cumsum(pw)
            base = np.cumsum(pw * 0.0 + 1 / PARTICLE_NUM) - 1 / PARTICLE_NUM
            resampleid = base + np.random.rand(base.shape[0]) / PARTICLE_NUM

            inds = []
            ind = 0
            for ip in range(PARTICLE_NUM):
                while ((ind < wcum.shape[0] - 1) and (resampleid[ip] > wcum[ind])):
                    ind += 1
                inds.append(ind)

            tparticles = self.particles[:]
            for i in range(len(inds)):
                self.particles[i].x = tparticles[inds[i]].x
                self.particles[i].y = tparticles[inds[i]].y
                self.particles[i].yaw = tparticles[inds[i]].yaw
                self.particles[i].lm = tparticles[inds[i]].lm[:, :]
                self.particles[i].lmP = tparticles[inds[i]].lmP[:, :]
                self.particles[i].w = 1.0 / PARTICLE_NUM

    def normalize_weight(self):

        sumw = sum([p.w for p in self.particles])

        try:
            for i in range(PARTICLE_NUM):
                self.particles[i].w /= sumw
        except ZeroDivisionError:
            for i in range(PARTICLE_NUM):
                self.particles[i].w = 1.0 / PARTICLE_NUM

            return self.particles

        return self.particles

    def calc_final_state(self):

        self.xSlam = np.zeros((STATE_SIZE, 1))
        self.particles = self.normalize_weight()

        for i in range(PARTICLE_NUM):
            self.xSlam[0, 0] += self.particles[i].w * self.particles[i].x
            self.xSlam[1, 0] += self.particles[i].w * self.particles[i].y
            self.xSlam[2, 0] += self.particles[i].w * self.particles[i].yaw

        self.xSlam[2, 0] = pi_2_pi(self.xSlam[2, 0])



class Particle:

    def __init__(self):

        self.w = 1.0 / PARTICLE_NUM
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # another robot x-y positions
        self.lmPos = np.zeros((1, 2)) # arPos = [[ar_x, ar_y]]
        self.lmStat = np.array([[False]]) # Find another robot or Not
        self.lm = np.hstack((self.lmPos, self.lmStat))

        # landmark position covariance
        self.lmP = np.zeros((LM_SIZE, LM_SIZE))


def calc_input1(time):

    if time <= 1.0:	# wait at first
        v = 0.0
        yawrate = 0.0
    else:
        #v = 1.0		# v[m/s]
        #yawrate = 0.0		# w[rad/s]
        v = np.random.rand()                 # v[m/s]
        yawrate = np.random.randn() * 1.5   # w[rad/s]

    u = np.array([v, yawrate]).reshape(2, 1)

    return u

def calc_input2(time):

    if time <= 1.0:	# wait at first
        v = 0.0
        yawrate = 0.0
    else:
        #v = 1.0		# v[m/s]
        #yawrate = 0.0		# w[rad/s]
        v = np.random.rand()                # v[m/s]
        yawrate = np.random.randn() * 1.5   # w[rad/s]

    u = np.array([v, yawrate]).reshape(2, 1)

    return u

def motion_model(x, u):

	F = np.array([[1.0, 0.0, 0.0],
				  [0.0, 1.0, 0.0],
				  [0.0, 0.0, 1.0]])

	B = np.array([[DT * math.cos(x[2, 0]), 0.0],
				  [DT * math.sin(x[2, 0]), 0.0],
				  [0.0, DT]])

	# motion model
	x = F @ x + B @ u

	x[2, 0] = pi_2_pi(x[2, 0])

	return x

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    print("running...")

    # Start Position
    xIni1 = np.array([0.0,  1.0, 0.0]).reshape(3, 1)
    xIni2 = np.array([0.0, -1.0, 0.0]).reshape(3, 1)

    r1 = Robot(xIni1, Q1, R1, OFFSET_YAWRATE_NOISE1)    # Instance robot1
    r2 = Robot(xIni2, Q2, R2, OFFSET_YAWRATE_NOISE2)    # Instance robot2

    time = 0.0;
    step = 0;

    while step <= MAX_STEP:
        step += 1
        time += DT

        u1 = calc_input1(time)
        u2 = calc_input2(time)

        if show_animation:  # pragma: no cover
            plt.cla()

            # Robot1
            r1.action(u1)               # Move robot1
            r1.observation(r2.xTrue)    # Observation robot1
            r1.fast_slam(u1)            # Slam robot1
            r1.calc_final_state()
            r1.store_history()

            for i in range(PARTICLE_NUM):
                # Particles1
                plt.plot(r1.particles[i].x, r1.particles[i].y, ".", c = "#5EC84E")
                plt.plot(r1.particles[i].lm[:, 0], r1.particles[i].lm[:, 1], "xb")

            # Robot1
            plt.plot(r1.hxTrue[0, :], r1.hxTrue[1, :], "-b")
            plt.plot(r1.hxDead[0, :], r1.hxDead[1, :], "-k")
            plt.plot(r1.hxSlam[0, :], r1.hxSlam[1, :], "-r")
            plt.plot(r1.xSlam[0], r1.xSlam[1], "Xk")

            # Robot2
            r2.action(u2)               # Move robot2
            r2.observation(r1.xTrue)    # Observation robot2
            r2.fast_slam(u2)            # Slam robot2
            r2.calc_final_state()
            r2.store_history()

            for i in range(PARTICLE_NUM):
                # Particles2
                plt.plot(r2.particles[i].x, r2.particles[i].y, ".", c = "#5EC84E")
                plt.plot(r2.particles[i].lm[:, 0], r2.particles[i].lm[:, 1], "xr")

            # Robot2
            plt.plot(r2.hxTrue[0, :], r2.hxTrue[1, :], "-b")
            plt.plot(r2.hxDead[0, :], r2.hxDead[1, :], "-k")
            plt.plot(r2.hxSlam[0, :], r2.hxSlam[1, :], "-r")
            plt.plot(r2.xSlam[0], r2.xSlam[1], "Xk")

            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.1)

if __name__ == '__main__':
    main()
