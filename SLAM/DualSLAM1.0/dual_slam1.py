"""
FastSLAM1.0 with EKF
author: Ota Satoshi
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# Fast SLAM covariance
Q1 = np.diag([0.3, np.deg2rad(4.0)])**2
R1 = np.diag([0.1, np.deg2rad(8.0)])**2
OFFSET_YAWRATE_NOISE1 = 0.01

#  Simulation parameter
Q2 = np.diag([0.3, np.deg2rad(2.0)])**2
R2 = np.diag([0.5, np.deg2rad(10.0)])**2
OFFSET_YAWRATE_NOISE2 = 0.01

DT = 0.1	# time delta
MAX_STEP = 3000 	# maximum step
SIM_TIME = 20.0	# simulation time
MAX_RANGE = 20.0	# maximum observation range
STATE_SIZE = 3 # Robot state(x, y, yaw)
LM_SIZE = 2 # Land mark(x, y)
PARTICLE_NUM = 100 # Nuber of particles
NTH = PARTICLE_NUM / 2.0  # Number of particle for re-sampling

show_animation = True

class Robot:

    def __init__(self, xIni, Q, R):

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

    def store_history(self, xTrue, xDead, xSlam):

        # store data history
        self.hxSlam = np.hstack((hxSlam, x_state))
        self.hxDead = np.hstack((hxDead, xDead))
        self.hxTrue = np.hstack((hxTrue, xTrue))

    def action(self, u):

        # Compute True positions
        self.xTrue = motion_model(self.xTrue, u)

        # Add noise to input
        uReal_v = u[0, 0] + np.random.randn() * self.R[0, 0]
        uReal_w = u[1, 0] + np.random.randn() * self.R[1, 1] + OFFSET_YAWRATE_NOISE
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

    def fast_slam(self, u):
        self.predict_particles(u)      # Estimate particles position from input
        self.update_with_observation() # Update with observation
        self.resampling()              # Resampling with weight

    def predict_particles(self, u):

        # calculate particles positions
        for i in range(PARTICLE_NUM):
            xTemp = np.zeros((STATE_SIZE, 1))

            xTemp[0, 0] = self.particles[i].x
            xTemp[1, 0] = self.particles[i].y
            xTemp[2, 0] = self.particles[i].yaw

            uN = u + (np.random.randn(1, 2) @ self.R).T # add noise
            xTemp = motion_model(xTemp, uN) # calculate particle position from motion model

            self.particles[i].x = xTemp[0, 0]
            self.particles[i].y = xTemp[1, 0]
            self.particles[i].yaw = xTemp[2, 0]

    def update_with_observation(self):

        # Update with observation
        for i in range(PARTICLE_NUM):
            # New landmark
            if self.particles[i].lm[0, 2] == False:
                self.particles[i] = self.add_new_lm()
            # Known landmark
            else:
                w = compute_weight(self.particles[i], self.z, self.Q)
                self.particles[i].w *= w
                self.particles[i] = self.update_landmark(self.particles[i], self.z, self.Q)

    def add_new_lm(self):

        # extract observation data
        r = self.z[0]
        b = self.z[1]

        # calculate landmark position
        s = math.sin(pi_2_pi(self.particle.yaw + b))
        c = math.cos(pi_2_pi(self.particle.yaw + b))

        # update landmark state
        self.particle.lm[lm_id, 0] = particle.x + r * c
        self.particle.lm[lm_id, 1] = particle.y + r * s
        self.particle.lm[lm_id, 2] = True

        # calculate Jacobian
        dx = self.particle.lm[lm_id, 0] - self.particle.x
        dy = self.particle.lm[lm_id, 1] - self.particle.y
        q = dx**2 + dy**2
        d = math.sqrt(q)

        H = np.array([[ dx / d, dy / d],
                      [-dy / q, dx / q]])

        # initialize covariance
        HInv = np.linalg.inv(H)
        particle.lmP[lm_id * 2 : (lm_id + 1) * 2] = HInv @ Q @ HInv.T

        return particle




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


def action(xTrue, xDead, u):

    # calculate True positions
    xTrue = motion_model(xTrue, u)

    # add noise to input
    uN_v = u[0, 0] + np.random.randn() * Rsim[0, 0]
    uN_w = u[1, 0] + np.random.randn() * Rsim[1, 1] + OFFSET_YAWRATE_NOISE
    uN = np.array([uN_v, uN_w]).reshape(2, 1)

    # calculate deadreconing
    xDead = motion_model(xDead, uN)

    return xTrue, xDead

def fast_slam(particles, u, zN):

    particles = predict_particles(particles, u) # estimate particles position from input

    particles = update_with_observation(particles, zN) #update with observation

    particles = resampling(particles)

    return particles


def normalize_weight(particles):

    sumw = sum([p.w for p in particles])

    try:
        for i in range(PARTICLE_NUM):
            particles[i].w /= sumw
    except ZeroDivisionError:
        for i in range(PARTICLE_NUM):
            particles[i].w = 1.0 / PARTICLE_NUM

        return particles

    return particles


def calc_final_state(particles):

    xSlam = np.zeros((STATE_SIZE, 1))

    particles = normalize_weight(particles)

    for i in range(PARTICLE_NUM):
        xSlam[0, 0] += particles[i].w * particles[i].x
        xSlam[1, 0] += particles[i].w * particles[i].y
        xSlam[2, 0] += particles[i].w * particles[i].yaw

    xSlam[2, 0] = pi_2_pi(xSlam[2, 0])
    #  print(xSlam)

    return xSlam


def predict_particles(particles, u):

    # calculate particles positions
    for i in range(PARTICLE_NUM):
        xTemp = np.zeros((STATE_SIZE, 1))

        xTemp[0, 0] = particles[i].x
        xTemp[1, 0] = particles[i].y
        xTemp[2, 0] = particles[i].yaw

        uN = u + (np.random.randn(1, 2) @ R1).T # add noise
        xTemp = motion_model(xTemp, uN) # calculate particle position from motion model

        particles[i].x = xTemp[0, 0]
        particles[i].y = xTemp[1, 0]
        particles[i].yaw = xTemp[2, 0]

    return particles


def add_new_lm(particle, zN, Q):

    # extract observation data
    r = zN[0]
    b = zN[1]
    lm_id = int(zN[2])

    # calculate landmark position
    s = math.sin(pi_2_pi(particle.yaw + b))
    c = math.cos(pi_2_pi(particle.yaw + b))

    # update landmark state
    particle.lm[lm_id, 0] = particle.x + r * c
    particle.lm[lm_id, 1] = particle.y + r * s
    particle.lm[lm_id, 2] = True

    # calculate Jacobian
    dx = particle.lm[lm_id, 0] - particle.x
    dy = particle.lm[lm_id, 1] - particle.y
    q = dx**2 + dy**2
    d = math.sqrt(q)

    H = np.array([[ dx / d, dy / d],
                  [-dy / q, dx / q]])

    # initialize covariance
    HInv = np.linalg.inv(H)
    particle.lmP[lm_id * 2 : (lm_id + 1) * 2] = HInv @ Q @ HInv.T

    return particle


def update_landmark(particle, zN, Q):

    lm_id = int(zN[2])
    lmx = np.array(particle.lm[lm_id, 0:2]).reshape(2, 1) # (lm_x[t-1], lm_y[t-1])
    lmP = np.array(particle.lmP[lm_id * 2 : (lm_id + 1) * 2, :]) # covariance[t-1]

    # calculate Jacobian
    dx = lmx[0, 0] - particle.x
    dy = lmx[1, 0] - particle.y
    q = dx**2 + dy**2
    d = math.sqrt(q)

    H = np.array([[ dx / d, dy / d],
                  [-dy / q, dx / q]])

    # calculate Inovation Vector
    zNh = np.array([d, math.atan2(dy, dx) - particle.yaw]).reshape(2, 1)
    zNh[1, 0] = pi_2_pi(zNh[1, 0])
    dz = zN[0:2].reshape(2, 1) - zNh # Inovation Vector
    dz[1, 0] = pi_2_pi(dz[1, 0])

    lmx, lmP = update_with_EKF(lmx, lmP, dz, H, Q)

    particle.lm[lm_id, 0:2] = lmx.T
    particle.lmP[lm_id * 2 : (lm_id + 1) * 2, :] = lmP
    #print("update")
    return particle


def compute_weight(particle, zN, Q):

    lm_id = int(zN[2])
    lmx = np.array(particle.lm[lm_id, 0:2]).reshape(2, 1) # (lm_x, lm_y)
    lmP = np.array(particle.lmP[lm_id * 2 : (lm_id + 1) * 2, :]) # covariance

    # calculate Jacobian
    dx = lmx[0, 0] - particle.x
    dy = lmx[1, 0] - particle.y
    q = dx**2 + dy**2
    d = math.sqrt(q)

    H = np.array([[ dx / d, dy / d],
                  [-dy / q, dx / q]])

    # calculate Inovation Vector
    zNh = np.array([d, math.atan2(dy, dx) - particle.yaw]).reshape(2, 1)
    zNh[1, 0] = pi_2_pi(zNh[1, 0])
    dz = zN[0:2].reshape(2, 1) - zNh # Inovation Vector
    dz[1, 0] = pi_2_pi(dz[1, 0])
    #print(H)
    Qt = H @ lmP @ H.T + Q   # observation covariance
    QtInv = np.linalg.inv(Qt)   # Q^-1

    # compute particle wight
    num = math.exp(-0.5 * dz.T @ QtInv @ dz)
    den = 2 * math.pi * math.sqrt(np.linalg.det(Qt))
    w = num / den

    return w

def update_with_EKF(lmx, lmP, dz, H, Q):

    Qt = H @ lmP @ H.T + Q   # observation covariance
    Kt = lmP @ H.T @ np.linalg.inv(Qt)   # calculate kalman gain

    lmx = lmx + Kt @ dz    # update average
    lmP = (np.eye(LM_SIZE) - (Kt @ H)) @ lmP  # update covariance

    return lmx, lmP

def update_with_observation(particles, zN):

    #update with observation
    for ip in range(PARTICLE_NUM):
        # new landmark
        if particles[ip].lm[0, 2] == False:
            particles[ip] = add_new_lm(particles[ip], zN, Q)
        # known landmark
        else:
            w = compute_weight(particles[ip], zN, Q)
            particles[ip].w *= w
            particles[ip] = update_landmark(particles[ip], zN, Q)

    return particles


def resampling(particles):
    """
    low variance re-sampling
    """

    particles = normalize_weight(particles)

    pw = []
    for i in range(PARTICLE_NUM):
        pw.append(particles[i].w)

    pw = np.array(pw)

    Neff = 1.0 / (pw @ pw.T)  # Effective particle number
    # print(Neff)

    if Neff < NTH:  # resampling
        wcum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / PARTICLE_NUM) - 1 / PARTICLE_NUM
        resampleid = base + np.random.rand(base.shape[0]) / PARTICLE_NUM

        inds = []
        ind = 0
        for ip in range(PARTICLE_NUM):
            while ((ind < wcum.shape[0] - 1) and (resampleid[ip] > wcum[ind])):
                ind += 1
            inds.append(ind)

        tparticles = particles[:]
        for i in range(len(inds)):
            particles[i].x = tparticles[inds[i]].x
            particles[i].y = tparticles[inds[i]].y
            particles[i].yaw = tparticles[inds[i]].yaw
            particles[i].lm = tparticles[inds[i]].lm[:, :]
            particles[i].lmP = tparticles[inds[i]].lmP[:, :]
            particles[i].w = 1.0 / PARTICLE_NUM

    return particles


def calc_input1(time):

	if time <= 1.0:	# wait at first
		v = 0.0
		yawrate = 0.0
	else:
		v = 1.0		# v[m/s]
		yawrate = 0.1		# w[rad/s]

	u = np.array([v, yawrate]).reshape(2, 1)

	return u

def calc_input2(time):

	if time <= 1.0:	# wait at first
		v = 0.0
		yawrate = 0.0
	else:
		v = 1.0		# v[m/s]
		yawrate = 0.1		# w[rad/s]

	u = np.array([v, yawrate]).reshape(2, 1)

	return u

def observation(xTrue, xLm):

    # calculate landmark positions
    zN = np.zeros((3, 0))
    dx = xLm[0, 0] - xTrue[0, 0]
    dy = xLm[1, 0] - xTrue[1, 0]

    d = math.sqrt(dx**2 + dy**2) # distanse to landmark
    angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0]) # angle for landmark

    if d <= MAX_RANGE:
        dN = d + np.random.randn() * Q1[0, 0]
        angleN = angle + np.random.randn() * Q1[1, 1]

        zN = np.array([dN, angleN]).reshape(2, 1) # observe another robot

    return zN


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

    r1 = Robot(xIni1, Q1, R1)    # instance robot1
    r2 = Robot(xIni2, Q2, R2)    # instance robot2

    time = 0.0;
    step = 0;

'''
    LM_list = np.array([[10.0, -2.0],
                        [15.0, 10.0],
                        [15.0, 15.0],
                        [10.0, 20.0],
                        [3.0, 15.0],
                        [-5.0, 20.0],
                        [-5.0, 5.0],
                        [-10.0, 15.0]])

    LM_NUM = LM_list.shape[0]
'''

    while step <= MAX_STEP:
        step += 1
        time += DT

        u1 = calc_input1(time)
        u2 = calc_input2(time)

        # Robot1
        r1.action(u1) # move robot1
        r1.observation(r2.xTrue) # observation robot1
        r1.particles = fast_slam(r1.particles, u1, zN1) # slam robot1
        r1.xSlam = calc_final_state(r1.particles)
        x_state1 = r1.xSlam[0: STATE_SIZE]

        # Robot2
        r2.xTrue, r2.xDead = action(r2.xTrue, r21.xDead, u) # move robot2
        zN2 = observation(r2.xTrue, r1.xTrue) # observation robot2
        r2.particles = fast_slam(r2.particles, uN2, zN2) # slam robot2
        r2.xSlam = calc_final_state(r2.particles)
        x_state2 = r1.xSlam[0: STATE_SIZE]

        if show_animation:  # pragma: no cover
            plt.cla()
            plt.plot(LM_list[:, 0], LM_list[:, 1], "*k")

            for i in range(PARTICLE_NUM):
                plt.plot(particles[i].x, particles[i].y, ".r")
                plt.plot(particles[i].lm[:, 0], particles[i].lm[:, 1], "xb")

            plt.plot(hxTrue[0, :], hxTrue[1, :], "-b")
            plt.plot(hxDead[0, :], hxDead[1, :], "-k")
            plt.plot(hxSlam[0, :], hxSlam[1, :], "-r")
            plt.plot(xSlam[0], xSlam[1], "Xk")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

if __name__ == '__main__':
    main()
