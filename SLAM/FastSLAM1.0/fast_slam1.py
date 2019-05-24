"""
FastSLAM1.0 with EKF
author: Ota Satoshi
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# Fast SLAM covariance
Q = np.diag([0.2, np.deg2rad(8.0)])**2
R = np.diag([0.4, np.deg2rad(15.0)])**2

OFFSET_YAWRATE_NOISE = 0.01

DT = 0.1	# time delta
MAX_STEP = 1000 	# maximum step
SIM_TIME = 20.0	# simulation time
MAX_RANGE = 20.0	# maximum observation range
STATE_SIZE = 3 # Robot state(x, y, yaw)
LM_SIZE = 2 # Land mark(x, y)
PARTICLE_NUM = 100 # Nuber of particles
NTH = PARTICLE_NUM / 8.0  # Number of particle for re-sampling

show_animation = True

class Particle:

    def __init__(self, LM_NUM):
        self.w = 1.0 / PARTICLE_NUM
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        # landmark x-y positions
        self.lmPos = np.zeros((LM_NUM, LM_SIZE)) # lmPos = [[lmx_0, lmy_0], [lmx_1, lmy_1], ... ,[lmx_N, lmy_N]]
        #self.lmStat = np.array((LM_NUM, 1))
        self.lmStat = np.array([[False], [False], [False], [False], [False], [False], [False], [False]])
        #self.lmStat[:, 0] = False                # lmStat = [[False], [False], ... ,[False]]

        self.lm = np.hstack((self.lmPos, self.lmStat))
        # landmark position covariance
        self.lmP = np.zeros((LM_NUM * LM_SIZE, LM_SIZE))


def move(xTrue, xDead, u):

    # calculate True positions
    xTrue = motion_model(xTrue, u)

    # add noise to input
    uN_v = u[0, 0] + np.random.randn() * R[0, 0]
    uN_w = u[1, 0] + np.random.randn() * R[1, 1] + OFFSET_YAWRATE_NOISE
    uN = np.array([uN_v, uN_w]).reshape(2, 1)

    # calculate deadreconing
    xDead = motion_model(xDead, uN)

    return xTrue, xDead, uN

def fast_slam1(particles, u, zN):

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

        uN_v = u[0, 0] + np.random.randn() * R[0, 0]
        uN_w = u[1, 0] + np.random.randn() * R[1, 1] + OFFSET_YAWRATE_NOISE
        uN = np.array([uN_v, uN_w]).reshape(2, 1)
        #uN = u + (np.random.randn(1, 2) @ R).T # add noise
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
    for iz in range(len(zN[0, :])):

        lmid = int(zN[2, iz])

        for ip in range(PARTICLE_NUM):

            # new landmark
            if particles[ip].lm[lmid, 2] == False:
                particles[ip] = add_new_lm(particles[ip], zN[:, iz], Q)
            # known landmark
            else:
                w = compute_weight(particles[ip], zN[:, iz], Q)
                particles[ip].w *= w
                particles[ip] = update_landmark(particles[ip], zN[:, iz], Q)
                #print("update")

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

        tparticles = particles[:]
        for i in range(len(inds)):
            particles[i].x = tparticles[inds[i]].x
            particles[i].y = tparticles[inds[i]].y
            particles[i].yaw = tparticles[inds[i]].yaw
            particles[i].lm = tparticles[inds[i]].lm[:, :]
            particles[i].lmP = tparticles[inds[i]].lmP[:, :]
            particles[i].w = 1.0 / PARTICLE_NUM

    return particles


def calc_input(time):

	if time <= 1.0:	# wait at first
		v = 0.0
		yawrate = 0.0
	else:
		v = 1.0		# v[m/s]
		yawrate = 0.1		# w[rad/s]

	u = np.array([v, yawrate]).reshape(2, 1)

	return u


def observation(xTrue, LM_list):

    # calculate landmark positions
    zN = np.zeros((3, 0))
    for i in range(len(LM_list[:, 0])):
        dx = LM_list[i, 0] - xTrue[0, 0]
        dy = LM_list[i, 1] - xTrue[1, 0]

        d = math.sqrt(dx**2 + dy**2) # distanse to landmark
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0]) # angle for landmark

        if d <= MAX_RANGE:
            dN = d + np.random.randn() * Q[0, 0]
            angleN = angle + np.random.randn() * Q[1, 1]

            zN_i = np.array([dN, angleN, i]).reshape(3, 1) # observe landmark
            zN = np.hstack((zN, zN_i)) # zN = [[dN_0, dN_1, ... , dN_N], [angleN_0, angleN_1, ... , angleN_N], [0, 1, ... ,N]]

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

    time = 0.0;
    step = 0;

    LM_list = np.array([[10.0, -2.0],
                        [15.0, 10.0],
                        [15.0, 15.0],
                        [10.0, 20.0],
                        [3.0, 15.0],
                        [-5.0, 20.0],
                        [-5.0, 5.0],
                        [-10.0, 15.0]])

    LM_NUM = LM_list.shape[0]

    xSlam = np.zeros((STATE_SIZE, 1)) # Estimate with fast_slam1.0
    xDead = np.zeros((STATE_SIZE, 1)) # Estimate with deadreconing
    xTrue = np.zeros((STATE_SIZE, 1)) # True position

	# history
    hxSlam = xSlam
    hxDead = xDead
    hxTrue = xTrue

    # Creat particle instance [P_0, P_1, ... , P_M]
    particles = [Particle(LM_NUM) for i in range(PARTICLE_NUM)]

    while step <= MAX_STEP:
        step += 1
        time += DT

        u = calc_input(time)

        xTrue, xDead, uN = move(xTrue, xDead, u) # move

        zN = observation(xTrue, LM_list) # observation

        particles = fast_slam1(particles, u, zN)

        xSlam = calc_final_state(particles)

        x_state = xSlam[0: STATE_SIZE]

        # store data history
        hxSlam = np.hstack((hxSlam, x_state))
        hxDead = np.hstack((hxDead, xDead))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:  # pragma: no cover
            plt.cla()
            plt.plot(LM_list[:, 0], LM_list[:, 1], "*k")

            for i in range(PARTICLE_NUM):
                plt.plot(particles[i].x, particles[i].y, ".", c = "#5EC84E")
                plt.plot(particles[i].lm[:, 0], particles[i].lm[:, 1], "xb")

            plt.plot(hxTrue[0, :], hxTrue[1, :], "-b")
            plt.plot(hxDead[0, :], hxDead[1, :], "-k")
            plt.plot(hxSlam[0, :], hxSlam[1, :], "-r")
            plt.plot(xSlam[0], xSlam[1], "xk")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

if __name__ == '__main__':
    main()
