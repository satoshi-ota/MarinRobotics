"""
FastSLAM1.0 with EKF
author: Ota Satoshi
"""

import numpy as np
import math
import matplotlib.pyplot as plt

Q    = np.diag([3.0, np.deg2rad(10.0)])**2
R = np.diaf([1.0, np.deg2rad(20.0)])**2

Qsim = np.diag([0.3, np.deg2rad(2.0)])**2
Rsim = np.diaf([0.5, np.deg2rad(10.0)])**2
OFFSET_YAWRATE_NOISE = 0.01

STATE_SIZE = 3 # Robot state(x, y, yaw)
LM_SIZE = 2 # Land mark(x, y)
PARTICLE_NUM = 100 # Nuber of particles
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling

class Particle:

    def __init__(self, LM_NUM):
        self.w = 1.0 / PARTICLE_NUM
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        # landmark x-y positions
        self.lmPos = np.zeros((LM_NUM, LM_SIZE)) # lmPos = [[lmx_0, lmy_0], [lmx_1, lmy_1], ... ,[lmx_N, lmy_N]]
        self.lmStat = np.array((LM_NUM, 1))
        self.lmStat[;, ;] = False                # lmStat = [[False], [False], ... ,[False]]

        self.lm = np.hstack((self.lmPos, self.lmStat))
        # landmark position covariance
        self.lmP = np.zeros((LM_NUM * LM_SIZE, LM_SIZE))


def move(xTrue, xDead, u):

    # calculate True positions
    xTrue = motion_model(xTrue, u)

    # add noise to input
    uN_v = u[0, 0] + np.random.randn() * Rsim[0, 0]
    uN_w = u[1, 0] + np.random.randn() * Rsim[1, 1] + OFFSET_YAWRATE_NOISE

    uN = np.array([uN_v, uN_w]).reshape(2, 1)

    # calculate deadreconing
    xDead = motion_model(xDead, uN)

    return xTrue, xDead, uN

def fast_slam1(particles, uN, zN):

    particles = predict(particles, uN) # estimate particles position from input

    particles = update_with_observation(particles, zN) #update with observation

    particles = resampling(particles)

    return particles


def normalize_weight(particles):

    sumw = sum([p.w for p in particles])

    try:
        for i in range(N_PARTICLE):
            particles[i].w /= sumw
    except ZeroDivisionError:
        for i in range(N_PARTICLE):
            particles[i].w = 1.0 / N_PARTICLE

        return particles

    return particles


def calc_final_state(particles):

    xSlam = np.zeros((STATE_SIZE, 1))

    particles = normalize_weight(particles)

    for i in range(N_PARTICLE):
        xSlam[0, 0] += particles[i].w * particles[i].x
        xSlam[1, 0] += particles[i].w * particles[i].y
        xSlam[2, 0] += particles[i].w * particles[i].yaw

    xSlam[2, 0] = pi_2_pi(xSlam[2, 0])
    #  print(xSlam)

    return xSlam


def predict(particles, uN):

    # calculate particles positions
    for i in range(PARTICLE_NUM):
        xTemp = np.zeros((STATE_SIZE, 1))

        xTemp[0, 0] = particles[i].x
        xTemp[1, 0] = particles[i].y
        xTemp[2, 0] = particles[i].yaw

        uN = uN + (np.random.randn() @ R).T # add noise
        xTemp = motion_model(xTemp, uN) # calculate particle position from motion model

        particles[i].x = xTemp[0, 0]
        particles[i].y = xTemp[0, 1]
        particles[i].z = xTemp[0, 2]

    return particles


def add_new_lm(particle, zN, Q):

    # extract observation data
    r = z[0]
    b = z[1]
    lm_id = z[2]

    # calculate landmark position
    s = math.sin(pi_2_pi(particle.yaw + b))
    c = math.cos(pi_2_pi(particle.yaw + b))

    # update landmark state
    particle.lm[lm_id, 0] = particle.x + r * c
    particle.lm[lm_id, 1] = particle.x + r * s
    particle.lm[lm_id, 2] == True

    # calculate Jacobian
    dx = particle.lm[lm_id, 0] - particle.x
    dy = particle.lm[lm_id, 1] - particle.y
    q = dx**2 + dy**2
    d = math.sqrt(q)

    H = np.array([[-dx / d, -dy / d, 0],
                  [dx / q, -dy / q, -1],
                  [0, 0, 0]])

    # update covariance
    particle.lmP[lm_id * 2 : (lm_id + 1) * 2] = H @ Q @ H.T

    return particle


def compute_jacobians(particle, xf, Pf, Q):
	dx = xf[0, 0] - particle.x
	dy = xf[1, 0] - particle.y

	d = math.sqrt(dx**2 + dy**2)



    return zp, Hv, Hf, Sf


def update_landmark(particle, zN, Q):

    lm_id = int(zN[2])
	mu = np.array(particle.lm[lm_id, 0:2]).reshape(2, 1) # (lm_x, lm_y)
	Si = np.array(particle.lmP[lm_id * 2 : (lm_id + 1) * 2]) # covariance

    # calculate Jacobian
    dx = mu[0] - particle.x
    dy = mu[1] - particle.y
    q = dx**2 + dy**2
    d = math.sqrt(q)

    H = np.array([[-dx / d, -dy / d, 0],
                  [dx / q, -dy / q, -1],
                  [0, 0, 0]])

    # calculate Inovation Vector
    zNh = np.array([d, math.atan2(dy, dx) - particle.yaw).reshape(3, 1)
    dz = (zN - zNh).T # Inovation Vector
    dz[1, 0] = pi_2_pi(dz[1, 0])

	mu, Si = update_with_EKF(mu, Si, dz, H, Q)

    particle.lm[lm_id, :] = mu.T
    particle.lmP[lm_id * 2 : (lm_id + 1) * 2] = Si

    return particle


def compute_weight(particle, zN, Q):

	lm_id = int(zN[2])
	mu = np.array(particle.lm[lm_id, 0:2]).reshape(2, 1) # (lm_x, lm_y)
	Si = np.array(particle.lmP[lm_id * 2 : (lm_id + 1) * 2]) # covariance

    # calculate Jacobian
    dx = mu[0] - particle.x
    dy = mu[1] - particle.y
    q = dx**2 + dy**2
    d = math.sqrt(q)

    H = np.array([[-dx / d, -dy / d, 0],
                  [dx / q, -dy / q, -1],
                  [0, 0, 0]])

    # calculate Inovation Vector
    zNh = np.array([d, math.atan2(dy, dx) - particle.yaw).reshape(3, 1)
    dz = (zN - zNh).T # Inovation Vector
    dz[1, 0] = pi_2_pi(dz[1, 0])

    Qp = H @ Si @ H.T + Q   # observation covariance
    QpInv = np.linalg.inv(Qp)   # Q^-1

    # compute particle wight
    num = math.exp(-0.5 * dz.T @ QpInv @dz)
    den = 2 * math.pi * math.sqrt(np.linalg.det(Q))
    w = num / den

    return w

def update_with_EKF(mu, Si, dz, H, Q):

    Qp = H @ Si @ H.T + Q   # observation covariance
    Kp = Si @ H.T @ np.linalg.inv(Qp)   # calculate kalman gain

    mu = mu + K @ dz    # update average
    Si = (np.eye(LM_SIZE) - (Kp @ H)) @ Si  # update covariance

    return mu, Si

def update_with_observation(particles, zN):

    #update with observation
    for iz in range(len(zN[0, :])):

        lmid = int(zN[2, iz])

        for ip in range(PARTICLE_NUM):

            # new landmark
            if particles[ip].lm[lmid, 2] == False:
                particles[ip] = add_new_lm(particles[ip], zN[;, iz], Q)
            # known landmark
            else:
		        w = compute_weight(particles[ip], zN[:, iz], Q)
		        particles[ip].w *= w
		        particles[ip] = update_landmark(particles[ip], z[:, iz], Q)

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


def calc_input(time):

    return u


def observation(xTrue, LM_list):

    # calculate landmark positions
    for i in range(len(LM_list[:, 0])):
        dx = LM_list[i, 0] - xTrue[0, 0]
        dy = LM_list[i, 1] - xTrue[1, 0]

        d = math.sqrt(dx**2 + dy**2) # distanse to landmark
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0]) # angle for landmark

        if d <= MAX_RANGE:
            dN = d + np.random.randn() * Qsim[0, 0]
            angleN = angle + np.random.randn() * Qsim[1, 1]

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

    LM_list = ([[10.0, -5.0],
                [4.0, 7.0],
                [8.0, -13.0],
                [-3.0, 6.0],
                [2.0, 8.0]])

    LM_NUM = LM_list.shape[0]

    xSlam = np.zeros((STATE_SIZE, 1)) # Estimate with fast_slam1.0
    xDead = np.zeros((STATE_SIZE, 1)) # Estimate with deadreconing
    xTrue = np.zeros((STATE_SIZE, 1)) # True position

    # Creat particle instance [P_0, P_1, ... , P_M]
    particles = [Particle(LM_NUM) for i in range(PARTICLE_NUM)]

    while step >= max_step:
        step += 1
        time += DT

        u = calc_input(time)

        xTrue, xDead, uN = move(xTrue, xDead, u) # move

        zN = observation(xTrue, LM_list) # observation

        particles = fast_slam1(particles, uN, zN)

        xSlam = calc_final_state(particles)

        x_state = xSlam[0: STATE_SIZE]

        # store data history
        hxSlam = np.hstack((hxSlam, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:  # pragma: no cover
            plt.cla()
            plt.plot(RFID[:, 0], RFID[:, 1], "*k")

            for i in range(N_PARTICLE):
                plt.plot(particles[i].x, particles[i].y, ".r")
                plt.plot(particles[i].lm[:, 0], particles[i].lm[:, 1], "xb")

            plt.plot(hxTrue[0, :], hxTrue[1, :], "-b")
            plt.plot(hxDR[0, :], hxDR[1, :], "-k")
            plt.plot(hxSlam[0, :], hxSlam[1, :], "-r")
            plt.plot(xSlam[0], xSlam[1], "xk")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

if __name__ == '__main__':
    main()
