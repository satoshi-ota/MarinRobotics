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

    particles = update(particles, zN) #update with observation

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

    xEst = np.zeros((STATE_SIZE, 1))

    particles = normalize_weight(particles)

    for i in range(N_PARTICLE):
        xEst[0, 0] += particles[i].w * particles[i].x
        xEst[1, 0] += particles[i].w * particles[i].y
        xEst[2, 0] += particles[i].w * particles[i].yaw

    xEst[2, 0] = pi_2_pi(xEst[2, 0])
    #  print(xEst)

    return xEst


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
    Ht = np.array([[c, -r * s],
                   [s, r * c]])

    # update covariance
    particle.lmP[lm_id * 2 : (lm_id + 1) * 2] = Ht @ Q @ Ht.T

    return particle


def compute_jacobians(particle, xf, Pf, Q):
	dx = xf[0, 0] - particle.x
	dy = xf[1, 0] - particle.y
	
	d = math.sqrt(dx**2 + dy**2)

		

    return zp, Hv, Hf, Sf


def update_KF_with_cholesky(xf, Pf, v, Q, Hf):

    return x, P


def update_landmark(particle, z, Q):


    return particle


def compute_weight(particle, zN, Q):

	lm_id = int(zN[2])
	xf = np.array(particle.lm[lm_id, 0:2]).reshape(2, 1)
	Pf = np.array(particle.lmP[lm_id * 2 : (lm_id + 1) * 2])

	zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)


    return w


def update(particles, zN):

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



if __name__ == '__main__':
    main()
