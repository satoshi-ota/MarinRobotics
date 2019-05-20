"""
FastSLAM1.0 with EKF
author: Ota Satoshi
"""

import numpy as np
import math
import matplotlib.pyplot as plt

Q_ekf = np.diag([3.0, np.deg2rad(10.0)])**2
R_ekf = np.diaf([1.0, np.deg2rad(20.0)])**2

Q_pf = np.diag([0.3, np.deg2rad(2.0)])**2
R_pf = np.diaf([0.5, np.deg2rad(10.0)])**2

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
        self.lm = np.zeros((LM_NUM, LM_SIZE))
        # landmark position covariance
        self.lmP = np.zeros((LM_NUM * LM_SIZE, LM_SIZE))


def move(xTrue, xDead, u):

    # calculate True positions
    xTrue = motion_model(xTrue, u)

    # add noise to input
    uN_v = u[0, 0] + np.random.randn() * R_pf[0, 0]
    uN_w = u[1, 0] + np.random.randn() * R_pf[1, 1]

    uN = np.array([uN_v, uN_w]).reshape(2, 1)

    # calculate deadreconing
    xDead = motion_model(xDead, uN)

    return xTrue, xDead, uN

def fast_slam1(particles, u, z):

    particles = predict_particles(particles, u)

    particles = update_with_observation(particles, z)

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


def predict_particles(particles, u):

    return particles


def add_new_lm(particle, z, Q):

    return particle


def compute_jacobians(particle, xf, Pf, Q):

    return zp, Hv, Hf, Sf


def update_KF_with_cholesky(xf, Pf, v, Q, Hf):

    return x, P


def update_landmark(particle, z, Q):


    return particle


def compute_weight(particle, z, Q):


    return w


def update_with_observation(particles, z):


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
            dN = d + np.random.randn() * Q_pf[0, 0]
            angleN = angle + np.random.randn() * Q_pf[1, 1]

            zN_i = np.array([dN, angleN, i]).reshape(3, 1) # observe landmark
            zN = np.hstack((zN, zN_i))

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

    # Creat particle instance [P_1, P_2, ... , P_n]
    particles = [Particle(LM_NUM) for i in range(PARTICLE_NUM)]

    while step >= max_step:
        step += 1
        time += DT

        u = calc_input(time)

        xTrue, xDead, uN = move(xTrue, xDead, u) # move

        zN = observation(xTrue, LM_list) # observation



if __name__ == '__main__':
    main()
