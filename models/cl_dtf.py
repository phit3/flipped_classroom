import random
from scipy import optimize
import math
import numpy as np

from .fr import FR


class CL_DTF_P_Lin(FR):
    def generate_randoms(self):
        return [random.random() for _ in range(self.output_steps)]

    def next_tf_ratio(self):
        m = self.curriculum_length
        k = 1.0
        self.tf_ratio = self.tf_factor * (max(0.0, k - self.epoch / m))


class CL_DTF_P_InvSig(FR):
    def generate_randoms(self):
        return [random.random() for _ in range(self.output_steps)]

    def next_tf_ratio(self):
        m = self.curriculum_length
        k = optimize.newton(lambda b: b / (b + math.exp(m/2/b)) - 0.5, m/10)
        self.tf_ratio = self.tf_factor * (k / (k + np.exp(self.epoch / k)))


class CL_DTF_P_Exp(FR):
    def generate_randoms(self):
        return [random.random() for _ in range(self.output_steps)]

    def next_tf_ratio(self):
        m = self.curriculum_length
        k = 0.5**(10 / m)
        self.tf_ratio = self.tf_factor * (k ** self.epoch)


class CL_DTF_D_Lin(CL_DTF_P_Lin):
    def generate_randoms(self):
        tf_steps = int(self.tf_ratio * self.output_steps)
        save_tf = np.zeros(tf_steps) - 1.0
        save_no_tf = np.ones(self.output_steps - tf_steps) + 1.0
        randoms = np.concatenate([save_tf, save_no_tf])
        return randoms


class CL_DTF_D_InvSig(CL_DTF_P_InvSig):
    def generate_randoms(self):
        tf_steps = int(self.tf_ratio * self.output_steps)
        save_tf = np.zeros(tf_steps) - 1.0
        save_no_tf = np.ones(self.output_steps - tf_steps) + 1.0
        randoms = np.concatenate([save_tf, save_no_tf])
        return randoms


class CL_DTF_D_Exp(CL_DTF_P_Exp):
    def generate_randoms(self):
        tf_steps = int(self.tf_ratio * self.output_steps)
        save_tf = np.zeros(tf_steps) - 1.0
        save_no_tf = np.ones(self.output_steps - tf_steps) + 1.0
        randoms = np.concatenate([save_tf, save_no_tf])
        return randoms
