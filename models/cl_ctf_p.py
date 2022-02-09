import random

from .fr import FR


class CL_CTF_P(FR):
    def generate_randoms(self):
        return [random.random()] * self.output_steps

    def next_tf_ratio(self):
        pass
