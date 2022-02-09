from .fr import FR


class TF(FR):
    def generate_randoms(self):
        return [0] * self.output_steps

    def next_tf_ratio(self):
        self.tf_ratio = 1.0
