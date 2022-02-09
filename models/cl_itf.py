
from .cl_dtf import CL_DTF_P_Lin
from .cl_dtf import CL_DTF_P_InvSig
from .cl_dtf import CL_DTF_P_Exp
from .cl_dtf import CL_DTF_D_Lin
from .cl_dtf import CL_DTF_D_InvSig
from .cl_dtf import CL_DTF_D_Exp


class CL_ITF_P_Lin(CL_DTF_P_Lin):
    def next_tf_ratio(self):
        super(CL_ITF_P_Lin, self).next_tf_ratio()
        self.teacher_forcing_probability = 1.0 - self.teacher_forcing_probability


class CL_ITF_P_InvSig(CL_DTF_P_InvSig):
    def next_tf_ratio(self):
        super(CL_ITF_P_InvSig, self).next_tf_ratio()
        self.teacher_forcing_probability = 1.0 - self.teacher_forcing_probability


class CL_ITF_P_Exp(CL_DTF_P_Exp):
    def next_tf_ratio(self):
        super(CL_ITF_P_Exp, self).next_tf_ratio()
        self.teacher_forcing_probability = 1.0 - self.teacher_forcing_probability


class CL_ITF_D_Lin(CL_DTF_D_Lin):
    def next_tf_ratio(self):
        super(CL_ITF_D_Lin, self).next_tf_ratio()
        self.teacher_forcing_probability = 1.0 - self.teacher_forcing_probability


class CL_ITF_D_InvSig(CL_DTF_D_InvSig):
    def next_tf_ratio(self):
        super(CL_ITF_D_InvSig, self).next_tf_ratio()
        self.teacher_forcing_probability = 1.0 - self.teacher_forcing_probability


class CL_ITF_D_Exp(CL_DTF_D_Exp):
    def next_tf_ratio(self):
        super(CL_ITF_D_Exp, self).next_tf_ratio()
        self.teacher_forcing_probability = 1.0 - self.teacher_forcing_probability
