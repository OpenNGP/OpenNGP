"""Module for learnable per-data/dataset parameter

dataset-level instrinsic parameter
per-data exposuire
per-data pose

"""


import torch
import torch.nn as nn


class ExposureRefine(nn.Module):
    def __init__(self, num_frame):
        super().__init__()

        self.num_frame = num_frame
        self.num_param = 3  # white balance
        self.vars = nn.Parameter(torch.zeros(self.num_frame, self.num_param))

    def get_exposure_scale(self, ids):
        expos = torch.exp(0.6931471805599453*self.vars[ids])
        return expos.squeeze(2)


class DepthScaleRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.vars = nn.Parameter(torch.zeros(1))
    
    def get_depth_scale(self):
        return torch.exp(0.6931471805599453*self.vars)


class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    @staticmethod
    def so3_to_SO3(w):  # [...,3]
        wx = Lie.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = Lie.taylor_A(theta)
        B = Lie.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R

    @staticmethod
    def SO3_to_so3(R, eps=1e-7):  # [...,3,3]
        trace = R[..., 0, 0]+R[..., 1, 1]+R[..., 2, 2]
        theta = ((trace-1)/2).clamp(-1+eps, 1-eps).acos_()[..., None, None] % np.pi  # ln(R) will explode if theta==pi
        lnR = 1/(2*Lie.taylor_A(theta)+1e-8)*(R-R.transpose(-2, -1))  # FIXME: wei-chiu finds it weird
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    @staticmethod
    def se3_to_SE3(wu):  # [...,3]
        w, u = wu.split([3, 3], dim=-1)
        wx = Lie.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = Lie.taylor_A(theta)
        B = Lie.taylor_B(theta)
        C = Lie.taylor_C(theta)
        R = I+A*wx+B*wx@wx
        V = I+B*wx+C*wx@wx
        Rt = torch.cat([R, (V@u[..., None])], dim=-1)
        h_pad = torch.tensor([[[0, 0, 0, 1]]], device=w.device, dtype=torch.float32)
        h_pad = h_pad.expand((Rt.shape[0], 1, 4))
        return torch.cat([Rt, h_pad], dim=1)

    @staticmethod
    def SE3_to_se3(Rt, eps=1e-8):  # [...,3,4]
        R, t = Rt.split([3, 1], dim=-1)
        w = Lie.SO3_to_so3(R)
        wx = Lie.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = Lie.taylor_A(theta)
        B = Lie.taylor_B(theta)
        invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        u = (invV@t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    @staticmethod
    def skew_symmetric(w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O, -w2, w1], dim=-1),
                          torch.stack([w2, O, -w0], dim=-1),
                          torch.stack([-w1, w0, O], dim=-1)], dim=-2)
        return wx

    @staticmethod
    def taylor_A(x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i > 0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

    @staticmethod
    def taylor_B(x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

    @staticmethod
    def taylor_C(x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans


class PoseRefine(nn.Module):
    def __init__(self, num_frame, affine_repr='lie'):
        super().__init__()

        self.num_frame = num_frame
        self.num_params = 6
        self.affine_repr = affine_repr

        self.lie = Lie()

        self.vars = nn.Parameter(torch.zeros(self.num_frame, self.num_params))

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[..., 0]
        return R_new, t_new

    def get_SE3(self, ids):
        return self.lie.se3_to_SE3(self.vars[ids])

    def get_SO3(self, ids):
        return self.lie.so3_to_SO3(self.vars[ids, 0:3])

    def get_rotations(self, ids):
        if 'lie' == self.affine_repr:
            return self.get_SO3(ids)
        else:
            return self.get_rotation_matrices(ids)

    def get_translations(self, ids):
        return self.vars[ids, 3:6]

    def get_eulers(self, ids):
        return self.vars[ids, 0:3]

    def get_rotation_matrices(self, ids):
        rotations = self.get_eulers(ids)  # [N_frames, 3]

        cos_alpha = torch.cos(rotations[:, 0])
        cos_beta = torch.cos(rotations[:, 1])
        cos_gamma = torch.cos(rotations[:, 2])
        sin_alpha = torch.sin(rotations[:, 0])
        sin_beta = torch.sin(rotations[:, 1])
        sin_gamma = torch.sin(rotations[:, 2])

        col1 = torch.stack([cos_alpha * cos_beta,
                            sin_alpha * cos_beta,
                            -sin_beta], -1)
        col2 = torch.stack([cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma,
                            sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma,
                            cos_beta * sin_gamma], -1)
        col3 = torch.stack([cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma,
                            sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma,
                            cos_beta * cos_gamma], -1)

        return torch.stack([col1, col2, col3], -1)
