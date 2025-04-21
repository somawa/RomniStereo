# ocam.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
# from Davide Scaramuzza's OcamCalib toolbox
# https://sites.google.com/site/scarabotix/ocamcalib-toolbox
#
import torch
import numpy as np
from scipy.io import loadmat
from easydict import EasyDict as Edict
from utils.log import *
from utils.common import *
from utils.geometry import *

class OcamModel:
    def __init__(self): pass
    def setConfig(self, cfg):
        self.id = cfg['cam_id']
        num_pol = cfg['poly'][0]
        if len(cfg['poly']) - 1 != num_pol :
            LOG_WARNING('Number of coeffs does not match in ocam\'s poly')
        self.poly = cfg['poly'][-1:0:-1] # make reverse
        LOG_INFO(f"self.poly: {self.poly}")
        num_invpol = cfg['inv_poly'][0]
        if len(cfg['inv_poly']) - 1 != num_invpol :
            LOG_WARNING('Number of coeffs does not match in ocam\'s inv_poly')
        self.inv_poly = cfg['inv_poly'][-1:0:-1] # make reverse
        self.xc, self.yc = cfg['center'] # x, y fliped
        self.c, self.d, self.e = cfg['affine']
        self.height, self.width = cfg['image_size']
        self.max_theta = np.deg2rad(cfg['max_fov']) / 2.0

        print(f"CamId {self.id} has invalid_mask of {cfg['invalid_mask'] }")
        # if cfg['invalid_mask'] == 0:
        #     print("No mask file")
        #     self.invalid_mask_file = ""
        # else:
        #     print("Mask file present")
        #     self.invalid_mask_file = cfg['invalid_mask']
        self.invalid_mask_file = cfg['invalid_mask']
         
        self.cam2rig = np.array(cfg['pose']).reshape((6, 1))
        self.rig2cam = inverseTransform(self.cam2rig)
    # end setConfig

    # p can be torch.Tensor
    def pixelToRay(self, p, out_theta=False, max_theta=None):
        if max_theta is None: max_theta = self.max_theta
        # flip axis 
        x = p[1,:].reshape((1, -1)) - self.xc
        y = p[0,:].reshape((1, -1)) - self.yc
        p = concat((x, y), axis=0)
        invdet = 1.0 / (self.c - self.d * self.e)
        A_inv = invdet * np.array([
            [      1, -self.d], 
            [-self.e,  self.c]]) 
        p = A_inv.dot(p)
        # flip axis 
        x = p[1,:].reshape((1, -1))
        y = p[0,:].reshape((1, -1))
        rho = sqrt(x * x + y * y)
        z = polyval(self.poly, rho).reshape((1, -1))
        # theta is angle from the optical axis.
        theta = atan2(rho, -z) 
        out = concat((x, y, -z), axis=0)
        out[:,theta.squeeze() > max_theta] = np.nan
        if out_theta:
            return out, theta
        else:
            return out
    # end pixelToRay


    

    # P can be torch.Tensor
    def rayToPixel(self, P, out_theta=False, max_theta=None):
        if max_theta is None: max_theta = self.max_theta
        norm = sqrt(P[0,:]**2 + P[1,:]**2) + EPS
        theta = atan2(-P[2,:], norm)
        print(f"R2P trigger polyval on {len(norm)} for norm {norm} and {len(theta)} for theta {theta}")
        rho = polyval(self.inv_poly, theta)
        # max_theta check : theta is the angle from xy-plane in ocam, 
        # thus add pi/2 to compute the angle from the optical axis.
        theta = theta + np.pi / 2
        # flip axis
        x = P[1,:] / norm * rho
        y = P[0,:] / norm * rho
        x2 = x * self.c + y * self.d + self.xc
        y2 = x * self.e + y          + self.yc
        x2 = x2.reshape((1, -1))
        y2 = y2.reshape((1, -1))
        out = concat((y2, x2), axis=0)
        out[:,theta.squeeze() > max_theta] = -1e5
        if out_theta:
            return out, theta
        else:
            return out
    # end rayToPixel


    # ########FTheta Model###########
    # # p can be torch.Tensor
    # def pixelToRay(self, p, out_theta=False, max_theta=None):
    #     if max_theta is None: max_theta = self.max_theta
    #     # flip axis 
    #     x = p[0,:].reshape((1, -1)) - self.xc
    #     y = p[1,:].reshape((1, -1)) - self.yc

    #     P_d_norm = sqrt(p[1,:] * p[1,:] + p[0,:] * p[0,:])


    #     p = concat((x, y), axis=0)
    #     # invdet = 1.0 / (self.c - self.d * self.e)
    #     # A_inv = invdet * np.array([
    #     #     [      1, -self.d], 
    #     #     [-self.e,  self.c]]) 
    #     # p = A_inv.dot(p)
    #     # flip axis 
    #     x = p[0,:].reshape((1, -1))
    #     y = p[1,:].reshape((1, -1))
    #     # rho = sqrt(x * x + y * y)
    #     # print("P2R trigger polyval")
    #     # z = polyval(self.poly, rho).reshape((1, -1))




    #     theta = P_d_norm / self.poly[0]
    #     print(f"pixeltoray being done on k value {self.poly[0]}")

    #     x = sin(theta) * x / P_d_norm
    #     y = sin(theta) * y / P_d_norm
    #     z = cos(theta)


    #     print("x is ", x.shape)
    #     print("y is ", y.shape)
    #     print("z is ", z.shape)
    #     # theta is angle from the optical axis.
    #     # theta = atan2(rho, -z) 
        
    #     out = concat((x, y, z), axis=0)
    #     # out[:,theta.squeeze() > max_theta] = np.nan
    #     if out_theta:
    #         return out, theta
    #     else:
    #         return out
    # # end pixelToRay
    # # 
    # # 
    # # 
    # # 
    # # # P can be torch.Tensor
    # def rayToPixel(self, P, out_theta=False, max_theta=None):
    #     if max_theta is None: max_theta = self.max_theta
    #     # norm = sqrt(P[0,:]**2 + P[1,:]**2) + EPS
    #     l2_norm = sqrt(P[1,:]**2 + P[0,:]**2 + P[2,:]**2)


    #     # theta = atan2(-P[2,:], norm)
    #     theta = acos(P[2,:] / l2_norm)


    #     # print(f"R2P trigger polyval on {len(norm)} for norm {norm} and {len(theta)} for theta {theta}")
    #     # rho = polyval(self.inv_poly, theta)
    #     r = self.poly[0] * theta
    #     print(f"raytopixel being done on k value {self.poly[0]}")

    #     R_p = sqrt(P[1,:] * P[1,:] + P[0,:] * P[0,:]);

    #     x2 = self.xc + r * P[0,:] / R_p
    #     y2 = self.yc + r * P[1,:] / R_p


    #     # # max_theta check : theta is the angle from xy-plane in ocam, 
    #     # # thus add pi/2 to compute the angle from the optical axis.
    #     # theta = theta + np.pi / 2
    #     # # flip axis
    #     # x = P[1,:] / norm * rho
    #     # y = P[0,:] / norm * rho
    #     # x2 = x * self.c + y * self.d + self.xc
    #     # y2 = x * self.e + y          + self.yc
    #     x2 = x2.reshape((1, -1))
    #     y2 = y2.reshape((1, -1))
    #     out = concat((x2, y2), axis=0)
    #     # out[:,theta.squeeze() > max_theta] = -1e5
    #     if out_theta:
    #         return out, theta
    #     else:
    #         return out
    # # end rayToPixel

    def makeInvisibleMask(self) -> np.ndarray:
        xs, ys = np.meshgrid(range(self.width), range(self.height))
        print("Invisible mask", xs, ys)
        p = np.concatenate((xs.reshape((1, -1)), ys.reshape((1, -1))), axis=0)
        ray = self.pixelToRay(p)
        invisible = np.isnan(ray[0, :])
        return invisible.reshape((self.height, self.width)).astype(np.bool)
