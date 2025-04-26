import cv2
import torch 
import numpy as np


def compute_normal_flow(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Computes the normal flow between two input images. 
    Used for generating ground-truth normal flow when training NFlowNet. 
    """

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    dense_optical_flow = cv2.calcOpticalFlowFarneback(img1, img2, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    u = dense_optical_flow[:, :, 0]
    v = dense_optical_flow[:, :, 1]
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)

    grad_mag_sq = Ix**2 + Iy**2
    grad_mag_sq[grad_mag_sq == 0] = 1e-5

    normal_flow_scalar = (u * Ix + v * Iy) / grad_mag_sq
    n_x = normal_flow_scalar * Ix
    n_y = normal_flow_scalar * Iy

    normal_flow = np.stack((n_x, n_y), axis=-1)
    return normal_flow


def ProjectionEndpointError():
    pass

def EndpointErrorMap():
    pass
    
