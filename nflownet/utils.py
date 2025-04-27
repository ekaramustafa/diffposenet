import cv2
import torch 
import numpy as np


import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision.transforms import Grayscale


def compute_normal_flow(opt_flow: torch.tensor, img_pair: torch.tensor):
    """
    Args:
        opt_flow: (B, 2, H, W) or (2, H, W) tensor containing horizontal (u) and vertical (v) optical flow components
        image_pair: (B, 6, H, W) or (6, H, W) tensor containing concatenated image pairs (e.g., [img1, img2])

    Returns:
        normal_flow_magnitude: (B, 1, H, W) tensor containing normal flow for the image pair
    """

    grayscale_transform = Grayscale(num_output_channels=1)
    if opt_flow.ndim == 3:
        opt_flow = opt_flow.unsqueeze(0)
    if img_pair.ndim == 3:
        img_pair = img_pair.unsqueeze(0)

    B, _, H, W = opt_flow.shape
    u = opt_flow[:, 0:1, :, :]
    v = opt_flow[:, 1:2, :, :]

    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=opt_flow.device).unsqueeze(0).unsqueeze(0) / 8.0
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=opt_flow.device).unsqueeze(0).unsqueeze(0) / 8.0

    img1 = img_pair[:, :3, :, :]
    img1_gray = grayscale_transform(img1)

    grad_x = F.conv2d(img1_gray, sobel_x, padding=1)  # ∂I/∂x
    grad_y = F.conv2d(img1_gray, sobel_y, padding=1)  # ∂I/∂y

    grad_mag_sq = grad_x ** 2 + grad_y ** 2 + 1e-6
    dot_product = u * grad_x + v * grad_y
    normal_flow = (dot_product / grad_mag_sq)
    return normal_flow



def pad_to_divisible_by_4(tensor):
    # Calculate padding to make height and width divisible by 4
    height, width = tensor.shape[2], tensor.shape[3]
    pad_h = (4 - height % 4) % 4  # Compute how much padding is needed for height
    pad_w = (4 - width % 4) % 4  # Compute how much padding is needed for width
    # Apply padding (padding is applied as [left, right, top, bottom])
    return nn.functional.pad(tensor, (0, pad_w, 0, pad_h))


def crop_to_target_size(tensor, target_height, target_width):
    """Crop tensor to the target size (height, width)."""
    _, _, h, w = tensor.shape
    start_h = (h - target_height) // 2
    start_w = (w - target_width) // 2
    return tensor[:, :, start_h:start_h+target_height, start_w:start_w+target_width]



# def compute_normal_flow(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
#     """
#     Computes the normal flow between two input images (as PyTorch tensors).
#     Used for generating ground-truth normal flow when training NFlowNet.

#     Parameters:
#     - img1: torch.Tensor, shape (C, H, W), the nth frame
#     - img2: torch.Tensor, shape (C, H, W), the (n+1)st frame

#     Returns:
#     - normal_flow: torch.Tensor, shape (H, W), the ground-truth normal flow
#     """

#     # Move to CPU and convert to numpy for OpenCV
#     img1_np = img1.detach().cpu().numpy()
#     img2_np = img2.detach().cpu().numpy()

#     if img1_np.shape[0] == 3:  # if (C,H,W), convert to (H,W,C) for OpenCV
#         img1_np = np.transpose(img1_np, (1, 2, 0))
#         img2_np = np.transpose(img2_np, (1, 2, 0))

#     img1_np = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)  # tensor is usually RGB
#     img2_np = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

#     # Farneback optical flow (still with OpenCV)
#     dense_optical_flow = cv2.calcOpticalFlowFarneback(
#         img1_np, img2_np, None, 
#         pyr_scale=0.5, levels=3, winsize=15, iterations=3, 
#         poly_n=5, poly_sigma=1.2, flags=0
#     )

#     u = dense_optical_flow[:, :, 0]
#     v = dense_optical_flow[:, :, 1]

#     Ix = cv2.Sobel(img1_np, cv2.CV_64F, 1, 0, ksize=5)
#     Iy = cv2.Sobel(img1_np, cv2.CV_64F, 0, 1, ksize=5)

#     grad_mag_sq = Ix**2 + Iy**2
#     grad_mag_sq[grad_mag_sq == 0] = 1e-5

#     normal_flow_scalar = (u * Ix + v * Iy) / grad_mag_sq
#     n_x = normal_flow_scalar * Ix
#     n_y = normal_flow_scalar * Iy

#     # Final normal flow calculation
#     normal_flow = np.stack((n_x, n_y), axis=-1)
#     normal_flow = np.sqrt(normal_flow[:,:,0]**2 + normal_flow[:,:,1]**2)

#     # Convert back to tensor
#     normal_flow_tensor = torch.from_numpy(normal_flow).float()

#     return normal_flow_tensor



def ProjectionEndpointError(grad_x, grad_y, u, n_hat):
    """
    Computes the Projection Endpoint Error (PEE).

    Parameters:
    - grad_x: np.ndarray, gradient of image along x-axis (∂I/∂x)
    - grad_y: np.ndarray, gradient of image along y-axis (∂I/∂y)
    - ue: np.ndarray of shape (..., 2), estimated optical flow vectors
    - n_hat: np.ndarray, ground-truth normal flow scalar per pixel

    Returns:
    - PEE: float, the mean projection endpoint error over all pixels
    """
    
    grad_norm_squared = grad_x**2 + grad_y**2
    grad_norm_squared[grad_norm_squared == 0] = 1e-5

    projection = (grad_x * u[..., 0] + grad_y * u[..., 1]) / grad_norm_squared
    error = np.abs(n_hat - projection)

    PEE = np.mean(error)
    return PEE



def EndpointErrorMap():
    pass
    
