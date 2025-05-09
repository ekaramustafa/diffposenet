import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import Grayscale



def compute_normal_flow(opt_flow: torch.Tensor, img_pair: torch.Tensor, magnitude: bool=False):                 
    """
    Computes the normal flow or its magnitude from optical flow and an image pair.

    Args:
    - opt_flow (torch.Tensor): (2, H, W) tensor containing the horizontal (u) and vertical (v) optical flow components.
    - image_pair (torch.Tensor): (6, H, W) tensor containing two concatenated RGB images (e.g., [img1, img2]).
    - magnitude (bool): Whether to return just the magnitude of the normal flow (default False).    
    
    Returns:
    - torch.Tensor: (2, H, W) normal flow or (1, H, W) magnitude, depending on `magnitude` flag.
    """
    grayscale_transform = Grayscale(num_output_channels=1)

    u = opt_flow[0:1, :, :]
    v = opt_flow[1:2, :, :]

    ref_img_gray = grayscale_transform(img_pair[:3, :, :])
    grad_x, grad_y = compute_image_gradients(ref_img_gray)
    grad_norm_sq = grad_x ** 2 + grad_y ** 2
    grad_norm_sq[grad_norm_sq == 0] = 1e-6

    dot_product = u * grad_x + v * grad_y
    scale = dot_product / grad_norm_sq

    n_x = scale * grad_x
    n_y = scale * grad_y

    if magnitude:
        return torch.sqrt(n_x ** 2 + n_y ** 2)
    else:
        return torch.cat([n_x, n_y], dim=0)



def compute_image_gradients(image_tensor: torch.tensor):
    """
    Computes the spatial image gradients ∂I/∂x and ∂I/∂y for a grayscale image tensor.
    
    Args: 
    - image_tensor (torch.Tensor): A single grayscale image, shape (1, H, W).

    Returns: 
    - grad_x (torch.Tensor): Horizontal gradients (∂I/∂x), shape (1, H, W).
    - grad_y (torch.Tensor): Vertical gradients (∂I/∂y), shape (1, H, W). 
    """
    x = image_tensor.unsqueeze(0).float() 
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=np.float32)

    conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).float()
    conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).float()
    conv_x.weight = nn.Parameter(torch.from_numpy(sobel_x).unsqueeze(0).unsqueeze(0))
    conv_y.weight = nn.Parameter(torch.from_numpy(sobel_y).unsqueeze(0).unsqueeze(0))

    grad_x = conv_x(x).squeeze(0)
    grad_y = conv_y(x).squeeze(0)
    return grad_x, grad_y



def pad_to_divisible_by_4(tensor):
    """
    Pads a 4D tensor (B, C, H, W) along the height and width dimensions so that both become divisible by 4.
    Padding is applied to the right and bottom ends of the image. 
    """
    height, width = tensor.shape[2], tensor.shape[3]
    pad_h = (4 - height % 4) % 4  
    pad_w = (4 - width % 4) % 4 
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))


def crop_to_target_size(tensor, target_height, target_width):
    """
    Crops a 4D tensor to the target size from the bottom and right ends of the tensor.
    """
    return tensor[:, :, :target_height, :target_width]


def interpolate_to_divisible_by_16(tensor):
    """
    Interpolates the (B, C, H, W) tensor so that H and W are divisible by 16.
    Uses bilinear interpolation and keeps the values centered.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W)

    Returns:
        torch.Tensor: Interpolated tensor with H and W divisible by 16
    """
    B, C, H, W = tensor.shape

    def next_multiple_of_16(x):
        return math.ceil(x / 16) * 16

    new_H = next_multiple_of_16(H)
    new_W = next_multiple_of_16(W)

    resized_tensor = F.interpolate(tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
    return resized_tensor


def ProjectionEndpointError(opt_flow:torch.Tensor, normal_pred:torch.Tensor):            
    """
    Computes the Projection Endpoint Error (PEE).
    Reference: https://arxiv.org/html/2412.11284v1#S2.E2

    Args:
    - opt_flow (torch.Tensor): ground-truth optical flow 
    - normal_pred (torch.Tensor): normal flow prediction

    Returns:
    - PEE: float, the mean projection endpoint error over all pixels
    """
    n_x = normal_pred[0:1, :, :]
    n_y = normal_pred[1:2, :, :]
    normal_pred_norm = torch.sqrt(n_x**2 + n_y**2)
    normal_pred_norm[normal_pred_norm == 0] = 1e-5

    projection = torch.dot(opt_flow, normal_pred) / normal_pred_norm
    error = np.abs(projection - normal_pred)

    PEE = np.mean(error)
    return PEE


def compute_magnitude_and_direction(flow: torch.Tensor):
    """
    Computes the magnitude and the phase (direction) of the optical/normal flow given its 2 components

    Args:
    - flow (torch.Tensor): A tensor of shape (2,H,W) denoting optical/normal flow

    Returns:
    - magnitude (torch.Tensor): A tensor of shape (1,H,W) denoting the magnitude of the flow at each pixel
    - phase (torch.Tensor): A tensor of shape (1,H,W) denoting the direction of the flow at each pixel
    """
    magnitude = torch.sqrt(flow[0]**2 + flow[1]**2).unsqueeze(0)
    phase = torch.atan2(flow[1], flow[0]).unsqueeze(0)
    return magnitude, phase



    

    
    
