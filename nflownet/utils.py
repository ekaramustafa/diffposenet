import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import Grayscale



def compute_normal_flow(opt_flow: torch.tensor, img_pair: torch.tensor, magnitude: bool=False):                 
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



def ProjectionEndpointError(grad_x, grad_y, u, n_hat):            # <- WILL BE CHANGED
    """
    Computes the Projection Endpoint Error (PEE).

    Args:
    - grad_x (torch.Tensor): gradient of the image along x-axis (∂I/∂x)
    - grad_y (torch.Tensor): gradient of the image along y-axis (∂I/∂y)
    - proj_opt_flow (torch.Tensor): the 
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
    
