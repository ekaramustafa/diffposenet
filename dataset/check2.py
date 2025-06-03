import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
import sys
from pathlib import Path

# Add parent directory to import from nflownet.utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from nflownet.utils import compute_normal_flow

def flow_to_rgb(flow_tensor):
    flow_np = flow_tensor.cpu().numpy()
    if flow_np.shape[0] == 2:
        flow_np = np.moveaxis(flow_np, 0, -1)
    hsv = np.zeros((*flow_np.shape[:2], 3), dtype=np.uint8)
    mag = np.linalg.norm(flow_np, axis=-1)
    ang = np.arctan2(flow_np[..., 1], flow_np[..., 0])
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip((mag / np.max(mag)) * 255, 0, 255).astype(np.uint8)
    import cv2
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def plot_normal_flow_sample(p_path):
    p_path = Path(p_path)
    image_dir = p_path / "image_left"
    flow_dir = p_path / "flow"

    imgs = sorted(image_dir.glob("*.png"))
    flows = sorted(flow_dir.glob("*_flow.npy"))
    masks = sorted(flow_dir.glob("*_mask.npy"))

    if len(imgs) < 2 or not flows or not masks:
        print(f"⚠️ Missing files in {p_path.name}")
        return

    # Load images and tensors
    img1 = Image.open(imgs[0]).convert("RGB")
    img2 = Image.open(imgs[1]).convert("RGB")
    img_pair = torch.cat([ToTensor()(img1), ToTensor()(img2)], dim=0)  # (6, H, W)

    flow_tensor = torch.from_numpy(np.load(flows[0])).permute(2, 0, 1).float()  # (2, H, W)
    mask_np = np.load(masks[0])  # (H, W)
    mask_tensor = (1.0 - torch.from_numpy(mask_np).float() / 100.0).unsqueeze(0)
    flow_tensor = flow_tensor * mask_tensor

    # Apply mask to flow
    flow_tensor = flow_tensor * mask_tensor  # broadcasts over both u and v channels

    # Compute normal flow
    normal_flow = compute_normal_flow(flow_tensor, img_pair)

    # Plotting
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(img1); axs[0].set_title("Image 1")
    axs[1].imshow(img2); axs[1].set_title("Image 2")
    axs[2].imshow(flow_to_rgb(flow_tensor)); axs[2].set_title("Masked Optical Flow")
    axs[3].imshow(mask_np, cmap="gray"); axs[3].set_title("Flow Mask")
    axs[4].imshow(flow_to_rgb(normal_flow)); axs[4].set_title("Normal Flow")
    for ax in axs: ax.axis("off")
    plt.tight_layout()

    # Save to file
    save_path = p_path / "normal_flow_visualization.png"
    plt.savefig(save_path)
    print(f"✅ Saved visualization to: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_normal_flow_sample("/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data/amusement/Easy/P001")
