import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

def flow_to_rgb(flow):
    # Very basic flow visualization
    flow_np = flow.cpu().numpy()
    if flow_np.shape[0] == 2:
        flow_np = np.moveaxis(flow_np, 0, -1)
    hsv = np.zeros((*flow_np.shape[:2], 3), dtype=np.uint8)
    mag = np.linalg.norm(flow_np, axis=-1)
    ang = np.arctan2(flow_np[..., 1], flow_np[..., 0])
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # Hue
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip((mag / np.max(mag)) * 255, 0, 255).astype(np.uint8)
    from cv2 import cvtColor, COLOR_HSV2RGB
    return cvtColor(hsv, COLOR_HSV2RGB)

def plot_sample(traj_path):
    image_dir = traj_path / "image_left"
    flow_dir = traj_path / "flow"
    normal_dir = traj_path / "normal_flow"

    # Get sorted files
    imgs = sorted(image_dir.glob("*.png"))
    flows = sorted(flow_dir.glob("*_flow.npy"))
    masks = sorted(flow_dir.glob("*_mask.npy"))
    normals = sorted(normal_dir.glob("normal_*.pt"))

    if len(imgs) < 2 or not flows or not masks or not normals:
        print(f"⚠️ Skipping {traj_path.name}: missing files. imgs={len(imgs)}, flows={len(flows)}, masks={len(masks)}, normals={len(normals)}")
        return

    # Load images
    img1 = Image.open(imgs[0]).convert("RGB")
    img2 = Image.open(imgs[1]).convert("RGB")

    # Load flow + mask + normal_flow
    flow = np.load(flows[0])
    mask = np.load(masks[0])
    normal = torch.load(normals[0]).detach().numpy()

    # Plot
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(img1)
    axs[0].set_title("Image 1")
    axs[1].imshow(img2)
    axs[1].set_title("Image 2")
    axs[2].imshow(flow_to_rgb(torch.from_numpy(flow).permute(2, 0, 1)))
    axs[2].set_title("Optical Flow")
    axs[3].imshow(mask.squeeze(), cmap="gray")
    axs[3].set_title("Flow Mask")
    axs[4].imshow(flow_to_rgb(torch.from_numpy(normal)))
    axs[4].set_title("Normal Flow")

    for ax in axs:
        ax.axis("off")

    plt.suptitle(traj_path.name)
    plt.tight_layout()
    
    out_dir = Path("flow_viz_outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{traj_path.name}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved visualization to: {out_path}")


if __name__ == "__main__":
    ROOT = Path("/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data/amusement/Easy")

    for traj in sorted(ROOT.glob("P*")):
        print(f"🔍 Checking {traj}")
        plot_sample(traj)