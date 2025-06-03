import os
import cv2
import torch
import numpy as np
from glob import glob
from torchvision.utils import flow_to_image

def create_combined_flow_video_with_torchvision(input_dir: str, image_dir: str, out_video: str = "combined_output.mp4"):
    # Limit to 20 frames
    flow_files = sorted(glob(os.path.join(input_dir, "*_flow.npy")))[:20]
    mask_files = sorted(glob(os.path.join(input_dir, "*_mask.npy")))[:20]
    image_files = sorted(glob(os.path.join(image_dir, "*.png")) + glob(os.path.join(image_dir, "*.jpg")))[:20]

    num_samples = min(len(flow_files), len(mask_files), len(image_files))
    if num_samples < 1:
        raise RuntimeError("Not enough valid input files (need at least 1 of each).")

    flow = np.load(flow_files[0])  # shape: (H, W, 2)
    H, W = flow.shape[:2]
    out_w = W * 3

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5  # 🔥 Set FPS to 5
    writer = cv2.VideoWriter(out_video, fourcc, fps, (out_w, H))

    for flow_path, mask_path, img_path in zip(flow_files, mask_files, image_files):
        # Load tensors
        flow_np = np.load(flow_path).astype(np.float32)
        mask_np = np.load(mask_path).astype(np.float32)

        flow_tensor = torch.from_numpy(flow_np).permute(2, 0, 1)

        # Normalize flow
        max_mag = torch.norm(flow_tensor, dim=0).max()
        flow_tensor = flow_tensor / (max_mag + 1e-8)
        flow_tensor = flow_tensor.clamp(-1, 1)

        flow_rgb = flow_to_image(flow_tensor)
        flow_rgb_np = flow_rgb.permute(1, 2, 0).cpu().numpy()

        # Masked flow
        mask_tensor = 1.0 - (torch.from_numpy(mask_np) / 100.0)
        mask_expanded = mask_tensor.unsqueeze(0).expand_as(flow_rgb.float() / 255.0)
        masked_flow = ((flow_rgb.float() / 255.0) * mask_expanded).clamp(0, 1)
        masked_flow_np = (masked_flow * 255).byte().permute(1, 2, 0).cpu().numpy()

        # Load image
        image = cv2.imread(img_path)
        if image.shape[:2] != (H, W):
            image = cv2.resize(image, (W, H))

        combined = np.hstack([flow_rgb_np, masked_flow_np, image])
        writer.write(combined)

    writer.release()
    print(f"✅ Saved: {out_video} with {num_samples} frames at 5 FPS")


if __name__  == "__main__":
    create_combined_flow_video_with_torchvision(
    input_dir="comp447_project/tartanair_dataset/train_data/amusement/Easy/P002/flow", 
    image_dir="comp447_project/tartanair_dataset/train_data/amusement/Easy/P002/image_left",
    out_video="tartanair_combined_visualization.mp4"
)