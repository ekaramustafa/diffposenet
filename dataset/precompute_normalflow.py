import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing

# === Project import ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nflownet.utils import compute_normal_flow

torch.set_grad_enabled(False)

IMG_TRANSFORM = transforms.ToTensor()
CROP_DIV = 16

def crop_to_divisible(img: torch.Tensor):
    C, H, W = img.shape
    new_H = H - (H % CROP_DIV)
    new_W = W - (W % CROP_DIV)
    return img[:, :new_H, :new_W]

def read_image(path):
    img = Image.open(path).convert("RGB")
    return crop_to_divisible(IMG_TRANSFORM(img))

def read_opt_flow(path):
    flow = np.load(path)
    return crop_to_divisible(torch.from_numpy(flow).permute(2, 0, 1).float())

def process_and_save(img1_path, img2_path, flow_path, save_path):
    try:
        if Path(save_path).exists():
            return

        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        paired = torch.cat([img1, img2], dim=0)
        flow = read_opt_flow(flow_path)
        normal_flow = compute_normal_flow(flow, paired)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(normal_flow, save_path)

    except Exception as e:
        rel_parts = Path(save_path).relative_to(ROOT_DIR).parts
        env = rel_parts[0] if len(rel_parts) > 0 else "UNKNOWN_ENV"
        traj = rel_parts[2] if len(rel_parts) > 2 else "UNKNOWN_TRAJ"
        print(f"❌ Failed in {env}/{traj} → {save_path.name}: {e}")

def collect_all_traj_paths(root_dir, env_name):
    samples = []
    root_dir = Path(root_dir)
    for env in root_dir.iterdir():
        if env.name != env_name:
            continue

        for diff in (env / "Easy").iterdir():
            traj_path = diff
            image_dir = traj_path / "image_left"
            flow_dir = traj_path / "flow"
            normal_dir = traj_path / "unmasked_normal_flow"

            if image_dir.exists() and flow_dir.exists():
                imgs = sorted(image_dir.glob("*.png"))
                flows = sorted(flow_dir.glob("*_flow.npy"))

                if len(imgs) == len(flows) + 1:
                    for i in range(len(flows)):
                        save_path = normal_dir / f"normal_{i:06d}.pt"
                        samples.append((
                            str(imgs[i]), str(imgs[i+1]),
                            str(flows[i]), save_path
                        ))
    return samples

if __name__ == "__main__":
    ROOT_DIR = Path("/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data")
    ENV_LIST = ["amusement", "japanesealley", "office", "office2", "oldtown", "seasidetown", "soulcity"]

    for env in ENV_LIST:
        ENV_NAME = env
        print(f"🔍 Scanning {ENV_NAME} dataset...")
        all_samples = collect_all_traj_paths(ROOT_DIR, ENV_NAME)
        print(f"🧾 Total samples to process: {len(all_samples)}")

        NUM_CPUS = 20
        print(f"🚀 Using {NUM_CPUS} CPU cores...")

        Parallel(n_jobs=NUM_CPUS, verbose=10)(
            delayed(process_and_save)(img1, img2, flow, out)
            for (img1, img2, flow, out) in all_samples
        )

        print("✅ All normal flows generated.")
