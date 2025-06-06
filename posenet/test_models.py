import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.tartanair import TartanAirDataset
from torch.utils.data import DataLoader, Subset
import logging
from model import PoseNetDinoImprovedContrastive

logger = logging.getLogger(__name__)

config = {
    "image_size": (224, 224),
    "train_seq_len": 6,
    "val_seq_len": 6,
    "train_subset_ratio": 0.01,
    "val_subset_ratio": 0.01,
    "skip": 1,
    "batch_size": 8,
    "num_workers": 4,
    "pin_memory": True,
    "drop_last": True
}

logger.info("============= Loading Datasets =============")
train_dataset = TartanAirDataset(
    root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data/", 
    size=config["image_size"], 
    seq_len=config["train_seq_len"],
    track_sequences=False,
    skip=config["skip"]
)

val_dataset = TartanAirDataset(
    root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/cvpr_data/", 
    size=config["image_size"], 
    seq_len=config["val_seq_len"],
    track_sequences=True
)
print(len(train_dataset))
print(len(val_dataset))
print(len(train_dataset.poses))
print(len(train_dataset.image_files))

print("================================================")

# Create subsets for faster training
train_size = int(len(train_dataset) * config["train_subset_ratio"])
val_size = int(len(val_dataset) * config["val_subset_ratio"])

train_indices = list(range(0, len(train_dataset), len(train_dataset) // train_size))
val_indices = list(range(0, len(val_dataset), len(val_dataset) // val_size))

train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)

logger.info(f"Train dataset size: {len(train_dataset)}")
logger.info(f"Validation dataset size: {len(val_dataset)}")

original_val_dataset = val_dataset.dataset
if hasattr(original_val_dataset, 'get_sequence_names') and original_val_dataset.track_sequences:
    sequence_names = original_val_dataset.get_sequence_names()
    logger.info(f"Sequences in validation set: {sequence_names}")

train_loader = DataLoader(
    train_dataset, 
    batch_size=config["batch_size"], 
    shuffle=True, 
    num_workers=config["num_workers"], 
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=config["batch_size"], 
    shuffle=False, 
    num_workers=config["num_workers"], 
    pin_memory=True,
    drop_last=False
)


model = PoseNetDinoImprovedContrastive(model_size='base', freeze_dino=True)

for batch in train_loader:
    images, t_gt, q_gt = batch
    t_pred, q_pred, dino_proj, trans_proj = model(images, return_contrastive_features=True)
    loss = model.pose_loss(t_pred, q_pred, t_gt, q_gt, dino_proj, trans_proj)
    print("loss: ", loss)

    print("t_pred.shape: ", t_pred.shape)
    print("q_pred.shape: ", q_pred.shape)
    print("dino_proj.shape: ", dino_proj.shape)
    print("trans_proj.shape: ", trans_proj.shape)

    break
