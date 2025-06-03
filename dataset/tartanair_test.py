import os
import numpy as np
import logging
from tartanair import TartanAirDataset

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting TartanAir dataset test")
    
    logger.info("Loading training dataset...")
    train_dataset = TartanAirDataset(
        root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data/", 
        size=(224, 224), 
        seq_len=2,
        track_sequences=False  # No need for training
    )

    logger.info("Loading validation dataset...")
    val_dataset = TartanAirDataset(
        root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/cvpr_data/", 
        size=(224, 224), 
        seq_len=2,
        track_sequences=True  # Enable for per-sequence evaluation
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    

if __name__ == "__main__":
    main()

