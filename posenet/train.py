import sys
import os
from torch.utils.data import DataLoader
from model import PoseNet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.tartanair import TartanAirDataset

def main():
    dataset = TartanAirDataset(root_dir="data/image_left")
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    pose_net = PoseNet()
    images, translations, rotations = next(iter(train_loader))
    for i in range(10):
        t, q = pose_net(images)
        loss = pose_net.pose_loss(t, q, translations[i], rotations[i])
        print(loss)
        
    return
    


if __name__ == "__main__":
    main()

