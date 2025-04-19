from model import PoseNet
import torch

def main():
    model = PoseNet()

    model = PoseNet()
    dummy = torch.randn(4, 6, 3, 224, 224)
    t, q = model(dummy)
    print(t.shape, q.shape)  # [4, 3], [4, 4]
    print(t)
    print(q)

if __name__ == '__main__':
    main()