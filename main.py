from dataset import VideoSuperResolution
from model import BetaVAE_H

from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset = VideoSuperResolution(240, 480, start=100, stop=1000)
    batch_size = 1
    shuffle = False
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    model = BetaVAE_H()

    model = model.cuda()

    for x, y in dataloader:
        print(x.shape)
        x = x.cuda()
        y = y.cuda()
        out, _, _ = model.forward(x)

        print(out.shape)

