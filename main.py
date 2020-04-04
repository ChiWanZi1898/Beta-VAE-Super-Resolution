from dataset import VideoSuperResolution
from model import BetaVAE_H
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import torch.optim as optim
import PIL.Image as Image
import os

IMAGE_FOLDER = 'export'
MODEL_FOLDER = 'saved_models'


def reconstruction_loss(x, x_recon, distribution="gaussian"):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None
    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld


def export_image(epoch, cnt, batch_a, batch_b):
    batch_size = batch_a.shape[0]
    batch_a = F.sigmoid(batch_a.clone())
    for i in range(batch_size):
        if int(batch_size / 4) > 0 and i % int(batch_size / 4) == 0:
            image_a = ToPILImage()(batch_a[i].cpu().clone())
            image_b = ToPILImage()(batch_b[i].cpu().clone())
            w, h = image_a.size
            merged = Image.new(image_a.mode, (w * 2, h))
            merged.paste(image_a, box=(0, 0))
            merged.paste(image_b, box=(w, 0))
            merged_name = f'e{epoch}_cnt{cnt}_b{i}.jpg'

            export_folder = os.path.join(IMAGE_FOLDER, f'e{epoch}')
            if not os.path.exists(export_folder):
                os.makedirs(export_folder)
            image_path = os.path.join(export_folder, merged_name)
            print(f"Exporting {image_path}")
            merged.save(image_path)


def save_model(model, epoch, loss):
    torch.save(model, os.path.join(MODEL_FOLDER, f'e_{epoch}_loss_{loss}.model'))


if __name__ == "__main__":
    dataset = VideoSuperResolution(240, 480, start=100, stop=5000)
    batch_size = 64
    beta = 4  # beta-VAE's beta

    shuffle = True
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    model = BetaVAE_H()

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    model = model.cuda()

    for epoch in range(1000000):
        total_loss = 0
        total_recon_loss = 0
        epoch_kld = 0
        cnt = 0
        for x, y in dataloader:
            x = x.cuda()
            y = y.cuda()
            x_recon, mu, logvar = model.forward(x)
            recon_loss = reconstruction_loss(y, x_recon)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            beta_vae_loss = recon_loss + beta * total_kld

            if epoch % 10 == 0:
                export_image(epoch, cnt, x_recon, y)
                save_model(model, epoch, beta_vae_loss.item())

            cnt += 1
            total_loss += beta_vae_loss.item()
            total_recon_loss += recon_loss.item()
            epoch_kld += total_kld.item()

            optimizer.zero_grad()
            beta_vae_loss.backward()
            optimizer.step()
        print(
            f'epoch:{epoch} | loss:{total_loss / cnt} | recon:{total_recon_loss / cnt} | kld:{epoch_kld / cnt}')
