#!/usr/bin/env python
# coding: utf-8

# ### Execute in the Terminal:
#
# Only once you have setup s3fs, you can execute the following command in Terminal:
#
# `s3fs your-name -o use_cache=/tmp -o allow_other -o uid=1001 -o mp_umask=002 -o multireq_max=5 ./DATA`

# In[ ]:


import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import pandas as pd

import torchvision
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from baseline import *


# In[ ]:


class BaseEncoder(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(BaseEncoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.Flatten(),  # B, 256
            nn.Linear(6400, z_dim * 2),  # B, z_dim*2
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]

        return mu, logvar

    def _encode(self, x):
        return self.encoder(x)


class BaseDecoder(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(BaseDecoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 6400),  # B, 256
            View((-1, 256, 5, 5)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        x_recon = self._decode(z)

        return x_recon

    def _decode(self, z):
        return self.decoder(z)


class BaseVAE(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(BaseVAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc

        self.encoder = BaseEncoder(z_dim, nc)
        self.decoder = BaseDecoder(z_dim, nc)

    def forward(self, x):
        mu, logvar = self._encode(x)
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


# In[ ]:


class StudentEncoder(nn.Module):
    def __init__(self, z_dim=20, nc=3):
        super(StudentEncoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.Flatten(),  # B, 256
            nn.Linear(6400, 3200),  # B, z_dim*2
            nn.Linear(3200, z_dim * 2),  # B, z_dim*2
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]

        return mu, logvar

    def _encode(self, x):
        return self.encoder(x)


class StudentDecoder(nn.Module):
    def __init__(self, z_dim=20, nc=3):
        super(StudentDecoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 3200),  # B, 256
            nn.Linear(3200, 6400),  # B, 256
            View((-1, 256, 5, 5)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        x_recon = self._decode(z)

        return x_recon

    def _decode(self, z):
        return self.decoder(z)


class StudentVAE(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(StudentVAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc

        self.encoder = StudentEncoder(z_dim, nc)
        self.decoder = StudentDecoder(z_dim, nc)

    def forward(self, x):
        mu, logvar = self._encode(x)
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


# In[ ]:


# !pip install torchsummary


# In[ ]:


x = torch.ones(1, 3, 128, 128)

x = BaseEncoder()(x)
print(x[0].shape)

# In[ ]:


from torchsummary import summary

model = BaseVAE().cuda()
summary(model, input_size=(3, 128, 128))

# In[ ]:


from torchsummary import summary

model = StudentVAE().cuda()
summary(model, input_size=(3, 128, 128))

# In[ ]:


# In[ ]:


import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

# In[ ]:


# !pip install tensorboard


# In[ ]:


import os
from torch.utils.tensorboard import SummaryWriter

# In[ ]:


from tqdm import tqdm


def train(net, C_max, use_cuda, max_iter, global_iter, decoder_dist,
          beta, C_stop_iter, objective, gamma, optim):
    net.train()

    logs_base_dir = "runs"
    os.makedirs(logs_base_dir, exist_ok=True)
    train_summary_writer = SummaryWriter()

    if use_cuda == True:
        C_max = Variable(torch.FloatTensor([C_max])).cuda()
    else:
        C_max = Variable(torch.FloatTensor([C_max]))
    out = False

    pbar = tqdm(total=max_iter)
    pbar.update(global_iter)

    BCE_list = []
    KLD_list = []
    TL_list = []

    test_BCE_list = []
    test_KLD_list = []
    test_TL_list = []

    while not out:
        for x in data_loader:
            x = x.cuda()
            net = net.cuda()
            net.train()

            global_iter += 1
            pbar.update(1)

            if use_cuda == True:
                x = Variable(x).cuda()
            else:
                x = Variable(x)

            x_recon, mu, logvar = net(x)
            recon_loss = reconstruction_loss(x, x_recon, decoder_dist)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            beta_vae_loss = recon_loss + beta * total_kld

            BCE = recon_loss
            KLD = total_kld
            total_loss = beta_vae_loss

            BCE_list.append(BCE.item())
            KLD_list.append(KLD.item())
            TL_list.append(total_loss.item())

            beta_vae_loss.backward()
            optim.step()
            optim.zero_grad()

            net.eval()

            test_KLD = 0
            test_BCE = 0
            test_TL = 0

            for y in test_loader:
                y = y.cuda()
                y = Variable(y)

                y_recon, mu, logvar = net(y)
                recon_loss = reconstruction_loss(y, y_recon, decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                beta_vae_loss = recon_loss + beta * total_kld

                test_KLD += total_kld.item()
                test_BCE += recon_loss.item()
                test_TL += beta_vae_loss.item()

                del y
                del dim_wise_kld
                del mean_kld
                del total_kld
                del recon_loss
                del beta_vae_loss

            test_BCE_list.append(test_BCE / 14196)
            test_KLD_list.append(test_KLD / 14196)
            test_TL_list.append(test_TL / 14196)

            if global_iter >= max_iter:
                out = True
                break

    pbar.write("[Training Finished]")
    pbar.close()

    return BCE_list, KLD_list, TL_list, test_BCE_list, test_KLD_list, test_TL_list


# In[ ]:


dataset = YourName()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, num_workers=8, shuffle=True, drop_last=True)

test_dataset = YourName(train=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, num_workers=8, shuffle=True, drop_last=True)


# In[ ]:


# # !rm -rf runs
# %reload_ext tensorboard
# %tensorboard --port=6006 --logdir=runs


# In[ ]:


def TRAIN(BETA_PARAM, return_model=False):
    C_max = 25
    max_iter = len(data_loader)
    # max_iter = 100
    global_iter = 0
    decoder_dist = "gaussian"
    beta = 0.5
    C_stop_iter = 1e5
    gamma = 1000
    use_cuda = True
    # use_cuda = False
    objective = "H"

    if use_cuda == False:
        net1 = BaseVAE()
    else:
        net1 = BaseVAE().cuda()

    optim1 = torch.optim.Adam(net1.parameters(),
                              lr=1e-4,
                              betas=(0.9, 0.999))

    PATH = "./base-model-" + str(BETA_PARAM) + ".pt"
    print("Now training:", PATH)
    if not os.path.exists(PATH):
        BASELINE_BCE, BASELINE_KLD, BASELINE_TL, BASELINE_TEST_BCE, BASELINE_TEST_KLD, BASELINE_TEST_TL = train(net1,
                                                                                                                C_max,
                                                                                                                use_cuda,
                                                                                                                max_iter,
                                                                                                                global_iter,
                                                                                                                decoder_dist,
                                                                                                                BETA_PARAM,
                                                                                                                C_stop_iter,
                                                                                                                objective,
                                                                                                                gamma,
                                                                                                                optim1)

        PATH = "./base-model-" + str(BETA_PARAM) + ".pt"

        torch.save({'model_state_dict': net1.state_dict(),
                    'BASELINE_BCE': BASELINE_BCE,
                    'BASELINE_KLD': BASELINE_KLD,
                    'BASELINE_TL': BASELINE_TL,
                    'BASELINE_TEST_BCE': BASELINE_TEST_BCE,
                    'BASELINE_TEST_KLD': BASELINE_TEST_KLD,
                    'BASELINE_TEST_TL': BASELINE_TEST_TL,
                    'BETA_PARAM': BETA_PARAM}, PATH)

    if use_cuda == False:
        net2 = StudentVAE()

    else:
        net2 = StudentVAE().cuda()

    optim2 = torch.optim.Adam(net2.parameters(),
                              lr=1e-4,
                              betas=(0.9, 0.999))
    PATH = "./student-model-" + str(BETA_PARAM) + ".pt"
    print("Now training:", PATH)
    if not os.path.exists(PATH):
        OUR_MODEL_BCE, OUR_MODEL_KLD, OUR_MODEL_TL, OUR_MODEL_TEST_BCE, OUR_MODEL_TEST_KLD, OUR_MODEL_TEST_TL = train(
            net2,
            C_max,
            use_cuda,
            max_iter,
            global_iter,
            decoder_dist,
            BETA_PARAM,
            C_stop_iter,
            objective,
            gamma,
            optim2)

        PATH = "./student-model-" + str(BETA_PARAM) + ".pt"

        torch.save({'model_state_dict': net2.state_dict(),
                    'OUR_MODEL_BCE': OUR_MODEL_BCE,
                    'OUR_MODEL_KLD': OUR_MODEL_KLD,
                    'OUR_MODEL_TL': OUR_MODEL_TL,
                    'OUR_MODEL_TEST_BCE': OUR_MODEL_TEST_BCE,
                    'OUR_MODEL_TEST_KLD': OUR_MODEL_TEST_KLD,
                    'OUR_MODEL_TEST_TL': OUR_MODEL_TEST_TL,
                    'BETA_PARAM': BETA_PARAM}, PATH)

    if return_model == True:
        return net1, net2


#     else:
#         return BASELINE_BCE, BASELINE_KLD, BASELINE_TL, OUR_MODEL_BCE, OUR_MODEL_KLD, OUR_MODEL_TL


# In[ ]:


# In[ ]:


# BCE_baseline_list = []
# KLD_baseline_list = []
# TL_baseline_list = []

# BCE_ours_list = []
# KLD_ours_list = []
# TL_ours_list = []

for beta_param in [2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 2]:
    TRAIN(beta_param)

#     BASELINE_BCE, BASELINE_KLD, BASELINE_TL, OUR_MODEL_BCE, OUR_MODEL_KLD, OUR_MODEL_TL = TRAIN(beta_param)

#     BCE_baseline_list.append(BASELINE_BCE)
#     KLD_baseline_list.append(BASELINE_KLD)
#     TL_baseline_list.append(BASELINE_TL)

#     BCE_ours_list.append(OUR_MODEL_BCE)
#     KLD_ours_list.append(OUR_MODEL_KLD)
#     TL_ours_list.append(OUR_MODEL_TL)


# In[ ]:


# ===


# In[ ]:


PATH1 = ["./base-model-0.25.pt",
         "./base-model-0.5.pt",
         "./base-model-1.pt",
         "./base-model-2.pt",
         "./base-model-4.pt"]

PATH2 = ["./student-model-0.25.pt",
         "./student-model-0.5.pt",
         "./student-model-1.pt",
         "./student-model-2.pt",
         "./student-model-4.pt"]

BASELINE_BCE = []
BASELINE_KLD = []
BASELINE_TL = []

BASELINE_TEST_BCE = []
BASELINE_TEST_KLD = []
BASELINE_TEST_TL = []

OUR_MODEL_BCE = []
OUR_MODEL_KLD = []
OUR_MODEL_TL = []

OUR_MODEL_TEST_BCE = []
OUR_MODEL_TEST_KLD = []
OUR_MODEL_TEST_TL = []

for path1, path2 in zip(PATH1, PATH2):
    checkpoint = torch.load(path1, map_location=torch.device('cpu'))
    BASELINE_BCE.append(checkpoint['BASELINE_BCE'])
    BASELINE_KLD.append(checkpoint['BASELINE_KLD'])
    BASELINE_TL.append(checkpoint['BASELINE_TL'])

    BASELINE_TEST_BCE.append(checkpoint['BASELINE_TEST_BCE'])
    BASELINE_TEST_KLD.append(checkpoint['BASELINE_TEST_KLD'])
    BASELINE_TEST_TL.append(checkpoint['BASELINE_TEST_TL'])

    checkpoint = torch.load(path2, map_location=torch.device('cpu'))
    OUR_MODEL_BCE.append(checkpoint['OUR_MODEL_BCE'])
    OUR_MODEL_KLD.append(checkpoint['OUR_MODEL_KLD'])
    OUR_MODEL_TL.append(checkpoint['OUR_MODEL_TL'])

    OUR_MODEL_TEST_BCE.append(checkpoint['OUR_MODEL_TEST_BCE'])
    OUR_MODEL_TEST_KLD.append(checkpoint['OUR_MODEL_TEST_KLD'])
    OUR_MODEL_TEST_TL.append(checkpoint['OUR_MODEL_TEST_TL'])

for path1, path2 in zip(PATH1, PATH2):
    checkpoint = torch.load(path1, map_location=torch.device('cpu'))

    checkpoint = torch.load(path2, map_location=torch.device('cpu'))

    del checkpoint

# In[ ]:


checkpoint.keys()

# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

plt.close('all')
fig = plt.figure(1, (25., 5.))

metric = "BCE"

res = ["Resolution = 128x128"]

cols = ["Beta = 0.25", "Beta = 0.5", "Beta = 1", "Beta = 2", "Beta = 4"]

# res = ["Resolution = 142p",
#        "Resolution = 243p",
#        "Resolution = 320p",
#        "Resolution = 480p",
#        "Resolution = 720p",
#        "Resolution = 1080p"]

grid = Grid(fig,
            rect=111,
            nrows_ncols=(len(res), len(cols)),
            axes_pad=0.1)

for i in range(len(cols) * len(res)):

    data1 = np.array([d for d in BASELINE_BCE[i]])
    data2 = np.array([d for d in OUR_MODEL_BCE[i]])

    iterations = range(len(data1))

    grid[i].plot(iterations, data1, 'g', label='Baseline')
    grid[i].plot(iterations, data2, 'b', label='Our Model')
    grid[i].legend(loc="upper left")

    if i % 5 == 0:
        grid[i].set_ylabel(metric, rotation=90, size='large')

    if i % 5 == 4:
        grid[i].annotate(res[i // 5], xy=(1.1, 0.5), rotation=270, size='large',
                         ha='center', va='center', xycoords='axes fraction')

    if i >= 24:
        grid[i].set_xlabel("Iteration", rotation=0, size='large')

for ax, col in zip(grid, cols):
    ax.set_title(col)

plt.show()

# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

plt.close('all')
fig = plt.figure(1, (25., 5.))

metric = "KLD"

# cols = ["Beta = 1"]
res = ["Resolution = 128x128"]

cols = ["Beta = 0.25", "Beta = 0.5", "Beta = 1", "Beta = 2", "Beta = 4"]

# res = ["Resolution = 142p",
#        "Resolution = 243p",
#        "Resolution = 320p",
#        "Resolution = 480p",
#        "Resolution = 720p",
#        "Resolution = 1080p"]

grid = Grid(fig,
            rect=111,
            nrows_ncols=(len(res), len(cols)),
            axes_pad=0.1)

for i in range(len(cols) * len(res)):

    data1 = np.array([d for d in BASELINE_KLD[i]])
    data2 = np.array([d for d in OUR_MODEL_KLD[i]])

    iterations = range(len(data1))

    grid[i].plot(iterations, data1, 'g', label='Baseline')
    grid[i].plot(iterations, data2, 'b', label='Our Model')
    grid[i].legend(loc="upper left")

    if i % 5 == 0:
        grid[i].set_ylabel(metric, rotation=90, size='large')

    if i % 5 == 4:
        grid[i].annotate(res[i // 5], xy=(1.1, 0.5), rotation=270, size='large',
                         ha='center', va='center', xycoords='axes fraction')

    grid[i].set_xlabel("Batch Iteration", rotation=0, size='large')

for ax, col in zip(grid, cols):
    ax.set_title(col)

plt.show()

# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

plt.close('all')
fig = plt.figure(1, (25., 5.))

metric = "Total Loss"

# cols = ["Beta = 1"]
res = ["Resolution = 128x128"]

cols = ["Beta = 0.25", "Beta = 0.5", "Beta = 1", "Beta = 2", "Beta = 4"]

# res = ["Resolution = 142p",
#        "Resolution = 243p",
#        "Resolution = 320p",
#        "Resolution = 480p",
#        "Resolution = 720p",
#        "Resolution = 1080p"]

grid = Grid(fig,
            rect=111,
            nrows_ncols=(len(res), len(cols)),
            axes_pad=0.1)

for i in range(len(cols) * len(res)):

    data1 = np.array([d for d in BASELINE_TL[i]])
    data2 = np.array([d for d in OUR_MODEL_TL[i]])

    iterations = range(len(data1))

    grid[i].plot(iterations, data1, 'g', label='Baseline')
    grid[i].plot(iterations, data2, 'b', label='Our Model')
    grid[i].legend(loc="upper left")

    if i % 5 == 0:
        grid[i].set_ylabel(metric, rotation=90, size='large')

    if i % 5 == 4:
        grid[i].annotate(res[i // 5], xy=(1.1, 0.5), rotation=270, size='large',
                         ha='center', va='center', xycoords='axes fraction')

    grid[i].set_xlabel("Batch Iteration", rotation=0, size='large')

for ax, col in zip(grid, cols):
    ax.set_title(col)

plt.show()

# ---

# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

plt.close('all')
fig = plt.figure(1, (25., 5.))

metric = "BCE"

# cols = ["Beta = 1"]
res = ["Resolution = 128x128"]

cols = ["Beta = 0.25", "Beta = 0.5", "Beta = 1", "Beta = 2", "Beta = 4"]

# res = ["Resolution = 142p",
#        "Resolution = 243p",
#        "Resolution = 320p",
#        "Resolution = 480p",
#        "Resolution = 720p",
#        "Resolution = 1080p"]

grid = Grid(fig,
            rect=111,
            nrows_ncols=(len(res), len(cols)),
            axes_pad=0.1)

for i in range(len(cols) * len(res)):

    data1 = np.array([d for d in BASELINE_TEST_BCE[i]])
    data2 = np.array([d for d in OUR_MODEL_TEST_BCE[i]])

    iterations = range(len(data1))

    grid[i].plot(iterations, data1, 'g', label='Baseline')
    grid[i].plot(iterations, data2, 'b', label='Our Model')
    grid[i].legend(loc="upper left")

    if i % 5 == 0:
        grid[i].set_ylabel(metric, rotation=90, size='large')

    if i % 5 == 4:
        grid[i].annotate(res[i // 5], xy=(1.1, 0.5), rotation=270, size='large',
                         ha='center', va='center', xycoords='axes fraction')

    grid[i].set_xlabel("Batch Iteration", rotation=0, size='large')

for ax, col in zip(grid, cols):
    ax.set_title(col)

plt.show()

# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

plt.close('all')
fig = plt.figure(1, (25., 5.))

metric = "KLD"

# cols = ["Beta = 1"]
res = ["Resolution = 128x128"]

cols = ["Beta = 0.25", "Beta = 0.5", "Beta = 1", "Beta = 2", "Beta = 4"]

# res = ["Resolution = 142p",
#        "Resolution = 243p",
#        "Resolution = 320p",
#        "Resolution = 480p",
#        "Resolution = 720p",
#        "Resolution = 1080p"]

grid = Grid(fig,
            rect=111,
            nrows_ncols=(len(res), len(cols)),
            axes_pad=0.1)

for i in range(len(cols) * len(res)):

    data1 = np.array([d for d in BASELINE_TEST_KLD[i]])
    data2 = np.array([d for d in OUR_MODEL_TEST_KLD[i]])

    iterations = range(len(data1))

    grid[i].plot(iterations, data1, 'g', label='Baseline')
    grid[i].plot(iterations, data2, 'b', label='Our Model')
    grid[i].legend(loc="upper left")

    if i % 5 == 0:
        grid[i].set_ylabel(metric, rotation=90, size='large')

    if i % 5 == 4:
        grid[i].annotate(res[i // 5], xy=(1.1, 0.5), rotation=270, size='large',
                         ha='center', va='center', xycoords='axes fraction')

    grid[i].set_xlabel("Batch Iteration", rotation=0, size='large')

for ax, col in zip(grid, cols):
    ax.set_title(col)

plt.show()

# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

plt.close('all')
fig = plt.figure(1, (25., 5.))

metric = "Total Loss"

# cols = ["Beta = 1"]
res = ["Resolution = 128x128"]

cols = ["Beta = 0.25", "Beta = 0.5", "Beta = 1", "Beta = 2", "Beta = 4"]

# res = ["Resolution = 142p",
#        "Resolution = 243p",
#        "Resolution = 320p",
#        "Resolution = 480p",
#        "Resolution = 720p",
#        "Resolution = 1080p"]

grid = Grid(fig,
            rect=111,
            nrows_ncols=(len(res), len(cols)),
            axes_pad=0.1)

for i in range(len(cols) * len(res)):

    data1 = np.array([d for d in BASELINE_TEST_TL[i]])
    data2 = np.array([d for d in OUR_MODEL_TEST_TL[i]])

    iterations = range(len(data1))

    grid[i].plot(iterations, data1, 'g', label='Baseline')
    grid[i].plot(iterations, data2, 'b', label='Our Model')
    grid[i].legend(loc="upper left")

    if i % 5 == 0:
        grid[i].set_ylabel(metric, rotation=90, size='large')

    if i % 5 == 4:
        grid[i].annotate(res[i // 5], xy=(1.1, 0.5), rotation=270, size='large',
                         ha='center', va='center', xycoords='axes fraction')

    grid[i].set_xlabel("Batch Iteration", rotation=0, size='large')

for ax, col in zip(grid, cols):
    ax.set_title(col)

plt.show()

# In[ ]:


### Section on Scene Identificiation


# In[ ]:


BETA_PARAM = 2
base, student = TRAIN(BETA_PARAM, return_model=True)

# In[ ]:


print(base)

# In[ ]:


print(student)

# In[ ]:


BaseLatentSpace = BaseEncoder().cpu()
BaseLatentSpace.load_state_dict(base.cpu().encoder.state_dict())

del base

StudentLatentSpace = StudentEncoder().cpu()
StudentLatentSpace.load_state_dict(student.cpu().encoder.state_dict())

del student

# In[ ]:


# In[ ]:


test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=True)


# In[ ]:


def KL_Div(mu0, mu1, var0, var1):
    KLD = 0.5 * (var0 / var1) + (mu1 - mu0) * (1 / var1) * (mu1 - mu0) - 1 + np.log(var1 / var0)

    return KLD


def KL_Div_Loss(mu0_vector, mu1_vector, var0_vector, var1_vector):
    total_KLD = 0

    for (mu0, mu1, var0, var1) in zip(mu0_vector, mu1_vector, var0_vector, var1_vector):
        total_KLD += KL_Div(mu0, mu1, var0, var1)

    return total_KLD


def multi_KLD(LatentSpace):
    mu = None
    var = None

    max_idx = len(data_loader)

    KLD = []
    for idx, batch in enumerate(test_loader):

        if idx == 0:
            mu0, var0 = LatentSpace(batch)
            mu0, var0 = mu0.detach().numpy().squeeze(0), var0.detach().numpy().squeeze(0)

            print("mu shape:", mu0.shape)
            print("mu shape:", var0.shape)

        else:
            mu1, var1 = LatentSpace(batch)
            mu1, var1 = mu1.detach().numpy().squeeze(0), var1.detach().numpy().squeeze(0)

            KLD.append(KL_Div_Loss(mu0, mu1, var0, var1))

            mu0, var0 = mu1, var1

        if (idx + 1) % 10000 == 0:
            print(idx, "/", max_idx)

    print("Done.")

    return KLD


# In[ ]:


# In[ ]:


# In[ ]:


BaseLatentSpace.eval()
baseKLD = multi_KLD(BaseLatentSpace)

# In[ ]:


StudentLatentSpace.eval()
studentKLD = multi_KLD(StudentLatentSpace)

# In[ ]:


len([KLD if KLD < 1000 else 0 for KLD in baseKLD])

# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

plt.close('all')
fig = plt.figure(1, (20., 10.))

# rows = ["KL-Div (Our Model)", "KL-Div (Baseline)", "L1 Loss", "L2 Loss"]
# rows = ["KL-Div (Our Model)", "KL-Div (Baseline)"]
rows = ["KL-Div (Our Model)"]

grid = Grid(fig,
            rect=111,
            nrows_ncols=(len(rows), 1),
            axes_pad=0.1)

# data = [studentKLD, baseKLD]
data = [[KLD if KLD < 1000 else 0 for KLD in baseKLD]]

for i in range(len(rows)):

    #     epochs = range(len(data[i]))
    iterations = range(0, 14195)

    TRUE_SCENES_LIST  # length 259
    PRED_SCENES_LIST  # length 259

    grid[i].plot(iterations, [data[i][j] for j in iterations], 'g', label='Distance')

    grid[i].set_ylabel(rows[i], rotation=90, size='large')
    if i == 3:
        grid[i].set_xlabel("Iteration", rotation=0, size='large')

#     # TODO: Check if top 259 scenes are correctly identified

#     assert(len(TRUE_SCENES_LIST) == 259)
#     assert(len(PRED_SCENES_LIST) == 259)

#     correct = 0
#     for true_scene in TRUE_SCENES_LIST:

#         if true_scene in PRED_SCENES_LIST:
#             plt.axvline(true_scene,  label='pyplot vertical line', color = "b")
#             correct += 1
#         else::
#             plt.axvline(true_scene,  label='pyplot vertical line', color = "r")


#     test_accuracy = correct/len(TRUE_SCENES_LIST)


plt.show()

# In[ ]:


# In[ ]:
