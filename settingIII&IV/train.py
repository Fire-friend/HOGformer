import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import AdaIRTrainDataset
from net.model import HOGformer as AdaIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.nn.functional as F

class HOGLayer(torch.nn.Module):
    def __init__(self,
                 nbins=9,               # Number of orientation bins
                 cell_size=8,           # Cell size
                 block_size=2,          # Block size (in cells)
                 signed_gradient=False, # Whether to use signed gradients
                 eps=1e-8):             # Numerical stability
        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.cell_size = cell_size
        self.block_size = block_size
        self.signed_gradient = signed_gradient
        self.eps = eps
        if not self.signed_gradient:
            angles = torch.tensor([(i * np.pi / self.nbins) for i in range(self.nbins)])
            self.bin_width = np.pi / self.nbins
        else:
            angles = torch.tensor([(i * 2 * np.pi / self.nbins) for i in range(self.nbins)])
            self.bin_width = 2 * np.pi / self.nbins
        self.register_buffer('angles', angles.view(1,-1, 1, 1))
        self.register_buffer('dx_filter', torch.tensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).float().view(1, 1, 3, 3))
        self.register_buffer('dy_filter', torch.tensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).float().view(1, 1, 3, 3))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        if channels == 3:
            x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
            x_gray = x_gray.unsqueeze(1)
        else:
            x_gray = x

        dx = F.conv2d(x_gray, self.dx_filter, padding=1)
        dy = F.conv2d(x_gray, self.dy_filter, padding=1)
        magnitude = torch.sqrt(dx**2 + dy**2 + self.eps)
        orientation = torch.atan2(dy, dx + self.eps)

        if not self.signed_gradient:
            # Map to [0,π]
            orientation = torch.abs(orientation)

        delta = torch.abs(orientation - self.angles)
        if self.signed_gradient:
            delta = torch.min(delta, 2 * np.pi - delta)
        else:
            delta = torch.min(delta, np.pi - delta)
        weights = torch.relu(1.0 - delta / self.bin_width)
        new_height = (height // self.cell_size) * self.cell_size
        new_width = (width // self.cell_size) * self.cell_size
        if height % self.cell_size != 0 or width % self.cell_size != 0:
            magnitude = magnitude[:, :, :new_height, :new_width]
            weights = weights[:, :, :new_height, :new_width]

        weighted_magnitude = weights * magnitude
        hist = F.avg_pool2d(weighted_magnitude, kernel_size=self.cell_size, stride=self.cell_size)

        # Block normalization
        if self.block_size > 1:
            B, C, Hc, Wc = hist.shape
            if Hc >= self.block_size and Wc >= self.block_size:
                blocks = F.unfold(hist, kernel_size=self.block_size, stride=1)
                blocks = blocks.permute(0, 2, 1).reshape(-1, C * self.block_size**2)
                block_norm = torch.norm(blocks, p=2, dim=1, keepdim=True)
                blocks = blocks / (block_norm + self.eps)
                num_blocks = (Hc - self.block_size + 1) * (Wc - self.block_size + 1)
                out_hist = blocks.reshape(B, num_blocks, -1)
                out_hist = out_hist.reshape(B, -1)
            else:
                out_hist = hist.reshape(B, -1)
        else:
            # Don't use blocks, directly flatten
            out_hist = hist.reshape(batch_size, -1)

        return out_hist

class HOGLoss(torch.nn.Module):
    """
    HOG feature loss function
    """
    def __init__(self,
                 nbins=9,               # Number of orientation bins
                 cell_size=8,           # Cell size
                 block_size=2,          # Block size (in cells)
                 signed_gradient=False, # Whether to use signed gradients
                 loss_type='l1',        # Loss type: 'l1' or 'l2'
                 eps=1e-8):             # Numerical stability
        super(HOGLoss, self).__init__()
        self.hog_layer = HOGLayer(
            nbins=nbins,
            cell_size=cell_size,
            block_size=block_size,
            signed_gradient=signed_gradient,
            eps=eps
        )
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        Calculate HOG loss between predicted and target images

        Args:
            pred: Predicted image, [B, C, H, W]
            target: Target image, [B, C, H, W]

        Returns:
            loss: HOG feature loss
        """
        hog_pred = self.hog_layer(pred)
        hog_target = self.hog_layer(target)
        if self.loss_type.lower() == 'l1':
            loss = F.l1_loss(hog_pred, hog_target)
        else:
            loss = F.mse_loss(hog_pred, hog_target)
        return loss

class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR()
        self.loss_fn  = nn.L1Loss()
        self.cri_seq = self.pearson_correlation_loss
        self.cri_HOGloss = HOGLoss()

    def pearson_correlation_loss(self, x1, x2):
        assert x1.shape == x2.shape
        b, c = x1.shape[:2]
        dim = -1
        x1, x2 = x1.reshape(b, -1), x2.reshape(b, -1)
        x1_mean, x2_mean = x1.mean(dim=dim, keepdims=True), x2.mean(dim=dim, keepdims=True)
        numerator = ((x1 - x1_mean) * (x2 - x2_mean)).sum( dim=dim, keepdims=True )

        std1 = (x1 - x1_mean).pow(2).sum(dim=dim, keepdims=True).sqrt()
        std2 = (x2 - x2_mean).pow(2).sum(dim=dim, keepdims=True).sqrt()
        denominator = std1 * std2
        corr = numerator.div(denominator + 1e-6)
        return corr

    def compute_correlation_loss(self, x1, x2):
        b, c = x1.shape[0:2]
        x1 = x1.view(b, -1)
        x2 = x2.view(b, -1)
#        print(x1, x2)
        pearson = (1. - self.cri_seq(x1, x2)) / 2.
        return pearson[~pearson.isnan()*~pearson.isinf()].mean()

    def forward(self,x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        l_pear = self.compute_correlation_loss(restored,clean_patch)
        l_l1 = self.loss_fn(restored,clean_patch)

        # Skip HOG loss in the last 3 epochs
        if self.current_epoch < opt.epochs - 5:
            l_hog = self.cri_HOGloss(restored,clean_patch)
            loss = l_l1+l_pear+l_hog
        else:
            l_hog = torch.tensor(0.0, device=restored.device)
            loss = l_l1+l_pear

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss_hog", l_hog)
        self.log("train_loss", loss)
        self.log("train_loss_l1", l_l1)
        self.log("train_loss_pearson", l_pear)
        return loss

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=180)

        return [optimizer],[scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="AdaIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = AdaIRTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    model = AdaIRModel()

    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()
