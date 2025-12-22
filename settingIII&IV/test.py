import argparse
import os
import subprocess

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# from net.model import AdaIR
from net.model import HOGformer as AdaIR
from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.image_io import save_image_tensor
from utils.val_utils import AverageMeter, compute_psnr_ssim


class HOGLayer(torch.nn.Module):
    def __init__(
        self,
        nbins=9,  # Number of orientation bins
        cell_size=8,  # Cell size
        block_size=2,  # Block size (in cells)
        signed_gradient=False,  # Whether to use signed gradients
        eps=1e-8,
    ):  # Numerical stability
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
            angles = torch.tensor(
                [(i * 2 * np.pi / self.nbins) for i in range(self.nbins)]
            )
            self.bin_width = 2 * np.pi / self.nbins
        self.register_buffer("angles", angles.view(1, -1, 1, 1))
        self.register_buffer(
            "dx_filter",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3),
        )
        self.register_buffer(
            "dy_filter",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3),
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        if channels == 3:
            x_gray = (
                0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
            )
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
        hist = F.avg_pool2d(
            weighted_magnitude, kernel_size=self.cell_size, stride=self.cell_size
        )

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

    def __init__(
        self,
        nbins=9,  # Number of orientation bins
        cell_size=8,  # Cell size
        block_size=2,  # Block size (in cells)
        signed_gradient=False,  # Whether to use signed gradients
        loss_type="l1",  # Loss type: 'l1' or 'l2'
        eps=1e-8,
    ):  # Numerical stability
        super(HOGLoss, self).__init__()
        self.hog_layer = HOGLayer(
            nbins=nbins,
            cell_size=cell_size,
            block_size=block_size,
            signed_gradient=signed_gradient,
            eps=eps,
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
        if self.loss_type.lower() == "l1":
            loss = F.l1_loss(hog_pred, hog_target)
        else:
            loss = F.mse_loss(hog_pred, hog_target)
        return loss


class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR()
        self.loss_fn = nn.L1Loss()
        self.cri_seq = self.pearson_correlation_loss
        self.cri_HOGloss = HOGLoss()

    def pearson_correlation_loss(self, x1, x2):
        assert x1.shape == x2.shape
        b, c = x1.shape[:2]
        dim = -1
        x1, x2 = x1.reshape(b, -1), x2.reshape(b, -1)
        x1_mean, x2_mean = (
            x1.mean(dim=dim, keepdims=True),
            x2.mean(dim=dim, keepdims=True),
        )
        numerator = ((x1 - x1_mean) * (x2 - x2_mean)).sum(dim=dim, keepdims=True)

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
        pearson = (1.0 - self.cri_seq(x1, x2)) / 2.0
        return pearson[~pearson.isnan() * ~pearson.isinf()].mean()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        l_pear = self.compute_correlation_loss(restored, clean_patch)
        l_hog = self.cri_HOGloss(restored, clean_patch)
        l_l1 = self.loss_fn(restored, clean_patch)
        loss = l_l1 + l_pear + l_hog

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("train_loss_l1", l_l1)
        self.log("train_loss_pearson", l_pear)
        self.log("train_loss_hog", l_hog)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=180
        )

        return [optimizer], [scheduler]


def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + "denoise/" + str(sigma) + "/"
    subprocess.check_output(["mkdir", "-p", output_path])

    dataset.set_sigma(sigma)
    testloader = DataLoader(
        dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0
    )

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for [clean_name], degrad_patch, clean_patch in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            # restored = restored.permute(0, 2, 3, 1)
            # clean_patch = clean_patch.permute(0, 2, 3, 1)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + ".png")

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + "/"
    subprocess.check_output(["mkdir", "-p", output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(
        dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0
    )

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for [degraded_name], degrad_patch, clean_patch in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + ".png")
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--mode",
        type=int,
        default=5,
        help="0 for denoise, 1 for derain, 2 for dehaze, 3 for deblur, 4 for enhance, 5 for all-in-one (three tasks), 6 for all-in-one (five tasks)",
    )

    parser.add_argument(
        "--gopro_path",
        type=str,
        default="/data1/wjw/all_in_one_set2/Test/deblur/",
        help="save path of test hazy images",
    )
    parser.add_argument(
        "--enhance_path",
        type=str,
        default="/data1/wjw/all_in_one_set2/Test/enhance/",
        help="save path of test hazy images",
    )
    parser.add_argument(
        "--denoise_path",
        type=str,
        default="/data1/wjw/all_in_one_set2/Test/denoise/",
        help="save path of test noisy images",
    )
    parser.add_argument(
        "--derain_path",
        type=str,
        default="/data1/wjw/all_in_one_set2/Test/derain/",
        help="save path of test raining images",
    )
    parser.add_argument(
        "--dehaze_path",
        type=str,
        default="/data1/wjw/all_in_one_set2/Test/dehaze/",
        help="save path of test hazy images",
    )

    parser.add_argument(
        "--output_path", type=str, default="AdaIR3_results/", help="output save path"
    )
    parser.add_argument(
        "--ckpt_name", type=str, default="adair3d.ckpt", help="checkpoint save path"
    )
    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = "ckpt/" + testopt.ckpt_name

    denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]
    deblur_splits = ["gopro/"]
    enhance_splits = ["lol/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path, i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)

    print("CKPT name : {}".format(ckpt_path))

    net = AdaIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    if testopt.mode == 0:
        for testset, name in zip(denoise_tests, denoise_splits):
            print("Start {} testing Sigma=15...".format(name))
            test_Denoise(net, testset, sigma=15)

            print("Start {} testing Sigma=25...".format(name))
            test_Denoise(net, testset, sigma=25)

            print("Start {} testing Sigma=50...".format(name))
            test_Denoise(net, testset, sigma=50)

    elif testopt.mode == 1:
        print("Start testing rain streak removal...")
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print("Start testing {} rain streak removal...".format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")

    elif testopt.mode == 2:
        print("Start testing SOTS...")
        derain_base_path = testopt.derain_path
        name = derain_splits[0]
        testopt.derain_path = os.path.join(derain_base_path, name)
        derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
        test_Derain_Dehaze(net, derain_set, task="dehaze")

    elif testopt.mode == 3:
        print("Start testing GOPRO...")
        deblur_base_path = testopt.gopro_path
        name = deblur_splits[0]
        testopt.gopro_path = os.path.join(deblur_base_path, name)
        derain_set = DerainDehazeDataset(
            testopt, addnoise=False, sigma=15, task="deblur"
        )
        test_Derain_Dehaze(net, derain_set, task="deblur")

    elif testopt.mode == 4:
        print("Start testing LOL...")
        enhance_base_path = testopt.enhance_path
        name = derain_splits[0]
        testopt.enhance_path = os.path.join(enhance_base_path, name, task="enhance")
        derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
        test_Derain_Dehaze(net, derain_set, task="enhance")

    elif testopt.mode == 5:
        for testset, name in zip(denoise_tests, denoise_splits):
            print("Start {} testing Sigma=15...".format(name))
            test_Denoise(net, testset, sigma=15)

            print("Start {} testing Sigma=25...".format(name))
            test_Denoise(net, testset, sigma=25)

            print("Start {} testing Sigma=50...".format(name))
            test_Denoise(net, testset, sigma=50)

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:
            print("Start testing {} rain streak removal...".format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=55)
            test_Derain_Dehaze(net, derain_set, task="derain")

        print("Start testing SOTS...")
        test_Derain_Dehaze(net, derain_set, task="dehaze")

    elif testopt.mode == 6:
        for testset, name in zip(denoise_tests, denoise_splits):
            print("Start {} testing Sigma=15...".format(name))
            test_Denoise(net, testset, sigma=15)

            print("Start {} testing Sigma=25...".format(name))
            test_Denoise(net, testset, sigma=25)

            print("Start {} testing Sigma=50...".format(name))
            test_Denoise(net, testset, sigma=50)

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:
            print("Start testing {} rain streak removal...".format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=55)
            test_Derain_Dehaze(net, derain_set, task="derain")

        print("Start testing SOTS...")
        test_Derain_Dehaze(net, derain_set, task="dehaze")

        deblur_base_path = testopt.gopro_path
        for name in deblur_splits:
            print("Start testing GOPRO...")

            # print('Start testing {} rain streak removal...'.format(name))
            testopt.gopro_path = os.path.join(deblur_base_path, name)
            deblur_set = DerainDehazeDataset(
                testopt, addnoise=False, sigma=55, task="deblur"
            )
            test_Derain_Dehaze(net, deblur_set, task="deblur")

        enhance_base_path = testopt.enhance_path
        for name in enhance_splits:
            print("Start testing LOL...")
            testopt.enhance_path = os.path.join(enhance_base_path, name)
            derain_set = DerainDehazeDataset(
                testopt, addnoise=False, sigma=55, task="enhance"
            )
            test_Derain_Dehaze(net, derain_set, task="enhance")
