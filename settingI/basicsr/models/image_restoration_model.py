import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
#from audtorch.metrics.functional import pearsonr
import torch.autograd as autograd


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
                 block_size=1,          # Block size (in cells)
                 signed_gradient=False, # Whether to use signed gradients
                 loss_type='l2',        # Loss type: 'l1' or 'l2'
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

    
class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()
        self.psnr_best = -1
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')
        if train_opt.get('seq_opt'):
#            from audtorch.metrics.functional import pearsonr
#            self.cri_seq = pearsonr
            self.cri_seq = self.pearson_correlation_loss #
        self.cri_HOGloss = HOGLoss().to(self.device)
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

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

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'label' in data:
            self.label = data['label']
#            self.label = torch.nn.functional.one_hot(data['label'], num_classes=3)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def check_inf_nan(self, x):
        x[x.isnan()] = 0
        x[x.isinf()] = 1e7
        return x
    def compute_correlation_loss(self, x1, x2):
        b, c = x1.shape[0:2]
        x1 = x1.view(b, -1)
        x2 = x2.view(b, -1)
#        print(x1, x2)
        pearson = (1. - self.cri_seq(x1, x2)) / 2.
        return pearson[~pearson.isnan()*~pearson.isinf()].mean()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, )

        loss_dict = OrderedDict()
        l_pix = self.cri_pix(self.output, self.gt)
        loss_dict['l_pix'] = l_pix
        l_pear = self.compute_correlation_loss(self.output, self.gt)
        loss_dict['l_pear'] = l_pear
        l_hog = self.cri_HOGloss(self.output, self.gt)
        loss_dict['l_hog'] = l_hog
        loss_total = l_pix + l_pear + l_hog
        loss_total.backward()
        
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01, error_if_nonfinite=False)
        self.optimizer_g.step()

        self.log_dict, self.loss_total = self.reduce_loss_dict(loss_dict)
        self.loss_dict = loss_dict
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx >= 60:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = max(current_metric, self.metric_results[metric])

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if metric == 'psnr' and value >= self.psnr_best:
                self.save(0, current_iter, best=True)
                self.psnr_best = value
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, best=False):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'], best=best)
        else:
            self.save_network(self.net_g, 'net_g', current_iter, best=best)
        self.save_training_state(epoch, current_iter, best=best)
