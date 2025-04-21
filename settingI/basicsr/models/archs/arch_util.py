import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.utils import get_root_logger

# try:
#     from basicsr.models.ops.dcn import (ModulatedDeformConvPack,
#                                         modulated_deform_conv)
# except ImportError:
#     # print('Cannot import dcn. Ignore this warning if dcn is not used. '
#     #       'Otherwise install BasicSR with compiling dcn.')
#

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# class DCNv2Pack(ModulatedDeformConvPack):
#     """Modulated deformable conv for deformable alignment.
#
#     Different from the official DCNv2Pack, which generates offsets and masks
#     from the preceding features, this DCNv2Pack takes another different
#     features to generate offsets and masks.
#
#     Ref:
#         Delving Deep into Deformable Alignment in Video Super-Resolution.
#     """
#
#     def forward(self, x, feat):
#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             logger = get_root_logger()
#             logger.warning(
#                 f'Offset abs mean is {offset_absmean}, larger than 50.')
#
#         return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
#                                      self.stride, self.padding, self.dilation,
#                                      self.groups, self.deformable_groups)










##########################
from math import pi

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy
except ImportError:
    numpy = None

if numpy is None and torch is None:
    raise ImportError("Must have either Numpy or PyTorch but both not found")


def set_framework_dependencies(x):
    if type(x) is numpy.ndarray:
        to_dtype = lambda a: a
        fw = numpy
    else:
        to_dtype = lambda a: a.to(x.dtype)
        fw = torch
    eps = fw.finfo(fw.float32).eps
    return fw, to_dtype, eps


def support_sz(sz):
    def wrapper(f):
        f.support_sz = sz
        return f
    return wrapper


@support_sz(4)
def cubic(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    absx = fw.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1.) * to_dtype(absx <= 1.) +
            (-0.5 * absx3 + 2.5 * absx2 - 4. * absx + 2.) *
            to_dtype((1. < absx) & (absx <= 2.)))


@support_sz(4)
def lanczos2(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return (((fw.sin(pi * x) * fw.sin(pi * x / 2) + eps) /
            ((pi**2 * x**2 / 2) + eps)) * to_dtype(abs(x) < 2))


@support_sz(6)
def lanczos3(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return (((fw.sin(pi * x) * fw.sin(pi * x / 3) + eps) /
            ((pi**2 * x**2 / 3) + eps)) * to_dtype(abs(x) < 3))


@support_sz(2)
def linear(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return ((x + 1) * to_dtype((-1 <= x) & (x < 0)) + (1 - x) *
            to_dtype((0 <= x) & (x <= 1)))

@support_sz(1)
def box(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return to_dtype((-1 <= x) & (x < 0)) + to_dtype((0 <= x) & (x <= 1))


import warnings
from math import ceil

from fractions import Fraction


class NoneClass:
    pass


try:
    import torch
    from torch import nn

    nnModuleWrapped = nn.Module
except ImportError:
    warnings.warn('No PyTorch found, will work only with Numpy')
    torch = None
    nnModuleWrapped = NoneClass

try:
    import numpy
except ImportError:
    warnings.warn('No Numpy found, will work only with PyTorch')
    numpy = None

if numpy is None and torch is None:
    raise ImportError("Must have either Numpy or PyTorch but both not found")


def resize_fine(input, scale_factors=None, out_shape=None,
           interp_method=lanczos3, support_sz=None,
           antialiasing=True, by_convs=False, scale_tolerance=None,
           max_numerator=10, pad_mode='constant', adv_weights=None):
    # get properties of the input tensor
    in_shape, n_dims = input.shape, input.ndim

    # fw stands for framework that can be either numpy or torch,
    # determined by the input type
    fw = numpy if type(input) is numpy.ndarray else torch
    eps = fw.finfo(fw.float32).eps
    device = input.device if fw is torch else None
    weights_container = []

    # set missing scale factors or output shapem one according to another,
    # scream if both missing. this is also where all the defults policies
    # take place. also handling the by_convs attribute carefully.
    scale_factors, out_shape, by_convs = set_scale_and_out_sz(in_shape,
                                                              out_shape,
                                                              scale_factors,
                                                              by_convs,
                                                              scale_tolerance,
                                                              max_numerator,
                                                              eps, fw)

    # sort indices of dimensions according to scale of each dimension.
    # since we are going dim by dim this is efficient
    sorted_filtered_dims_and_scales = [(dim, scale_factors[dim], by_convs[dim],
                                        in_shape[dim], out_shape[dim])
                                       for dim in sorted(range(n_dims),
                                                         key=lambda ind: scale_factors[ind])
                                       if scale_factors[dim] != 1.]

    # unless support size is specified by the user, it is an attribute
    # of the interpolation method
    if support_sz is None:
        support_sz = interp_method.support_sz

    # output begins identical to input and changes with each iteration
    output = input

    # iterate over dims
    for i, (dim, scale_factor, dim_by_convs, in_sz, out_sz
            ) in enumerate(sorted_filtered_dims_and_scales):
        # STEP 1- PROJECTED GRID: The non-integer locations of the projection
        # of output pixel locations to the input tensor
        projected_grid = get_projected_grid(in_sz, out_sz,
                                            scale_factor, fw, dim_by_convs,
                                            device)

        # STEP 1.5: ANTIALIASING- If antialiasing is taking place, we modify
        # the window size and the interpolation method (see inside function)
        cur_interp_method, cur_support_sz = apply_antialiasing_if_needed(
            interp_method,
            support_sz,
            scale_factor,
            antialiasing)

        # STEP 2- FIELDS OF VIEW: for each output pixels, map the input pixels
        # that influence it. Also calculate needed padding and update grid
        # accoedingly
        field_of_view = get_field_of_view(projected_grid, cur_support_sz, fw,
                                          eps, device)

        # STEP 2.5- CALCULATE PAD AND UPDATE: according to the field of view,
        # the input should be padded to handle the boundaries, coordinates
        # should be updated. actual padding only occurs when weights are
        # aplied (step 4). if using by_convs for this dim, then we need to
        # calc right and left boundaries for each filter instead.
        pad_sz, projected_grid, field_of_view = calc_pad_sz(in_sz, out_sz,
                                                            field_of_view,
                                                            projected_grid,
                                                            scale_factor,
                                                            dim_by_convs, fw,
                                                            device)

        # STEP 3- CALCULATE WEIGHTS: Match a set of weights to the pixels in
        # the field of view for each output pixel
        if adv_weights != None:
            weights = adv_weights[i]
        else:
            weights = get_weights(cur_interp_method, projected_grid, field_of_view)
            weights_container.append(weights)

        # STEP 4- APPLY WEIGHTS: Each output pixel is calculated by multiplying
        # its set of weights with the pixel values in its field of view.
        # We now multiply the fields of view with their matching weights.
        # We do this by tensor multiplication and broadcasting.
        # if by_convs is true for this dim, then we do this action by
        # convolutions. this is equivalent but faster.
        if not dim_by_convs:
            output = apply_weights(output, field_of_view, weights, dim, n_dims,
                                   pad_sz, pad_mode, fw)
        else:
            output = apply_convs(output, scale_factor, in_sz, out_sz, weights,
                                 dim, pad_sz, pad_mode, fw)
    return output


def get_projected_grid(in_sz, out_sz, scale_factor, fw, by_convs, device=None):
    # we start by having the ouput coordinates which are just integer locations
    # in the special case when usin by_convs, we only need two cycles of grid
    # points. the first and last.
    grid_sz = out_sz if not by_convs else scale_factor.numerator
    out_coordinates = fw_arange(grid_sz, fw, device)

    # This is projecting the ouput pixel locations in 1d to the input tensor,
    # as non-integer locations.
    # the following fomrula is derived in the paper
    # "From Discrete to Continuous Convolutions" by Shocher et al.
    return (out_coordinates / float(scale_factor) +
            (in_sz - 1) / 2 - (out_sz - 1) / (2 * float(scale_factor)))


def get_field_of_view(projected_grid, cur_support_sz, fw, eps, device):
    # for each output pixel, map which input pixels influence it, in 1d.
    # we start by calculating the leftmost neighbor, using half of the window
    # size (eps is for when boundary is exact int)
    left_boundaries = fw_ceil(projected_grid - cur_support_sz / 2 - eps, fw)

    # then we simply take all the pixel centers in the field by counting
    # window size pixels from the left boundary
    ordinal_numbers = fw_arange(ceil(cur_support_sz - eps), fw, device)
    return left_boundaries[:, None] + ordinal_numbers


def calc_pad_sz(in_sz, out_sz, field_of_view, projected_grid, scale_factor,
                dim_by_convs, fw, device):
    if not dim_by_convs:
        # determine padding according to neighbor coords out of bound.
        # this is a generalized notion of padding, when pad<0 it means crop
        pad_sz = [-field_of_view[0, 0].item(),
                  field_of_view[-1, -1].item() - in_sz + 1]

        # since input image will be changed by padding, coordinates of both
        # field_of_view and projected_grid need to be updated
        field_of_view += pad_sz[0]
        projected_grid += pad_sz[0]

    else:
        # only used for by_convs, to calc the boundaries of each filter the
        # number of distinct convolutions is the numerator of the scale factor
        num_convs, stride = scale_factor.numerator, scale_factor.denominator

        # calculate left and right boundaries for each conv. left can also be
        # negative right can be bigger than in_sz. such cases imply padding if
        # needed. however if# both are in-bounds, it means we need to crop,
        # practically apply the conv only on part of the image.
        left_pads = -field_of_view[:, 0]

        # next calc is tricky, explanation by rows:
        # 1) counting output pixels between the first position of each filter
        #    to the right boundary of the input
        # 2) dividing it by number of filters to count how many 'jumps'
        #    each filter does
        # 3) multiplying by the stride gives us the distance over the input
        #    coords done by all these jumps for each filter
        # 4) to this distance we add the right boundary of the filter when
        #    placed in its leftmost position. so now we get the right boundary
        #    of that filter in input coord.
        # 5) the padding size needed is obtained by subtracting the rightmost
        #    input coordinate. if the result is positive padding is needed. if
        #    negative then negative padding means shaving off pixel columns.
        right_pads = (((out_sz - fw_arange(num_convs, fw, device) - 1)  # (1)
                       // num_convs)  # (2)
                      * stride  # (3)
                      + field_of_view[:, -1]  # (4)
                      - in_sz + 1)  # (5)

        # in the by_convs case pad_sz is a list of left-right pairs. one per
        # each filter

        pad_sz = list(zip(left_pads, right_pads))

    return pad_sz, projected_grid, field_of_view


def get_weights(interp_method, projected_grid, field_of_view):
    # the set of weights per each output pixels is the result of the chosen
    # interpolation method applied to the distances between projected grid
    # locations and the pixel-centers in the field of view (distances are
    # directed, can be positive or negative)
    weights = interp_method(projected_grid[:, None] - field_of_view)

    # we now carefully normalize the weights to sum to 1 per each output pixel
    sum_weights = weights.sum(1, keepdims=True)
    sum_weights[sum_weights == 0] = 1
    return weights / sum_weights


def apply_weights(input, field_of_view, weights, dim, n_dims, pad_sz, pad_mode,
                  fw):
    # for this operation we assume the resized dim is the first one.
    # so we transpose and will transpose back after multiplying
    tmp_input = fw_swapaxes(input, dim, 0, fw)

    # apply padding
    tmp_input = fw_pad(tmp_input, fw, pad_sz, pad_mode)

    # field_of_view is a tensor of order 2: for each output (1d location
    # along cur dim)- a list of 1d neighbors locations.
    # note that this whole operations is applied to each dim separately,
    # this is why it is all in 1d.
    # neighbors = tmp_input[field_of_view] is a tensor of order image_dims+1:
    # for each output pixel (this time indicated in all dims), these are the
    # values of the neighbors in the 1d field of view. note that we only
    # consider neighbors along the current dim, but such set exists for every
    # multi-dim location, hence the final tensor order is image_dims+1.
    neighbors = tmp_input[field_of_view]

    # weights is an order 2 tensor: for each output location along 1d- a list
    # of weights matching the field of view. we augment it with ones, for
    # broadcasting, so that when multiplies some tensor the weights affect
    # only its first dim.
    tmp_weights = fw.reshape(weights, (*weights.shape, *[1] * (n_dims - 1)))

    # now we simply multiply the weights with the neighbors, and then sum
    # along the field of view, to get a single value per out pixel
    tmp_output = (neighbors * tmp_weights).sum(1)

    # we transpose back the resized dim to its original position
    return fw_swapaxes(tmp_output, 0, dim, fw)


def apply_convs(input, scale_factor, in_sz, out_sz, weights, dim, pad_sz,
                pad_mode, fw):
    # for this operations we assume the resized dim is the last one.
    # so we transpose and will transpose back after multiplying
    input = fw_swapaxes(input, dim, -1, fw)

    # the stride for all convs is the denominator of the scale factor
    stride, num_convs = scale_factor.denominator, scale_factor.numerator

    # prepare an empty tensor for the output
    tmp_out_shape = list(input.shape)
    tmp_out_shape[-1] = out_sz
    tmp_output = fw_empty(tuple(tmp_out_shape), fw, input.device)

    # iterate over the conv operations. we have as many as the numerator
    # of the scale-factor. for each we need boundaries and a filter.
    for conv_ind, (pad_sz, filt) in enumerate(zip(pad_sz, weights)):
        # apply padding (we pad last dim, padding can be negative)
        pad_dim = input.ndim - 1
        tmp_input = fw_pad(input, fw, pad_sz, pad_mode, dim=pad_dim)

        # apply convolution over last dim. store in the output tensor with
        # positional strides so that when the loop is comlete conv results are
        # interwind
        tmp_output[..., conv_ind::num_convs] = fw_conv(tmp_input, filt, stride)

    return fw_swapaxes(tmp_output, -1, dim, fw)


def set_scale_and_out_sz(in_shape, out_shape, scale_factors, by_convs,
                         scale_tolerance, max_numerator, eps, fw):
    # eventually we must have both scale-factors and out-sizes for all in/out
    # dims. however, we support many possible partial arguments
    if scale_factors is None and out_shape is None:
        raise ValueError("either scale_factors or out_shape should be "
                         "provided")
    if out_shape is not None:
        # if out_shape has less dims than in_shape, we defaultly resize the
        # first dims for numpy and last dims for torch
        out_shape = (list(out_shape) + list(in_shape[len(out_shape):])
                     if fw is numpy
                     else list(in_shape[:-len(out_shape)]) + list(out_shape))
        if scale_factors is None:
            # if no scale given, we calculate it as the out to in ratio
            # (not recomended)
            scale_factors = [out_sz / in_sz for out_sz, in_sz
                             in zip(out_shape, in_shape)]
    if scale_factors is not None:
        # by default, if a single number is given as scale, we assume resizing
        # two dims (most common are images with 2 spatial dims)
        scale_factors = (scale_factors
                         if isinstance(scale_factors, (list, tuple))
                         else [scale_factors, scale_factors])
        # if less scale_factors than in_shape dims, we defaultly resize the
        # first dims for numpy and last dims for torch
        scale_factors = (list(scale_factors) + [1] *
                         (len(in_shape) - len(scale_factors)) if fw is numpy
                         else [1] * (len(in_shape) - len(scale_factors)) +
                              list(scale_factors))
        if out_shape is None:
            # when no out_shape given, it is calculated by multiplying the
            # scale by the in_shape (not recomended)
            out_shape = [ceil(scale_factor * in_sz)
                         for scale_factor, in_sz in
                         zip(scale_factors, in_shape)]
        # next part intentionally after out_shape determined for stability
        # we fix by_convs to be a list of truth values in case it is not
        if not isinstance(by_convs, (list, tuple)):
            by_convs = [by_convs] * len(out_shape)

        # next loop fixes the scale for each dim to be either frac or float.
        # this is determined by by_convs and by tolerance for scale accuracy.
        for ind, (sf, dim_by_convs) in enumerate(zip(scale_factors, by_convs)):
            # first we fractionaize
            if dim_by_convs:
                frac = Fraction(1 / sf).limit_denominator(max_numerator)
                frac = Fraction(numerator=frac.denominator, denominator=frac.numerator)

            # if accuracy is within tolerance scale will be frac. if not, then
            # it will be float and the by_convs attr will be set false for
            # this dim
            if scale_tolerance is None:
                scale_tolerance = eps
            if dim_by_convs and abs(frac - sf) < scale_tolerance:
                scale_factors[ind] = frac
            else:
                scale_factors[ind] = float(sf)
                by_convs[ind] = False

        return scale_factors, out_shape, by_convs


def apply_antialiasing_if_needed(interp_method, support_sz, scale_factor,
                                 antialiasing):
    # antialiasing is "stretching" the field of view according to the scale
    # factor (only for downscaling). this is low-pass filtering. this
    # requires modifying both the interpolation (stretching the 1d
    # function and multiplying by the scale-factor) and the window size.
    scale_factor = float(scale_factor)
    if scale_factor >= 1.0 or not antialiasing:
        return interp_method, support_sz
    cur_interp_method = (lambda arg: scale_factor *
                                     interp_method(scale_factor * arg))
    cur_support_sz = support_sz / scale_factor
    return cur_interp_method, cur_support_sz


def fw_ceil(x, fw):
    if fw is numpy:
        return fw.int_(fw.ceil(x))
    else:
        return x.ceil().long()


def fw_floor(x, fw):
    if fw is numpy:
        return fw.int_(fw.floor(x))
    else:
        return x.floor().long()


def fw_cat(x, fw):
    if fw is numpy:
        return fw.concatenate(x)
    else:
        return fw.cat(x)


def fw_swapaxes(x, ax_1, ax_2, fw):
    if fw is numpy:
        return fw.swapaxes(x, ax_1, ax_2)
    else:
        return x.transpose(ax_1, ax_2)


def fw_pad(x, fw, pad_sz, pad_mode, dim=0):
    if pad_sz == (0, 0):
        return x
    if fw is numpy:
        pad_vec = [(0, 0)] * x.ndim
        pad_vec[dim] = pad_sz
        return fw.pad(x, pad_width=pad_vec, mode=pad_mode)
    else:
        if x.ndim < 3:
            x = x[None, None, ...]

        pad_vec = [0] * ((x.ndim - 2) * 2)
        pad_vec[0:2] = pad_sz
        return fw.nn.functional.pad(x.transpose(dim, -1), pad=pad_vec,
                                    mode=pad_mode).transpose(dim, -1)


def fw_conv(input, filter, stride):
    # we want to apply 1d conv to any nd array. the way to do it is to reshape
    # the input to a 4D tensor. first two dims are singeletons, 3rd dim stores
    # all the spatial dims that we are not convolving along now. then we can
    # apply conv2d with a 1xK filter. This convolves the same way all the other
    # dims stored in the 3d dim. like depthwise conv over these.
    # TODO: numpy support
    reshaped_input = input.reshape(1, 1, -1, input.shape[-1])
    reshaped_output = torch.nn.functional.conv2d(reshaped_input,
                                                 filter.view(1, 1, 1, -1),
                                                 stride=(1, stride))
    return reshaped_output.reshape(*input.shape[:-1], -1)


def fw_arange(upper_bound, fw, device):
    if fw is numpy:
        return fw.arange(upper_bound)
    else:
        return fw.arange(upper_bound, device=device)


def fw_empty(shape, fw, device):
    if fw is numpy:
        return fw.empty(shape)
    else:
        return fw.empty(size=(*shape,), device=device)
