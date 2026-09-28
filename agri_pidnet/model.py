"""Agri-PIDNet-S decode head for MMSegmentation 1.2.2.

The implementation contains only PIDNet-S and the three modules evaluated in
the paper: CA-MKIR, SaE-D2T, and BWR-Bag. Boolean switches expose the complete
2^3 factorial ablation without duplicating model files.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS


BN_MOMENTUM = 0.1
ALIGN_CORNERS = False


@MODELS.register_module()
class AgriDiceLoss(nn.Module):
    """Class-weighted multi-class Dice loss used in the reported setup."""

    def __init__(self,
                 smooth=1.0,
                 exponent=2.0,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_dice',
                 **kwargs):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Unsupported reduction: {reduction}')
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        reduction = reduction_override or self.reduction
        ignore_index = self.ignore_index if ignore_index is None else ignore_index
        probabilities = F.softmax(pred, dim=1)
        num_classes = probabilities.shape[1]
        valid = (target != ignore_index).to(probabilities.dtype)
        one_hot = F.one_hot(
            target.long().clamp(0, num_classes - 1),
            num_classes=num_classes).permute(0, 3, 1, 2)
        one_hot = one_hot.to(probabilities.dtype)

        valid = valid.unsqueeze(1)
        dims = (2, 3)
        numerator = 2 * (probabilities * one_hot * valid).sum(dims)
        numerator = numerator + self.smooth
        denominator = (
            ((probabilities.pow(self.exponent) +
              one_hot.pow(self.exponent)) * valid).sum(dims) + self.smooth)
        loss = 1 - numerator / denominator
        if self.class_weight is not None:
            class_weight = probabilities.new_tensor(self.class_weight)
            if class_weight.numel() != num_classes:
                raise ValueError('class_weight must match num_classes')
            loss = loss * class_weight.view(1, -1)
        loss = loss.mean(dim=1)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss * self.loss_weight

    @property
    def loss_name(self):
        return self._loss_name


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + residual
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) + residual
        return out if self.no_relu else self.relu(out)


class SegHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(
            inplanes, interplanes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, 1, bias=True)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        return self.conv2(self.relu(self.bn2(x)))


class PAPPM(nn.Module):
    """Parallel aggregation pyramid pooling module used by PIDNet-S."""

    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()

        def pooled_scale(kernel=None, stride=None, padding=None):
            pool = (nn.AdaptiveAvgPool2d((1, 1)) if kernel is None else
                    nn.AvgPool2d(kernel, stride, padding))
            return nn.Sequential(
                pool,
                nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, branch_planes, 1, bias=False),
            )

        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, 1, bias=False),
        )
        self.scale1 = pooled_scale(5, 2, 2)
        self.scale2 = pooled_scale(9, 4, 4)
        self.scale3 = pooled_scale(17, 8, 8)
        self.scale4 = pooled_scale()
        self.scale_process = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_planes * 4, branch_planes * 4, 3,
                padding=1, groups=4, bias=False),
        )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 5, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, 1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, 1, bias=False),
        )

    def forward(self, x):
        size = x.shape[-2:]
        base = self.scale0(x)
        scales = [
            F.interpolate(layer(x), size=size, mode='bilinear',
                          align_corners=ALIGN_CORNERS) + base
            for layer in (self.scale1, self.scale2, self.scale3, self.scale4)
        ]
        scales = self.scale_process(torch.cat(scales, dim=1))
        return self.compression(torch.cat([base, scales], dim=1)) + \
            self.shortcut(x)


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )

    def forward(self, x, y):
        y_query = F.interpolate(
            self.f_y(y), size=x.shape[-2:], mode='bilinear',
            align_corners=False)
        similarity = torch.sigmoid(
            torch.sum(self.f_x(x) * y_query, dim=1, keepdim=True))
        y = F.interpolate(
            y, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return (1 - similarity) * x + similarity * y


class LightBag(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, p, i, d):
        edge = torch.sigmoid(d)
        return (self.conv_p((1 - edge) * i + p) +
                self.conv_i(i + edge * p))


class HardSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HardSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = HardSigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = HardSwish()
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        _, _, height, width = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [height, width], dim=2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()
        return a_w * a_h


def channel_shuffle(x, groups):
    batch, channels, height, width = x.shape
    x = x.view(batch, groups, channels // groups, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch, channels, height, width)


class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, channels, kernel_sizes=(1, 3, 5), stride=1):
        super().__init__()
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels, channels, kernel, stride=stride,
                    padding=kernel // 2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
            )
            for kernel in kernel_sizes
        ])

    def forward(self, x):
        return [branch(x) for branch in self.dwconvs]


class MultiKernelInvertedResidual(nn.Module):
    """CA-MKIR unit with expansion t=2 and kernels 1/3/5."""

    def __init__(self, in_channels, out_channels, stride=1, use_ca=True):
        super().__init__()
        expanded = in_channels * 2
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, expanded, 1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.ReLU6(inplace=True),
        )
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(
            expanded, kernel_sizes=(1, 3, 5), stride=stride)
        self.coord_att = CoordinateAttention(expanded) if use_ca else None
        self.pconv2 = nn.Sequential(
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shuffle_groups = math.gcd(expanded, out_channels)

    def forward(self, x):
        features = self.multi_scale_dwconv(self.pconv1(x))
        out = sum(features)
        out = channel_shuffle(out, self.shuffle_groups)
        if self.coord_att is not None:
            # Kept identical to the reported implementation.
            out = out + self.coord_att(out)
        return self.pconv2(out)


class MKBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 no_relu=False, use_ca=True):
        super().__init__()
        self.block = MultiKernelInvertedResidual(
            inplanes, planes, stride=stride, use_ca=use_ca)
        self.downsample = downsample
        self.no_relu = no_relu
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.block(x) + residual
        return out if self.no_relu else self.relu(out)


class MultiBranchChannelRecalibration(nn.Module):
    """MBCR: four parallel C-to-C/4 branches followed by channel gating."""

    def __init__(self, channels=256, reduction=4, cardinality=4):
        super().__init__()
        if cardinality != 4:
            raise ValueError('The reported MBCR uses cardinality=4.')
        hidden = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True))
        self.fc4 = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Linear(hidden * cardinality, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, _, _ = x.shape
        pooled = self.avg_pool(x).view(batch, channels)
        weight = self.fc(torch.cat([
            self.fc1(pooled), self.fc2(pooled),
            self.fc3(pooled), self.fc4(pooled)], dim=1))
        return x * weight.view(batch, channels, 1, 1)


class MultiScaleChannelWeighting(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.attn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        pooled = x.mean(dim=1, keepdim=True).expand_as(x)
        return torch.sigmoid(self.attn(x + pooled))


class SaED2T(nn.Module):
    """SaE-D2T block used before PAPPM in the deep I branch."""

    def __init__(self, channels=256, num_heads=8, num_prototypes=2,
                 use_mbcr=True):
        super().__init__()
        effective_heads = max(1, min(num_heads, 2))
        if channels % effective_heads:
            raise ValueError('channels must be divisible by effective heads')
        self.norm1 = nn.LayerNorm(channels)
        self.pool = nn.AvgPool2d(2, 2)
        self.self_attn1 = nn.MultiheadAttention(
            channels, effective_heads, dropout=0.0, batch_first=True)
        self.mscw1 = MultiScaleChannelWeighting(channels)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, padding=1,
                groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.Mheads = nn.Linear(
            channels, num_prototypes, bias=False)
        self.mscw2 = MultiScaleChannelWeighting(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.sae = (MultiBranchChannelRecalibration(channels)
                    if use_mbcr else None)

    def forward(self, x):
        batch, channels, height, width = x.shape
        x_down = F.avg_pool2d(x, 2, 2)
        down_h, down_w = x_down.shape[-2:]
        query = x_down.flatten(2).transpose(1, 2)

        key_value = self.pool(x_down).flatten(2).transpose(1, 2)
        sampled_query = query[:, ::4, :]
        attended = self.self_attn1(
            sampled_query, key_value, key_value, need_weights=False)[0]
        attended = self.norm1(attended + sampled_query)
        attended = attended.transpose(1, 2).reshape(
            batch, channels, down_h // 2, down_w // 2)
        attended = F.interpolate(
            attended, size=(down_h, down_w), mode='bilinear',
            align_corners=False).flatten(2).transpose(1, 2)

        local = self.conv3x3(x_down).flatten(2).transpose(1, 2)
        assignment = F.softmax(self.Mheads(local), dim=1)
        prototypes = assignment.transpose(-1, -2) @ query
        prototype_weights = self.mscw2(prototypes)
        reconstructed = assignment @ prototype_weights
        reconstructed = self.norm2(reconstructed + query)

        out = (attended + reconstructed).transpose(1, 2).reshape(
            batch, channels, down_h, down_w)
        out = F.interpolate(
            out, size=(height, width), mode='bilinear',
            align_corners=False)
        if self.sae is not None:
            out = self.sae(out) + out
        return out


class CCTB(nn.Module):
    """Compatibility wrapper retaining the trained checkpoint hierarchy."""

    def __init__(self, channels=256, num_heads=8, use_mbcr=True):
        super().__init__()
        self.block = SaED2T(
            channels=channels, num_heads=num_heads,
            num_prototypes=2, use_mbcr=use_mbcr)

    def forward(self, x):
        return self.block(x)


def split_windows(x, window_size):
    batch, channels, height, width = x.shape
    x = x.permute(0, 2, 3, 1).reshape(
        batch, height // window_size, window_size,
        width // window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(
        -1, window_size * window_size, channels)


class WindowAttention(nn.Module):
    def __init__(self, channels=128, window_size=4, num_heads=2):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(
            channels, num_heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        batch, channels, height, width = x.shape
        size = self.window_size
        pad_h = (size - height % size) % size
        pad_w = (size - width % size) % size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        padded_h, padded_w = x.shape[-2:]
        tokens = split_windows(x, size)
        attended = self.attn(tokens, tokens, tokens, need_weights=False)[0]
        attended = self.norm(attended + tokens)
        n_h, n_w = padded_h // size, padded_w // size
        attended = attended.reshape(
            batch, n_h, n_w, size, size, channels)
        attended = attended.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, channels, padded_h, padded_w)
        return attended[:, :, :height, :width]


class WindowRefinementBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.attn = WindowAttention(channels, window_size=4, num_heads=2)
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.ffn(x)


class BWRBag(nn.Module):
    def __init__(self, channels=128, use_window_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        self.enhance = (WindowRefinementBlock(channels)
                        if use_window_attention else None)

    def forward(self, p, i, d):
        edge = torch.sigmoid(d)
        out = self.conv(edge * p + (1 - edge) * i)
        return out + self.enhance(out) if self.enhance is not None else out


def boundary_target_tensor(labels, edge_width=4, ignore_index=255):
    """Torch implementation of adjacent-label edge extraction and dilation."""
    labels = labels.squeeze(1)
    valid = labels != ignore_index
    edge = torch.zeros_like(labels, dtype=torch.bool)
    horizontal = ((labels[:, :, 1:] != labels[:, :, :-1]) &
                  valid[:, :, 1:] & valid[:, :, :-1])
    vertical = ((labels[:, 1:, :] != labels[:, :-1, :]) &
                valid[:, 1:, :] & valid[:, :-1, :])
    edge[:, :, 1:] |= horizontal
    edge[:, :, :-1] |= horizontal
    edge[:, 1:, :] |= vertical
    edge[:, :-1, :] |= vertical
    edge = edge.float().unsqueeze(1)
    if edge_width > 1:
        before = (edge_width - 1) // 2
        after = edge_width - 1 - before
        edge = F.max_pool2d(
            F.pad(edge, (before, after, before, after)),
            kernel_size=edge_width, stride=1)
    return edge, valid.unsqueeze(1)


@MODELS.register_module()
class AgriPIDNetSHead(BaseDecodeHead):
    """PIDNet-S with independently switchable CA-MKIR, SaE-D2T, BWR-Bag."""

    def __init__(self,
                 m=2,
                 n=3,
                 num_classes=4,
                 planes=32,
                 ppm_planes=96,
                 head_planes=128,
                 use_ca_mkir=True,
                 use_sae_d2t=True,
                 use_bwr_bag=True,
                 use_ca=True,
                 use_mbcr=True,
                 use_window_attention=True,
                 use_boundary_loss=True,
                 boundary_loss_weight=0.20,
                 edge_width=4,
                 **kwargs):
        # BaseDecodeHead supplies the MMSegmentation loss and pixel-sampling
        # interfaces; the actual logits are produced by ``final_layer``.
        super().__init__(
            in_channels=3,
            channels=head_planes,
            num_classes=num_classes,
            dropout_ratio=0.0,
            **kwargs)
        if (m, n, planes, ppm_planes, head_planes) != (2, 3, 32, 96, 128):
            raise ValueError('This clean release implements the reported PIDNet-S only.')

        self.use_ca_mkir = use_ca_mkir
        self.use_sae_d2t = use_sae_d2t
        self.use_bwr_bag = use_bwr_bag
        self.use_boundary_loss = use_boundary_loss
        self.boundary_loss_weight = boundary_loss_weight
        self.edge_width = edge_width

        mk_block = MKBasicBlock if use_ca_mkir else BasicBlock
        mk_kwargs = {'use_ca': use_ca} if use_ca_mkir else {}

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, 3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            mk_block, planes, planes, m, block_kwargs=mk_kwargs)
        self.layer2 = self._make_layer(
            mk_block, planes, planes * 2, m, stride=2,
            block_kwargs=mk_kwargs)
        self.layer3 = self._make_layer(
            BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(
            BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(
            Bottleneck, planes * 8, planes * 8, 2, stride=2)

        self.layer3_ = self._make_layer(
            mk_block, planes * 2, planes * 2, m,
            block_kwargs=mk_kwargs)
        self.layer4_ = self._make_layer(
            mk_block, planes * 2, planes * 2, m,
            block_kwargs=mk_kwargs)
        self.layer5_ = self._make_layer(
            Bottleneck, planes * 2, planes * 2, 1)

        self.layer3_d = self._make_single_layer(
            mk_block, planes * 2, planes, block_kwargs=mk_kwargs)
        self.layer4_d = self._make_layer(
            Bottleneck, planes, planes, 1)
        self.layer5_d = self._make_layer(
            Bottleneck, planes * 2, planes * 2, 1)

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, 1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, 1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)
        self.diff3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
        )
        self.diff4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )

        self.cswin_block = (CCTB(planes * 8, num_heads=8, use_mbcr=use_mbcr)
                            if use_sae_d2t else None)
        self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
        self.dfm = (BWRBag(planes * 4, use_window_attention)
                    if use_bwr_bag else LightBag(planes * 4))
        self.final_layer = SegHead(planes * 4, head_planes, num_classes)
        if self.use_boundary_loss:
            self.seghead_d = SegHead(planes * 2, planes, 1)

        self._initialize_weights()

    @staticmethod
    def _downsample(inplanes, outplanes, stride):
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride=stride, bias=False),
            nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM),
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1,
                    block_kwargs=None):
        block_kwargs = block_kwargs or {}
        outplanes = planes * block.expansion
        downsample = None
        if stride != 1 or inplanes != outplanes:
            downsample = self._downsample(inplanes, outplanes, stride)
        layers = [block(inplanes, planes, stride, downsample,
                        **block_kwargs)]
        for index in range(1, blocks):
            layers.append(block(
                outplanes, planes, no_relu=(index == blocks - 1),
                **block_kwargs))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1,
                           block_kwargs=None):
        block_kwargs = block_kwargs or {}
        outplanes = planes * block.expansion
        downsample = None
        if stride != 1 or inplanes != outplanes:
            downsample = self._downsample(inplanes, outplanes, stride)
        return block(inplanes, planes, stride, downsample,
                     no_relu=True, **block_kwargs)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, return_boundary=False):
        target_size = (x.shape[-2] // 8, x.shape[-1] // 8)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        p = self.layer3_(x)
        d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        p = self.pag3(p, self.compression3(x))
        d = d + F.interpolate(
            self.diff3(x), size=target_size, mode='bilinear',
            align_corners=ALIGN_CORNERS)

        x = self.relu(self.layer4(x))
        p = self.layer4_(self.relu(p))
        d = self.layer4_d(self.relu(d))
        p = self.pag4(p, self.compression4(x))
        d = d + F.interpolate(
            self.diff4(x), size=target_size, mode='bilinear',
            align_corners=ALIGN_CORNERS)
        boundary_features = d

        p = self.layer5_(self.relu(p))
        d = self.layer5_d(self.relu(d))

        if self.cswin_block is not None:
            x = F.interpolate(
                x, size=(16, 16), mode='bilinear', align_corners=True)
            x = self.cswin_block(x) + x
            x = F.interpolate(
                x, size=(8, 16), mode='bilinear', align_corners=True)
        x = self.layer5(x)
        x = F.interpolate(
            self.spp(x), size=target_size, mode='bilinear',
            align_corners=ALIGN_CORNERS)
        logits = self.final_layer(self.dfm(p, x, d))

        if return_boundary:
            if not self.use_boundary_loss:
                return logits, None
            return logits, self.seghead_d(boundary_features)
        return logits

    def loss(self, inputs, batch_data_samples, train_cfg):
        logits, boundary_logits = self(inputs, return_boundary=True)
        losses = super().loss_by_feat(logits, batch_data_samples)
        if (self.use_boundary_loss and boundary_logits is not None and
                self.boundary_loss_weight > 0):
            gt_semantic_seg = self._stack_batch_gt(batch_data_samples)
            target, valid = boundary_target_tensor(
                gt_semantic_seg, self.edge_width, self.ignore_index)
            boundary_logits = resize(
                boundary_logits, size=target.shape[-2:], mode='bilinear',
                align_corners=False)
            raw = F.binary_cross_entropy_with_logits(
                boundary_logits, target, reduction='none')
            valid = valid.to(dtype=raw.dtype)
            losses['loss_boundary'] = (
                (raw * valid).sum() / valid.sum().clamp_min(1.0) *
                self.boundary_loss_weight)
        return losses
