# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'CSPLayer', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'ConvNormLayer', 'BasicBlock', 
           'BottleNeck', 'Blocks')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class CSPLayer(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through CSPLayer layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

################################### RT-DETR PResnet ###################################
"""
RT-DETR PResnet相关模块
这部分代码定义了用于构建类似ResNet的主干网络的一些基础模块，
特别是ConvNormLayer、BasicBlock和BottleNeck。
"""

def get_activation(act: str, inpace: bool=True):
    """
    根据字符串名称获取激活函数实例。
    这是一个辅助函数，用于方便地通过字符串指定常用的激活函数。
    
    参数:
        act (str): 激活函数的名称 (例如, 'silu', 'relu', 'gelu')。
        inpace (bool): 是否使用inplace操作（如果激活函数支持）。默认为True。
        
    返回:
        nn.Module: 对应的激活函数模块实例。
        
    异常:
        RuntimeError: 如果提供的激活函数名称不支持。
    """
    # 将激活函数名称转换为小写，以便不区分大小写匹配
    act = act.lower()
    
    # 根据名称返回对应的PyTorch激活函数模块
    if act == 'silu':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    # 注意：这里有一个重复的'silu'检查，可能是笔误，但保持原样
    elif act == 'silu':
        m = nn.SiLU()
    elif act == 'gelu':
        m = nn.GELU()
    # 如果act为None，则返回一个恒等映射，即不应用激活函数
    elif act is None:
        m = nn.Identity()
    # 如果act本身就是一个nn.Module实例，则直接返回它
    elif isinstance(act, nn.Module):
        m = act
    # 如果以上条件都不满足，则抛出运行时错误
    else:
        raise RuntimeError(f'不支持的激活函数类型: {act}')  

    # 检查返回的模块是否有inplace属性，如果有并且inpace参数为True，则设置它
    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 

class ConvNormLayer(nn.Module):
    """
    卷积-归一化层
    
    将卷积层、批归一化层和可选的激活函数组合在一起的标准模块。
    这是构建深度卷积网络的基本组件。
    """
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        """
        初始化ConvNormLayer
        
        参数:
            ch_in (int): 输入通道数。
            ch_out (int): 输出通道数。
            kernel_size (int): 卷积核大小。
            stride (int): 卷积步长。
            padding (int, optional): 填充大小。如果为None，则自动计算以保持空间维度。
            bias (bool): 卷积层是否使用偏置。默认为False，因为批归一化层包含了偏置。
            act (str or nn.Module, optional): 激活函数的名称或实例。如果为None，则不使用激活函数。
        """
        super().__init__()
        # 定义卷积层
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            # 如果padding为None，则自动计算填充大小以使输出尺寸在stride=1时保持不变
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        # 定义批归一化层
        self.norm = nn.BatchNorm2d(ch_out)
        # 获取激活函数层，如果act为None则使用恒等映射
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        """
        前向传播函数：Conv -> BN -> Act
        """
        return self.act(self.norm(self.conv(x)))
    
    def forward_fuse(self, x):
        """
        用于融合推理的前向传播函数：Conv -> Act
        
        在模型部署或推理时，可以将卷积层和批归一化层融合成一个单独的卷积层，
        这个函数模拟了融合后的行为（假设BN层已被融合进Conv层）。
        """
        # """执行二维数据的转置卷积。""" (原英文注释翻译，但似乎与代码功能不符，代码实际是 Conv -> Act)
        return self.act(self.conv(x))

class BasicBlock(nn.Module):
    """
    ResNet基础残差块
    
    包含两个3x3卷积层和一个可选的shortcut连接。
    通道扩展系数为1。
    """
    # 定义通道扩展系数，BasicBlock不改变通道数（相对于ch_out）
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        """
        初始化BasicBlock
        
        参数:
            ch_in (int): 输入通道数。
            ch_out (int): 输出通道数。
            stride (int): 第一个卷积层的步长，用于下采样。
            shortcut (bool): 是否使用shortcut连接（输入直接添加到输出）。
            act (str): 激活函数类型。默认为'relu'。
            variant (str): ResNet变体类型，影响shortcut的实现方式。默认为'd'。
        """
        super().__init__()

        self.shortcut = shortcut

        # 如果不使用恒等shortcut（例如，当输入输出通道数或空间维度不同时）
        if not shortcut:
            # ResNet-D变体：如果需要下采样(stride=2)，shortcut路径使用平均池化+1x1卷积
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            # 其他情况或非ResNet-D变体：shortcut路径使用步长为stride的1x1卷积
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        # 残差路径的第一个卷积层，步长为stride
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        # 残差路径的第二个卷积层，步长为1，不带激活函数（激活函数在与shortcut相加后应用）
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        # 获取最终的激活函数
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        """
        前向传播函数
        """
        # 计算残差路径的输出
        out = self.branch2a(x)
        out = self.branch2b(out)
        
        # 获取shortcut路径的输出
        if self.shortcut:
            # 如果使用恒等shortcut，直接使用输入x
            short = x
        else:
            # 否则，通过定义的short模块处理输入x
            short = self.short(x)
        
        # 将残差路径和shortcut路径的输出相加
        out = out + short
        # 应用最终的激活函数
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    """
    ResNet瓶颈残差块
    
    包含一个1x1卷积（降维），一个3x3卷积和一个1x1卷积（升维），以及可选的shortcut连接。
    通道扩展系数通常为4。
    """
    # 定义通道扩展系数，瓶颈块的输出通道数是内部通道数(ch_out)的4倍
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        """
        初始化BottleNeck块
        
        参数:
            ch_in (int): 输入通道数。
            ch_out (int): 内部（隐藏层）通道数。最终输出通道数为 ch_out * expansion。
            stride (int): 步长，通常应用于中间的3x3卷积（变体b）或第一个1x1卷积（变体a）。
            shortcut (bool): 是否使用shortcut连接。
            act (str): 激活函数类型。默认为'relu'。
            variant (str): ResNet变体类型。'a'表示步长在第一个1x1卷积，'b'/'d'表示步长在3x3卷积。
        """
        super().__init__()

        # 根据变体类型确定步长应用在哪一层卷积
        if variant == 'a':
            # 变体a: 步长在第一个1x1卷积
            stride1, stride2 = stride, 1
        else:
            # 变体b/d: 步长在中间的3x3卷积
            stride1, stride2 = 1, stride

        # 内部通道数
        width = ch_out 

        # 残差路径的第一个1x1卷积（降维），步长为stride1
        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        # 残差路径的3x3卷积，步长为stride2
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        # 残差路径的第二个1x1卷积（升维），步长为1，不带激活函数
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        # 如果不使用恒等shortcut
        if not shortcut:
            # ResNet-D变体：如果需要下采样(stride=2)，shortcut路径使用平均池化+1x1卷积（输出通道调整为expansion倍）
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            # 其他情况或非ResNet-D变体：shortcut路径使用步长为stride的1x1卷积（输出通道调整为expansion倍）
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        # 获取最终的激活函数
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        """
        前向传播函数
        """
        # 计算残差路径输出
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        # 获取shortcut路径输出
        if self.shortcut:
            # 如果使用恒等shortcut，直接使用输入x
            short = x
        else:
            # 否则，通过定义的short模块处理输入x
            short = self.short(x)

        # 将残差路径和shortcut路径的输出相加
        out = out + short
        # 应用最终的激活函数
        out = self.act(out)

        return out


class Blocks(nn.Module):
    """
    Blocks类：用于构建由多个相同类型的模块组成的序列网络块组
    这个类在神经网络中扮演着重要的角色，用于创建重复的网络块序列，是构建复杂网络结构的基础组件
    """
    def __init__(self, ch_in, ch_out, block, count, stage_num, act='relu', input_resolution=None, sr_ratio=None, kernel_size=None, kan_name=None, variant='d'):
        """
        初始化Blocks模块
        
        参数:
            ch_in (int): 输入通道数
            ch_out (int): 输出通道数
            block (nn.Module): 要重复使用的基础模块类型，如CSPLayer、C3等
            count (int): 要堆叠的模块数量
            stage_num (int): 当前所处的网络阶段编号，影响第一个块的步长设置
            act (str): 激活函数类型，默认为'relu'
            input_resolution (tuple): 输入特征图的分辨率，某些注意力机制需要
            sr_ratio (int): 空间缩减比率，用于注意力机制
            kernel_size (int): 卷积核大小
            kan_name (str): 可能用于核注意力网络的名称指定
            variant (str): 模块变体类型，默认为'd'，影响网络结构细节
        """
        super().__init__()

        # 创建一个模块列表，用于存放所有的block实例
        self.blocks = nn.ModuleList()
        
        # 循环创建count个block实例
        for i in range(count):
            # 根据不同的参数组合，创建不同类型的block
            
            # 情况1: 如果提供了input_resolution和sr_ratio，用于创建带有注意力机制的块
            if input_resolution is not None and sr_ratio is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        # 如果是第一个块且不是第2阶段，则使用步长2进行下采样；否则使用步长1
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        # 如果是第一个块则不使用shortcut连接；否则使用
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        input_resolution=input_resolution,
                        sr_ratio=sr_ratio)
                )
            # 情况2: 如果提供了kernel_size，创建指定卷积核大小的块
            elif kernel_size is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kernel_size=kernel_size)
                )
            # 情况3: 如果提供了kan_name，创建带有核注意力网络的块
            elif kan_name is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kan_name=kan_name)
                )
            # 情况4: 默认情况，创建基本的block
            else:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act)
                )
            
            # 第一个块之后，更新输入通道数为输出通道数乘以block的扩展系数
            # 这是因为许多块（如ResNet中的Bottleneck）在内部会扩展通道数
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        """
        前向传播函数
        
        按顺序通过所有blocks处理输入特征
        
        参数:
            x (torch.Tensor): 输入特征图
            
        返回:
            torch.Tensor: 经过所有block处理后的特征图
        """
        out = x
        # 依次通过每个block进行特征提取
        for block in self.blocks:
            out = block(out)
        return out