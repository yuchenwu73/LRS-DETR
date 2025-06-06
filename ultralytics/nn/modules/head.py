# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
模型头部模块。
"""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors

from .block import DFL, Proto
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_

__all__ = 'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder'


class Detect(nn.Module):
    """YOLOv8检测模型的检测头部。"""
    dynamic = False  # 强制网格重建
    export = False  # 导出模式
    shape = None
    anchors = torch.empty(0)  # 初始化
    strides = torch.empty(0)  # 初始化

    def __init__(self, nc=80, ch=()):
        """使用指定的类别数量和通道数初始化YOLOv8检测层。"""
        super().__init__()
        self.nc = nc  # 类别数量
        self.nl = len(ch)  # 检测层数量
        self.reg_max = 16  # DFL通道数 (ch[0] // 16 用于缩放4/8/12/16/20，对应n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # 每个锚框的输出数量
        self.stride = torch.zeros(self.nl)  # 构建时计算的步长
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # 通道数
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """连接并返回预测的边界框和类别概率。"""
        shape = x[0].shape  # 批次、通道、高度、宽度
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # 避免TF FlexSplitV操作
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # 使用图像大小归一化xywh，以减轻TFLite整数模型的量化误差，如YOLOv5中所示：
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # 详情请参见此PR：https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """初始化Detect()的偏置，警告：需要步长可用。"""
        m = self  # self.model[-1]  # Detect()模块
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # 标称类别频率
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # 从
            a[-1].bias.data[:] = 1.0  # 边界框
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # 分类（0.01个目标，80个类别，640图像）


class Segment(Detect):
    """YOLOv8分割模型的分割头部。"""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """初始化YOLO模型的属性，如掩码数量、原型数量和卷积层。"""
        super().__init__(nc, ch)
        self.nm = nm  # 掩码数量
        self.npr = npr  # 原型数量
        self.proto = Proto(ch[0], self.npr, self.nm)  # 原型
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """如果处于训练模式，返回模型输出和掩码系数，否则返回输出和掩码系数。"""
        p = self.proto(x[0])  # 掩码原型
        bs = p.shape[0]  # 批次大小

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # 掩码系数
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose(Detect):
    """YOLOv8关键点模型的关键点头部。"""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """使用默认参数和卷积层初始化YOLO网络。"""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # 关键点数量，维度数（2表示x,y，3表示x,y,可见性）
        self.nk = kpt_shape[0] * kpt_shape[1]  # 关键点总数
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """执行YOLO模型的前向传播并返回预测结果。"""
        bs = x[0].shape[0]  # 批次大小
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """解码关键点。"""
        ndim = self.kpt_shape[1]
        if self.export:  # 导出TFLite时需要，以避免'PLACEHOLDER_FOR_GREATER_OP_CODES'错误
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # 原地sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8分类头部，即x(b,c1,20,20)到x(b,c2)。"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """使用指定的输入和输出通道、核大小、步长、填充和分组初始化YOLOv8分类头部。"""
        super().__init__()
        c_ = 1280  # efficientnet_b0大小
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 转换为x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # 转换为x(b,c2)

    def forward(self, x):
        """对输入图像数据执行YOLO模型的前向传播。"""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):
    """
    实时可变形Transformer解码器(RTDETRDecoder)模块，用于目标检测。

    该解码器模块利用Transformer架构和可变形卷积来预测图像中目标的边界框和类别标签。
    它整合了多个层的特征，并通过一系列Transformer解码器层来输出最终预测结果。
    """
    export = False  # 导出模式标志

    def __init__(
            self,
            nc=80,  # 类别数量
            ch=(512, 1024, 2048),  # 骨干网络特征图的通道数
            hd=256,  # 隐藏层维度
            nq=300,  # 查询点数量
            ndp=4,  # 解码器点数
            nh=8,  # 注意力头数
            ndl=6,  # 解码器层数
            d_ffn=1024,  # 前馈网络维度
            eval_idx=-1,  # 评估索引
            dropout=0.,  # dropout比率
            act=nn.ReLU(),  # 激活函数
            # 训练参数
            nd=100,  # 去噪数量
            label_noise_ratio=0.5,  # 标签噪声比例
            box_noise_scale=1.0,  # 边界框噪声尺度
            learnt_init_query=False):  # 是否学习初始查询嵌入
        """
        初始化RTDETRDecoder模块

        参数说明:
            nc (int): 类别数量，默认为80
            ch (tuple): 骨干网络特征图的通道数，默认为(512, 1024, 2048)
            hd (int): 隐藏层维度，默认为256
            nq (int): 查询点数量，默认为300
            ndp (int): 解码器点数，默认为4
            nh (int): 注意力头数，默认为8
            ndl (int): 解码器层数，默认为6
            d_ffn (int): 前馈网络维度，默认为1024
            dropout (float): dropout比率，默认为0
            act (nn.Module): 激活函数，默认为nn.ReLU
            eval_idx (int): 评估索引，默认为-1
            nd (int): 去噪数量，默认为100
            label_noise_ratio (float): 标签噪声比例，默认为0.5
            box_noise_scale (float): 边界框噪声尺度，默认为1.0
            learnt_init_query (bool): 是否学习初始查询嵌入，默认为False
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc + 1, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """
        前向传播函数，返回输入图像的边界框和分类分数
        
        参数:
            x: 输入特征
            batch: 批次信息
            
        返回:
            训练模式: 返回解码器输出、编码器输出和去噪元数据
            推理模式: 返回预测结果
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """
        为给定形状生成锚框，并进行验证
        
        参数:
            shapes: 特征图形状列表
            grid_size: 网格大小
            dtype: 数据类型
            device: 计算设备
            eps: 数值稳定性参数
            
        返回:
            anchors: 生成的锚框
            valid_mask: 有效锚框的掩码
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """
        处理并返回编码器输入
        
        该函数将输入特征投影到指定维度，并重新排列特征维度以适应Transformer架构
        
        参数:
            x: 输入特征列表
            
        返回:
            feats: 处理后的特征
            shapes: 特征图形状列表
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """
        生成并准备解码器所需的输入
        
        该函数处理特征和形状信息，生成查询嵌入和参考边界框
        
        参数:
            feats: 编码器特征
            shapes: 特征图形状
            dn_embed: 去噪嵌入
            dn_bbox: 去噪边界框
            
        返回:
            embeddings: 查询嵌入
            refer_bbox: 参考边界框
            enc_bboxes: 编码器边界框
            enc_scores: 编码器分数
        """
        bs = len(feats)
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """
        初始化或重置模型各组件的参数
        
        该函数使用预定义的权重和偏置初始化模型的各个组件，包括：
        - 分类头和边界框头的初始化
        - 编码器输出的初始化
        - 查询嵌入的初始化
        - 位置编码头的初始化
        - 输入投影层的初始化
        """
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
