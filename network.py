import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Parameter,Sigmoid, Softmax
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np
import torchvision.models as models

class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out

class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        nc = 32
        out_channels = nc*4
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


class ResBlock(nn.Module):
    """
    Res Conv Block 
    """
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        self.bn =  nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding = 1),
            nn.BatchNorm2d(out_ch),
        )
        self.residual = nn.Sequential()
        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False),
                nn.BatchNorm2d(out_ch),
            )
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.bn(x)+self.residual(x))


class PF_Module(Module):
    def __init__(self, in_channels):
        super(PF_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.sigmoid = Sigmoid()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 1, 1, padding=0, bias=False),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, 1, 1, padding=0, bias=False),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, 1, 1, padding=0, bias=False),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, 1, 1, padding=0, bias=False),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU())

    def forward(self,x1,x2,x3,x4):
        x_cat = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)], 1)
        x1_ = self.conv1(x1)
        x2_ = self.conv2(x2)
        x3_ = self.conv3(x3)
        x4_ = self.conv4(x4)
        x = torch.cat([x1_, x2_, x3_, x4_], 1)

        m_batchsize, T, C, height, width = x_cat.size()
        proj_q = x.view(m_batchsize, T, -1)
        proj_k = x.view(m_batchsize, T, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_q, proj_k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        channel_scale = torch.sum(attention, 2).unsqueeze(-1).unsqueeze(-1)
        
        channel_scale = self.sigmoid(channel_scale)
        proj_value = x_cat.view(m_batchsize, T, C, -1) 
        out = channel_scale * proj_value
        out = out.view(m_batchsize, T, C, height, width)
        output = self.gamma * out + x_cat
        output = output.view(m_batchsize, -1, height, width)
        return output


class CA_Module(Module):
    def __init__(self, in_channels):
        super(CA_Module, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)

    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_q = self.query(x).view(m_batchsize, C, -1)
        proj_k = self.key(x).view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_q, proj_k)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(m_batchsize, C, -1)
        out = torch.bmm(attention,proj_value)
        out = out.view(m_batchsize, C, height, width)
        output = self.gamma * out + x

        return output


class MCDA(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(FusionHead, self).__init__()
        inter_channels = in_channels//2
        phase_num = 4
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.pf = PF_Module(inter_channels)
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels*phase_num, in_channels, 3, padding=1, bias=False),
                                    norm_layer(in_channels),
                                    nn.ReLU())
        self.ca = CA_Module(in_channels)

    def forward(self, x_dp, x_vp, x_ap, x_pre):
        feat1 = self.conv1(x_dp)
        feat2 = self.conv2(x_vp)
        feat3 = self.conv3(x_ap)
        feat4 = self.conv4(x_pre)
        pf_feat = self.pf(feat1,feat2,feat3,feat4)
        out = self.conv5(ta_feat)
        output = self.ca(out)
        return output
    

class Encoder(nn.Module):
    def __init__(self, num_channels=1, nc=32):
        super(Encoder, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, nc, 3, padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, nc, 3, padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
        )
        self.second_1 = ResBlock(nc, nc*2)
        self.second_2 = ResBlock(nc*2, nc*2)

        self.third_1 = ResBlock(nc*2, nc*4)
        self.third_2 = ResBlock(nc*4, nc*4)

        self.fourth_1 = ResBlock(nc*4, nc*8)
        self.fourth_2 = ResBlock(nc*8, nc*8)

        self.fifth_1 = ResBlock(nc*8,nc*16)
        self.fifth_2 = ResBlock(nc*16,nc*16)

        self.sixth_1 = ResBlock(nc*16,nc*16)
        self.sixth_2 = ResBlock(nc*16,nc*16)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.MCDA_1 = MCDA(nc*4)
        self.MCDA_2 = MCDA(nc*8)
        self.MCDA_3 = MCDA(nc*16)
        self.MCDA_4 = MCDA(nc*16)

    def forward(self, x_dp, x_vp, x_ap, x_pre):
        dp_scale_1 = self.first(x_dp)
        vp_scale_1 = self.first(x_vp)
        ap_scale_1 = self.first(x_ap)
        pre_scale_1 = self.first(x_pre)

        dp_scale_2 = self.second_2(self.second_1(self.pool(dp_scale_1)))
        vp_scale_2 = self.second_2(self.second_1(self.pool(vp_scale_1)))
        ap_scale_2 = self.second_2(self.second_1(self.pool(ap_scale_1)))
        pre_scale_2 = self.second_2(self.second_1(self.pool(pre_scale_1)))

        dp_scale_3 = self.third_2(self.third_1(self.pool(dp_scale_2)))
        vp_scale_3 = self.third_2(self.third_1(self.pool(vp_scale_2)))
        ap_scale_3 = self.third_2(self.third_1(self.pool(ap_scale_2)))
        pre_scale_3 = self.third_2(self.third_1(self.pool(pre_scale_2)))
        aggr_scale_3 = self.MCDA_1(dp_scale_3, vp_scale_3, ap_scale_3, pre_scale_3)

        dp_scale_4 = self.fourth_2(self.fourth_1(self.pool(dp_scale_3)))
        vp_scale_4 = self.fourth_2(self.fourth_1(self.pool(vp_scale_3)))
        ap_scale_4 = self.fourth_2(self.fourth_1(self.pool(ap_scale_3)))
        pre_scale_4 = self.fourth_2(self.fourth_1(self.pool(pre_scale_3)))
        aggr_scale_4 = self.MCDA_2(dp_scale_4, vp_scale_4, ap_scale_4, pre_scale_4)

        dp_scale_5 = self.fifth_2(self.fifth_1(self.pool(dp_scale_4)))
        vp_scale_5 = self.fifth_2(self.fifth_1(self.pool(vp_scale_4)))
        ap_scale_5 = self.fifth_2(self.fifth_1(self.pool(ap_scale_4)))
        pre_scale_5 = self.fifth_2(self.fifth_1(self.pool(pre_scale_4)))
        aggr_scale_5 = self.MCDA_3(dp_scale_5, vp_scale_5, ap_scale_5, pre_scale_5)
        
        dp_scale_6 = self.sixth_2(self.sixth_1(self.pool(dp_scale_5)))
        vp_scale_6 = self.sixth_2(self.sixth_1(self.pool(vp_scale_5)))
        ap_scale_6 = self.sixth_2(self.sixth_1(self.pool(ap_scale_5)))
        pre_scale_6 = self.sixth_2(self.sixth_1(self.pool(pre_scale_5)))
        aggr_scale_6 = self.MCDA_4(dp_scale_6, vp_scale_6, ap_scale_6, pre_scale_6)

        return aggr_scale_3, aggr_scale_4, aggr_scale_5, aggr_scale_6


class Seg_head(nn.Module):
    def __init__(self, num_classes=2, nc=32):
        super().__init__()
        self.aspp = _ASPP(nc*16, [6, 12, 18], norm_layer=nn.BatchNorm2d, norm_kwargs=None)
        self.conv_5 = _ConvBNReLU(nc*4, nc*4, 3, padding=1)
        self.conv_4 = _ConvBNReLU(nc*16, nc*4, 3, padding=1)
        self.conv_3 = _ConvBNReLU(nc*8, nc*4, 3, padding=1)
        self.conv_2 = _ConvBNReLU(nc*4, nc*4, 3, padding=1)

        self.conv_block = nn.Sequential(
            _ConvBNReLU(nc*4*4, nc*4, 3, padding=1),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            _ConvBNReLU(nc*4, nc*4, 3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(nc*4, num_classes, 1)
        )
        self.cf5 = nn.Sequential(
            _ConvBNReLU(nc*4, nc*4, 3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(nc*4, num_classes, 1)
        )
        self.cf4 = nn.Sequential(
            _ConvBNReLU(nc*4, nc*4, 3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(nc*4, num_classes, 1)
        )
        self.cf3 = nn.Sequential(
            _ConvBNReLU(nc*4, nc*4, 3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(nc*4, num_classes, 1)
        )
        self.cf2 = nn.Sequential(
            _ConvBNReLU(nc*4, nc*4, 3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(nc*4, num_classes, 1)
        )

    def forward(self, aggr_5, aggr_4, aggr_3, aggr_2):
        size = aggr_2.size()[2:]
        aggr_5 = self.aspp(aggr_5)
        aggr_5 = F.interpolate(aggr_5, size, mode='bilinear', align_corners=True)
        aggr_5 = self.conv_5(aggr_5)

        aggr_4 = F.interpolate(aggr_4, size, mode='bilinear', align_corners=True)
        aggr_4 = self.conv_4(aggr_4)

        aggr_3 = F.interpolate(aggr_3, size, mode='bilinear', align_corners=True)
        aggr_3 = self.conv_3(aggr_3)

        aggr_2 = self.conv_2(aggr_2)
        features = self.conv_block(torch.cat([aggr_5, aggr_4, aggr_3, aggr_2], dim=1))
        maps = self.classifier(features)
        map5 = self.cf5(aggr_5)
        map4 = self.cf4(aggr_4)
        map3 = self.cf3(aggr_3)
        map2 = self.cf2(aggr_2)
        return features, maps, map5, map4, map3, map2
    

class Network(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = Encoder()
        self.head = Seg_head(num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_dp, x_vp, x_ap, x_pre):
        size = x_dp.size()[2:]
        aggr_2, aggr_3, aggr_4, aggr_5 = self.encoder(x_dp, x_vp, x_ap, x_pre)

        _, maps, map5, map4, map3, map2 \
            = self.head(aggr_5, aggr_4, aggr_3, aggr_2)
        initial_seg = F.interpolate(maps, size, mode='bilinear', align_corners=True)
        initial_seg = self.softmax(initial_seg) 

        seg5 = self.softmax(F.interpolate(map5, size, mode='bilinear', align_corners=True))
        seg4 = self.softmax(F.interpolate(map4, size, mode='bilinear', align_corners=True))
        seg3 = self.softmax(F.interpolate(map3, size, mode='bilinear', align_corners=True))
        seg2 = self.softmax(F.interpolate(map2, size, mode='bilinear', align_corners=True))

        return initial_seg, seg5, seg4, seg3, seg2


