import torch
from torch import nn
from model.build_contextpath import build_contextpath
import warnings
import numpy as np
import time
warnings.filterwarnings(action='ignore')

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))
    
class Seg_head(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        output = self.gamma*x1 + x2
        return output
    
class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

# probablity attention
class ProbAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProbAttention, self).__init__()
        self.out_channels = out_channels    
        self.factor = 50
        self.head = 1
        self.inchannels = in_channels
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K
        index_sample = torch.randint(L_K, (sample_k,))
        K_sample = K_expand[:, :, index_sample, :]
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1))

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] 
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        return contex

    def _update_context(self, context_in, V, scores, index):
        B, H, L_V, D = V.shape
        attn = torch.softmax(scores, dim=-1) 
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        return context_in

    def forward(self, input_1, input_2):
        input = torch.cat((input_1, input_2), dim=1)
        m_batchsize,C,width ,height = input.size()
        queries = self.query_conv(input).view(m_batchsize,self.head,width*height,-1)
        keys = self.key_conv(input).view(m_batchsize,self.head,width*height,-1)
        values = self.value_conv(input).view(m_batchsize,self.head,width*height,-1)
        
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 
        # add scale factor
        scale = 1.0/(self.inchannels**0.5)
        scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index).transpose(2,1).contiguous().view(m_batchsize, self.out_channels, width, height)
        return context

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class FeatureFusionModule(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(FeatureFusionModule, self).__init__()
        self.in_channels = in_channels
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.spatial_attention = SpatialAttention()

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        attention = self.spatial_attention(feature)
        feature = attention * feature
        return feature

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.in_channels = in_channels
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=out_channels, stride=1)
        self.channel_attention = ChannelAttention(in_channels=out_channels)

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        attention_feature = self.channel_attention(feature) * feature
        return attention_feature

class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, context_path, model):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = build_contextpath(name=context_path)

        if context_path == 'resnet18':
            self.attention_refinement_module = AttentionRefinementModule(768, 384)
            self.supervision1 = Seg_head(256, num_classes)
            self.supervision2 = Seg_head(512, num_classes)
            if model == 'BiSeNet-Pro':
                self.feature_fusion_module = FeatureFusionModule(num_classes, 640)
            else:
                self.feature_fusion_module = ProbAttention(640, num_classes)
        else:
            print('Error: unspport context_path network \n')

        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        self.init_weight()

    def init_weight(self):
        for name, m in self.named_modules():
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)

        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx2 = torch.mul(cx2, tail)
        
        cx2_up = torch.nn.functional.interpolate(cx2, size=cx1.size()[-2:], mode='bilinear', align_corners=True)
        
        cx = self.attention_refinement_module(cx1, cx2_up)
        cx = torch.nn.functional.interpolate(cx, size = sx.size()[-2:], mode='bilinear')

        if self.training == True:
            # # upsampling
            cx1 = torch.nn.functional.interpolate(cx1, size=sx.size()[-2:], mode='bilinear')
            cx2 = torch.nn.functional.interpolate(cx2, size=sx.size()[-2:], mode='bilinear')
            cx1_sup = self.supervision1(cx1)
            cx2_sup = self.supervision2(cx2)
            cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=input.size()[-2:], mode='bilinear')
            cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=input.size()[-2:], mode='bilinear')

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
        result = self.conv(result)

        if self.training == True:
            return result, cx1_sup, cx2_sup
        return result
