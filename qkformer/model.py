import torch
import torch.nn as nn
# from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model

__all__ = ['QKFormer']

T=4

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = 4.0
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha

        return grad_x, None
sig_fn=sigmoid.apply
class MultiStepLIFNode(nn.Module):
    def __init__(self,tau,  v_threshold=1,detach_reset=0, backend=0):
        super().__init__()
        self.spike_fn = sigmoid.apply
        self.back=backend
        self.lin1 = nn.Linear(tau * T, tau * T)
        self.lin2 = nn.Linear(tau * T, tau * T)
        if self.back=='ATT'  or self.back=='ATTOUT' :

            self.leake=nn.Parameter(torch.ones(T,tau)*0.5)
            self.o = nn.Parameter(torch.ones(T, tau) * 0.25)
            self.input = nn.Parameter(torch.ones(T,tau)*0.5)
            self.th1= nn.Parameter(torch.ones(T,tau)*0.1 )
            self.th2 = nn.Parameter(torch.ones(T, tau)*0.1 )
            self.th3 = nn.Parameter(torch.ones(T, tau)*0.1  )
        else:
            self.leake = nn.Parameter(torch.ones( T) * 0.5)
            self.input = nn.Parameter(torch.ones(T) * 0.5)
        self.v_threshold =v_threshold
        self.sig=nn.Sigmoid()
        self.tau=tau

    def forward(self, x_seq: torch.Tensor,leake=0,input=0,threshold=1):
        # 脉冲打乱
        # index=torch.randperm(x_seq.size(0))
        # x_seq = x_seq[index]

        if self.back=='ATT':

            leake=self.leake.reshape([T,1,-1,1])
            input = self.input.reshape([T, 1, -1,1])

            o=self.o.reshape([T, 1, -1,1])

        elif self.back=='ATTOUT':
            #T, B, C, H, W


            leake = self.leake.reshape([T, 1, -1,1,1])
            input = self.input.reshape([T, 1, -1,1,1])
            o = self.o.reshape([T, 1, -1,1,1])

        else:
            leake = self.leake
            input = self.input


        spike_seq = []

        for t in range(x_seq.shape[0]):
            #lif
            if t == 0:

                mem = x_seq[t] *0.5

            else:
                mem = mem *0.5+ x_seq[t] *  0.5
            # if t == 0:
            #     mem = x_seq[t] *input [t]
            #
            # else:
            #     mem = mem *leake[t] + x_seq[t] *  input [t]

            x= self.spike_fn( mem -self.v_threshold)
            mem=mem-x
            spike_seq.append((x).unsqueeze(0))

        spike_seq = torch.cat(spike_seq, 0)
        # print(spike_seq.shape)
        # indice=torch.randperm(len(spike_seq))
        # spike_seq=spike_seq[indice]

        # print("发射率：{}".format( spike_seq.sum() / spike_seq.numel()))
        return spike_seq


ration=0.005
class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        # print(dim)
        self.num_heads = num_heads
        # self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=dim, detach_reset=True, backend='ATT')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=dim, detach_reset=True, backend='ATT')

        self.attn_lif = MultiStepLIFNode(tau=num_heads, v_threshold=0.5, detach_reset=True, backend='ATTOUT')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=dim, detach_reset=True, backend='ATTOUT')


    def forward(self, x,loss_a,loss_b):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        q = torch.sum(q, dim=3, keepdim=True)
        attn = self.attn_lif(q)

        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)


        return x, loss_a,loss_b


class Spiking_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,tau=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.tau=tau
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=dim, detach_reset=True, backend='ATT')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=dim, detach_reset=True, backend='ATT')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=dim, detach_reset=True, backend='ATT')
        self.attn_lif = MultiStepLIFNode(tau=dim, v_threshold=0.5, detach_reset=True, backend='ATT')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=dim, detach_reset=True, backend='ATTOUT')

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x,loss_a,loss_b):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()


        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)

        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)

        x = x.flatten(0,1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T,B,C,W,H))

        return x,loss_a,loss_b

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.,tau=2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp1_lif = MultiStepLIFNode(tau=hidden_features, detach_reset=True, backend='ATTOUT')

        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = MultiStepLIFNode(tau=out_features, detach_reset=True, backend='ATTOUT')

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.tau=tau
    def forward(self, x,loss_a,loss_b):
        T, B, C, H, W = x.shape


        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)
        x = self.mlp1_lif(x)

        x = self.mlp2_conv(x.flatten(0, 1))  # b ct hw
        x = self.mlp2_bn(x).reshape(T, B, self.c_output, H, W)
        x = self.mlp2_lif(x)

        return x,loss_a,loss_b


class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x,loss_a,loss_b):
        out,loss_a,loss_b=self.tssa(x, loss_a,loss_b)
        x = x + out
        out,loss_a,loss_b=  self.mlp(x,loss_a,loss_b)
        x = x + out
        return x,loss_a,loss_b


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.ssa = Spiking_Self_Attention(dim, num_heads,tau=20)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop,tau=20)

    def forward(self, x,loss_a,loss_b):


        out, loss_a ,loss_b=  self.ssa(x,loss_a,loss_b)
        x = x + out
        out, loss_a,loss_b = self.mlp(x, loss_a,loss_b)
        x = x + out
        return x,loss_a,loss_b

class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)


        self.proj_lif = MultiStepLIFNode(tau=embed_dims // 2, detach_reset=True, backend='ATTOUT')

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 1)
        self.proj1_lif = MultiStepLIFNode(tau=embed_dims // 1, detach_reset=True, backend='ATTOUT')

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims //1, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=embed_dims, detach_reset=True, backend='ATTOUT')

        self.scale=nn.Parameter(torch.ones(4,1,3,32,32))
        self.bias = nn.Parameter(torch.ones(4, 1, 3, 32, 32))

    def forward(self, x,loss_a,loss_b):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x)



        x_feat = x

        x = self.proj1_conv(x.flatten(0, 1))
        x = self.proj1_bn(x).reshape(T, B, -1, H, W)
        x = self.proj1_lif(x)

        x_feat = self.proj_res_conv(x_feat.flatten(0, 1))
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H, W).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x,loss_a,loss_b


class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_lif = MultiStepLIFNode(tau=embed_dims, detach_reset=True, backend='ATTOUT')

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_lif = MultiStepLIFNode(tau=embed_dims, detach_reset=True, backend='ATTOUT')

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=embed_dims, detach_reset=True, backend='ATTOUT')

    def forward(self, x,loss_a,loss_b):
        T, B, C, H, W = x.shape
        # Downsampling + Res

        x = x.flatten(0, 1).contiguous()
        x_feat = x
        x=x.reshape(T, B, -1, H, W)

        x = self.proj3_conv(x.flatten(0, 1))
        x = self.proj3_bn(x).reshape(T, B, -1, H, W)
        x = self.proj3_lif(x).contiguous()


        x = self.proj4_conv(x.flatten(0, 1))
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)


        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//2, W//2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x,loss_a,loss_b





class spiking_transformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[8, 8, 8], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        num_heads = [8, 8, 8]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        self.infn=quant4.apply
        patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 4)

        stage1 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 4, num_heads=num_heads[0], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(1)])

        patch_embed2 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 2)

        stage2 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 2, num_heads=num_heads[1], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(1)])

        patch_embed3 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims)

        stage3 = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads[2], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths - 2)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        self.sig_fn = sigmoid.apply
        self.scale = nn.Parameter(torch.ones([self.T, num_classes]))
        self.offset = nn.Parameter(torch.ones([self.T, num_classes]))
        self.encodercon2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.encoderbn2 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")
        stage3 = getattr(self, f"stage3")
        patch_embed3 = getattr(self, f"patch_embed3")
        loss_a=0
        loss_b=0
        x,loss_a,loss_b = patch_embed1(x,loss_a,loss_b)
        print("发射率：{}".format(x.sum() / x.numel()))
        print("real发射率：{}".format(self.sig_fn(x).sum() / x.numel()))
        print("3的发射率：{}".format(torch.where(x>2.5,1,0).sum() / x.numel()))
        for blk in stage1:

            x ,loss_a,loss_b= blk(x,loss_a,loss_b)
            print("发射率：{}".format(x.sum() / x.numel()))
            print("real发射率：{}".format(self.sig_fn(x).sum() / x.numel()))
            print("3的发射率：{}".format(torch.where(x>2.5,1,0).sum() / x.numel()))

        x ,loss_a,loss_b= patch_embed2(x,loss_a,loss_b)
        print("发射率：{}".format(x.sum() / x.numel()))
        print("real发射率：{}".format(self.sig_fn(x).sum() / x.numel()))
        print("3的发射率：{}".format(torch.where(x>2.5,1,0).sum() / x.numel()))
        for blk in stage2:
            x ,loss_a,loss_b= blk(x,loss_a,loss_b)
            print("发射率：{}".format(x.sum() / x.numel()))
            print("real发射率：{}".format(self.sig_fn(x).sum() / x.numel()))
            print("3的发射率：{}".format(torch.where(x>2.5,1,0).sum() / x.numel()))

        x,loss_a,loss_b = patch_embed3(x,loss_a,loss_b)
        print("发射率：{}".format(x.sum() / x.numel()))
        print("real发射率：{}".format(self.sig_fn(x).sum() / x.numel()))
        print("3的发射率：{}".format(torch.where(x>2.5,1,0).sum() / x.numel()))
        for blk in stage3:
            x ,loss_a,loss_b= blk(x,loss_a,loss_b)
            print("发射率：{}".format(x.sum() / x.numel()))
            print("real发射率：{}".format(self.sig_fn(x).sum() / x.numel()))
            print("3的发射率：{}".format(torch.where(x>2.5,1,0).sum() / x.numel()))

        return x.flatten(3).mean(3),loss_a,loss_b

    def forward(self, x):

        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x,loss_a ,loss_b= self.forward_features(x)
        x = self.head(x)
        print("*" * 30)

        return x .mean(0),loss_a,loss_b


@register_model
def QKFormer(pretrained=False, **kwargs):
    model = spiking_transformer(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    input = torch.randn(2, 3, 32, 32).cuda(0)
    model = create_model(
        'QKFormer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=100, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        T=4,
    ).cuda(0)

    from torchinfo import summary
    summary(model, input_size=(2, 3, 32, 32))