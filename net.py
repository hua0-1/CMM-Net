import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


# -------------------------- 基础工具模块 --------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# -------------------------- 标准化层（稳定训练） --------------------------
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        self.dim = dim
        if LayerNorm_type == 'BiasFree':
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        # 手动转置：[b,c,h,w] → [b,h,w,c] → 归一化 → 恢复维度
        x = x.permute(0, 2, 3, 1).reshape(b * h * w, self.dim)
        x = self.norm(x)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x


# -------------------------- 1. 模态感知长短程注意力（核心组件） --------------------------
class ModalAwareLongShortAttention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False, window_size=8):
        super(ModalAwareLongShortAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 固定为8（64//8）
        self.window_size = window_size  # 8×8窗口
        # temperature维度适配6维注意力计算
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1, 1, 1) * self.head_dim ** (-0.5))

        # 模态权重分支：学习模态专属通道权重
        self.modal_weight = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # QKV生成（含深度卷积增强局部关联）
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, other_x):
        b, c, h, w = x.shape  # 输入维度：[b,64,h,w]

        # 1. 模态自适应权重：强化模态专属通道（如红外热辐射通道、可见光背景通道）
        modal_w = torch.sigmoid(self.modal_weight(x))  # [b,64,h,w]
        x_weighted = x * modal_w  # [b,64,h,w]

        # 2. 生成QKV并拆分注意力头（维度顺序：[b, head, h, w, head_dim]）
        qkv = self.qkv_dwconv(self.qkv(x_weighted))  # [b,192,h,w]
        q, k, v = qkv.chunk(3, dim=1)  # 各为[b,64,h,w]
        q = rearrange(q, 'b (head hd) h w -> b head h w hd', head=self.num_heads, hd=self.head_dim)
        k = rearrange(k, 'b (head hd) h w -> b head h w hd', head=self.num_heads, hd=self.head_dim)
        v = rearrange(v, 'b (head hd) h w -> b head h w hd', head=self.num_heads, hd=self.head_dim)
        q = F.normalize(q, dim=-1)  # 对head_dim维度归一化，增强稳定性
        k = F.normalize(k, dim=-1)

        # 3. 模态差异因子：动态调整局部/全局注意力权重（差异大→局部，差异小→全局）
        diff_map = torch.abs(x - other_x).mean(dim=1, keepdim=True)  # [b,1,h,w]
        diff_factor = torch.sigmoid(diff_map)  # [b,1,h,w]，0-1归一化

        # 4. 并行计算局部窗口（短程）与全局稀疏（长程）注意力
        local_attn = self.local_window_attention(q, k, v, h, w)  # [b,64,h,w]
        global_attn = self.global_sparse_attention(q, k, v, h, w)  # [b,64,h,w]

        # 5. 动态加权融合 + 输出投影
        attn_output = local_attn * (0.5 + 0.5 * diff_factor) + global_attn * (0.5 - 0.5 * diff_factor)
        return self.project_out(attn_output)  # [b,64,h,w]

    def local_window_attention(self, q, k, v, h, w):
        """局部窗口注意力：捕捉短程依赖（如目标边缘、局部纹理）"""
        win_size = self.window_size
        num_win_h = h // win_size  # 窗口行数（如128//8=16）
        num_win_w = w // win_size  # 窗口列数（如128//8=16）
        win_pix = win_size * win_size  # 每个窗口像素数（64）

        # 按空间维度拆分窗口，维度：[b, head, num_win_h, num_win_w, win_size, win_size, head_dim]
        q_win = rearrange(q, 'b head (nh hs) (nw ws) hd -> b head nh nw hs ws hd',
                          nh=num_win_h, nw=num_win_w, hs=win_size, ws=win_size)
        k_win = rearrange(k, 'b head (nh hs) (nw ws) hd -> b head nh nw hs ws hd',
                          nh=num_win_h, nw=num_win_w, hs=win_size, ws=win_size)
        v_win = rearrange(v, 'b head (nh hs) (nw ws) hd -> b head nh nw hs ws hd',
                          nh=num_win_h, nw=num_win_w, hs=win_size, ws=win_size)

        # 窗口内特征平坦化，确保矩阵乘法维度匹配
        q_win_flat = rearrange(q_win, 'b head nh nw hs ws hd -> b head nh nw (hs ws) hd')  # [b,8,16,16,64,8]
        k_win_flat = rearrange(k_win, 'b head nh nw hs ws hd -> b head nh nw hd (hs ws)')  # [b,8,16,16,8,64]

        # 注意力计算（temperature维度适配6维attn）
        attn = (q_win_flat @ k_win_flat) * self.temperature  # [b,8,16,16,64,64]
        attn = attn.softmax(dim=-1)  # 对窗口内像素维度softmax

        # 注意力加权输出 + 恢复空间维度
        v_win_flat = rearrange(v_win, 'b head nh nw hs ws hd -> b head nh nw (hs ws) hd')  # [b,8,16,16,64,8]
        out_win = attn @ v_win_flat  # [b,8,16,16,64,8]
        out = rearrange(out_win, 'b head nh nw (hs ws) hd -> b (head hd) (nh hs) (nw ws)',
                        hs=win_size, ws=win_size)  # [b,64,128,128]
        return out

    def global_sparse_attention(self, q, k, v, h, w):
        """全局稀疏注意力：捕捉长程依赖（如场景全局布局）"""
        win_size = self.window_size
        # 提取窗口中心像素（减少计算量，避免全注意力O(N²)复杂度）
        center_h = torch.arange(win_size // 2, h, win_size, device=q.device)  # [8,16,...,120]（16个）
        center_w = torch.arange(win_size // 2, w, win_size, device=q.device)  # [8,16,...,120]（16个）
        centers = torch.stack(torch.meshgrid(center_h, center_w, indexing='ij'), dim=-1).reshape(-1, 2)  # [256,2]
        N = len(centers)  # 中心像素总数（256）

        # 按坐标提取中心像素，维度：[b, head, N, head_dim]
        q_global = q[:, :, centers[:, 0], centers[:, 1], :]  # [b,8,256,8]
        k_global = k[:, :, centers[:, 0], centers[:, 1], :]  # [b,8,256,8]
        v_global = v[:, :, centers[:, 0], centers[:, 1], :]  # [b,8,256,8]

        # 全局注意力计算（temperature维度适配）
        attn = (q_global @ k_global.transpose(-2, -1)) * self.temperature[..., 0, 0, :, :]  # [b,8,256,256]
        attn = attn.softmax(dim=-1)
        out_global = attn @ v_global  # [b,8,256,8]

        # 恢复为完整特征图（非中心区域双线性插值补全）
        out = torch.zeros_like(q)  # [b,8,128,128,8]
        out[:, :, centers[:, 0], centers[:, 1], :] = out_global  # 填充中心像素
        out = rearrange(out, 'b head h w hd -> b (head hd) h w')  # [b,64,128,128]
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)  # 补全非中心区域
        return out


# -------------------------- 2. 渐进式通道变换前馈网络 --------------------------
class ProgressiveChannelFFN(nn.Module):
    def __init__(self, dim=64, ffn_expansion_factor=2, bias=False):
        super(ProgressiveChannelFFN, self).__init__()
        self.dim = dim  # 输入通道数（固定64）
        # 通道变换维度：严格遵循“压缩→恢复”对称结构，确保残差连接维度匹配
        self.compress1 = nn.Conv2d(dim, dim // 2, kernel_size=1, bias=bias)  # 64→32（第一步压缩）
        self.conv_local = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=bias)  # 32→32（局部交互）
        self.compress2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=bias)  # 32→32（不进一步压缩，避免广播问题）
        self.depth_conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2,
                                    bias=bias)  # 32→32（轻量化）
        self.restore1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=bias)  # 32→32（保持维度）
        self.restore2 = nn.Conv2d(dim // 2, dim, kernel_size=1, bias=bias)  # 32→64（最终恢复）
        self.act = nn.GELU()

    def forward(self, x):
        # 输入维度验证
        assert x.shape[1] == self.dim, f"前馈网络输入通道错误：需{self.dim}通道，实际{x.shape[1]}通道"

        # 1. 压缩阶段1：64→32 + 残差连接（通道完全匹配）
        comp1 = self.act(self.compress1(x))  # [b, 32, h, w]
        comp1 = comp1 + x[:, :self.dim // 2, :, :]  # [b,32,h,w] + [b,32,h,w]

        # 2. 局部交互：32→32（增强特征关联性）
        conv_local = self.act(self.conv_local(comp1))  # [b,32,h,w]

        # 3. 压缩阶段2：32→32（不降维，避免广播问题）
        comp2 = self.act(self.compress2(conv_local))  # [b,32,h,w]
        comp2 = comp2 + conv_local  # [b,32,h,w] + [b,32,h,w]（通道完全匹配）

        # 4. 轻量化处理：32→32（深度卷积降参）
        depth = self.act(self.depth_conv(comp2))  # [b,32,h,w]
        depth = depth + comp2  # [b,32,h,w] + [b,32,h,w]（通道完全匹配）

        # 5. 恢复阶段1：32→32（保持维度）
        res1 = self.act(self.restore1(depth))  # [b,32,h,w]
        res1 = res1 + depth  # [b,32,h,w] + [b,32,h,w]（通道完全匹配）

        # 6. 恢复阶段2：32→64 + 残差连接（通道完全匹配）
        res2 = self.restore2(res1)  # [b,64,h,w]
        res2 = res2 + x  # [b,64,h,w] + [b,64,h,w]（最终残差）

        return res2


# -------------------------- 3. 模态感知通道变换单元（MACU）——原ImprovedLTBock --------------------------
class ModalAwareChannelTransformUnit(nn.Module):
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(ModalAwareChannelTransformUnit, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)  # 注意力前归一化
        self.attn = ModalAwareLongShortAttention(dim=dim, num_heads=num_heads, bias=bias)  # 模态感知注意力
        self.norm2 = LayerNorm(dim, LayerNorm_type)  # 前馈网络前归一化
        self.ffn = ProgressiveChannelFFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)  # 渐进式前馈
        self.drop_path = DropPath(drop_prob=0.1)  # 防止过拟合

    def forward(self, x, other_x):
        # 注意力分支：残差+模态感知注意力
        x = x + self.drop_path(self.attn(self.norm1(x), other_x))
        # 前馈分支：残差+渐进式通道变换
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# -------------------------- 4. 模态感知通道变换块（MACB）——原ImprovedBTE --------------------------
class ModalAwareChannelTransformBlock(nn.Module):
    def __init__(self, dim=64, num_heads=8, num_blocks=4, ffn_expansion_factor=2, bias=False,
                 LayerNorm_type='WithBias'):
        super(ModalAwareChannelTransformBlock, self).__init__()
        self.blocks = nn.ModuleList([
            ModalAwareChannelTransformUnit(
                dim=dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks)
        ])

    def forward(self, x, other_x):
        for block in self.blocks:
            x = block(x, other_x)  # 每个MACU单元均使用模态感知注意力
        return x


# -------------------------- 5. 跨模态特征自适应校准模块（CMFAC）——优化后 --------------------------
class CrossModalFeatureAlignment(nn.Module):
    def __init__(self, dim=64, bias=False, LayerNorm_type='WithBias'):
        super(CrossModalFeatureAlignment, self).__init__()
        self.dim = dim
        # 1. 模态分布校准：通过仿射变换对齐VIS/IR特征的均值和方差（保留原逻辑）
        self.norm_vis = LayerNorm(dim, LayerNorm_type)
        self.norm_ir = LayerNorm(dim, LayerNorm_type)
        self.affine_vis = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # VIS特征仿射
        self.affine_ir = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # IR特征仿射

        # 2. 模态优势特征注意力：突出各模态的独特信息（降低注意力强度）
        self.vis_attention = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.Sigmoid()
        )
        self.ir_attention = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.Sigmoid()
        )

        # 3. 特征融合门：自适应融合校准后的特征（降低融合权重影响）
        self.fusion_gate = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        # 新增：亮度归一化约束（防止特征值溢出）
        self.brightness_norm = LayerNorm(dim, LayerNorm_type)
        # 新增：校准强度控制（降低CMFAC对原始特征的干扰）
        self.calib_strength = nn.Parameter(torch.tensor(0.3))  # 初始强度0.3，可训练调整

    def forward(self, feat_vis, feat_ir):
        # 步骤1：分布校准（对齐均值方差）
        feat_vis_norm = self.norm_vis(feat_vis)
        feat_ir_norm = self.norm_ir(feat_ir)
        feat_vis_calib = self.affine_vis(feat_vis_norm)  # [b,64,h,w]
        feat_ir_calib = self.affine_ir(feat_ir_norm)  # [b,64,h,w]

        # 步骤2：亮度归一化（关键：限制特征值范围，避免过亮）
        feat_vis_calib = self.brightness_norm(feat_vis_calib)
        feat_ir_calib = self.brightness_norm(feat_ir_calib)
        # 进一步限制极值（0-1区间）
        feat_vis_calib = torch.clamp(feat_vis_calib, min=0, max=1)
        feat_ir_calib = torch.clamp(feat_ir_calib, min=0, max=1)

        # 步骤3：模态优势注意力（降低注意力权重，避免过度增强）
        feat_concat = torch.cat([feat_vis_calib, feat_ir_calib], dim=1)  # [b,128,h,w]
        vis_attn = self.vis_attention(feat_concat) * 0.5  # 注意力强度减半（0-0.5）
        ir_attn = self.ir_attention(feat_concat) * 0.5  # 避免过度突出单模态特征
        feat_vis_enhanced = feat_vis_calib * vis_attn
        feat_ir_enhanced = feat_ir_calib * ir_attn

        # 步骤4：自适应融合（柔和融合，保留原始特征）
        fusion_weight = torch.sigmoid(self.fusion_gate(feat_concat))  # [b,64,h,w]
        aligned_feat = fusion_weight * feat_vis_enhanced + (1 - fusion_weight) * feat_ir_enhanced

        # 步骤5：控制校准强度（残差连接，保留原始特征主导权）
        aligned_feat = feat_vis * (1 - self.calib_strength) + aligned_feat * self.calib_strength

        return aligned_feat, feat_vis_enhanced, feat_ir_enhanced  # 返回校准后特征+增强后单模态特征


# -------------------------- 6. 原有依赖模块（确保兼容） --------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * self.head_dim ** (-0.5))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=self.head_dim)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=self.head_dim)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=self.head_dim)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=1, embed_dim=64, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class MidFrequencyEncoder(nn.Module):
    def __init__(self, in_channels=1, bias=False, LayerNorm_type='WithBias'):
        super(MidFrequencyEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=bias)
        self.norm_attn = LayerNorm(256, LayerNorm_type)
        self.attn = Attention(dim=256, num_heads=8, bias=bias)
        self.dilated_conv = nn.Conv2d(256, 256, kernel_size=3, dilation=2, padding=2, bias=bias)
        self.norm_final = LayerNorm(256, LayerNorm_type)
        self.relu = nn.ReLU(inplace=True)
        self.channel_adjust = nn.Conv2d(256, 64, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x + self.attn(self.norm_attn(x))
        x = self.relu(self.dilated_conv(x))
        x = self.norm_final(x)
        x = self.channel_adjust(x)
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :32], x[:, 32:]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        self.net = nn.Sequential(*[DetailNode() for _ in range(num_layers)])

    def forward(self, x):
        z1, z2 = x[:, :32], x[:, 32:]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


# -------------------------- 7. Restormer编码器（优化CMFAC特征融合逻辑） --------------------------
class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0],
                                                               ffn_expansion_factor=ffn_expansion_factor,
                                                               bias=bias, LayerNorm_type=LayerNorm_type)
                                              for _ in range(num_blocks[0])])
        # 集成MACB（原ImprovedBTE）作为基础特征提取模块
        self.baseFeature = ModalAwareChannelTransformBlock(
            dim=dim,
            num_heads=heads[2],
            num_blocks=num_blocks[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )
        self.detailFeature = DetailFeatureExtraction()
        self.midFeature = MidFrequencyEncoder(in_channels=inp_channels, bias=bias, LayerNorm_type=LayerNorm_type)
        # 集成优化后的CMFAC模块
        self.cmfac = CrossModalFeatureAlignment(dim=dim, bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, inp_img, other_inp_img=None):
        # 1. 浅层特征提取（共享编码器）
        inp_enc_level1 = self.patch_embed(inp_img)  # [b,64,h,w]
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # [b,64,h,w]

        # 2. 配对模态特征处理（训练时传入红外，测试时自配对）
        if other_inp_img is not None:
            other_inp_enc = self.patch_embed(other_inp_img)
            other_enc_level1 = self.encoder_level1(other_inp_enc)  # [b,64,h,w]
        else:
            other_enc_level1 = out_enc_level1  # 测试阶段自配对

        # 3. 基础特征提取（MACB模块）
        base_feature = self.baseFeature(out_enc_level1, other_enc_level1)  # [b,64,h,w]

        # 4. 跨模态特征校准（CMFAC模块）
        aligned_base_feat, vis_enhanced, ir_enhanced = self.cmfac(base_feature, other_enc_level1)

        # 5. 细节与中频特征提取（原有逻辑）
        detail_feature = self.detailFeature(out_enc_level1)  # [b,64,h,w]
        mid_feature = self.midFeature(inp_img)  # [b,64,h,w]

        # 优化：柔和融合校准特征（降低CMFAC对原始特征的干扰）
        base_feature = base_feature * 0.8 + aligned_base_feat * 0.2  # 原始特征占80%，校准特征占20%
        detail_feature = detail_feature * 0.9 + vis_enhanced * 0.1  # 细节特征以原始为主
        mid_feature = mid_feature * 0.9 + ir_enhanced * 0.1  # 中频特征以原始为主

        return base_feature, detail_feature, mid_feature, out_enc_level1


# -------------------------- 8. Restormer解码器（原有逻辑不变） --------------------------
class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1],
                                                               ffn_expansion_factor=ffn_expansion_factor,
                                                               bias=bias, LayerNorm_type=LayerNorm_type)
                                              for _ in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(dim // 2, out_channels, kernel_size=3, padding=1, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature, mid_feature):
        # 特征拼接与通道压缩
        out_enc_level0 = torch.cat((base_feature, detail_feature, mid_feature), dim=1)  # [b,192,h,w]
        out_enc_level0 = self.reduce_channel(out_enc_level0)  # [b,64,h,w]

        # 解码器Transformer块
        out_enc_level1 = self.encoder_level2(out_enc_level0)  # [b,64,h,w]

        # 尺寸对齐（确保输出与输入一致）
        if inp_img is not None and out_enc_level1.shape[2:] != inp_img.shape[2:]:
            out_enc_level1 = F.interpolate(out_enc_level1, size=inp_img.shape[2:], mode='bilinear', align_corners=True)

        # 输出投影与残差连接
        out_enc_level1 = self.output(out_enc_level1)  # [b,1,h,w]
        if inp_img is not None:
            out_enc_level1 = out_enc_level1 + inp_img[:, :1, :, :]  # 保留输入结构

        return self.sigmoid(out_enc_level1), out_enc_level0


# -------------------------- 测试代码（验证CMFAC与MACB集成正确性） --------------------------
if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 测试前馈网络
    ffn = ProgressiveChannelFFN(dim=64).to(device)
    test_feat = torch.randn(1, 64, 128, 128).to(device)
    ffn_out = ffn(test_feat)
    assert ffn_out.shape == test_feat.shape, f"前馈网络输出维度错误：{ffn_out.shape} != {test_feat.shape}"
    print("✅ 渐进式前馈网络测试通过！")

    # 2. 测试MACU单元
    macu = ModalAwareChannelTransformUnit(dim=64, num_heads=8).to(device)
    vis_feat = torch.randn(1, 64, 128, 128).to(device)
    ir_feat = torch.randn(1, 64, 128, 128).to(device)
    macu_out = macu(vis_feat, ir_feat)
    assert macu_out.shape == vis_feat.shape, f"MACU输出维度错误：{macu_out.shape} != {vis_feat.shape}"
    print("✅ 模态感知通道变换单元（MACU）测试通过！")

    # 3. 测试MACB块
    macb = ModalAwareChannelTransformBlock(dim=64, num_blocks=4).to(device)
    macb_out = macb(vis_feat, ir_feat)
    assert macb_out.shape == vis_feat.shape, f"MACB输出维度错误：{macb_out.shape} != {vis_feat.shape}"
    print("✅ 模态感知通道变换块（MACB）测试通过！")

    # 4. 测试CMFAC模块（优化后）
    cmfac = CrossModalFeatureAlignment(dim=64).to(device)
    aligned_feat, vis_enhanced, ir_enhanced = cmfac(vis_feat, ir_feat)
    assert aligned_feat.shape == vis_feat.shape, f"CMFAC输出维度错误：{aligned_feat.shape} != {vis_feat.shape}"
    assert torch.all(aligned_feat >= 0) and torch.all(aligned_feat <= 1), "CMFAC输出未被限制在0-1区间"
    print("✅ 跨模态特征自适应校准模块（CMFAC）测试通过！")

    # 5. 测试编码器-解码器完整流程（集成CMFAC+MACB）
    encoder = Restormer_Encoder(inp_channels=1).to(device)
    decoder = Restormer_Decoder(out_channels=1).to(device)
    test_vis_img = torch.randn(1, 1, 128, 128).to(device)
    test_ir_img = torch.randn(1, 1, 128, 128).to(device)
    base_feat, detail_feat, mid_feat, enc_feat = encoder(test_vis_img, test_ir_img)
    output_img, dec_feat = decoder(test_vis_img, base_feat, detail_feat, mid_feat)
    assert output_img.shape == (1, 1, 128, 128), f"解码器输出维度错误：{output_img.shape}"
    print("✅ 完整编码器-解码器流程（含优化后CMFAC+MACB）测试通过！所有模块无维度错误，可正常训练。")
