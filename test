# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse


# 图像读取/保存工具
def image_read_cv2(img_path, mode="GRAY"):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ 图像文件不存在: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"❌ 图像文件损坏: {img_path}")
    if mode == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def img_save(img, save_name, out_dir, normalize=True):
    """保存图像，自动归一化到0-255"""
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{save_name}.png")

    if img.dtype != np.uint8:
        if normalize:
            # 归一化到0-255范围
            img_min, img_max = img.min(), img.max()
            if img_max - img_min > 1e-8:
                img = (img - img_min) / (img_max - img_min)
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    cv2.imwrite(save_path, img)
    return save_path


# 导入整合模型（与训练结构完全一致）
from net import Restormer_Encoder, Restormer_Decoder, ModalAwareChannelTransformBlock, DetailFeatureExtraction, \
    MidFrequencyEncoder


class FusionNet(torch.nn.Module):
    """融合网络定义（与训练结构完全一致）"""

    def __init__(self):
        super(FusionNet, self).__init__()
        # 编码器（与训练时完全相同的配置）
        self.encoder = Restormer_Encoder(
            inp_channels=1, dim=64, num_blocks=[4, 4], heads=[8, 8, 8],
            ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'
        )
        # 解码器
        self.decoder = Restormer_Decoder(
            inp_channels=1, out_channels=1, dim=64, num_blocks=[4, 4], heads=[8, 8, 8],
            ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'
        )
        # 融合层（与训练时完全相同）
        self.base_fuse = ModalAwareChannelTransformBlock(
            dim=64, num_heads=8, num_blocks=4, ffn_expansion_factor=2,
            bias=False, LayerNorm_type='WithBias'
        )
        self.detail_fuse = DetailFeatureExtraction(num_layers=1)
        self.mid_fuse = MidFrequencyEncoder(
            in_channels=64, bias=False, LayerNorm_type='WithBias'
        )

    def forward(self, vis, ir):
        """前向传播（与训练时完全一致）"""
        # 编码器特征提取
        encoder_outputs_vis = self.encoder(vis, ir)
        f_V_B, f_V_D, f_V_M = encoder_outputs_vis[:3]

        encoder_outputs_ir = self.encoder(ir, vis)
        f_I_B, f_I_D, f_I_M = encoder_outputs_ir[:3]

        # 三特征融合（MACB + Detail + Mid）
        f_F_B = self.base_fuse(f_V_B, f_I_B)
        f_F_D = self.detail_fuse(f_V_D + f_I_D)
        f_F_M = self.mid_fuse(f_V_M + f_I_M)

        # 解码器生成融合图
        fused, _ = self.decoder(vis, f_F_B, f_F_D, f_F_M)
        return fused


def remove_module_prefix(state_dict):
    """移除DataParallel包装的'module.'前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        # 处理可能的'module.'前缀
        if k.startswith("module."):
            new_key = k[7:]  # 移除'module.'
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def pad_to_multiple(img, multiple=8):
    """将图像填充到指定倍数的尺寸（适配MACB窗口注意力）"""
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    # 对称填充，保持中心内容
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if len(img.shape) == 2:
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
    else:
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)

    return padded, (top, bottom, left, right)


def get_matched_image_pairs(vis_dir, ir_dir):
    """获取匹配的图像对"""
    support_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # 获取所有有效文件
    vis_files = {}
    for f in os.listdir(vis_dir):
        if f.lower().endswith(support_ext):
            name = os.path.splitext(f)[0]
            vis_files[name] = os.path.join(vis_dir, f)

    ir_files = {}
    for f in os.listdir(ir_dir):
        if f.lower().endswith(support_ext):
            name = os.path.splitext(f)[0]
            ir_files[name] = os.path.join(ir_dir, f)

    # 匹配相同名称的图像对
    common_names = sorted(set(vis_files.keys()) & set(ir_files.keys()))

    if not common_names:
        raise ValueError(f"❌ 无匹配图像对！可见光目录:{len(vis_files)}张，红外目录:{len(ir_files)}张")

    matched_pairs = [(vis_files[name], ir_files[name], name) for name in common_names]
    print(f"✅ 找到{len(matched_pairs)}对有效图像")
    return matched_pairs


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='CDDFuse-MFE 图像融合测试')
    parser.add_argument('--dataset', type=str, default='LLVIP', help='数据集名称')
    parser.add_argument('--model_path', type=str, default='models/CDDFuse_MFE_CMFAC_MACB_FusionFix_11-27-10-37_epoch120.pth', help='模型权重路径')
    parser.add_argument('--test_dir', type=str, default='test_img', help='测试图像目录')
    parser.add_argument('--out_dir', type=str, default='result/test_result_CMFAC_MACB2', help='结果保存目录')
    parser.add_argument('--device', type=str, default='cpu', help='设备选择')
    args = parser.parse_args()

    # 设备配置
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 推理设备: {device}")

    # 路径配置
    base_dir = os.path.abspath(os.path.dirname(__file__))
    test_ir_dir = os.path.join(base_dir, args.test_dir, args.dataset, "ir")
    test_vis_dir = os.path.join(base_dir, args.test_dir, args.dataset, "vis")
    out_dir = os.path.join(base_dir, args.out_dir, f"test_{args.dataset}")

    # 默认模型路径（根据训练时间戳调整）
    if args.model_path is None:
        model_path = os.path.join(base_dir, "models", "CDDFuse_MFE_CMFAC_MACB_FusionFix_11-27-10-37_epoch120.pth")
        # 如果默认路径不存在，尝试查找最新模型
        if not os.path.exists(model_path):
            model_dir = os.path.join(base_dir, "models")
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') and 'CDDFuse' in f]
                if model_files:
                    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                    model_path = os.path.join(model_dir, model_files[0])
                    print(f"🔍 使用最新模型: {model_files[0]}")
    else:
        model_path = args.model_path

    # 路径有效性检查
    for path, desc in [(test_ir_dir, "红外目录"), (test_vis_dir, "可见光目录")]:
        if not os.path.isdir(path):
            raise NotADirectoryError(f"❌ {desc}不存在: {path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型权重不存在: {model_path}")

    print(f"📦 加载模型: {os.path.basename(model_path)}")

    # 初始化模型
    model = FusionNet().to(device)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)

    try:
        # 加载各个模块的权重（与训练保存格式一致）
        if "DIDF_Encoder" in checkpoint:
            model.encoder.load_state_dict(remove_module_prefix(checkpoint["DIDF_Encoder"]), strict=False)
            print("  ✅ 编码器权重加载成功")

        if "DIDF_Decoder" in checkpoint:
            model.decoder.load_state_dict(remove_module_prefix(checkpoint["DIDF_Decoder"]), strict=False)
            print("  ✅ 解码器权重加载成功")

        if "BaseFuseLayer" in checkpoint:
            model.base_fuse.load_state_dict(remove_module_prefix(checkpoint["BaseFuseLayer"]), strict=False)
            print("  ✅ Base融合层权重加载成功")

        if "DetailFuseLayer" in checkpoint:
            model.detail_fuse.load_state_dict(remove_module_prefix(checkpoint["DetailFuseLayer"]), strict=False)
            print("  ✅ Detail融合层权重加载成功")

        if "MidFuseLayer" in checkpoint:
            model.mid_fuse.load_state_dict(remove_module_prefix(checkpoint["MidFuseLayer"]), strict=False)
            print("  ✅ Mid融合层权重加载成功")

        print("✅ 所有模型权重加载完成")

    except Exception as e:
        print(f"⚠️ 权重加载警告: {str(e)}")
        # 尝试加载完整模型状态字典
        try:
            model.load_state_dict(remove_module_prefix(checkpoint), strict=False)
            print("✅ 使用完整模型状态字典加载成功")
        except Exception as e2:
            print(f"❌ 权重加载失败: {str(e2)}")
            return

    model.eval()
    print(f"📁 结果保存至: {out_dir}")

    # 获取匹配图像对
    matched_pairs = get_matched_image_pairs(test_vis_dir, test_ir_dir)

    # 批量推理
    success_count = 0

    with torch.no_grad():
        for vis_path, ir_path, img_name in tqdm(matched_pairs, desc="🔍 推理进度"):
            try:
                # 图像读取与预处理
                ir_img = image_read_cv2(ir_path, mode="GRAY")
                vis_img = image_read_cv2(vis_path, mode="GRAY")

                # 保存原始尺寸
                orig_h, orig_w = ir_img.shape

                # 检查尺寸是否一致
                if vis_img.shape != ir_img.shape:
                    print(f"  ⚠️ 尺寸不匹配: VIS{vis_img.shape} vs IR{ir_img.shape}，调整可见光尺寸")
                    vis_img = cv2.resize(vis_img, (orig_w, orig_h))

                # 尺寸填充（确保为8的倍数）
                ir_padded, (top, bottom, left, right) = pad_to_multiple(ir_img, multiple=8)
                vis_padded, _ = pad_to_multiple(vis_img, multiple=8)

                # 张量转换
                ir_tensor = torch.FloatTensor(ir_padded / 255.0).unsqueeze(0).unsqueeze(0).to(device)
                vis_tensor = torch.FloatTensor(vis_padded / 255.0).unsqueeze(0).unsqueeze(0).to(device)

                # 前向推理
                fused_tensor = model(vis_tensor, ir_tensor)

                # 结果后处理
                fused_padded = fused_tensor.squeeze().cpu().numpy()

                # 裁剪回原始尺寸
                fused_orig = fused_padded[top:top + orig_h, left:left + orig_w]

                # 确保值在有效范围内
                fused_orig = np.clip(fused_orig, 0, 1)

                # 保存结果
                save_path = img_save(fused_orig, img_name, out_dir)

                tqdm.write(f"✅ {img_name} 融合完成")
                success_count += 1

            except Exception as e:
                tqdm.write(f"❌ 处理{img_name}失败: {str(e)[:100]}")
                import traceback
                traceback.print_exc()
                continue

    # 推理总结
    total_count = len(matched_pairs)
    print(f"\n🎉 推理完成! 成功处理{success_count}/{total_count}对图像")
    print(f"📁 结果目录: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序终止: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
