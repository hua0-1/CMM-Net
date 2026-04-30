# Modality-Aware Fusion Network with Cross-Modal Alignment and Mid-Frequency Enhancement for Infrared and Visible Image Integration

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19915232.svg)](https://doi.org/10.5281/zenodo.19915232)

This repository contains the official implementation of our paper submitted to **The Visual Computer**.

Abstract
Infrared and visible image fusion integrates complementary multimodal information to generate high-quality images for surveillance and recognition tasks. Existing methods suffer from target information loss, color imbalance, and edge blurring due to insufficient cross-modal alignment and mid-frequency feature mining. This paper proposes a modality-aware fusion network built with three core components to address these issues. A modality-aware channel transformation block enhances infrared thermal and visible texture features via long-short range attention. A cross-modal feature adaptive calibration module aligns feature distributions end-to-end to balance fusion colors. A mid-frequency feature extractor captures edge and transition details to reduce blurring. A multi-constraint loss function ensures feature alignment and color consistency. Experiments on three datasets show the method outperforms seven advanced approaches, with 11% MI improvement, over 10% PSNR gain, and over 14% SSIM increase, demonstrating superior target integrity, color balance, and edge clarity for real-world visual computing applications.Furthermore, detection tests based on YOLOv12 indicate that, compared with other fusion techniques, our method achieves higher confidence scores and better overall detection performance.

# Training
The full training code will be uploaded upon paper acceptance. Stay tuned!
# Testing
python test.py --checkpoint CDDFuse_MFE_CMFAC_MACB_FusionFix_11-27-10-37_epoch120.pth --ir ir.png --vis vis.png

# Data Preparation
Download the MSRS dataset and place it in the folder ./data/.

## 📦 Model Weights

Download the pre-trained model: [百度网盘](https://pan.baidu.com/s/1OXp3ag4sg_GzfQh-qElPSw) 提取码: `3y3z`

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

Training Code
The full training code will be uploaded to this repository upon paper acceptance. Stay tuned!

📝 Citation
If you find this code useful, please cite our paper:

bibtex
@article{cmmnet2025,
  title={Modality-Aware Fusion Network with Cross-Modal Alignment and Mid-Frequency Enhancement for Infrared and Visible Image Integration},
  author={...},
  journal={The Visual Computer},
  year={2025}
}
