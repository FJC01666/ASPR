# -*- coding: utf-8 -*-
"""
DeepGlobe 一步到位预处理：
- 从 raw_root/train 中按 <id>_sat.* 与 <id>_mask.* 成对读取
- 每对切成 6x6 patch
- 过滤：单色或“最小颜色占比 <= 0.048”的 patch 丢弃
- 对每个 patch 生成 6 类二值掩码（按给定 RGB 颜色）
- 直接导出到： out_root/Deepglobe/<1..6>/test/{origin,groundtruth}
"""
import os, glob, argparse, shutil
import numpy as np
import cv2
from PIL import Image

# ===== 配置（与文章一致） =====
NUM_SPLIT = 6
FG_THRESH = 0.048
# 6 类颜色（RGB）
LABELSET_RGB = [
    [0, 255, 255],    # class 1
    [255, 255, 0],    # class 2
    [255, 0, 255],    # class 3
    [0, 255, 0],      # class 4
    [0,0,255],      # class 5
    [255, 255, 255],  # class 6
]

def ensure(d): os.makedirs(d, exist_ok=True)

def build_pairs(train_dir):
    """收集 *_sat 与 *_mask 成对样本"""
    sats, masks = {}, {}
    for f in os.listdir(train_dir):
        p = os.path.join(train_dir, f)
        if not os.path.isfile(p): continue
        name = os.path.basename(f)
        if "_sat" in name:
            stem = name.split("_sat")[0]; sats[stem] = p
        elif "_mask" in name:
            stem = name.split("_mask")[0]; masks[stem] = p
    pairs = []
    for k, ip in sats.items():
        mp = masks.get(k)
        if mp and os.path.exists(ip) and os.path.exists(mp):
            pairs.append((ip, mp, k))
    return sorted(pairs, key=lambda x: x[2])

def cut_patches(img, num=NUM_SPLIT):
    H, W = img.shape[:2]
    h, w = H // num, W // num
    out = []
    for i in range(num):
        for j in range(num):
            y0, y1 = i*h, (i+1)*h if i < num-1 else H
            x0, x1 = j*w, (j+1)*w if j < num-1 else W
            out.append(((i, j), img[y0:y1, x0:x1]))
    return out

def is_valid_mask(mask_img, fg_thresh=FG_THRESH):
    """过滤：单色或‘最小颜色占比 <= 阈值’则无效"""
    pil = Image.fromarray(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB))
    W, H = pil.size
    colors = pil.getcolors(maxcolors=W*H+10) or []
    if len(colors) <= 1:  # 单色
        return False
    total = float(sum(c for c, _ in colors))
    min_ratio = min(c/total for c, _ in colors)
    return min_ratio > fg_thresh

def binary_masks_from_color(mask_bgr):
    """按颜色表生成 {class_id: 0/255 二值掩码}"""
    m_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    out = {}
    for idx, color in enumerate(LABELSET_RGB, start=1):
        binm = np.all(m_rgb == np.array(color, dtype=np.uint8), axis=2).astype(np.uint8) * 255
        if binm.sum() > 0:
            out[idx] = binm
    return out

def export_patch(ipatch, bin_dict, stem_ij, dst_root):
    """把同一个 patch 复制到包含该类的每个类目录（origin/groundtruth）"""
    for cls_id, binm in bin_dict.items():
        dst_img = os.path.join(dst_root, str(cls_id), "test", "origin", f"{stem_ij}.jpg")
        dst_gt  = os.path.join(dst_root, str(cls_id), "test", "groundtruth", f"{stem_ij}.png")
        ensure(os.path.dirname(dst_img)); ensure(os.path.dirname(dst_gt))
        cv2.imwrite(dst_img, ipatch)
        cv2.imwrite(dst_gt,  binm)

def process(raw_root, out_root, clear=False):
    train_dir = os.path.join(raw_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"未找到 {train_dir}")

    dst_root = os.path.join(out_root, "Deepglobe")
    if clear and os.path.isdir(dst_root):
        shutil.rmtree(dst_root)
    ensure(dst_root)

    pairs = build_pairs(train_dir)
    if len(pairs) == 0:
        raise FileNotFoundError("在 train/ 中未匹配到 *_sat 与 *_mask 成对样本。")

    print(f"[INFO] 匹配到 {len(pairs)} 对样本，开始处理…")
    kept = 0
    for ipath, mpath, stem in pairs:
        img = cv2.imread(ipath, cv2.IMREAD_COLOR)
        msk = cv2.imread(mpath, cv2.IMREAD_COLOR)
        if img is None or msk is None: continue

        ipatches = cut_patches(img, NUM_SPLIT)
        mpatches = cut_patches(msk, NUM_SPLIT)

        for ((i, j), ip), ((_, _), mp) in zip(ipatches, mpatches):
            if not is_valid_mask(mp):  # 过滤弱前景/单色
                continue
            bin_dict = binary_masks_from_color(mp)
            if not bin_dict:  # 没任何类像素
                continue
            stem_ij = f"{stem}_{i}{j}"
            export_patch(ip, bin_dict, stem_ij, dst_root)
            kept += 1

    print(f"[Done] 共导出 {kept} 个有效 patch 到：{dst_root}")
    print("结构示例：")
    print(os.path.join(dst_root, "1", "test", "origin", "<id_ij>.jpg"))
    print(os.path.join(dst_root, "1", "test", "groundtruth", "<id_ij>.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="原始 DeepGlobe 根目录（含 train/，文件为 *_sat 与 *_mask）")
    ap.add_argument("--out_root", required=True, help="输出根目录，如 E:/DeepGlobe/preprocessed")
    ap.add_argument("--clear", action="store_true", help="若已存在目标目录，先清空再写入")
    args = ap.parse_args()
    process(args.raw_root, args.out_root, clear=args.clear)

if __name__ == "__main__":
    main()

