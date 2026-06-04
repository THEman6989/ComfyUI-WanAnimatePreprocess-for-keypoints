import json
import math
import numpy as np
import torch

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def _mask_np(m):
    a = m.detach().cpu().float().numpy() if isinstance(m, torch.Tensor) else np.asarray(m, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, :, :]
    if a.ndim != 3:
        raise ValueError(f"Expected MASK [B,H,W], got {a.shape}")
    return a.astype(np.float32)


def _img_np(x):
    a = x.detach().cpu().float().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
    if a.ndim == 3:
        a = a[None, :, :, :]
    if a.ndim != 4:
        raise ValueError(f"Expected IMAGE [B,H,W,C], got {a.shape}")
    return np.clip(a.astype(np.float32), 0, 1)


def _mt(a):
    return torch.from_numpy(np.clip(a, 0, 1).astype(np.float32))


def _it(a):
    return torch.from_numpy(np.clip(a, 0, 1).astype(np.float32))


def _stats(mask, th=0.5):
    b = mask > th
    area = int(b.sum())
    if area <= 0:
        return {"area": 0, "mean": float(mask.mean()), "bbox": None, "cx": None, "cy": None}
    ys, xs = np.where(b)
    return {"area": area, "mean": float(mask.mean()), "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], "cx": float(xs.mean()), "cy": float(ys.mean())}


def _resize(a, size, nearest=False):
    if not HAS_CV2:
        raise RuntimeError("opencv-python is required")
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(a.astype(np.float32), (int(size), int(size)), interpolation=interp)


class MaskQualityFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",), "area_threshold": ("INT", {"default": 64, "min": 1, "max": 1000000}), "mean_threshold": ("FLOAT", {"default": 0.001, "min": 0, "max": 1, "step": 0.0001}), "max_area_fraction": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 1, "step": 0.01}), "max_centroid_jump_px": ("FLOAT", {"default": 160.0, "min": 0, "max": 10000, "step": 1}), "max_area_jump_ratio": ("FLOAT", {"default": 5.0, "min": 1, "max": 100, "step": 0.1}), "mode": (["zero_invalid", "keep_original"], {"default": "zero_invalid"}), "threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01})}}
    RETURN_TYPES = ("MASK", "MASK", "STRING")
    RETURN_NAMES = ("filtered_masks", "invalid_frame_mask", "report")
    FUNCTION = "run"
    CATEGORY = "mask/video"

    def run(self, masks, area_threshold, mean_threshold, max_area_fraction, max_centroid_jump_px, max_area_jump_ratio, mode, threshold):
        arr = _mask_np(masks)
        n, h, w = arr.shape
        st = [_stats(arr[i], threshold) for i in range(n)]
        valid = [True] * n
        reasons = {}
        def bad(i, r):
            valid[i] = False
            reasons[r] = reasons.get(r, 0) + 1
        for i, s in enumerate(st):
            if s["area"] < area_threshold: bad(i, "area_small")
            if s["mean"] < mean_threshold: bad(i, "mean_low")
            if s["area"] > max_area_fraction * h * w: bad(i, "area_large")
        last = None
        for i, s in enumerate(st):
            if not valid[i]:
                continue
            if last is not None and st[last]["cx"] is not None and s["cx"] is not None:
                d = math.hypot(s["cx"] - st[last]["cx"], s["cy"] - st[last]["cy"])
                if d > max_centroid_jump_px:
                    bad(i, "centroid_jump")
                    continue
                ratio = max(s["area"] / max(1, st[last]["area"]), st[last]["area"] / max(1, s["area"]))
                if ratio > max_area_jump_ratio:
                    bad(i, "area_jump")
                    continue
            last = i
        out = arr.copy()
        inv = np.zeros_like(arr, dtype=np.float32)
        for i, ok in enumerate(valid):
            if not ok:
                inv[i] = 1
                if mode == "zero_invalid": out[i] = 0
        return (_mt(out), _mt(inv), json.dumps({"frames": n, "valid": int(sum(valid)), "invalid": int(n - sum(valid)), "reasons": reasons}, indent=2))


class MaskInterpolatorPro:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",), "area_threshold": ("INT", {"default": 64, "min": 1, "max": 1000000}), "mean_threshold": ("FLOAT", {"default": 0.001, "min": 0, "max": 1, "step": 0.0001}), "soft_blend": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}), "threshold": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}), "max_gap": ("INT", {"default": 24, "min": 1, "max": 1000})}, "optional": {"images": ("IMAGE",)}}
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("masks", "report")
    FUNCTION = "run"
    CATEGORY = "mask/video"

    def run(self, masks, area_threshold, mean_threshold, soft_blend, threshold, max_gap, images=None):
        arr = _mask_np(masks)
        n = arr.shape[0]
        ok = [int((arr[i] > 0.5).sum()) >= area_threshold and float(arr[i].mean()) >= mean_threshold for i in range(n)]
        idx = [i for i, v in enumerate(ok) if v]
        out = arr.copy()
        if not idx:
            return (_mt(out), "No valid masks found")
        for i in range(0, idx[0]): out[i] = out[idx[0]]
        for i in range(idx[-1] + 1, n): out[i] = out[idx[-1]]
        fixed = 0
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if gap <= 0: continue
            for j in range(1, gap + 1):
                t = j / (gap + 1)
                k = a + j
                if gap > max_gap:
                    out[k] = out[a] if t < 0.5 else out[b]
                else:
                    lin = (1 - t) * out[a] + t * out[b]
                    out[k] = (1 - soft_blend) * (lin > threshold).astype(np.float32) + soft_blend * lin
                fixed += 1
        return (_mt(out), json.dumps({"frames": n, "valid_before": int(sum(ok)), "repaired": fixed}, indent=2))


class MaskCropStabilizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "masks": ("MASK",), "output_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}), "padding_percent": ("FLOAT", {"default": 0.18, "min": 0, "max": 1, "step": 0.01}), "smoothing": ("FLOAT", {"default": 0.65, "min": 0, "max": 0.98, "step": 0.01}), "square_crop": ("BOOLEAN", {"default": True}), "mask_mode": (["image_crop", "masked_black", "masked_white"], {"default": "image_crop"}), "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01})}}
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("crops", "crop_masks", "report")
    FUNCTION = "run"
    CATEGORY = "mask/video"

    def run(self, images, masks, output_size, padding_percent, smoothing, square_crop, mask_mode, mask_threshold):
        imgs = _img_np(images); ms = _mask_np(masks)
        n, h, w, c = imgs.shape
        boxes = []
        for i in range(n):
            s = _stats(ms[i], mask_threshold)
            boxes.append(s["bbox"] if s["bbox"] is not None else None)
        good = [i for i, b in enumerate(boxes) if b is not None]
        if not good:
            boxes = [[0, 0, w, h] for _ in range(n)]
        else:
            x = np.arange(n, dtype=np.float32)
            filled = np.zeros((n, 4), dtype=np.float32)
            for q in range(4): filled[:, q] = np.interp(x, np.array(good, dtype=np.float32), np.array([boxes[i][q] for i in good], dtype=np.float32))
            s = float(np.clip(smoothing, 0, 0.98)); f = filled.copy(); r = filled.copy()
            for i in range(1, n): f[i] = s * f[i-1] + (1-s) * filled[i]
            for i in range(n-2, -1, -1): r[i] = s * r[i+1] + (1-s) * filled[i]
            boxes = ((f + r) * 0.5).tolist()
        crops = np.zeros((n, output_size, output_size, c), dtype=np.float32); cms = np.zeros((n, output_size, output_size), dtype=np.float32)
        final = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box; bw = max(1, x2-x1); bh = max(1, y2-y1); pad = max(bw, bh) * padding_percent
            x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
            if square_crop:
                side = max(x2-x1, y2-y1); cx = (x1+x2)/2; cy = (y1+y2)/2; x1 = cx-side/2; x2 = cx+side/2; y1 = cy-side/2; y2 = cy+side/2
            x1 = int(max(0, min(w-1, round(x1)))); y1 = int(max(0, min(h-1, round(y1)))); x2 = int(max(x1+1, min(w, round(x2)))); y2 = int(max(y1+1, min(h, round(y2))))
            final.append([x1,y1,x2,y2])
            im = imgs[i, y1:y2, x1:x2, :]; ma = ms[i, y1:y2, x1:x2]
            if mask_mode != "image_crop":
                m3 = np.expand_dims(np.clip(ma, 0, 1), -1)
                im = im * m3 if mask_mode == "masked_black" else im * m3 + (1 - m3)
            crops[i] = _resize(im, output_size); cms[i] = _resize(ma, output_size)
        return (_it(crops), _mt(cms), json.dumps({"frames": n, "valid_bboxes": len(good), "first_bbox": final[0], "last_bbox": final[-1]}, indent=2))


NODE_CLASS_MAPPINGS = {"MaskQualityFilter": MaskQualityFilter, "MaskInterpolatorPro": MaskInterpolatorPro, "MaskCropStabilizer": MaskCropStabilizer}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskQualityFilter": "Mask Quality Filter", "MaskInterpolatorPro": "Mask Interpolator Pro", "MaskCropStabilizer": "Mask Crop Stabilizer"}
