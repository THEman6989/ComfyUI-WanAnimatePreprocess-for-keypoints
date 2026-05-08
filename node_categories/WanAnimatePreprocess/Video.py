import os
import copy
import math
from collections import defaultdict

import random
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import json
import logging
import traceback
script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import load_torch_file, ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

from ...models.onnx_models import ViTPose, Yolo
from ...pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, crop, bbox_from_detector
from ...utils import get_face_bboxes, padding_resize, resize_by_area, resize_to_bounds
from ...pose_utils.human_visualization import AAPoseMeta, draw_aapose_by_meta_new, draw_aaface_by_meta
from ...retarget_pose import get_retarget_pose
from ...pose_data_editor_alone_automatic import PoseDataEditorAloneAutomaticChatyNode


BODY_GROUPS = {
    "ALL": list(range(20)),
    "TORSO": [1, 2, 5, 8, 11],
    "SHOULDERS": [2, 5],
    "ARMS": [2, 3, 4, 5, 6, 7],
    "LEGS": [8, 9, 10, 11, 12, 13],
    "FEET": [10, 13, 18, 19],
    "HEAD": [0, 14, 15, 16, 17],
    "HIP_WIDTH": [8, 11],
    "KNEE_WIDTH": [9, 12],
}

HAND_GROUPS = {
    "LEFT_HAND": "left",
    "RIGHT_HAND": "right",
    "HANDS": "both",
}

FACE_GROUP = {
    "FACE": True,
}

TARGET_OPTIONS = [
    "ALL",
    "BODY",
    "TORSO",
    "SHOULDERS",
    "ARMS",
    "LEGS",
    "FEET",
    "HEAD",
    "HIP_WIDTH",
    "KNEE_WIDTH",
    "HANDS",
    "LEFT_HAND",
    "RIGHT_HAND",
    "FACE",
]

TORSO_LENGTH_PAIRS = [
    (1, 2),  # neck to right shoulder
    (1, 5),  # neck to left shoulder
    (1, 8),  # neck to right hip
    (1, 11),  # neck to left hip
    (8, 11),  # hip width
]

FULL_BODY_LENGTH_PAIRS = TORSO_LENGTH_PAIRS + [
    (2, 3),  # right shoulder to right elbow
    (3, 4),  # right elbow to right wrist
    (5, 6),  # left shoulder to left elbow
    (6, 7),  # left elbow to left wrist
    (8, 9),  # right hip to right knee
    (9, 10),  # right knee to right ankle
    (11, 12),  # left hip to left knee
    (12, 13),  # left knee to left ankle
]



FRAME_DEPTH_SUBSAMPLE_INFO_TYPE = "FRAME_DEPTH_SUBSAMPLE_INFO"
FRAME_DEPTH_BATCH_INFO_TYPE = "FRAME_DEPTH_BATCH_INFO"
FRAME_DEPTH_BATCH_FEEDBACK_TYPE = "FRAME_DEPTH_BATCH_FEEDBACK"
FRAME_DEPTH_BATCH_FLOW_TYPE = "FRAME_DEPTH_BATCH_FLOW"


def _ceil_to_multiple(value, divisor):
    divisor = max(1, int(divisor))
    return int(math.ceil(float(value) / divisor) * divisor)


def _aspect_preserving_canvas_size(width, height, target_megapixels, divisible_by):
    width = int(width)
    height = int(height)
    divisor = max(1, int(divisible_by))
    target_area = max(1.0, float(target_megapixels) * 1_000_000.0)

    if width <= 0 or height <= 0:
        return width, height, width, height, 0, 0, False, False

    original_area = width * height
    if target_area >= original_area and width % divisor == 0 and height % divisor == 0:
        return width, height, width, height, 0, 0, False, False

    ratio_gcd = math.gcd(width, height)
    base_w = width // ratio_gcd
    base_h = height // ratio_gcd

    start_multiplier = min(
        ratio_gcd,
        max(1, int(math.floor(math.sqrt(target_area / float(base_w * base_h))))),
    )

    best = None
    for multiplier in range(start_multiplier, 0, -1):
        content_w = base_w * multiplier
        content_h = base_h * multiplier
        canvas_w = _ceil_to_multiple(content_w, divisor)
        canvas_h = _ceil_to_multiple(content_h, divisor)

        if canvas_w > width or canvas_h > height:
            continue

        best = (content_w, content_h, canvas_w, canvas_h)
        if canvas_w * canvas_h <= target_area:
            break

    if best is None:
        content_w = width
        content_h = height
        canvas_w = _ceil_to_multiple(width, divisor)
        canvas_h = _ceil_to_multiple(height, divisor)
    else:
        content_w, content_h, canvas_w, canvas_h = best

    pad_left = max(0, (canvas_w - content_w) // 2)
    pad_top = max(0, (canvas_h - content_h) // 2)
    did_resize = content_w != width or content_h != height
    did_pad = canvas_w != content_w or canvas_h != content_h

    return (
        int(content_w),
        int(content_h),
        int(canvas_w),
        int(canvas_h),
        int(pad_left),
        int(pad_top),
        did_resize,
        did_pad,
    )


def _resize_image_batch_cv2(images, width, height, interpolation, clamp_output=False):
    if int(images.shape[2]) == int(width) and int(images.shape[1]) == int(height):
        out = images
    else:
        device = images.device
        dtype = images.dtype
        frames = images.detach().cpu().numpy()
        resized_frames = []

        for frame in frames:
            resized = cv2.resize(frame, (int(width), int(height)), interpolation=interpolation)
            if resized.ndim == 2:
                resized = resized[:, :, None]
            resized_frames.append(resized.astype(np.float32, copy=False))

        out = torch.from_numpy(np.stack(resized_frames, axis=0)).to(device=device, dtype=dtype)

    if clamp_output:
        out = torch.clamp(out, 0.0, 1.0)

    return out


def _pad_image_batch(images, width, height, pad_left, pad_top):
    batch, image_h, image_w, channels = images.shape
    width = int(width)
    height = int(height)
    pad_left = int(pad_left)
    pad_top = int(pad_top)

    if image_w == width and image_h == height:
        return images

    out = torch.zeros((batch, height, width, channels), device=images.device, dtype=images.dtype)
    out[:, pad_top:pad_top + image_h, pad_left:pad_left + image_w, :] = images
    return out


def _crop_image_batch(images, width, height, pad_left, pad_top):
    image_h = int(images.shape[1])
    image_w = int(images.shape[2])
    width = min(int(width), image_w)
    height = min(int(height), image_h)
    pad_left = min(max(0, int(pad_left)), max(0, image_w - width))
    pad_top = min(max(0, int(pad_top)), max(0, image_h - height))

    return images[:, pad_top:pad_top + height, pad_left:pad_left + width, :]


def _concat_image_batches(batches):
    valid_batches = [batch for batch in batches if batch is not None and int(batch.shape[0]) > 0]
    if not valid_batches:
        return None
    if len(valid_batches) == 1:
        return valid_batches[0]
    return torch.cat(valid_batches, dim=0)


def _prepare_depth_source_images(
    images,
    current_fps,
    target_fps,
    target_megapixels,
    divisible_by,
    frame_limit_mode,
    max_frames,
    allow_fractional_fps,
):
    total_frames, original_h, original_w = images.shape[:3]
    (
        content_w,
        content_h,
        canvas_w,
        canvas_h,
        pad_left,
        pad_top,
        did_resize,
        did_pad,
    ) = _aspect_preserving_canvas_size(
        original_w,
        original_h,
        target_megapixels,
        divisible_by,
    )
    valid_indices, sampled_fps, resolved_frame_mode = _resolve_temporal_plan(
        total_frames,
        current_fps,
        target_fps,
        frame_limit_mode,
        max_frames,
        allow_fractional_fps,
    )

    source_images = images[valid_indices]
    if did_resize:
        source_images = _resize_image_batch_cv2(
            source_images,
            content_w,
            content_h,
            cv2.INTER_AREA,
            clamp_output=True,
        )
    if did_pad:
        source_images = _pad_image_batch(source_images, canvas_w, canvas_h, pad_left, pad_top)

    info = {
        "version": 2,
        "node": "FrameSubsamplerForDepthBatch",
        "original_width": int(original_w),
        "original_height": int(original_h),
        "content_width": int(content_w),
        "content_height": int(content_h),
        "sampled_width": int(canvas_w),
        "sampled_height": int(canvas_h),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "divisible_by": int(divisible_by),
        "target_megapixels": float(target_megapixels),
        "original_frame_count": int(total_frames),
        "sampled_frame_count": int(source_images.shape[0]),
        "current_fps": float(current_fps),
        "target_fps": float(target_fps),
        "sampled_fps": float(sampled_fps),
        "frame_limit_mode": str(resolved_frame_mode),
        "max_frames": int(max_frames),
        "allow_fractional_fps": bool(allow_fractional_fps),
        "valid_indices": [int(i) for i in valid_indices],
    }

    return source_images, info, resolved_frame_mode


def _slice_image_batch(images, start, end):
    start = max(0, int(start))
    end = min(int(images.shape[0]), max(start, int(end)))
    return images[start:end]


def _take_tail(images, count):
    count = max(0, int(count))
    if images is None or count <= 0 or int(images.shape[0]) <= 0:
        return None
    count = min(count, int(images.shape[0]))
    return images[-count:]


def _raw_link_value(value):
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        return value
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        return list(value)
    return None


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _duration_seconds(frame_count, fps):
    fps = _safe_float(fps, 0.0)
    frame_count = int(frame_count)
    if frame_count <= 0 or fps <= 0.0:
        return 0.0
    return frame_count / fps


def _sample_count_for_fps(total_frames, current_fps, target_fps):
    total_frames = int(total_frames)
    current_fps = max(0.001, float(current_fps))
    target_fps = max(0.001, float(target_fps))

    if total_frames <= 1 or target_fps >= current_fps:
        return total_frames

    duration = total_frames / current_fps
    return max(1, min(total_frames, int(math.ceil(duration * target_fps))))


def _resample_positions(input_count, output_count):
    input_count = int(input_count)
    output_count = int(output_count)

    if input_count <= 0 or output_count <= 0:
        return []

    if input_count == 1:
        return [0 for _ in range(output_count)]

    if output_count == 1:
        return [0]

    scale = (input_count - 1) / float(output_count - 1)
    return [min(input_count - 1, max(0, int(round(i * scale)))) for i in range(output_count)]


def _make_frame_indices_by_count(total_frames, target_frame_count):
    total_frames = int(total_frames)
    target_frame_count = max(1, min(total_frames, int(target_frame_count)))

    if target_frame_count >= total_frames:
        return list(range(total_frames))

    return sorted(set(_resample_positions(total_frames, target_frame_count))) or [0]


def _resolve_temporal_plan(total_frames, current_fps, target_fps, frame_limit_mode, max_frames, allow_fractional_fps):
    total_frames = int(total_frames)
    current_fps = max(0.001, float(current_fps))
    target_fps = max(0.001, float(target_fps))
    max_frames = max(1, int(max_frames))
    duration = _duration_seconds(total_frames, current_fps)

    if total_frames <= 1 or duration <= 0.0:
        return list(range(total_frames)), current_fps, "single_frame"

    mode = str(frame_limit_mode or "target_fps")

    if mode == "max_frames":
        target_count = min(total_frames, max_frames)

        if allow_fractional_fps:
            sampled_fps = target_count / duration
        else:
            sampled_fps = max(1.0, math.floor(target_count / duration))
            while sampled_fps > 1.0 and _sample_count_for_fps(total_frames, current_fps, sampled_fps) > target_count:
                sampled_fps -= 1.0
            target_count = _sample_count_for_fps(total_frames, current_fps, sampled_fps)

        indices = _make_frame_indices_by_count(total_frames, target_count)
        actual_fps = len(indices) / duration
        if not allow_fractional_fps:
            actual_fps = float(int(round(actual_fps)))
        return indices, min(current_fps, actual_fps), "max_frames"

    sampled_fps = min(current_fps, target_fps)
    if not allow_fractional_fps:
        sampled_fps = max(1.0, float(int(round(sampled_fps))))
        sampled_fps = min(current_fps, sampled_fps)

    target_count = _sample_count_for_fps(total_frames, current_fps, sampled_fps)
    indices = _make_frame_indices_by_count(total_frames, target_count)
    actual_fps = len(indices) / duration
    if not allow_fractional_fps:
        actual_fps = float(int(round(actual_fps)))
    return indices, min(current_fps, actual_fps), "target_fps"


def _make_frame_indices(total_frames, current_fps, target_fps):
    target_frame_count = _sample_count_for_fps(total_frames, current_fps, target_fps)
    return _make_frame_indices_by_count(total_frames, target_frame_count)


def _nearest_sample_positions(sampled_indices, total_frames):
    sampled_indices = list(sampled_indices)
    if not sampled_indices:
        return [0 for _ in range(total_frames)]

    positions = []
    sample_pos = 0

    for frame_idx in range(total_frames):
        while (
            sample_pos + 1 < len(sampled_indices)
            and abs(sampled_indices[sample_pos + 1] - frame_idx) <= abs(sampled_indices[sample_pos] - frame_idx)
        ):
            sample_pos += 1
        positions.append(sample_pos)

    return positions


class FrameSubsamplerForDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "current_fps": ("FLOAT", {"default": 30.0, "min": 0.001, "max": 240.0, "step": 0.001}),
                "target_fps": ("FLOAT", {"default": 10.0, "min": 0.001, "max": 240.0, "step": 0.001}),
                "target_megapixels": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.01,
                    "max": 64.00,
                    "step": 0.01,
                    "tooltip": "Maximale Ziel-Megapixel. Es wird nicht hochskaliert."
                }),
                "divisible_by": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Zielbreite und Zielhöhe bleiben durch diesen Wert teilbar, wenn das mit exaktem Seitenverhältnis möglich ist."
                }),
                "frame_limit_mode": (["target_fps", "max_frames"], {
                    "default": "target_fps",
                    "tooltip": "target_fps nutzt den FPS-Wert. max_frames nutzt eine maximale Framezahl und berechnet die Sample-FPS daraus."
                }),
                "max_frames": ("INT", {
                    "default": 150,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Nur im max_frames-Modus: nicht mehr als diese Anzahl Frames behalten."
                }),
                "allow_fractional_fps": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wenn aus, wird die Sample-FPS auf ganze FPS beschraenkt."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", FRAME_DEPTH_SUBSAMPLE_INFO_TYPE, "STRING")
    RETURN_NAMES = ("sampled_images", "valid_indices", "subsample_info", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Video"
    DESCRIPTION = "Skaliert ein Video fuer Depth/Removal VRAM-sparend herunter und reduziert optional die FPS ohne Cropping."

    def process(
        self,
        images,
        current_fps,
        target_fps,
        target_megapixels,
        divisible_by,
        frame_limit_mode,
        max_frames,
        allow_fractional_fps,
    ):
        total_frames, original_h, original_w = images.shape[:3]
        (
            content_w,
            content_h,
            canvas_w,
            canvas_h,
            pad_left,
            pad_top,
            did_resize,
            did_pad,
        ) = _aspect_preserving_canvas_size(
            original_w,
            original_h,
            target_megapixels,
            divisible_by,
        )
        valid_indices, sampled_fps, resolved_frame_mode = _resolve_temporal_plan(
            total_frames,
            current_fps,
            target_fps,
            frame_limit_mode,
            max_frames,
            allow_fractional_fps,
        )

        sampled_images = images[valid_indices]
        if did_resize:
            sampled_images = _resize_image_batch_cv2(
                sampled_images,
                content_w,
                content_h,
                cv2.INTER_AREA,
                clamp_output=True,
            )
        if did_pad:
            sampled_images = _pad_image_batch(sampled_images, canvas_w, canvas_h, pad_left, pad_top)

        info = {
            "version": 1,
            "node": "FrameSubsamplerForDepth",
            "original_width": int(original_w),
            "original_height": int(original_h),
            "content_width": int(content_w),
            "content_height": int(content_h),
            "sampled_width": int(canvas_w),
            "sampled_height": int(canvas_h),
            "pad_left": int(pad_left),
            "pad_top": int(pad_top),
            "divisible_by": int(divisible_by),
            "target_megapixels": float(target_megapixels),
            "original_frame_count": int(total_frames),
            "sampled_frame_count": int(sampled_images.shape[0]),
            "current_fps": float(current_fps),
            "target_fps": float(target_fps),
            "sampled_fps": sampled_fps,
            "frame_limit_mode": resolved_frame_mode,
            "max_frames": int(max_frames),
            "allow_fractional_fps": bool(allow_fractional_fps),
            "valid_indices": [int(i) for i in valid_indices],
        }

        indices_str = ",".join(map(str, valid_indices))
        log_output = (
            "Frame Subsampler For Depth\n"
            f"Frames: {total_frames} -> {sampled_images.shape[0]} ({current_fps:g} fps -> {sampled_fps:g} fps)\n"
            f"Frame mode: {resolved_frame_mode}\n"
            f"Content: {original_w}x{original_h} -> {content_w}x{content_h}\n"
            f"Canvas: {canvas_w}x{canvas_h}, divisible by {int(divisible_by)}"
        )

        return (sampled_images, indices_str, info, log_output)


class FrameSubsamplerForDepthRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "subsample_info": (FRAME_DEPTH_SUBSAMPLE_INFO_TYPE,),
                "restore_original_fps": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wenn aktiv, wird die originale Frameanzahl durch Duplizieren der naechsten Samples wiederhergestellt."
                }),
                "clamp_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Begrenzt Lanczos-Output auf 0..1 fuer ComfyUI IMAGE."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("restored_images", "output_fps", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Video"
    DESCRIPTION = "Skaliert Depth/Removal Frames per Lanczos exakt auf die Originalgroesse zurueck und gleicht optional die Framezahl an."

    def process(self, images, subsample_info, restore_original_fps, clamp_output):
        info = dict(subsample_info or {})
        original_w = int(info.get("original_width", images.shape[2]))
        original_h = int(info.get("original_height", images.shape[1]))
        original_frames = max(1, int(info.get("original_frame_count", images.shape[0])))
        content_w = int(info.get("content_width", info.get("sampled_width", images.shape[2])))
        content_h = int(info.get("content_height", info.get("sampled_height", images.shape[1])))
        pad_left = int(info.get("pad_left", 0))
        pad_top = int(info.get("pad_top", 0))
        valid_indices = [int(i) for i in info.get("valid_indices", [])]
        sampled_frame_count = int(info.get("sampled_frame_count", len(valid_indices) or images.shape[0]))
        original_fps = _safe_float(info.get("current_fps"), 0.0)
        if original_fps <= 0.0:
            original_fps = _safe_float(info.get("original_fps"), 0.0)
        if original_fps <= 0.0:
            original_fps = _safe_float(info.get("output_fps"), 0.0)
        if original_fps <= 0.0:
            original_fps = _safe_float(info.get("target_fps"), 30.0)

        images = _crop_image_batch(images, content_w, content_h, pad_left, pad_top)
        input_frame_count = int(images.shape[0])
        original_duration = _duration_seconds(original_frames, original_fps)
        inferred_input_fps = (
            input_frame_count / original_duration
            if original_duration > 0.0
            else _safe_float(info.get("sampled_fps", info.get("target_fps")), original_fps)
        )

        restored = _resize_image_batch_cv2(
            images,
            original_w,
            original_h,
            cv2.INTER_LANCZOS4,
            clamp_output=bool(clamp_output),
        )

        output_fps = inferred_input_fps
        temporal_mode = "kept incoming frame count"

        if restore_original_fps:
            if valid_indices and input_frame_count == sampled_frame_count:
                positions = _nearest_sample_positions(valid_indices, original_frames)
                temporal_mode = "restored from original sample indices"
            else:
                positions = _resample_positions(input_frame_count, original_frames)
                if input_frame_count < original_frames:
                    temporal_mode = "filled remaining frames after interpolation"
                elif input_frame_count > original_frames:
                    temporal_mode = "downsampled extra interpolated frames"
                else:
                    temporal_mode = "already at original frame count"

            max_pos = max(0, restored.shape[0] - 1)
            positions = [min(max(pos, 0), max_pos) for pos in positions]
            restored = restored[positions]
            output_fps = original_fps

        log_output = (
            "Frame Subsampler For Depth Restore\n"
            f"Content crop: {images.shape[2]}x{images.shape[1]} -> {original_w}x{original_h} (Lanczos)\n"
            f"Frames: {images.shape[0]} -> {restored.shape[0]}\n"
            f"Inferred input FPS: {inferred_input_fps:g}\n"
            f"Output FPS: {output_fps:g}\n"
            f"Temporal mode: {temporal_mode}"
        )

        return (restored, output_fps, log_output)


class FrameSubsamplerForDepthBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "current_fps": ("FLOAT", {"default": 30.0, "min": 0.001, "max": 240.0, "step": 0.001}),
                "target_fps": ("FLOAT", {"default": 10.0, "min": 0.001, "max": 240.0, "step": 0.001}),
                "target_megapixels": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.01,
                    "max": 64.00,
                    "step": 0.01,
                    "tooltip": "Maximale Ziel-Megapixel. Es wird nicht hochskaliert."
                }),
                "divisible_by": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Arbeits-Canvas bleibt durch diesen Wert teilbar."
                }),
                "frame_limit_mode": (["target_fps", "max_frames"], {
                    "default": "target_fps",
                    "tooltip": "target_fps nutzt den FPS-Wert. max_frames nutzt eine maximale Framezahl und berechnet die Sample-FPS daraus."
                }),
                "max_frames": ("INT", {
                    "default": 150,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Nur im max_frames-Modus: nicht mehr als diese Anzahl Frames behalten."
                }),
                "allow_fractional_fps": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wenn aus, wird die Sample-FPS auf ganze FPS beschraenkt."
                }),
                "enable_batching": ("BOOLEAN", {"default": True}),
                "batch_size": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Neue Frames pro Batch. Overlap-Frames kommen bei Folge-Batches zusaetzlich davor."
                }),
                "overlap_frames": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "So viele rohe Model-Output-Frames werden als Kontext vor den naechsten Batch gesetzt."
                }),
            },
            "optional": {
                "pre_batching": (FRAME_DEPTH_BATCH_FEEDBACK_TYPE,),
            },
        }

    RETURN_TYPES = (FRAME_DEPTH_BATCH_FLOW_TYPE, "IMAGE", "STRING", FRAME_DEPTH_BATCH_INFO_TYPE, "STRING")
    RETURN_NAMES = ("flow_control", "batch_images", "valid_indices", "batch_info", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Video"
    DESCRIPTION = "Batch-faehiger Frame Subsampler fuer Depth/Removal. Nutzt rohe Model-Outputs als Overlap-Kontext fuer Folge-Batches."

    def process(
        self,
        images,
        current_fps,
        target_fps,
        target_megapixels,
        divisible_by,
        frame_limit_mode,
        max_frames,
        allow_fractional_fps,
        enable_batching,
        batch_size,
        overlap_frames,
        pre_batching=None,
    ):
        if isinstance(pre_batching, dict) and pre_batching.get("source_images") is not None:
            source_images = pre_batching["source_images"]
            base_info = dict(pre_batching["base_info"])
            accumulated_images = pre_batching.get("accumulated_images")
            previous_tail = pre_batching.get("previous_tail")
            next_index = int(pre_batching.get("next_index", 0))
            iteration = int(pre_batching.get("iteration", 0))
            resolved_frame_mode = base_info.get("frame_limit_mode", "feedback")
        else:
            source_images, base_info, resolved_frame_mode = _prepare_depth_source_images(
                images,
                current_fps,
                target_fps,
                target_megapixels,
                divisible_by,
                frame_limit_mode,
                max_frames,
                allow_fractional_fps,
            )
            accumulated_images = None
            previous_tail = None
            next_index = 0
            iteration = 0

        total_source_frames = int(source_images.shape[0])
        batch_size = max(1, int(batch_size))
        overlap_frames = max(0, int(overlap_frames))
        use_batching = bool(enable_batching) and total_source_frames > batch_size

        unique_start = max(0, min(next_index, total_source_frames))
        if use_batching:
            unique_end = min(total_source_frames, unique_start + batch_size)
        else:
            unique_end = total_source_frames

        new_source_frames = _slice_image_batch(source_images, unique_start, unique_end)
        context_frames = 0
        context_images = None

        if use_batching and unique_start > 0 and previous_tail is not None and overlap_frames > 0:
            context_images = _take_tail(previous_tail, overlap_frames)
            context_frames = int(context_images.shape[0]) if context_images is not None else 0

        batch_images = _concat_image_batches([context_images, new_source_frames])
        if batch_images is None:
            batch_images = _slice_image_batch(source_images, max(0, total_source_frames - 1), total_source_frames)

        batch_info = dict(base_info)
        batch_info.update({
            "batching_enabled": bool(use_batching),
            "batch_size": int(batch_size),
            "overlap_frames": int(overlap_frames),
            "iteration": int(iteration),
            "unique_start": int(unique_start),
            "unique_end": int(unique_end),
            "context_frames": int(context_frames),
            "total_source_frames": int(total_source_frames),
            "next_index": int(unique_end),
            "is_last_batch": bool(unique_end >= total_source_frames),
            "accumulated_images": accumulated_images,
            "source_images": source_images,
        })

        indices_str = ",".join(map(str, base_info.get("valid_indices", [])))
        log_output = (
            "Frame Subsampler For Depth Batch\n"
            f"Frames: {base_info.get('original_frame_count')} -> {total_source_frames} ({base_info.get('current_fps'):g} fps -> {base_info.get('sampled_fps'):g} fps)\n"
            f"Frame mode: {resolved_frame_mode}\n"
            f"Batch: {iteration + 1}, unique {unique_start}:{unique_end}, context {context_frames}, output {int(batch_images.shape[0])}\n"
            f"Batching active: {use_batching}"
        )

        return ("stub", batch_images, indices_str, batch_info, log_output)


class FrameSubsamplerForDepthBatchRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_control": (FRAME_DEPTH_BATCH_FLOW_TYPE, {"rawLink": True}),
                "images": ("IMAGE",),
                "batch_info": (FRAME_DEPTH_BATCH_INFO_TYPE,),
                "restore_original_fps": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wenn aktiv, wird die finale Frameanzahl auf die Originalframezahl gebracht."
                }),
                "clamp_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Begrenzt Lanczos-Output auf 0..1 fuer ComfyUI IMAGE."
                }),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", FRAME_DEPTH_BATCH_FEEDBACK_TYPE, "FLOAT", "STRING")
    RETURN_NAMES = ("restored_images", "pre_batching", "output_fps", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Video"
    DESCRIPTION = "Batch-Ende fuer Depth/Removal. Sammelt rohe Model-Outputs, loopt bei Bedarf weiter und restored am Ende."

    def _explore_dependencies(self, node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        if not node_info or "inputs" not in node_info:
            return
        for value in node_info["inputs"].values():
            link = _raw_link_value(value)
            if link is None:
                continue
            parent_id = link[0]
            if parent_id not in upstream:
                upstream[parent_id] = []
                self._explore_dependencies(parent_id, dynprompt, upstream)
            upstream[parent_id].append(node_id)

    def _collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self._collect_contained(child_id, upstream, contained)

    def _expand_next_batch(self, flow_control, feedback, dynprompt, unique_id):
        from comfy_execution.graph_utils import GraphBuilder

        open_link = _raw_link_value(flow_control)
        if open_link is None:
            raise ValueError("flow_control must be connected from FrameSubsamplerForDepthBatch.flow_control")

        open_node_id = open_link[0]
        upstream = {}
        self._explore_dependencies(unique_id, dynprompt, upstream)

        contained = {}
        self._collect_contained(open_node_id, upstream, contained)
        contained[unique_id] = True
        contained[open_node_id] = True

        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            clone_id = "Recurse" if node_id == unique_id else node_id
            node = graph.node(original_node["class_type"], clone_id)
            node.set_override_display_id(node_id)

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            clone_id = "Recurse" if node_id == unique_id else node_id
            node = graph.lookup_node(clone_id)
            for key, value in original_node.get("inputs", {}).items():
                link = _raw_link_value(value)
                if link is not None and link[0] in contained:
                    parent = graph.lookup_node(link[0])
                    node.set_input(key, parent.out(link[1]))
                else:
                    node.set_input(key, value)

        new_open = graph.lookup_node(open_node_id)
        if new_open is None:
            raise ValueError("Could not find cloned batch start node")
        new_open.set_input("pre_batching", feedback)

        my_clone = graph.lookup_node("Recurse")
        if my_clone is None:
            raise ValueError("Could not find cloned batch restore node")

        return {
            "result": (my_clone.out(0), my_clone.out(1), my_clone.out(2), my_clone.out(3)),
            "expand": graph.finalize(),
        }

    def process(self, flow_control, images, batch_info, restore_original_fps, clamp_output, dynprompt=None, unique_id=None):
        info = dict(batch_info or {})
        context_frames = max(0, int(info.get("context_frames", 0)))
        unique_end = int(info.get("unique_end", images.shape[0]))
        total_source_frames = int(info.get("total_source_frames", info.get("sampled_frame_count", images.shape[0])))
        iteration = int(info.get("iteration", 0))
        overlap_frames = max(0, int(info.get("overlap_frames", 0)))

        raw_model_output = images
        new_processed = images[context_frames:] if context_frames > 0 else images
        accumulated_images = _concat_image_batches([info.get("accumulated_images"), new_processed])
        previous_tail = _take_tail(raw_model_output, overlap_frames)

        feedback = {
            "source_images": info.get("source_images"),
            "base_info": {k: v for k, v in info.items() if k not in {
                "accumulated_images",
                "source_images",
            }},
            "accumulated_images": accumulated_images,
            "previous_tail": previous_tail,
            "next_index": unique_end,
            "iteration": iteration + 1,
        }

        if unique_end < total_source_frames:
            if dynprompt is None or unique_id is None:
                raise ValueError("Batch loop needs ComfyUI dynprompt/unique_id hidden inputs.")
            return self._expand_next_batch(flow_control, feedback, dynprompt, unique_id)

        final_images = accumulated_images if accumulated_images is not None else images
        original_w = int(info.get("original_width", final_images.shape[2]))
        original_h = int(info.get("original_height", final_images.shape[1]))
        original_frames = max(1, int(info.get("original_frame_count", final_images.shape[0])))
        content_w = int(info.get("content_width", info.get("sampled_width", final_images.shape[2])))
        content_h = int(info.get("content_height", info.get("sampled_height", final_images.shape[1])))
        pad_left = int(info.get("pad_left", 0))
        pad_top = int(info.get("pad_top", 0))
        valid_indices = [int(i) for i in info.get("valid_indices", [])]
        sampled_frame_count = int(info.get("sampled_frame_count", len(valid_indices) or final_images.shape[0]))
        original_fps = _safe_float(info.get("current_fps"), 30.0)
        original_duration = _duration_seconds(original_frames, original_fps)
        inferred_input_fps = (
            int(final_images.shape[0]) / original_duration
            if original_duration > 0.0
            else _safe_float(info.get("sampled_fps", info.get("target_fps")), original_fps)
        )

        final_images = _crop_image_batch(final_images, content_w, content_h, pad_left, pad_top)
        restored = _resize_image_batch_cv2(
            final_images,
            original_w,
            original_h,
            cv2.INTER_LANCZOS4,
            clamp_output=bool(clamp_output),
        )

        output_fps = inferred_input_fps
        temporal_mode = "kept accumulated frame count"

        if restore_original_fps:
            if valid_indices and int(restored.shape[0]) == sampled_frame_count:
                positions = _nearest_sample_positions(valid_indices, original_frames)
                temporal_mode = "restored from original sample indices"
            else:
                positions = _resample_positions(int(restored.shape[0]), original_frames)
                if int(restored.shape[0]) < original_frames:
                    temporal_mode = "filled remaining frames after interpolation"
                elif int(restored.shape[0]) > original_frames:
                    temporal_mode = "downsampled extra interpolated frames"
                else:
                    temporal_mode = "already at original frame count"

            max_pos = max(0, int(restored.shape[0]) - 1)
            positions = [min(max(pos, 0), max_pos) for pos in positions]
            restored = restored[positions]
            output_fps = original_fps

        log_output = (
            "Frame Subsampler For Depth Batch Restore\n"
            f"Batches completed: {iteration + 1}\n"
            f"Accumulated low-res frames: {int(accumulated_images.shape[0]) if accumulated_images is not None else int(images.shape[0])}\n"
            f"Restored frames: {int(restored.shape[0])}\n"
            f"Inferred input FPS: {inferred_input_fps:g}\n"
            f"Output FPS: {output_fps:g}\n"
            f"Temporal mode: {temporal_mode}"
        )

        return (restored, feedback, output_fps, log_output)
