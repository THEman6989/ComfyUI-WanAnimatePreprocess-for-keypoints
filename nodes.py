import os
import copy
import math
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import json
import logging
from typing import Any, Dict, Iterable, List, Tuple, Union, Optional
script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import load_torch_file, ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

from .models.onnx_models import ViTPose, Yolo
from .pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, crop, bbox_from_detector
from .utils import get_face_bboxes, padding_resize, resize_by_area, resize_to_bounds
from .pose_utils.human_visualization import AAPoseMeta, draw_aapose_by_meta_new, draw_aaface_by_meta
from .retarget_pose import get_retarget_pose

COCO_BODY_KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

KEY_FRAME_BODY_INDICES: Tuple[int, ...] = tuple(range(len(COCO_BODY_KEYPOINT_NAMES)))
LEGACY_KEY_FRAME_BODY_INDICES: Tuple[int, ...] = (0, 1, 2, 5, 8, 11, 10, 13)

class OnnxDetectionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": (folder_paths.get_filename_list("detection"), {"tooltip": "These models are loaded from the 'ComfyUI/models/detection' -folder",}),
                "yolo_model": (folder_paths.get_filename_list("detection"), {"tooltip": "These models are loaded from the 'ComfyUI/models/detection' -folder",}),
                "onnx_device": (["CUDAExecutionProvider", "CPUExecutionProvider"], {"default": "CUDAExecutionProvider", "tooltip": "Device to run the ONNX models on"}),
            },
        }

    RETURN_TYPES = ("POSEMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Loads ONNX models for pose and face detection. ViTPose for pose estimation and YOLO for object detection."

    def loadmodel(self, vitpose_model, yolo_model, onnx_device):

        vitpose_model_path = folder_paths.get_full_path_or_raise("detection", vitpose_model)
        yolo_model_path = folder_paths.get_full_path_or_raise("detection", yolo_model)

        vitpose = ViTPose(vitpose_model_path, onnx_device)
        yolo = Yolo(yolo_model_path, onnx_device)

        model = {
            "vitpose": vitpose,
            "yolo": yolo,
        }

        return (model, )

class PoseAndFaceDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Height of the generation"}),
            },
            "optional": {
                "retarget_image": ("IMAGE", {"default": None, "tooltip": "Optional reference image for pose retargeting"}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "STRING", "BBOX", "BBOX,")
    RETURN_NAMES = ("pose_data", "face_images", "key_frame_body_points", "bboxes", "face_bboxes")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Detects human poses and face images from input images. Optionally retargets poses based on a reference image."

    def process(self, model, images, width, height, retarget_image=None):
        detector = model["yolo"]
        pose_model = model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()
        if retarget_image is not None:
            refer_img = resize_by_area(retarget_image[0].numpy() * 255, width * height, divisor=16) / 255.0
            ref_bbox = (detector(
                cv2.resize(refer_img.astype(np.float32), (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])

            if ref_bbox is None or ref_bbox[-1] <= 0 or (ref_bbox[2] - ref_bbox[0]) < 10 or (ref_bbox[3] - ref_bbox[1]) < 10:
                ref_bbox = np.array([0, 0, refer_img.shape[1], refer_img.shape[0]])

            center, scale = bbox_from_detector(ref_bbox, input_resolution, rescale=rescale)
            refer_img = crop(refer_img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (refer_img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            ref_keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            refer_pose_meta = load_pose_metas_from_kp2ds_seq(ref_keypoints, width=retarget_image.shape[2], height=retarget_image.shape[1])[0]

        comfy_pbar = ProgressBar(B*2)
        progress = 0
        bboxes = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting bboxes"):
            bboxes.append(detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        kp2ds = []
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting keypoints"):
            if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])

            bbox_xywh = bbox
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        kp2ds = np.concatenate(kp2ds, 0)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)

        face_images = []
        face_bboxes = []
        for idx, meta in enumerate(pose_metas):
            face_bbox_for_image = get_face_bboxes(meta['keypoints_face'][:, :2], scale=1.3, image_shape=(H, W))
            x1, x2, y1, y2 = face_bbox_for_image
            face_bboxes.append((x1, y1, x2, y2))
            face_image = images_np[idx][y1:y2, x1:x2]
            # Check if face_image is valid before resizing
            if face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
                logging.warning(f"Empty face crop on frame {idx}, creating fallback image.")
                # Create a fallback image (black or use center crop)
                fallback_size = int(min(H, W) * 0.3)
                fallback_x1 = (W - fallback_size) // 2
                fallback_x2 = fallback_x1 + fallback_size
                fallback_y1 = int(H * 0.1)
                fallback_y2 = fallback_y1 + fallback_size
                face_image = images_np[idx][fallback_y1:fallback_y2, fallback_x1:fallback_x2]
                
                # If still empty, create a black image
                if face_image.size == 0:
                    face_image = np.zeros((fallback_size, fallback_size, C), dtype=images_np.dtype)
            face_image = cv2.resize(face_image, (512, 512))
            face_images.append(face_image)

        face_images_np = np.stack(face_images, 0)
        face_images_tensor = torch.from_numpy(face_images_np)

        if retarget_image is not None and refer_pose_meta is not None:
            retarget_pose_metas = get_retarget_pose(pose_metas[0], refer_pose_meta, pose_metas, None, None)
        else:
            retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas]

        bbox = np.array(bboxes[0]).flatten()
        if bbox.shape[0] >= 4:
            bbox_ints = tuple(int(v) for v in bbox[:4])
        else:
            bbox_ints = (0, 0, 0, 0)

        key_frame_num = 4 if B >= 4 else 1
        key_frame_step = len(pose_metas) // key_frame_num
        key_frame_index_list = list(range(0, len(pose_metas), key_frame_step))

        key_points_index = list(KEY_FRAME_BODY_INDICES)

        points_dict_list: List[Dict[str, Any]] = []
        for key_frame_index in key_frame_index_list:
            keypoints_body_list = []
            body_key_points = pose_metas[key_frame_index]['keypoints_body']
            for each_index in key_points_index:
                if each_index >= len(body_key_points):
                    continue
                each_keypoint = body_key_points[each_index]
                if each_keypoint is None:
                    continue
                keypoints_body_list.append((each_index, each_keypoint))

            if not keypoints_body_list:
                continue

            meta_wh = np.array([
                float(pose_metas[key_frame_index].get('width', pose_metas[0]['width'])),
                float(pose_metas[key_frame_index].get('height', pose_metas[0]['height'])),
            ], dtype=np.float32)

            frame_points: List[Dict[str, Any]] = []
            for original_index, keypoint in keypoints_body_list:
                coords = np.asarray(keypoint[:2], dtype=np.float32) * meta_wh
                entry: Dict[str, Any] = {
                    "index": int(original_index),
                    "name": COCO_BODY_KEYPOINT_NAMES[original_index]
                    if original_index < len(COCO_BODY_KEYPOINT_NAMES)
                    else f"keypoint_{original_index}",
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                }
                if len(keypoint) >= 3 and keypoint[2] is not None:
                    try:
                        score_val = float(keypoint[2])
                    except (TypeError, ValueError):
                        score_val = None
                    if score_val is not None:
                        clipped_score = float(np.clip(score_val, 0.0, 1.0))
                        entry["score"] = clipped_score
                        entry["confidence"] = clipped_score
                frame_points.append(entry)

            if frame_points:
                points_dict_list = frame_points

        pose_data = {
            "retarget_image": refer_img if retarget_image is not None else None,
            "pose_metas": retarget_pose_metas,
            "refer_pose_meta": refer_pose_meta if retarget_image is not None else None,
            "pose_metas_original": pose_metas,
        }

        return (pose_data, face_images_tensor, json.dumps(points_dict_list), [bbox_ints], face_bboxes)

class DrawViTPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Height of the generation"}),
                "retarget_padding": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1, "tooltip": "When > 0, the retargeted pose image is padded and resized to the target size"}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks. Set to 0 to disable body drawing, -1 for auto"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the hand sticks. Set to 0 to disable hand drawing, -1 for auto"}),
                "draw_head": ("BOOLEAN", {"default": "True", "tooltip": "Whether to draw head keypoints"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("pose_images", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Draws pose images from pose data."

    def process(self, pose_data, width, height, body_stick_width, hand_stick_width, draw_head, retarget_padding=64):

        retarget_image = pose_data.get("retarget_image", None)
        pose_metas = pose_data["pose_metas"]

        draw_hand = hand_stick_width != 0
        use_retarget_resize = retarget_padding > 0 and retarget_image is not None

        comfy_pbar = ProgressBar(len(pose_metas))
        progress = 0
        crop_target_image = None
        pose_images = []

        for idx, meta in enumerate(tqdm(pose_metas, desc="Drawing pose images")):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            pose_image = draw_aapose_by_meta_new(canvas, meta, draw_hand=draw_hand, draw_head=draw_head, body_stick_width=body_stick_width, hand_stick_width=hand_stick_width)

            if crop_target_image is None:
                crop_target_image = pose_image

            if use_retarget_resize:
                pose_image = resize_to_bounds(pose_image, height, width, crop_target_image=crop_target_image, extra_padding=retarget_padding)
            else:
                pose_image = padding_resize(pose_image, height, width)

            pose_images.append(pose_image)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_images_np = np.stack(pose_images, 0)
        pose_images_tensor = torch.from_numpy(pose_images_np).float() / 255.0

        return (pose_images_tensor, )

class PoseRetargetPromptHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", )
    RETURN_NAMES = ("prompt", "retarget_prompt", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Generates text prompts for pose retargeting based on visibility of arms and legs in the template pose. Originally used for Flux Kontext"

    def process(self, pose_data):
        refer_pose_meta = pose_data.get("refer_pose_meta", None)
        if refer_pose_meta is None:
            return ("Change the person to face forward.", "Change the person to face forward.", )
        tpl_pose_metas = pose_data["pose_metas_original"]
        arm_visible = False
        leg_visible = False

        for tpl_pose_meta in tpl_pose_metas:
            tpl_keypoints = tpl_pose_meta['keypoints_body']
            tpl_keypoints = np.array(tpl_keypoints)
            if np.any(tpl_keypoints[3]) != 0 or np.any(tpl_keypoints[4]) != 0 or np.any(tpl_keypoints[6]) != 0 or np.any(tpl_keypoints[7]) != 0:
                if (tpl_keypoints[3][0] <= 1 and tpl_keypoints[3][1] <= 1 and tpl_keypoints[3][2] >= 0.75) or (tpl_keypoints[4][0] <= 1 and tpl_keypoints[4][1] <= 1 and tpl_keypoints[4][2] >= 0.75) or \
                    (tpl_keypoints[6][0] <= 1 and tpl_keypoints[6][1] <= 1 and tpl_keypoints[6][2] >= 0.75) or (tpl_keypoints[7][0] <= 1 and tpl_keypoints[7][1] <= 1 and tpl_keypoints[7][2] >= 0.75):
                    arm_visible = True
            if np.any(tpl_keypoints[9]) != 0 or np.any(tpl_keypoints[12]) != 0 or np.any(tpl_keypoints[10]) != 0 or np.any(tpl_keypoints[13]) != 0:
                if (tpl_keypoints[9][0] <= 1 and tpl_keypoints[9][1] <= 1 and tpl_keypoints[9][2] >= 0.75) or (tpl_keypoints[12][0] <= 1 and tpl_keypoints[12][1] <= 1 and tpl_keypoints[12][2] >= 0.75) or \
                    (tpl_keypoints[10][0] <= 1 and tpl_keypoints[10][1] <= 1 and tpl_keypoints[10][2] >= 0.75) or (tpl_keypoints[13][0] <= 1 and tpl_keypoints[13][1] <= 1 and tpl_keypoints[13][2] >= 0.75):
                    leg_visible = True
            if arm_visible and leg_visible:
                break

        if leg_visible:
            if tpl_pose_meta['width'] > tpl_pose_meta['height']:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."

            if refer_pose_meta['width'] > refer_pose_meta['height']:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."
        elif arm_visible:
            if tpl_pose_meta['width'] > tpl_pose_meta['height']:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."

            if refer_pose_meta['width'] > refer_pose_meta['height']:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."
        else:
            tpl_prompt = "Change the person to face forward."
            refer_prompt = "Change the person to face forward."

        return (tpl_prompt, refer_prompt, )


class PoseDataManipulator:
    PART_DEFINITIONS: Dict[str, Dict[str, Union[List[int], str]]] = {
        "feet": {"body": [15, 16]},
        "Only Legs": {"body": [13, 14, 15, 16], "special": "only_legs"},
        "full_legs": {"body": [11, 12, 13, 14, 15, 16]},
        "both_legs": {"body": [11, 12, 13, 14, 15, 16]},
        "torso": {"body": [5, 6, 11, 12]},
        "shoulders": {"body": [5, 6]},
        "hands": {"body": [9, 10], "lhand": "all", "rhand": "all"},
        "arms": {"body": [5, 6, 7, 8, 9, 10]},
        "body": {"body": "all"},
        "head": {"body": [0, 1, 2, 3, 4], "face": "all"},
        "face": {"face": "all"},
        "full_body": {"body": "all", "lhand": "all", "rhand": "all", "face": "all"},
    }
    PART_ALIASES: Dict[str, str] = {
        "legs": "Only Legs",
        "only_legs": "Only Legs",
        "Only_Legs": "Only Legs",
    }

    @classmethod
    def _part_choices(cls) -> List[str]:
        return list(cls.PART_DEFINITIONS.keys())

    def _resolve_part(self, part: str) -> Tuple[str, Union[List[int], str, Dict[str, Any]]]:
        canonical = self.PART_ALIASES.get(part, part)
        spec = self.PART_DEFINITIONS.get(canonical)
        if spec is None and canonical != part:
            spec = self.PART_DEFINITIONS.get(part)
            if spec is not None:
                canonical = part
        if spec is None:
            spec = []
        return canonical, spec

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "part": (cls._part_choices(), {"default": "feet"}),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                                          "tooltip": "Horizontal offset applied to the selected keypoints (normalized)."}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                                          "tooltip": "Vertical offset applied to the selected keypoints (normalized)."}),
                "scale": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                                       "tooltip": "Scale factor applied around the part centroid."}),
                "leg_scale_bidirectional": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When manipulating Only Legs, stretch equally above and below the leg center instead of downwards only.",
                }),
                "confidence_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01,
                                                  "tooltip": "Multiplier for confidence scores of the manipulated keypoints."}),
                "apply_to_retarget": ("BOOLEAN", {"default": True,
                                                     "tooltip": "Apply manipulation to retargeted pose metas."}),
                "apply_to_original": ("BOOLEAN", {"default": True,
                                                     "tooltip": "Apply manipulation to original pose metas."}),
                "apply_to_reference": ("BOOLEAN", {"default": False,
                                                      "tooltip": "Apply manipulation to the reference pose meta if present."}),
                "clamp_to_unit": ("BOOLEAN", {"default": True,
                                                 "tooltip": "Clamp manipulated coordinates to the [0,1] range before rescaling."}),
            },
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "manipulate"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Manipulates sections of pose data such as legs, torso, or arms by applying offsets and scaling."

    def _collect_indices(self, part: str) -> Tuple[str, Dict[str, Union[List[int], str]]]:
        canonical, spec = self._resolve_part(part)
        mapping: Dict[str, Union[List[int], str]] = spec if isinstance(spec, dict) else {}
        return canonical, mapping

    def _valid_indices(self, count: int, spec: Union[List[int], str]) -> List[int]:
        if isinstance(spec, str):
            if spec == "all":
                return list(range(count))
            return []
        return [idx for idx in spec if isinstance(idx, int) and 0 <= idx < count]

    def _clone_meta(self, meta: Any) -> Any:
        if isinstance(meta, AAPoseMeta):
            cloned = AAPoseMeta()
            cloned.image_id = getattr(meta, "image_id", "")
            cloned.width = getattr(meta, "width", 0)
            cloned.height = getattr(meta, "height", 0)
            cloned.kps_body = None if meta.kps_body is None else np.array(meta.kps_body, copy=True)
            cloned.kps_body_p = None if meta.kps_body_p is None else np.array(meta.kps_body_p, copy=True)
            cloned.kps_lhand = None if meta.kps_lhand is None else np.array(meta.kps_lhand, copy=True)
            cloned.kps_lhand_p = None if meta.kps_lhand_p is None else np.array(meta.kps_lhand_p, copy=True)
            cloned.kps_rhand = None if meta.kps_rhand is None else np.array(meta.kps_rhand, copy=True)
            cloned.kps_rhand_p = None if meta.kps_rhand_p is None else np.array(meta.kps_rhand_p, copy=True)
            cloned.kps_face = None if meta.kps_face is None else np.array(meta.kps_face, copy=True)
            cloned.kps_face_p = None if meta.kps_face_p is None else np.array(meta.kps_face_p, copy=True)
            return cloned
        if isinstance(meta, dict):
            cloned_dict: Dict[str, Any] = {}
            for key, value in meta.items():
                if isinstance(value, np.ndarray):
                    cloned_dict[key] = value.copy()
                else:
                    cloned_dict[key] = copy.deepcopy(value)
            return cloned_dict
        return copy.deepcopy(meta)

    def _ensure_array(self, data: Any) -> Optional[np.ndarray]:
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data
        try:
            return np.asarray(data, dtype=np.float32)
        except (TypeError, ValueError):
            return np.asarray(data, dtype=object)

    def _transform_block(self, coords: np.ndarray, indices: List[int], width: float, height: float,
                         offset_x: float, offset_y: float, scale: float, clamp: bool) -> List[int]:
        if coords is None or not indices:
            return []

        view = np.asarray(coords)
        if view.ndim != 2 or view.shape[1] < 2:
            return []

        width = max(float(width), 1.0)
        height = max(float(height), 1.0)

        valid_indices: List[int] = []
        selected_coords: List[List[float]] = []
        for idx in indices:
            if not (0 <= idx < view.shape[0]):
                continue
            try:
                x_val = float(view[idx, 0])
                y_val = float(view[idx, 1])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x_val) and math.isfinite(y_val)):
                continue
            valid_indices.append(idx)
            selected_coords.append([x_val, y_val])

        if not selected_coords:
            return []

        selected = np.asarray(selected_coords, dtype=np.float32)
        norm = selected / np.array([width, height], dtype=np.float32)
        center = norm.mean(axis=0, keepdims=True)
        norm = (norm - center) * scale + center
        norm[:, 0] += offset_x
        norm[:, 1] += offset_y
        if clamp:
            norm = np.clip(norm, 0.0, 1.0)

        for list_idx, (nx, ny) in zip(valid_indices, norm):
            coords[list_idx, 0] = float(nx * width)
            coords[list_idx, 1] = float(ny * height)

        return valid_indices

    def _scale_confidences(self, conf: Any, indices: List[int], factor: float) -> None:
        if conf is None or not indices:
            return
        if factor == 1.0:
            return
        for idx in indices:
            if not (isinstance(idx, int) and idx >= 0):
                continue
            try:
                current = float(conf[idx])
            except (TypeError, ValueError, IndexError):
                continue
            if not math.isfinite(current):
                continue
            new_val = float(np.clip(current * factor, 0.0, 1.0))
            try:
                conf[idx] = new_val
            except TypeError:
                if isinstance(conf, np.ndarray):
                    conf[idx] = new_val

    def _stretch_only_legs_aapose(self, meta: AAPoseMeta, offset_x: float, offset_y: float, scale: float,
                                  clamp: bool, bidirectional: bool) -> List[int]:
        coords = self._ensure_array(meta.kps_body)
        if coords is None or coords.ndim != 2 or coords.shape[1] < 2:
            return []

        width = max(float(getattr(meta, "width", 1.0)), 1.0)
        height = max(float(getattr(meta, "height", 1.0)), 1.0)
        scale_val = float(scale)
        if not math.isfinite(scale_val):
            scale_val = 1.0

        offset_px = float(offset_x) * width
        offset_py = float(offset_y) * height

        modified: List[int] = []

        def valid_point(index: int) -> Optional[Tuple[float, float]]:
            if not (0 <= index < coords.shape[0]):
                return None
            try:
                x_val = float(coords[index, 0])
                y_val = float(coords[index, 1])
            except (TypeError, ValueError):
                return None
            if not (math.isfinite(x_val) and math.isfinite(y_val)):
                return None
            return x_val, y_val

        leg_triplets = ((11, 13, 15), (12, 14, 16))

        for hip_idx, knee_idx, ankle_idx in leg_triplets:
            hip_point = valid_point(hip_idx)
            if hip_point is None:
                continue
            hip_y = hip_point[1]

            knee_point = valid_point(knee_idx)
            ankle_point = valid_point(ankle_idx)

            if not bidirectional and knee_point is None and ankle_point is None:
                continue

            target_indices: List[int] = []

            if bidirectional:
                if ankle_point is None:
                    continue
                center_y = (hip_y + ankle_point[1]) * 0.5
                for idx in (hip_idx, knee_idx, ankle_idx):
                    point = valid_point(idx)
                    if point is None:
                        continue
                    new_y = center_y + (point[1] - center_y) * scale_val
                    coords[idx, 1] = float(new_y)
                    target_indices.append(idx)
            else:
                for idx, point in ((knee_idx, knee_point), (ankle_idx, ankle_point)):
                    if point is None:
                        continue
                    new_y = hip_y + (point[1] - hip_y) * scale_val
                    coords[idx, 1] = float(new_y)
                    target_indices.append(idx)

            modified.extend(target_indices)

        if not modified and offset_px == 0.0 and offset_py == 0.0:
            return []

        unique_indices = sorted(set(modified))

        if offset_px != 0.0 or offset_py != 0.0:
            for idx in unique_indices:
                try:
                    coords[idx, 0] = float(coords[idx, 0]) + offset_px
                    coords[idx, 1] = float(coords[idx, 1]) + offset_py
                except (TypeError, ValueError):
                    continue

        if clamp and unique_indices:
            for idx in unique_indices:
                try:
                    nx = float(coords[idx, 0]) / width
                    ny = float(coords[idx, 1]) / height
                except (TypeError, ValueError):
                    continue
                coords[idx, 0] = float(np.clip(nx, 0.0, 1.0) * width)
                coords[idx, 1] = float(np.clip(ny, 0.0, 1.0) * height)

        meta.kps_body = coords
        return unique_indices

    def _manipulate_aapose(self, meta: AAPoseMeta, part_map: Dict[str, Union[List[int], str]], offset_x: float,
                            offset_y: float, scale: float, confidence_scale: float, clamp: bool,
                            bidirectional: bool) -> None:
        if meta is None:
            return
        meta.kps_body = self._ensure_array(meta.kps_body)
        meta.kps_body_p = self._ensure_array(meta.kps_body_p)
        meta.kps_lhand = self._ensure_array(meta.kps_lhand)
        meta.kps_lhand_p = self._ensure_array(meta.kps_lhand_p)
        meta.kps_rhand = self._ensure_array(meta.kps_rhand)
        meta.kps_rhand_p = self._ensure_array(meta.kps_rhand_p)
        meta.kps_face = self._ensure_array(meta.kps_face)
        meta.kps_face_p = self._ensure_array(meta.kps_face_p)

        if part_map.get("special") == "only_legs":
            modified = self._stretch_only_legs_aapose(meta, offset_x, offset_y, scale, clamp, bidirectional)
            self._scale_confidences(meta.kps_body_p, modified, confidence_scale)
            return

        body_indices = self._valid_indices(
            meta.kps_body.shape[0] if isinstance(meta.kps_body, np.ndarray) else 0,
            part_map.get("body", []),
        )
        modified = self._transform_block(meta.kps_body, body_indices, meta.width, meta.height, offset_x, offset_y, scale, clamp)
        self._scale_confidences(meta.kps_body_p, modified, confidence_scale)

        lhand_spec = part_map.get("lhand")
        if isinstance(meta.kps_lhand, np.ndarray) and lhand_spec is not None:
            l_indices = self._valid_indices(meta.kps_lhand.shape[0], lhand_spec)
            modified = self._transform_block(meta.kps_lhand, l_indices, meta.width, meta.height, offset_x, offset_y, scale, clamp)
            self._scale_confidences(meta.kps_lhand_p, modified, confidence_scale)

        rhand_spec = part_map.get("rhand")
        if isinstance(meta.kps_rhand, np.ndarray) and rhand_spec is not None:
            r_indices = self._valid_indices(meta.kps_rhand.shape[0], rhand_spec)
            modified = self._transform_block(meta.kps_rhand, r_indices, meta.width, meta.height, offset_x, offset_y, scale, clamp)
            self._scale_confidences(meta.kps_rhand_p, modified, confidence_scale)

        face_spec = part_map.get("face")
        if isinstance(meta.kps_face, np.ndarray) and face_spec is not None:
            f_indices = self._valid_indices(meta.kps_face.shape[0], face_spec)
            modified = self._transform_block(meta.kps_face, f_indices, meta.width, meta.height, offset_x, offset_y, scale, clamp)
            self._scale_confidences(meta.kps_face_p, modified, confidence_scale)

    def _stretch_only_legs_dict(self, meta: Dict[str, Any], offset_x: float, offset_y: float, scale: float,
                                confidence_scale: float, clamp: bool, bidirectional: bool) -> None:
        arr = meta.get("keypoints_body")
        if arr is None:
            return

        arr_np = np.asarray(arr)
        if arr_np.ndim != 2 or arr_np.shape[1] < 2:
            return

        try:
            arr_np = arr_np.astype(np.float32, copy=False)
        except (TypeError, ValueError):
            arr_np = arr_np.astype(np.float32)

        width = max(float(meta.get("width", 1.0)), 1.0)
        height = max(float(meta.get("height", 1.0)), 1.0)
        scale_val = float(scale)
        if not math.isfinite(scale_val):
            scale_val = 1.0

        offset_px = float(offset_x) * width
        offset_py = float(offset_y) * height

        modified: List[int] = []

        def valid_point(index: int) -> Optional[Tuple[float, float]]:
            if not (0 <= index < arr_np.shape[0]):
                return None
            try:
                x_val = float(arr_np[index, 0])
                y_val = float(arr_np[index, 1])
            except (TypeError, ValueError):
                return None
            if not (math.isfinite(x_val) and math.isfinite(y_val)):
                return None
            return x_val, y_val

        leg_triplets = ((11, 13, 15), (12, 14, 16))

        for hip_idx, knee_idx, ankle_idx in leg_triplets:
            hip_point = valid_point(hip_idx)
            if hip_point is None:
                continue
            _, hip_y = hip_point
            knee_point = valid_point(knee_idx)
            ankle_point = valid_point(ankle_idx)

            if not bidirectional and knee_point is None and ankle_point is None:
                continue

            target_indices: List[int] = []

            if bidirectional:
                if ankle_point is None:
                    continue
                center_y = (hip_y + ankle_point[1]) * 0.5
                for idx in (hip_idx, knee_idx, ankle_idx):
                    point = valid_point(idx)
                    if point is None:
                        continue
                    new_y = center_y + (point[1] - center_y) * scale_val
                    arr_np[idx, 1] = float(new_y)
                    target_indices.append(idx)
            else:
                for idx, point in ((knee_idx, knee_point), (ankle_idx, ankle_point)):
                    if point is None:
                        continue
                    new_y = hip_y + (point[1] - hip_y) * scale_val
                    arr_np[idx, 1] = float(new_y)
                    target_indices.append(idx)

            modified.extend(target_indices)

        if not modified and offset_px == 0.0 and offset_py == 0.0:
            return

        unique_indices = sorted(set(modified))

        if offset_px != 0.0 or offset_py != 0.0:
            for idx in unique_indices:
                try:
                    arr_np[idx, 0] = float(arr_np[idx, 0]) + offset_px
                    arr_np[idx, 1] = float(arr_np[idx, 1]) + offset_py
                except (TypeError, ValueError):
                    continue

        if clamp and unique_indices:
            for idx in unique_indices:
                try:
                    nx = float(arr_np[idx, 0]) / width
                    ny = float(arr_np[idx, 1]) / height
                except (TypeError, ValueError):
                    continue
                arr_np[idx, 0] = float(np.clip(nx, 0.0, 1.0) * width)
                arr_np[idx, 1] = float(np.clip(ny, 0.0, 1.0) * height)

        if arr_np.shape[1] >= 3 and confidence_scale != 1.0 and unique_indices:
            for idx in unique_indices:
                try:
                    score_val = float(arr_np[idx, 2])
                except (TypeError, ValueError):
                    continue
                arr_np[idx, 2] = float(np.clip(score_val * confidence_scale, 0.0, 1.0))

        if isinstance(arr, list):
            meta["keypoints_body"] = arr_np.tolist()
        else:
            meta["keypoints_body"] = arr_np

    def _manipulate_dict_meta(self, meta: Dict[str, Any], part_map: Dict[str, Union[List[int], str]], offset_x: float,
                              offset_y: float, scale: float, confidence_scale: float, clamp: bool,
                              bidirectional: bool) -> None:
        if meta is None:
            return
        width = float(meta.get("width", 1.0))
        height = float(meta.get("height", 1.0))

        if part_map.get("special") == "only_legs":
            self._stretch_only_legs_dict(meta, offset_x, offset_y, scale, confidence_scale, clamp, bidirectional)
            return

        def transform_array(key: str, indices_spec: Union[List[int], str]):
            if indices_spec is None:
                return
            arr = meta.get(key)
            if arr is None:
                return
            arr_np = np.asarray(arr)
            if arr_np.ndim != 2 or arr_np.shape[1] < 2:
                return
            indices = self._valid_indices(arr_np.shape[0], indices_spec)
            if not indices:
                return
            modified = self._transform_block(arr_np, indices, width, height, offset_x, offset_y, scale, clamp)
            if not modified:
                return
            if arr_np.shape[1] >= 3:
                for idx in modified:
                    try:
                        score_val = float(arr_np[idx, 2])
                    except (TypeError, ValueError):
                        continue
                    arr_np[idx, 2] = float(np.clip(score_val * confidence_scale, 0.0, 1.0))
            if isinstance(arr, list):
                meta[key] = arr_np.tolist()
            else:
                meta[key] = arr_np

        transform_array("keypoints_body", part_map.get("body"))
        transform_array("keypoints_left_hand", part_map.get("lhand"))
        transform_array("keypoints_right_hand", part_map.get("rhand"))
        transform_array("keypoints_face", part_map.get("face"))

    def _manipulate_meta(self, meta: Any, part_map: Dict[str, Union[List[int], str]], offset_x: float, offset_y: float,
                          scale: float, confidence_scale: float, clamp: bool, bidirectional: bool) -> Any:
        if isinstance(meta, AAPoseMeta):
            self._manipulate_aapose(meta, part_map, offset_x, offset_y, scale, confidence_scale, clamp, bidirectional)
            return meta
        if isinstance(meta, dict):
            self._manipulate_dict_meta(meta, part_map, offset_x, offset_y, scale, confidence_scale, clamp, bidirectional)
            return meta
        return meta

    def manipulate(self, pose_data, part, offset_x, offset_y, scale, leg_scale_bidirectional, confidence_scale,
                   apply_to_retarget=True, apply_to_original=True, apply_to_reference=False, clamp_to_unit=True):
        updated_pose_data = dict(pose_data)
        if "pose_metas" in pose_data:
            updated_pose_data["pose_metas"] = [self._clone_meta(meta) for meta in pose_data.get("pose_metas", [])]
        if "pose_metas_original" in pose_data:
            updated_pose_data["pose_metas_original"] = [
                self._clone_meta(meta) for meta in pose_data.get("pose_metas_original", [])
            ]
        if pose_data.get("refer_pose_meta") is not None:
            updated_pose_data["refer_pose_meta"] = self._clone_meta(pose_data.get("refer_pose_meta"))
        _, part_map = self._collect_indices(part)
        use_bidirectional = bool(leg_scale_bidirectional) if part_map.get("special") == "only_legs" else False

        if apply_to_retarget:
            metas = updated_pose_data.get("pose_metas", [])
            for idx, meta in enumerate(metas):
                metas[idx] = self._manipulate_meta(
                    meta,
                    part_map,
                    offset_x,
                    offset_y,
                    scale,
                    confidence_scale,
                    clamp_to_unit,
                    use_bidirectional,
                )
            updated_pose_data["pose_metas"] = metas

        if apply_to_original:
            metas = updated_pose_data.get("pose_metas_original", [])
            for idx, meta in enumerate(metas):
                metas[idx] = self._manipulate_meta(
                    meta,
                    part_map,
                    offset_x,
                    offset_y,
                    scale,
                    confidence_scale,
                    clamp_to_unit,
                    use_bidirectional,
                )
            updated_pose_data["pose_metas_original"] = metas

        if apply_to_reference and updated_pose_data.get("refer_pose_meta") is not None:
            updated_pose_data["refer_pose_meta"] = self._manipulate_meta(
                updated_pose_data["refer_pose_meta"],
                part_map,
                offset_x,
                offset_y,
                scale,
                confidence_scale,
                clamp_to_unit,
                use_bidirectional,
            )

        return (updated_pose_data,)


class PoseDataToOpenPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "use_retarget_pose": ("BOOLEAN", {"default": False, "tooltip": "When true, convert the retargeted pose instead of the original detection."}),
                "include_face": ("BOOLEAN", {"default": True, "tooltip": "Include face keypoints in the exported data."}),
                "include_hands": ("BOOLEAN", {"default": True, "tooltip": "Include hand keypoints in the exported data."}),
            },
            "optional": {
                "confidence_override": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Set all confidence scores to this value. Use -1 to keep original confidences.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("openpose_json",)
    FUNCTION = "convert"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Converts pose data into the standard OpenPose JSON dictionary format."

    def _meta_to_arrays(self, meta: Union[Dict[str, Any], AAPoseMeta]) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(meta, AAPoseMeta):
            width = int(meta.width)
            height = int(meta.height)

            def _combine(points: Union[np.ndarray, None], scores: Union[np.ndarray, None]) -> np.ndarray:
                if points is None:
                    return np.zeros((0, 3), dtype=np.float32)
                points = np.asarray(points, dtype=np.float32)
                if points.ndim != 2 or points.shape[1] != 2:
                    return np.zeros((0, 3), dtype=np.float32)
                if scores is None:
                    scores_arr = np.ones(points.shape[0], dtype=np.float32)
                else:
                    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
                    if scores_arr.shape[0] != points.shape[0]:
                        scores_arr = np.ones(points.shape[0], dtype=np.float32)
                return np.concatenate([points, scores_arr[:, None]], axis=1)

            body = _combine(meta.kps_body, meta.kps_body_p)
            lhand = _combine(meta.kps_lhand, meta.kps_lhand_p)
            rhand = _combine(meta.kps_rhand, meta.kps_rhand_p)
            face = _combine(meta.kps_face, meta.kps_face_p)
        else:
            width = int(meta["width"])
            height = int(meta["height"])
            body = np.asarray(meta.get("keypoints_body", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
            lhand = np.asarray(meta.get("keypoints_left_hand", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
            rhand = np.asarray(meta.get("keypoints_right_hand", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
            face = np.asarray(meta.get("keypoints_face", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)

        return width, height, body, lhand, rhand, face

    def _to_openpose_list(
        self,
        keypoints: np.ndarray,
        width: int,
        height: int,
        confidence_override: Optional[float] = None,
    ) -> List[float]:
        if keypoints.size == 0:
            return []

        coords = keypoints[:, :2]
        conf = keypoints[:, 2] if keypoints.shape[1] > 2 else np.ones(keypoints.shape[0], dtype=np.float32)

        if confidence_override is not None:
            override_val = float(np.clip(confidence_override, 0.0, 5.0))
            conf = np.full(conf.shape, override_val, dtype=np.float32)

        max_coord = np.max(coords) if coords.size else 0
        if max_coord <= 1.5:
            xs = coords[:, 0] * width
            ys = coords[:, 1] * height
        else:
            xs = coords[:, 0]
            ys = coords[:, 1]

        openpose_kps: List[float] = []
        for x, y, c in zip(xs, ys, conf):
            openpose_kps.extend([float(x), float(y), float(c)])
        return openpose_kps

    def _meta_to_openpose_frame(
        self,
        meta: Any,
        include_face: bool,
        include_hands: bool,
        frame_index: int,
        confidence_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        width, height, body, lhand, rhand, face = self._meta_to_arrays(meta)

        frame_entry: Dict[str, Any] = {
            "version": 1.3,
            "people": [
                {
                    "person_id": [-1],
                    "pose_keypoints_2d": self._to_openpose_list(body, width, height, confidence_override),
                    "pose_keypoints_3d": [],
                    "face_keypoints_2d": self._to_openpose_list(face, width, height, confidence_override)
                    if include_face
                    else [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_2d": self._to_openpose_list(lhand, width, height, confidence_override)
                    if include_hands
                    else [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_2d": self._to_openpose_list(rhand, width, height, confidence_override)
                    if include_hands
                    else [],
                    "hand_right_keypoints_3d": [],
                }
            ],
            "canvas_width": int(width),
            "canvas_height": int(height),
            "frame_index": int(frame_index),
        }

        return frame_entry

    def convert(
        self,
        pose_data,
        use_retarget_pose=False,
        include_face=True,
        include_hands=True,
        confidence_override: Optional[float] = None,
    ):
        metas_source_key = "pose_metas" if use_retarget_pose else "pose_metas_original"
        metas: Iterable[Any] = pose_data.get(metas_source_key, [])
        if not metas:
            metas = pose_data.get("pose_metas_original", [])

        override_value: Optional[float] = None
        if confidence_override is not None and float(confidence_override) >= 0.0:
            override_value = float(np.clip(confidence_override, 0.0, 5.0))

        openpose_frames: List[Dict[str, Any]] = []
        for idx, meta in enumerate(metas):
            frame_entry = self._meta_to_openpose_frame(
                meta,
                include_face,
                include_hands,
                idx,
                override_value,
            )
            openpose_frames.append(frame_entry)

        json_output = json.dumps(openpose_frames, ensure_ascii=False)
        return (json_output,)


class PoseDataToOpenPoseKeypoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "use_retarget_pose": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When true, keypoints are taken from the retargeted pose meta list.",
                    },
                ),
                "frame_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Frame index to extract keypoints from. Clamped to the available range.",
                    },
                ),
                "confidence_override": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Set all confidence scores to this value. Use -1 to keep original confidences.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "POSE_KEYPOINT")
    RETURN_NAMES = ("pose_keypoint", "pose_keypoint_sequence")
    FUNCTION = "convert"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = (
        "Converts stored pose metadata into POSE_KEYPOINT dictionaries compatible with the Ultimate OpenPose Editor, returning "
        "both a selected frame and the entire sequence."
    )

    def convert(self, pose_data, use_retarget_pose=False, frame_index=0, confidence_override=-1.0):
        source_key = "pose_metas" if use_retarget_pose else "pose_metas_original"
        metas: Iterable[Any] = pose_data.get(source_key, [])
        if not metas:
            metas = pose_data.get("pose_metas_original", [])

        converter = PoseDataToOpenPose()
        override_value: Optional[float] = None
        if confidence_override is not None and float(confidence_override) >= 0.0:
            override_value = float(np.clip(confidence_override, 0.0, 5.0))
        frames: List[Dict[str, Any]] = []
        for idx, meta in enumerate(metas):
            frame_entry = converter._meta_to_openpose_frame(
                meta,
                include_face=True,
                include_hands=True,
                frame_index=idx,
                confidence_override=override_value,
            )
            frames.append(frame_entry)

        if not frames:
            return ([], [])

        clamped_index = max(0, min(int(frame_index), len(frames) - 1))
        selected_frame = copy.deepcopy(frames[clamped_index])
        full_sequence = copy.deepcopy(frames)
        return (selected_frame, full_sequence)


class KeyFrameBodyPointsManipulator:
    PART_DEFINITIONS: Dict[str, Union[List[int], str, Dict[str, Any]]] = {
        "feet": [15, 16],
        "Only Legs": {"indices": [13, 14, 15, 16], "special": "only_legs"},
        "full_legs": [11, 12, 13, 14, 15, 16],
        "both_legs": [11, 12, 13, 14, 15, 16],
        "torso": [5, 6, 11, 12],
        "shoulders": [5, 6],
        "hands": [9, 10],
        "arms": [5, 6, 7, 8, 9, 10],
        "body": list(range(len(COCO_BODY_KEYPOINT_NAMES))),
        "head": [0, 1, 2, 3, 4],
        "face": [0, 1, 2, 3, 4],
        "full_body": list(range(len(COCO_BODY_KEYPOINT_NAMES))),
    }
    PART_ALIASES: Dict[str, str] = {
        "legs": "Only Legs",
        "only_legs": "Only Legs",
        "Only_Legs": "Only Legs",
    }

    @classmethod
    def _part_choices(cls) -> List[str]:
        return list(cls.PART_DEFINITIONS.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key_frame_body_points": (
                    "STRING",
                    {"tooltip": "Key frame body points JSON as emitted by Pose and Face Detection."},
                ),
                "part": (cls._part_choices(), {"default": "feet"}),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                                          "tooltip": "Horizontal pixel offset for the selected keypoints."}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                                          "tooltip": "Vertical pixel offset for the selected keypoints."}),
                "scale": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                                       "tooltip": "Scale factor applied around the centroid of the selected points."}),
                "leg_scale_bidirectional": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When manipulating Only Legs, stretch equally above and below the leg center instead of downwards only.",
                }),
                "confidence_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01,
                                                  "tooltip": "Multiplier applied to score/confidence fields when available."}),
                "clamp_to_positive": ("BOOLEAN", {"default": True,
                                                    "tooltip": "Clamp manipulated coordinates to stay at or above zero."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("key_frame_body_points",)
    FUNCTION = "manipulate"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Applies offsets and scaling to selected key frame body points such as legs, torso, or hands."

    def _part_indices(self, part: str) -> List[int]:
        _, spec = self._resolve_part(part)
        if isinstance(spec, dict):
            spec = spec.get("indices", [])
        if isinstance(spec, str):
            if spec == "all":
                return list(range(len(COCO_BODY_KEYPOINT_NAMES)))
            return []
        return list(spec)

    def _resolve_entry_indices(self, entries: List[Dict[str, Any]]) -> List[Union[int, None]]:
        fallback = KEY_FRAME_BODY_INDICES
        if len(entries) == len(LEGACY_KEY_FRAME_BODY_INDICES):
            fallback = LEGACY_KEY_FRAME_BODY_INDICES

        resolved: List[Union[int, None]] = []
        fallback_iter = iter(fallback)
        for entry in entries:
            idx = None
            if isinstance(entry, dict):
                candidate = entry.get("index")
                if isinstance(candidate, int):
                    idx = candidate
            if idx is None:
                idx = next(fallback_iter, None)
            resolved.append(idx)
        return resolved

    def _collect_points(self, entries: List[Dict[str, Any]], indices: List[Union[int, None]],
                         target: Iterable[int]) -> Tuple[List[int], np.ndarray]:
        selected_indices: List[int] = []
        coords: List[List[float]] = []
        target_set = set(target)
        for list_idx, (entry, original_idx) in enumerate(zip(entries, indices)):
            if original_idx is None or original_idx not in target_set:
                continue
            try:
                x = float(entry.get("x", 0.0))
                y = float(entry.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            selected_indices.append(list_idx)
            coords.append([x, y])
        if not coords:
            return [], np.zeros((0, 2), dtype=np.float32)
        return selected_indices, np.asarray(coords, dtype=np.float32)

    def _apply_confidence_scale(self, entry: Dict[str, Any], factor: float) -> None:
        for key in ("confidence", "score"):
            if key in entry:
                try:
                    value = float(entry[key]) * factor
                except (TypeError, ValueError):
                    continue
                entry[key] = float(np.clip(value, 0.0, 1.0))

    def _stretch_only_legs_entries(
        self,
        entries: List[Dict[str, Any]],
        resolved_indices: List[Union[int, None]],
        scale: float,
        offset_x: float,
        offset_y: float,
        clamp_to_positive: bool,
        bidirectional: bool,
    ) -> List[int]:
        index_lookup: Dict[int, int] = {}
        for list_idx, body_idx in enumerate(resolved_indices):
            if isinstance(body_idx, int):
                index_lookup[body_idx] = list_idx

        scale_val = float(scale)
        if not math.isfinite(scale_val):
            scale_val = 1.0
        offset_x_val = float(offset_x)
        offset_y_val = float(offset_y)

        modified: List[int] = []

        def point_for(list_index: int) -> Optional[Tuple[float, float]]:
            if not (0 <= list_index < len(entries)):
                return None
            entry = entries[list_index]
            if not isinstance(entry, dict):
                return None
            try:
                x_val = float(entry.get("x", 0.0))
                y_val = float(entry.get("y", 0.0))
            except (TypeError, ValueError):
                return None
            if not (math.isfinite(x_val) and math.isfinite(y_val)):
                return None
            return x_val, y_val

        leg_triplets = ((11, 13, 15), (12, 14, 16))

        for hip_idx, knee_idx, ankle_idx in leg_triplets:
            hip_entry_idx = index_lookup.get(hip_idx)
            if hip_entry_idx is None:
                continue
            hip_point = point_for(hip_entry_idx)
            if hip_point is None:
                continue

            knee_entry_idx = index_lookup.get(knee_idx)
            ankle_entry_idx = index_lookup.get(ankle_idx)

            if not bidirectional and knee_entry_idx is None and ankle_entry_idx is None:
                continue

            current_targets: List[int] = []

            if bidirectional:
                if ankle_entry_idx is None:
                    continue
                ankle_point = point_for(ankle_entry_idx)
                if ankle_point is None:
                    continue
                center_y = (hip_point[1] + ankle_point[1]) * 0.5
                for idx in (hip_entry_idx, knee_entry_idx, ankle_entry_idx):
                    if idx is None:
                        continue
                    point = point_for(idx)
                    if point is None:
                        continue
                    new_y = center_y + (point[1] - center_y) * scale_val
                    entries[idx]["y"] = float(new_y)
                    current_targets.append(idx)
            else:
                for idx in (knee_entry_idx, ankle_entry_idx):
                    if idx is None:
                        continue
                    point = point_for(idx)
                    if point is None:
                        continue
                    new_y = hip_point[1] + (point[1] - hip_point[1]) * scale_val
                    entries[idx]["y"] = float(new_y)
                    current_targets.append(idx)

            modified.extend(current_targets)

        if not modified and offset_x_val == 0.0 and offset_y_val == 0.0:
            return []

        unique_indices = sorted(set(modified))

        if offset_x_val != 0.0 or offset_y_val != 0.0:
            for idx in unique_indices:
                entry = entries[idx]
                try:
                    entry["x"] = float(entry.get("x", 0.0)) + offset_x_val
                    entry["y"] = float(entry.get("y", 0.0)) + offset_y_val
                except (TypeError, ValueError):
                    continue

        if clamp_to_positive and unique_indices:
            for idx in unique_indices:
                entry = entries[idx]
                try:
                    entry["x"] = float(max(0.0, entry.get("x", 0.0)))
                    entry["y"] = float(max(0.0, entry.get("y", 0.0)))
                except (TypeError, ValueError):
                    continue

        return unique_indices

    def manipulate(self, key_frame_body_points, part, offset_x, offset_y, scale,
                   leg_scale_bidirectional, confidence_scale, clamp_to_positive=True):
        try:
            parsed = json.loads(key_frame_body_points)
        except (json.JSONDecodeError, TypeError):
            logging.warning("KeyFrameBodyPointsManipulator received invalid JSON payload.")
            return (key_frame_body_points,)

        if isinstance(parsed, dict):
            entries = parsed.get("points", [])
            container = parsed
            use_dict = True
        else:
            entries = parsed
            container = None
            use_dict = False

        if not isinstance(entries, list):
            logging.warning("KeyFrameBodyPointsManipulator expected a list of points.")
            return (key_frame_body_points,)

        indices = self._resolve_entry_indices(entries)
        _, part_spec = self._resolve_part(part)
        special = part_spec.get("special") if isinstance(part_spec, dict) else None

        if special == "only_legs":
            modified_entries = self._stretch_only_legs_entries(
                entries,
                indices,
                float(scale),
                float(offset_x),
                float(offset_y),
                bool(clamp_to_positive),
                bool(leg_scale_bidirectional),
            )
            if modified_entries and float(confidence_scale) != 1.0:
                factor = float(confidence_scale)
                for idx in modified_entries:
                    if 0 <= idx < len(entries):
                        entry = entries[idx]
                        if isinstance(entry, dict):
                            self._apply_confidence_scale(entry, factor)
            updated_payload = container if use_dict else entries
            return (json.dumps(updated_payload),)

        target_indices = self._part_indices(part)
        list_indices, coords = self._collect_points(entries, indices, target_indices)
        if coords.size == 0:
            return (key_frame_body_points,)

        center = coords.mean(axis=0, keepdims=True)
        transformed = (coords - center) * float(scale) + center
        transformed[:, 0] += float(offset_x)
        transformed[:, 1] += float(offset_y)
        if clamp_to_positive:
            transformed = np.clip(transformed, 0.0, None)

        for idx, point in zip(list_indices, transformed):
            entry = entries[idx]
            entry["x"] = float(point[0])
            entry["y"] = float(point[1])
            if confidence_scale != 1.0:
                self._apply_confidence_scale(entry, float(confidence_scale))

        updated_payload = container if use_dict else entries
        return (json.dumps(updated_payload),)


class KeyFrameBodyPointsToOpenPoseKeypoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key_frame_body_points": (
                    "STRING",
                    {
                        "tooltip": "Key frame body points in JSON format as produced by Pose and Face Detection.",
                    },
                ),
            },
            "optional": {
                "default_confidence": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Fallback confidence value applied when none is provided for a keypoint.",
                    },
                ),
                "canvas_width": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Canvas width for the generated OpenPose frame. Set to -1 to infer from the data.",
                    },
                ),
                "canvas_height": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Canvas height for the generated OpenPose frame. Set to -1 to infer from the data.",
                    },
                ),
                "confidence_override": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Set all exported confidences to this value. Use -1 to keep the detected confidences.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "convert"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = (
        "Converts key frame body points into an OpenPose POSE_KEYPOINT frame matching the Ultimate OpenPose Editor format."
    )

    def convert(
        self,
        key_frame_body_points,
        default_confidence=1.0,
        canvas_width=-1,
        canvas_height=-1,
        confidence_override=-1.0,
    ):
        try:
            parsed = json.loads(key_frame_body_points)
        except (json.JSONDecodeError, TypeError):
            logging.warning("KeyFrameBodyPointsToOpenPoseKeypoints received invalid JSON payload.")
            return ([],)

        if isinstance(parsed, dict):
            points_iterable = parsed.get("points", [])
        else:
            points_iterable = parsed

        if not isinstance(points_iterable, list):
            logging.warning("KeyFrameBodyPointsToOpenPoseKeypoints expected a list of points.")
            return ([],)

        indices_resolver = KeyFrameBodyPointsManipulator()
        resolved_indices = indices_resolver._resolve_entry_indices(points_iterable)

        body_keypoints = [0.0] * (len(COCO_BODY_KEYPOINT_NAMES) * 3)
        xs: List[float] = []
        ys: List[float] = []

        override_value: Optional[float] = None
        if confidence_override is not None and float(confidence_override) >= 0.0:
            override_value = float(np.clip(confidence_override, 0.0, 5.0))

        for entry, body_index in zip(points_iterable, resolved_indices):
            if body_index is None or not isinstance(entry, dict):
                continue
            try:
                x = float(entry.get("x", 0.0))
                y = float(entry.get("y", 0.0))
            except (TypeError, ValueError):
                continue

            if override_value is not None:
                c_val = override_value
            else:
                confidence = entry.get("confidence")
                if confidence is None:
                    confidence = entry.get("score", default_confidence)
                try:
                    c_val = float(confidence)
                except (TypeError, ValueError):
                    c_val = float(default_confidence)
                c_val = float(np.clip(c_val, 0.0, 5.0))

            slot = int(body_index) * 3
            if 0 <= slot <= len(body_keypoints) - 3:
                body_keypoints[slot] = x
                body_keypoints[slot + 1] = y
                body_keypoints[slot + 2] = c_val
                xs.append(x)
                ys.append(y)

        if not xs and not ys:
            return ([],)

        def _resolve_canvas(value_list: List[float], provided: int) -> int:
            if isinstance(provided, int) and provided > 0:
                return provided
            max_val = max(value_list) if value_list else 0.0
            if max_val > 2.0:
                return max(1, int(math.ceil(max_val)))
            return 512

        width = _resolve_canvas(xs, int(canvas_width) if canvas_width is not None else -1)
        height = _resolve_canvas(ys, int(canvas_height) if canvas_height is not None else -1)

        frame_entry = {
            "version": 1.3,
            "people": [
                {
                    "person_id": [-1],
                    "pose_keypoints_2d": body_keypoints,
                    "pose_keypoints_3d": [],
                    "face_keypoints_2d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_2d": [],
                    "hand_right_keypoints_3d": [],
                }
            ],
            "canvas_width": width,
            "canvas_height": height,
            "frame_index": 0,
        }

        return (frame_entry,)

NODE_CLASS_MAPPINGS = {
    "OnnxDetectionModelLoader10": OnnxDetectionModelLoader,
    "PoseAndFaceDetection10": PoseAndFaceDetection,
    "DrawViTPose10": DrawViTPose,
    "PoseRetargetPromptHelper10": PoseRetargetPromptHelper,
    "PoseDataManipulator10": PoseDataManipulator,
    "PoseDataToOpenPose10": PoseDataToOpenPose,
    "PoseDataToOpenPoseKeypoints10": PoseDataToOpenPoseKeypoints,
    "KeyFrameBodyPointsManipulator10v2": KeyFrameBodyPointsManipulator,
    "KeyFrameBodyPointsToOpenPoseKeypoints10": KeyFrameBodyPointsToOpenPoseKeypoints,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnnxDetectionModelLoader10": "ONNX Detection Model Loader 10",
    "PoseAndFaceDetection10": "Pose and Face Detection 10",
    "DrawViTPose10": "Draw ViT Pose 10",
    "PoseRetargetPromptHelper10": "Pose Retarget Prompt Helper 10",
    "PoseDataManipulator10": "Pose Data Manipulator 10",
    "PoseDataToOpenPose10": "Pose Data ➜ OpenPose JSON 10",
    "PoseDataToOpenPoseKeypoints10": "Pose Data ➜ OpenPose Keypoints 10",
    "KeyFrameBodyPointsManipulator10v2": "Key Frame Body Manipulator 10 v2",
    "KeyFrameBodyPointsToOpenPoseKeypoints10": "Key Frame Body ➜ OpenPose Keypoints 10",
}
