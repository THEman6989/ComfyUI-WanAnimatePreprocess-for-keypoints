# Mask Researcher Tools

This repo now includes three extra ComfyUI helper nodes for video mask analysis:

- `Mask Quality Filter`
- `Mask Interpolator Pro`
- `Mask Crop Stabilizer`

They are meant for a researcher workflow before DINOv2, SigLIP, CLIP, or any other embedding comparison.

## Recommended chain

```text
Video Frames
-> SAM / SAM3 / SAM3.1 masks
-> Mask Quality Filter
-> Mask Interpolator Pro
-> Mask Crop Stabilizer
-> DINOv2 / SigLIP / CLIP embeddings
```

## Why these nodes exist

Segmentation can fail for one frame or produce a wrong mask. If that bad mask is sent into an embedding model, it can create a false positive and look like an outfit change.

These nodes reduce that problem:

1. `Mask Quality Filter` marks broken masks as invalid.
2. `Mask Interpolator Pro` fills missing mask frames from nearby valid frames.
3. `Mask Crop Stabilizer` creates stable image crops so embeddings do not jump just because the crop moved.

## Nodes

### Mask Quality Filter

Input: `MASK`

Outputs:

- `filtered_masks`
- `invalid_frame_mask`
- `report`

It checks area, mean value, large masks, centroid jumps, and area jumps.

### Mask Interpolator Pro

Input: `MASK`

Optional input: `IMAGE`

Outputs:

- `masks`
- `report`

It repairs invalid or empty masks between good frames.

### Mask Crop Stabilizer

Inputs:

- `IMAGE`
- `MASK`

Outputs:

- `crops`
- `crop_masks`
- `report`

It creates stable square crops for embedding models.

## Dependencies

`opencv-python` is already listed in this repo requirements. The compact version of these nodes does not require SciPy.

## Suggested defaults

Quality Filter:

```text
area_threshold = 64
mean_threshold = 0.001
max_area_fraction = 0.75
max_centroid_jump_px = 160
max_area_jump_ratio = 5.0
mode = zero_invalid
threshold = 0.5
```

Interpolator:

```text
area_threshold = 64
mean_threshold = 0.001
soft_blend = 0.5
threshold = 0.5
max_gap = 24
```

Crop Stabilizer:

```text
output_size = 512
padding_percent = 0.18
smoothing = 0.65
square_crop = true
mask_mode = image_crop
mask_threshold = 0.5
```
