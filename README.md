# Semantic 3D Scene Reconstruction from Monocular Video

A pipeline that takes a short video recorded on a standard camera and produces a dense, coloured 3D point cloud with approximate semantic object labels placed in world space.

***

## Demo

**Input video:** [View on Google Drive](https://drive.google.com/file/d/14mE2FYkp679d-xNwr-J1tQJ_Qy8PqTdm/view?usp=sharing) — 50-second handheld phone recording, no depth sensor

**Output:** Dense coloured 3D point cloud with floating semantic labels
![semantic reconstruction](assets/semantic.png)

---

## Approach

Many classical reconstruction pipelines rely on COLMAP-style structure-from-motion for camera pose estimation, which often works best with strong camera motion, sufficient texture, and relatively stable scene geometry. This pipeline instead uses a learned reconstruction approach based on MASt3R to reduce reliance on classical feature matching.

### Geometric Reconstruction: MASt3R

[MASt3R](https://github.com/naver/mast3r) (Matching And Stereo 3D Reconstruction) is a transformer-based model that estimates 3D structure from pairs of images using learned dense correspondences and per-pixel point maps. A global alignment step then fuses the frames into a coherent world-space point cloud.

**Why MASt3R over COLMAP:**

- Works well on casual, unconstrained video without requiring slow orbital motion.
- Uses learned dense matching rather than classical sparse keypoint matching.
- Can generalise to real-world phone video.
- Produces dense geometric outputs rather than only sparse keypoints.


### Semantic Understanding: YOLO

[YOLOv11](https://github.com/ultralytics/ultralytics) runs on each frame to produce bounding boxes with class labels. The centre pixel of each bounding box is projected into 3D world space using MASt3R point maps, giving an approximate 3D position for each detected object.

**Why YOLO over prompt-based detectors:**

- Zero prompting — no object list required.
- Bounding boxes provide a practical anchor for approximate 3D localisation.
- Fast enough for per-frame inference on a consumer GPU.
- COCO classes cover many common indoor objects relevant to robot navigation.


### Visualisation: Viser

[Viser](https://github.com/nerfstudio-project/viser) renders the point cloud and floating semantic labels in a browser-based 3D viewer. It is lightweight, interactive, and well suited for robotics demos.

***

## Design Decisions

**Complete graph pairing over sequential:** MASt3R’s global alignment can benefit from pairing each frame with multiple other frames rather than relying only on adjacent pairs. For 12 frames, this gives 132 pairs and is still tractable on a consumer GPU.

**Confidence-weighted position averaging:** YOLO detections vary in confidence across frames. Object 3D positions are computed as a confidence-weighted average across frames where the object is detected, so higher-confidence detections contribute more.

**12 frames from 101:** Rather than processing every frame, 12 are sampled evenly across the video. This keeps compute manageable while preserving enough viewpoint diversity for reconstruction.

**Known limitation:** Label positions are approximate. The bounding box centre maps to a single 3D point, which may not correspond to the geometric centre of the object, especially for large objects like desks. A more accurate approach would aggregate 3D points from the projected bounding box region.

***

## Installation

```bash
git clone [https://github.com/ao2-yekeen/humanoid-3d-reconstruction](https://github.com/ao2-yekeen/humanoid-3d-reconstruction)
cd humanoid-3d-reconstruction
bash install.sh
```

`install.sh` clones MASt3R into `extern/mast3r`, installs Python dependencies, and downloads YOLO weights. MASt3R model weights are downloaded automatically on first run via Hugging Face.

Requires:

- Python 3.10
- CUDA 12.4
- gcc-11

***

## Usage

```bash
# Run full pipeline on a video
python run.py --video path/to/video.mp4 --output output/

# View results in browser
python visualise.py --pointcloud output/reconstruction.ply --semantic output/semantic_map.json
# Open http://localhost:8080
```


### Output files

| File | Description |
| :-- | :-- |
| `output/reconstruction.ply` | Dense coloured point cloud |
| `output/semantic_map.json` | Object labels with 3D positions |


***

## Pipeline Overview

```
Video
  │
  ├── Frame extraction (OpenCV, blur-filtered)
  │
  ├── YOLO detection per frame
  │     └── bounding boxes + class labels
  │
  ├── MASt3R reconstruction
  │     ├── Pairwise inference (complete graph)
  │     ├── Global alignment
  │     └── Dense point cloud in world space
  │
  ├── 3D label projection
  │     └── BBox centre → approximate 3D world position via MASt3R point maps
  │
  └── Viser visualisation
        ├── Point cloud
        └── Floating semantic labels
```


***

## Results

Tested on a 50-second handheld phone video of a student room. Detected objects include monitor, laptop, keyboard, chair, bottle, book, and shelf, placed plausibly in 3D space relative to the reconstructed geometry.

The reconstruction is geometrically coherent for the desk and near-field objects. Background elements such as walls and the ceiling are less consistent due to limited parallax at distance.

***

## Stack

| Component | Tool |
| :-- | :-- |
| 3D Reconstruction | MASt3R |
| Object Detection | YOLOv11n |
| Visualisation | Viser |
| Point Cloud I/O | Trimesh |
| Frame Extraction | OpenCV |


***

## Future Work

- Per-pixel semantic segmentation for more accurate 3D label placement.
- Temporal consistency filtering to reduce spurious detections.
- Real-time incremental reconstruction for live robot use.
- Integration with a robot navigation stack such as ROS2 costmaps.


