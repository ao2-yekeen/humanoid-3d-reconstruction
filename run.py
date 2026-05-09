"""
Semantic 3D Scene Reconstruction from Monocular Video

Takes a short video and produces:
  - A dense coloured 3D point cloud (.ply)
  - A semantic map of detected objects with 3D positions (.json)

Usage:
    python run.py --video path/to/video.mp4 --output output/
    python run.py --video path/to/video.mp4 --output output/ --frames 12 --iterations 500
"""

import argparse
import os
import sys
import cv2
import torch
import numpy as np
import trimesh
import json
from pathlib import Path
from ultralytics import YOLO

# MASt3R imports — cloned to extern/mast3r by install.sh
MAST3R_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extern', 'mast3r')
if not os.path.exists(MAST3R_DIR):
    raise RuntimeError(
        "MASt3R not found. Run install.sh first:\n  bash install.sh"
    )
sys.path.insert(0, MAST3R_DIR)
sys.path.insert(0, os.path.join(MAST3R_DIR, 'dust3r'))

from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


# ── Config ────────────────────────────────────────────────────────────────────

MAST3R_MODEL = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
YOLO_MODEL   = 'yolo11n.pt'
IMAGE_SIZE   = 224

PALETTE = [
    [220, 50,  50], [50, 180,  50], [50,  80, 220],
    [220,160,  50], [160, 50, 220], [50, 200, 200],
    [220,120,  50], [140,220,  50], [220, 50, 150],
    [80, 220, 180],
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, output_dir: str, fps: int = 2) -> list[str]:
    """Extract frames from video, skipping blurry ones."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    paths, idx, saved = [], 0, 0
    interval = max(1, int(cap.get(cv2.CAP_PROP_FPS) / fps))

    print(f"Extracting frames at {fps} fps...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() > 50:
                path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
                cv2.imwrite(path, frame)
                paths.append(path)
                saved += 1
        idx += 1

    cap.release()
    print(f"Extracted {saved} frames")
    return paths


def detect_objects(yolo: YOLO, frame_paths: list[str]) -> list[list[dict]]:
    """Run YOLO on each frame, return normalised bounding boxes."""
    print("Running YOLO object detection...")
    all_dets = []
    for fp in frame_paths:
        img = cv2.imread(fp)
        h, w = img.shape[:2]
        results = yolo(img, verbose=False, conf=0.3)[0]
        dets = []
        if results.boxes is not None:
            for box, cls, conf in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.cls.cpu().numpy(),
                results.boxes.conf.cpu().numpy()
            ):
                x1, y1, x2, y2 = box
                dets.append({
                    'label':   results.names[int(cls)],
                    'cx_norm': float((x1 + x2) / 2 / w),
                    'cy_norm': float((y1 + y2) / 2 / h),
                    'conf':    float(conf),
                })
        all_dets.append(dets)
        if dets:
            print(f"  {Path(fp).name}: {', '.join(d['label'] for d in dets)}")
    return all_dets


def reconstruct(frame_paths: list[str], device: str,
                niter: int = 500) -> tuple:
    """Run MASt3R reconstruction, return (pts3d, imgs, scene)."""
    print("\nRunning MASt3R reconstruction...")
    model = AsymmetricMASt3R.from_pretrained(MAST3R_MODEL).to(device)
    images = load_images([str(p) for p in frame_paths], size=IMAGE_SIZE)
    pairs  = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    print(f"{len(pairs)} image pairs")

    output = inference(pairs, model, device, batch_size=1, verbose=False)
    scene  = global_aligner(
        output, device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer
    )
    scene.compute_global_alignment(init='mst', niter=niter,
                                   schedule='cosine', lr=0.01)
    return scene.get_pts3d(), scene.imgs


def project_detections(pts3d, frame_detections: list[list[dict]]) -> list[dict]:
    """
    Project YOLO bounding box centres into 3D world space using
    MASt3R's per-pixel point maps. Confidence-weighted average
    across frames gives robust object positions.
    """
    print("\nProjecting detections into 3D...")
    tag_colour = {}
    colour_idx = 0
    objects = {}

    for pts, dets in zip(pts3d, frame_detections):
        pts_np = pts.reshape(IMAGE_SIZE, IMAGE_SIZE, 3).detach().cpu().numpy()

        for det in dets:
            label = det['label']
            conf  = det['conf']
            px = int(np.clip(det['cx_norm'] * IMAGE_SIZE, 0, IMAGE_SIZE - 1))
            py = int(np.clip(det['cy_norm'] * IMAGE_SIZE, 0, IMAGE_SIZE - 1))

            # Sample patch around detection centre
            r = 5
            patch = pts_np[max(0,py-r):py+r, max(0,px-r):px+r].reshape(-1, 3)
            valid = np.linalg.norm(patch, axis=1) > 0.01
            if valid.sum() == 0:
                continue

            pos = patch[valid].mean(axis=0)

            if label not in objects:
                if label not in tag_colour:
                    tag_colour[label] = PALETTE[colour_idx % len(PALETTE)]
                    colour_idx += 1
                objects[label] = {
                    'label':    label,
                    'colour':   tag_colour[label],
                    'weighted_pos': np.zeros(3),
                    'conf_sum': 0.0,
                    'count':    0,
                }

            objects[label]['weighted_pos'] += pos * conf
            objects[label]['conf_sum']     += conf
            objects[label]['count']        += 1

    semantic_map = []
    for obj in objects.values():
        avg_pos = (obj['weighted_pos'] / obj['conf_sum']).tolist()
        semantic_map.append({
            'label':      obj['label'],
            'position':   avg_pos,
            'colour':     obj['colour'],
            'seen_count': obj['count'],
        })

    return semantic_map


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Semantic 3D Scene Reconstruction from Monocular Video'
    )
    parser.add_argument('--video',      required=True, help='Path to input video')
    parser.add_argument('--output',     default='output/', help='Output directory')
    parser.add_argument('--frames',     type=int, default=12,
                        help='Number of frames to use (default: 12)')
    parser.add_argument('--iterations', type=int, default=500,
                        help='MASt3R alignment iterations (default: 500)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs(args.output, exist_ok=True)
    frames_dir = os.path.join(args.output, 'frames')
    ply_path   = os.path.join(args.output, 'reconstruction.ply')
    json_path  = os.path.join(args.output, 'semantic_map.json')

    # Step 1: Extract frames
    all_frames = extract_frames(args.video, frames_dir)
    if not all_frames:
        print("No frames extracted.")
        return

    # Sample evenly
    indices     = np.linspace(0, len(all_frames) - 1, args.frames, dtype=int)
    frame_paths = [all_frames[i] for i in indices]
    print(f"Using {len(frame_paths)} frames")

    # Step 2: YOLO detection
    yolo = YOLO(YOLO_MODEL)
    frame_dets = detect_objects(yolo, frame_paths)

    # Step 3: MASt3R reconstruction
    pts3d, imgs = reconstruct(frame_paths, device, niter=args.iterations)

    # Step 4: Project detections into 3D
    semantic_map = project_detections(pts3d, frame_dets)

    # Step 5: Save outputs
    print("\nSaving outputs...")
    all_pts  = np.concatenate([p.reshape(-1,3).detach().cpu().numpy() for p in pts3d])
    all_cols = np.concatenate([(img.reshape(-1,3)*255).astype(np.uint8) for img in imgs])

    pcd = trimesh.PointCloud(vertices=all_pts, colors=all_cols)
    pcd.export(ply_path)
    print(f"Point cloud: {len(all_pts):,} points -> {ply_path}")

    with open(json_path, 'w') as f:
        json.dump(semantic_map, f, indent=2)
    print(f"Semantic map: {len(semantic_map)} objects -> {json_path}")

    print("\nDetected objects:")
    for obj in sorted(semantic_map, key=lambda x: -x['seen_count'])[:10]:
        print(f"  {obj['label']:20s}  seen {obj['seen_count']}x  "
              f"pos={np.round(obj['position'], 2)}")

    print(f"\nDone. Visualise with:")
    print(f"  python visualise.py --pointcloud {ply_path} --semantic {json_path}")


if __name__ == '__main__':
    main()
