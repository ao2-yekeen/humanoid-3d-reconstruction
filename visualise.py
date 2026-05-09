"""
Semantic 3D Scene Visualiser

Loads a point cloud and semantic map and displays them in a
browser-based 3D viewer using Viser.

Usage:
    python visualise.py --pointcloud output/reconstruction.ply --semantic output/semantic_map.json
    # Then open http://localhost:8080
"""

import argparse
import json
import time
import numpy as np
import trimesh
import viser


def main():
    parser = argparse.ArgumentParser(description='Semantic 3D Scene Visualiser')
    parser.add_argument('--pointcloud', required=True, help='Path to .ply point cloud')
    parser.add_argument('--semantic',   required=True, help='Path to semantic_map.json')
    parser.add_argument('--port',       type=int, default=8080, help='Port (default: 8080)')
    parser.add_argument('--point_size', type=float, default=0.003, help='Point size (default: 0.003)')
    args = parser.parse_args()

    # Load point cloud
    print(f"Loading point cloud from {args.pointcloud}...")
    pcd  = trimesh.load(args.pointcloud)
    pts  = np.array(pcd.vertices, dtype=np.float32)
    cols = np.array(pcd.colors)[:, :3].astype(np.uint8) if hasattr(pcd, 'colors') else None
    print(f"Loaded {len(pts):,} points")

    # Load semantic map
    print(f"Loading semantic map from {args.semantic}...")
    with open(args.semantic) as f:
        semantic_map = json.load(f)
    print(f"Loaded {len(semantic_map)} objects")

    # Start viser server
    server = viser.ViserServer(port=args.port)
    print(f"\nOpen http://localhost:{args.port} in your browser")

    # Add point cloud
    server.scene.add_point_cloud(
        name='scene/reconstruction',
        points=pts,
        colors=cols,
        point_size=args.point_size,
    )

    # Add semantic labels
    for obj in semantic_map:
        pos    = np.array(obj['position'], dtype=np.float32)
        label  = obj['label']
        count  = obj['seen_count']

        server.scene.add_label(
            name=f'semantics/{label}',
            text=f'{label}',
            position=pos,
        )

    # Print summary
    print("\nDetected objects:")
    for obj in sorted(semantic_map, key=lambda x: -x['seen_count']):
        print(f"  {obj['label']:20s}  seen {obj['seen_count']}x  "
              f"pos={np.round(obj['position'], 2)}")

    print("\nPress Ctrl+C to exit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting.")


if __name__ == '__main__':
    main()
