#!/usr/bin/env python3
"""
Generate vertical heightmaps from LAS point cloud.
Projects the cloud onto XZ and YZ planes.
4 views: XZ (+Y / -Y) and YZ (+X / -X).

Depth is measured from a "backplane" placed at the far boundary for each view,
so all points are in front of it (like a backlight/background).
"""

import laspy
import numpy as np
from PIL import Image
from pathlib import Path

# Load the LAS file
las_path = Path("./tree_16104.las")
las = laspy.read(las_path)

# Extract coordinates
x = np.array(las.x)
y = np.array(las.y)
z = np.array(las.z)

print("=" * 50)
print("POINT CLOUD STATISTICS")
print("=" * 50)
print(f"Total points: {len(x)}")
print(f"\nX: min={x.min():.3f}, max={x.max():.3f}, extent={x.max()-x.min():.3f}")
print(f"Y: min={y.min():.3f}, max={y.max():.3f}, extent={y.max()-y.min():.3f}")
print(f"Z: min={z.min():.3f}, max={z.max():.3f}, extent={z.max()-z.min():.3f}")

# Center coordinates (subtract min to get 0-based)
x = x - x.min()
y = y - y.min()
z = z - z.min()

x_extent = x.max()
y_extent = y.max()
z_extent = z.max()

print(f"\nAfter centering to origin:")
print(f"X: 0 to {x_extent:.3f}")
print(f"Y: 0 to {y_extent:.3f}")
print(f"Z: 0 to {z_extent:.3f}")

# Resolution in pixels per meter
pixels_per_meter = 10
resolution_x = max(1, int(np.ceil(x_extent * pixels_per_meter)))
resolution_y = max(1, int(np.ceil(y_extent * pixels_per_meter)))
resolution_z = max(1, int(np.ceil(z_extent * pixels_per_meter)))

print(f"\n" + "=" * 50)
print("OUTPUT")
print("=" * 50)
print(f"Resolution: {pixels_per_meter} px/m")
print(f"XZ image size: {resolution_x} x {resolution_z}")
print(f"YZ image size: {resolution_y} x {resolution_z}")


def create_heightmap(horiz_coords, depth_values, vert_coords,
                     horiz_extent, vert_extent,
                     max_depth,
                     res_h, res_v):
    """
    Create a heightmap where pixel intensity = depth *amplitude* in the pixel:
    (max depth - min depth) among points that land in that pixel.

    If a pixel contains fewer than 2 points, its value is 0.
    
    Args:
        horiz_coords: horizontal axis coordinates (X for XZ, Y for YZ)
        depth_values: pre-computed depth from backplane (always positive)
        vert_coords: vertical axis (always Z)
        horiz_extent: max horizontal extent
        vert_extent: max vertical extent
        max_depth: maximum possible depth (for normalization)
        res_h: horizontal resolution in pixels
        res_v: vertical resolution in pixels
    """
    heightmap = np.zeros((res_v, res_h), dtype=float)
    
    # Bin edges
    h_bins = np.linspace(0, horiz_extent, res_h + 1)
    v_bins = np.linspace(0, vert_extent, res_v + 1)
    
    h_indices = np.clip(np.digitize(horiz_coords, h_bins) - 1, 0, res_h - 1)
    v_indices = np.clip(np.digitize(vert_coords, v_bins) - 1, 0, res_v - 1)
    
    # For each pixel, compute amplitude (max depth - min depth) if >=2 points.
    n_pixels = res_h * res_v
    flat = (v_indices * res_h + h_indices).astype(np.int64, copy=False)

    counts = np.bincount(flat, minlength=n_pixels)

    mins = np.full(n_pixels, np.inf, dtype=float)
    maxs = np.full(n_pixels, -np.inf, dtype=float)
    depth_values = np.asarray(depth_values, dtype=float)

    np.minimum.at(mins, flat, depth_values)
    np.maximum.at(maxs, flat, depth_values)

    amplitudes = np.where(counts >= 2, maxs - mins, 0.0)
    heightmap = amplitudes.reshape((res_v, res_h))
    
    # Normalize to [0, 1]
    if max_depth > 0:
        heightmap = heightmap / max_depth
    
    return heightmap


def save_heightmap(heightmap, path):
    """Save heightmap as grayscale PNG with Z=0 at bottom."""
    cells_before = np.sum(heightmap > 0)
    
    # Normalize to 0-255
    if heightmap.max() > 0:
        hm_norm = heightmap / heightmap.max()
    else:
        hm_norm = heightmap
    
    hm_uint8 = (hm_norm * 255).astype(np.uint8)
    
    # Flip vertically so Z=0 is at the bottom of the image
    hm_uint8 = np.flipud(hm_uint8)
    
    img = Image.fromarray(hm_uint8, mode='L')
    img.save(path)
    print(f"Saved: {path} ({img.width}x{img.height}, {cells_before} cells)")


# --- Backplanes / view setup ---
print("\n" + "=" * 50)
print("VIEW PLANES")
print("=" * 50)
print("Using backplanes on the far boundary so all points are in front.")
print(f"XZ views depth axis = Y, planes at Y=0 and Y={y_extent:.3f}")
print(f"YZ views depth axis = X, planes at X=0 and X={x_extent:.3f}")

# Maximum depth for normalization (full extent along depth axis)
max_depth_y = y_extent
max_depth_x = x_extent

# --- Generate 4 heightmaps ---
print("\nGenerating heightmaps...")

# XZ from +Y: backplane at Y=0, depth increases with Y
x_xz_pos = x
z_xz_pos = z
depth_xz_pos = y

xz_pos_hm = create_heightmap(x_xz_pos, depth_xz_pos, z_xz_pos,
                              x_extent, z_extent, max_depth_y,
                              resolution_x, resolution_z)
save_heightmap(xz_pos_hm, "heightmap_XZ_posY.png")

# XZ from -Y: backplane at Y=max, depth increases towards smaller Y
x_xz_neg = x
z_xz_neg = z
depth_xz_neg = y_extent - y

xz_neg_hm = create_heightmap(x_xz_neg, depth_xz_neg, z_xz_neg,
                              x_extent, z_extent, max_depth_y,
                              resolution_x, resolution_z)
save_heightmap(xz_neg_hm, "heightmap_XZ_negY.png")

# YZ from +X: backplane at X=0, depth increases with X
y_yz_pos = y
z_yz_pos = z
depth_yz_pos = x

yz_pos_hm = create_heightmap(y_yz_pos, depth_yz_pos, z_yz_pos,
                              y_extent, z_extent, max_depth_x,
                              resolution_y, resolution_z)
save_heightmap(yz_pos_hm, "heightmap_YZ_posX.png")

# YZ from -X: backplane at X=max, depth increases towards smaller X
y_yz_neg = y
z_yz_neg = z
depth_yz_neg = x_extent - x

yz_neg_hm = create_heightmap(y_yz_neg, depth_yz_neg, z_yz_neg,
                              y_extent, z_extent, max_depth_x,
                              resolution_y, resolution_z)
save_heightmap(yz_neg_hm, "heightmap_YZ_negX.png")

print("\nDone!")
