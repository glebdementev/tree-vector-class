#!/usr/bin/env python3
"""
Generate vertical heightmaps from LAS point cloud.
Projects XZ and YZ planes by slicing the cloud in the middle.
4 views: XZ+, XZ-, YZ+, YZ- (from both sides of each plane)

Depth = distance from the central slice plane.
Each view only uses points on its side of the center.
"""

import laspy
import numpy as np
from scipy import ndimage
from PIL import Image
from pathlib import Path

# Load the LAS file
las_path = Path("/home/gleb/dev/ol-class/tree_16104.las")
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

# Calculate center
x_center = x_extent / 2
y_center = y_extent / 2

# Slice thickness - 20% of the dimension extent (10% on each side of center)
slice_thickness_x = x_extent * 0.2
slice_thickness_y = y_extent * 0.2

print(f"\nAfter centering to origin:")
print(f"X: 0 to {x_extent:.3f}, center={x_center:.3f}")
print(f"Y: 0 to {y_extent:.3f}, center={y_center:.3f}")
print(f"Z: 0 to {z_extent:.3f}")

# Resolution in pixels per meter
pixels_per_meter = 10
resolution_x = int(np.ceil(x_extent * pixels_per_meter))
resolution_y = int(np.ceil(y_extent * pixels_per_meter))
resolution_z = int(np.ceil(z_extent * pixels_per_meter))

print(f"\n" + "=" * 50)
print("OUTPUT")
print("=" * 50)
print(f"Resolution: {pixels_per_meter} px/m")
print(f"XZ image size: {resolution_x} x {resolution_z}")
print(f"YZ image size: {resolution_y} x {resolution_z}")


def apply_median_filter_preserve(image):
    """
    Apply 3x3 median filter but preserve cells that have data.
    """
    has_data = image > 0
    filtered = ndimage.median_filter(image, size=3)
    result = np.where(has_data, np.maximum(image, filtered), filtered)
    return result


def create_heightmap(horiz_coords, depth_values, vert_coords,
                     horiz_extent, vert_extent,
                     max_depth,
                     res_h, res_v):
    """
    Create a heightmap where pixel intensity = depth from center plane.
    
    Args:
        horiz_coords: horizontal axis coordinates (X for XZ, Y for YZ)
        depth_values: pre-computed depth from center plane (always positive)
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
    
    # For each cell, find the point with maximum depth (closest to viewer)
    for hi, vi, d in zip(h_indices, v_indices, depth_values):
        if d > heightmap[vi, hi]:
            heightmap[vi, hi] = d
    
    # Normalize to [0, 1]
    if max_depth > 0:
        heightmap = heightmap / max_depth
    
    return heightmap


def save_heightmap(heightmap, path, apply_filter=True):
    """Save heightmap as grayscale PNG with Z=0 at bottom."""
    cells_before = np.sum(heightmap > 0)
    
    if apply_filter:
        heightmap = apply_median_filter_preserve(heightmap)
    
    cells_after = np.sum(heightmap > 0)
    
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
    print(f"Saved: {path} ({img.width}x{img.height}, {cells_before}->{cells_after} cells)")


# --- Get slices ---
print("\n" + "=" * 50)
print("SLICE STATISTICS")
print("=" * 50)

# XZ slice bounds
y_slice_min = y_center - slice_thickness_y / 2
y_slice_max = y_center + slice_thickness_y / 2

# Points in the XZ slice region
xz_mask = (y >= y_slice_min) & (y <= y_slice_max)

# Split into +Y side and -Y side
xz_pos_mask = xz_mask & (y > y_center)  # +Y side: y > y_center
xz_neg_mask = xz_mask & (y < y_center)  # -Y side: y < y_center

print(f"\nXZ slice (Y center = {y_center:.3f}, thickness = {slice_thickness_y:.3f}):")
print(f"  +Y side (y > {y_center:.3f}): {np.sum(xz_pos_mask)} points")
print(f"  -Y side (y < {y_center:.3f}): {np.sum(xz_neg_mask)} points")

# YZ slice bounds
x_slice_min = x_center - slice_thickness_x / 2
x_slice_max = x_center + slice_thickness_x / 2

# Points in the YZ slice region
yz_mask = (x >= x_slice_min) & (x <= x_slice_max)

# Split into +X side and -X side
yz_pos_mask = yz_mask & (x > x_center)  # +X side: x > x_center
yz_neg_mask = yz_mask & (x < x_center)  # -X side: x < x_center

print(f"\nYZ slice (X center = {x_center:.3f}, thickness = {slice_thickness_x:.3f}):")
print(f"  +X side (x > {x_center:.3f}): {np.sum(yz_pos_mask)} points")
print(f"  -X side (x < {x_center:.3f}): {np.sum(yz_neg_mask)} points")

# Maximum depth for normalization (half the slice thickness)
max_depth_y = slice_thickness_y / 2
max_depth_x = slice_thickness_x / 2

# --- Generate 4 heightmaps ---
print("\nGenerating heightmaps...")

# XZ from +Y: points where y > y_center, depth = y - y_center
x_xz_pos = x[xz_pos_mask]
z_xz_pos = z[xz_pos_mask]
depth_xz_pos = y[xz_pos_mask] - y_center  # distance from center plane

xz_pos_hm = create_heightmap(x_xz_pos, depth_xz_pos, z_xz_pos,
                              x_extent, z_extent, max_depth_y,
                              resolution_x, resolution_z)
save_heightmap(xz_pos_hm, "/home/gleb/dev/ol-class/heightmap_XZ_posY.png")

# XZ from -Y: points where y < y_center, depth = y_center - y
x_xz_neg = x[xz_neg_mask]
z_xz_neg = z[xz_neg_mask]
depth_xz_neg = y_center - y[xz_neg_mask]  # distance from center plane

xz_neg_hm = create_heightmap(x_xz_neg, depth_xz_neg, z_xz_neg,
                              x_extent, z_extent, max_depth_y,
                              resolution_x, resolution_z)
save_heightmap(xz_neg_hm, "/home/gleb/dev/ol-class/heightmap_XZ_negY.png")

# YZ from +X: points where x > x_center, depth = x - x_center
y_yz_pos = y[yz_pos_mask]
z_yz_pos = z[yz_pos_mask]
depth_yz_pos = x[yz_pos_mask] - x_center

yz_pos_hm = create_heightmap(y_yz_pos, depth_yz_pos, z_yz_pos,
                              y_extent, z_extent, max_depth_x,
                              resolution_y, resolution_z)
save_heightmap(yz_pos_hm, "/home/gleb/dev/ol-class/heightmap_YZ_posX.png")

# YZ from -X: points where x < x_center, depth = x_center - x
y_yz_neg = y[yz_neg_mask]
z_yz_neg = z[yz_neg_mask]
depth_yz_neg = x_center - x[yz_neg_mask]

yz_neg_hm = create_heightmap(y_yz_neg, depth_yz_neg, z_yz_neg,
                              y_extent, z_extent, max_depth_x,
                              resolution_y, resolution_z)
save_heightmap(yz_neg_hm, "/home/gleb/dev/ol-class/heightmap_YZ_negX.png")

print("\nDone!")
