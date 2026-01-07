import os
import tempfile

import mrcfile
import trimesh
import numpy as np
from skimage.filters import gaussian
from skimage.measure import marching_cubes


def choose_downsample_factor(shape, max_dim=256):
    nz, ny, nx = shape
    factor = 1
    while max(nz, ny, nx) > max_dim and factor < 16:
        factor *= 2
        nz //= 2; ny //= 2; nx //= 2
    return factor

def downsample_stride(vol, factor):
    factor = int(factor)
    return vol if factor <= 1 else vol[::factor, ::factor, ::factor]

def load_mrc(file_path):
    if file_path is None or not os.path.exists(file_path):
        return None, None, None
    with mrcfile.open(file_path, permissive=True) as m:
        vol = np.array(m.data, dtype=np.float32)  # (z,y,x)
        voxel_size = m.voxel_size
        apix_tuple = (float(voxel_size.x), float(voxel_size.y), float(voxel_size.z))
        # Use X value as a fallback if needed, but we'll store the full tuple
    stats = {
        "min": float(np.min(vol)),
        "max": float(np.max(vol)),
        "mean": float(np.mean(vol)),
        "std": float(np.std(vol)),
        "shape": vol.shape,
        "p95": float(np.percentile(vol, 95)),
        "p98": float(np.percentile(vol, 98.5)),
    }
    return vol, apix_tuple, stats

def safe_iso_from_percentile(vol, iso_percentile):
    iso_percentile = float(np.clip(iso_percentile, 50.0, 99.9))
    return float(np.percentile(vol, iso_percentile))

def center_and_scale(verts):
    c = verts.mean(axis=0, keepdims=True)
    v = verts - c
    extent = np.max(np.linalg.norm(v, axis=1))
    if extent > 0:
        v = v / extent
    return v

def build_glb(
    vol,
    apix_tuple,
    iso_percentile=98.5,
    smooth_sigma=1.0,
    max_dim=256,
    step_size=1,
    outline_strength=0.012,
):
    if vol is None:
        return None, "No volume loaded."

    auto_factor = choose_downsample_factor(vol.shape, int(max_dim))
    v = downsample_stride(vol, auto_factor)

    smooth_sigma = float(smooth_sigma)
    if smooth_sigma > 0:
        v = gaussian(v, sigma=smooth_sigma, preserve_range=True).astype(np.float32)

    iso = safe_iso_from_percentile(v, iso_percentile)
    
    # spacing: handle tuple of (x, y, z)
    if apix_tuple:
        spacing = (apix_tuple[2] * auto_factor, apix_tuple[1] * auto_factor, apix_tuple[0] * auto_factor)
    else:
        spacing = (1.0, 1.0, 1.0)

    try:
        verts, faces, normals, values = marching_cubes(
            volume=v,
            level=iso,
            spacing=spacing,
            step_size=int(step_size),
            allow_degenerate=False,
        )
    except Exception as e:
        return None, f"Marching cubes error: {e}"

    verts = center_and_scale(verts)
    faces = faces.astype(np.int32)
    
    # Main mesh
    main_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    main_mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[0.55, 0.55, 0.55, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.35,
    )
    main_mesh.visual = trimesh.visual.TextureVisuals(material=main_mat)

    # Outline shell
    vn = main_mesh.vertex_normals
    eps = float(outline_strength)
    shell_verts = verts + eps * vn
    shell_mesh = trimesh.Trimesh(vertices=shell_verts, faces=faces, process=False)
    shell_mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[0.30, 0.30, 0.30, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.75,
    )
    shell_mesh.visual = trimesh.visual.TextureVisuals(material=shell_mat)

    scene = trimesh.Scene()
    scene.add_geometry(shell_mesh, node_name="outline_shell")
    scene.add_geometry(main_mesh, node_name="main_surface")

    out_dir = tempfile.mkdtemp(prefix="mrc_glb_")
    glb_path = os.path.join(out_dir, "mesh.glb")
    scene.export(glb_path)

    if apix_tuple:
        info = f"**Shape:** {vol.shape} | **Voxel Size:** ({apix_tuple[0]:.4f}, {apix_tuple[1]:.4f}, {apix_tuple[2]:.4f})"
    else:
        info = f"**Shape:** {vol.shape} | **Voxel Size:** N/A"
        
    return glb_path, info

