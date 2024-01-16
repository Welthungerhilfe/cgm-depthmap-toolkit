import open3d as o3d
import logging
from pathlib import Path
from typing import Union

import numpy as np

from cgmml.common.depthmap_toolkit.depthmap import Depthmap

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)


def convert_to_open3d_pointcloud(dmap: Depthmap,
                                 floor_altitude_in_meters: float):
    """Converts depthmap into Open3D pointcloud.

    floor_altitude_in_meters is the floor altitude to align floor to Y=zero"""
    points = []
    normals = []
    points_3d_arr = dmap.convert_2d_to_3d_oriented()
    normal_3d_arr = dmap.calculate_normalmap_array(points_3d_arr)
    for x in range(2, dmap.width - 2):
        for y in range(2, dmap.height - 2):
            depth = dmap.depthmap_arr[x, y]
            if not depth:
                continue

            x_coord = points_3d_arr[0, x, y]
            y_coord = points_3d_arr[1, x, y] - floor_altitude_in_meters
            z_coord = points_3d_arr[2, x, y]
            x_normal = normal_3d_arr[0, x, y]
            y_normal = normal_3d_arr[1, x, y]
            z_normal = normal_3d_arr[2, x, y]
            points.append([x_coord, y_coord, z_coord])
            normals.append([x_normal, y_normal, z_normal])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def convert_to_open3d_mesh(dmap: Depthmap,
                           floor_altitude_in_meters: float):
    """Converts depthmap into Open3D mesh, postprocessed with Poisson reconstruction.

    floor_altitude_in_meters is the floor altitude to align floor to Y=zero"""

    pcd = convert_to_open3d_pointcloud(dmap, floor_altitude_in_meters)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
    return mesh


def export_obj(fpath: Union[str, Path],
               dmap: Depthmap,
               floor_altitude_in_meters: float,
               triangulate: bool):
    """Export .obj file, which can be visualized in tools like Meshlab.

    floor_altitude_in_meters is the floor altitude to align floor to Y=zero
    triangulate=True generates OBJ of type mesh
    triangulate=False generates OBJ of type pointcloud
    """
    fpath = Path(fpath)
    count = 0
    indices = np.zeros((dmap.width, dmap.height))

    # Create MTL file (a standart extension of OBJ files to define geometry materials and textures)
    material_fpath = fpath.with_suffix('.mtl')
    if dmap.has_rgb:
        with open(material_fpath, 'w') as f:
            f.write('newmtl default\n')
            f.write(f'map_Kd {str(dmap.rgb_fpath.absolute())}\n')

    with open(fpath, 'w') as f:
        if dmap.has_rgb:
            f.write(f'mtllib {material_fpath.name}\n')
            f.write('usemtl default\n')

        points_3d_arr = dmap.convert_2d_to_3d_oriented()
        for x in range(2, dmap.width - 2):
            for y in range(2, dmap.height - 2):
                depth = dmap.depthmap_arr[x, y]
                if not depth:
                    continue
                count = count + 1
                indices[x, y] = count  # add index of written vertex into array

                x_coord = points_3d_arr[0, x, y]
                y_coord = points_3d_arr[1, x, y] - floor_altitude_in_meters
                z_coord = points_3d_arr[2, x, y]
                f.write(f'v {x_coord} {y_coord} {z_coord}\n')
                f.write(f'vt {x / dmap.width} {y / dmap.height}\n')

        if triangulate:
            _do_triangulation(dmap, indices, f)
        logger.info('Mesh exported into %s', fpath)


def export_extrapolated_obj(fpath: Union[str, Path],
                            dmap: Depthmap,
                            floor_altitude_in_meters: float):
    """Export .obj file, postprocessed with Poisson reconstruction which can be visualized in tools like Meshlab.

    floor_altitude_in_meters is the floor altitude to align floor to Y=zero"""

    mesh = convert_to_open3d_mesh(dmap, floor_altitude_in_meters)
    o3d.io.write_triangle_mesh(str(fpath), mesh)
    logger.info('Mesh exported into %s', fpath)


def export_renderable_obj(fpath: Union[str, Path],
                          dmap: Depthmap,
                          floor_altitude_in_meters: float,
                          point_size_in_meters: float):
    """Export pointcloud as .obj file, which can be rendered in tools like Blender.

    floor_altitude_in_meters is the floor altitude to align floor to Y=zero
    point_size_in_meters is point size in meters
    """
    fpath = Path(fpath)
    count = 1

    # Create MTL file (a standart extension of OBJ files to define geometry materials and textures)
    material_fpath = fpath.with_suffix('.mtl')
    if dmap.has_rgb:
        with open(material_fpath, 'w') as f:
            f.write('newmtl default\n')
            f.write(f'map_Kd {str(dmap.rgb_fpath.absolute())}\n')

    with open(fpath, 'w') as f:
        if dmap.has_rgb:
            f.write(f'mtllib {material_fpath.name}\n')
            f.write('usemtl default\n')

        points_3d_arr = dmap.convert_2d_to_3d_oriented()
        for x in range(2, dmap.width - 2):
            for y in range(2, dmap.height - 2):
                depth = dmap.depthmap_arr[x, y]
                if not depth:
                    continue

                x_coord = points_3d_arr[0, x, y]
                y_coord = points_3d_arr[1, x, y] - floor_altitude_in_meters
                z_coord = points_3d_arr[2, x, y]
                _write_obj_cube(f, dmap, x, y, count, x_coord, y_coord, z_coord, point_size_in_meters)
                count = count + 8

        logger.info('Mesh exported into %s', fpath)


def export_ply(fpath: Union[str, Path],
               dmap: Depthmap,
               floor_altitude_in_meters: float):
    """Export .ply pointcloud file which can be visualized in tools like Meshlab.

    floor_altitude_in_meters is the floor altitude to align floor to Y=zero"""

    pcd = convert_to_open3d_pointcloud(dmap, floor_altitude_in_meters)
    o3d.io.write_point_cloud(str(fpath), pcd)
    logger.info('Pointcloud exported into %s', fpath)


def _do_triangulation(dmap: Depthmap, indices, filehandle):
    max_diff = 0.2
    for x in range(2, dmap.width - 2):
        for y in range(2, dmap.height - 2):
            # get depth of all points of 2 potential triangles
            d00 = dmap.depthmap_arr[x, y]
            d10 = dmap.depthmap_arr[x + 1, y]
            d01 = dmap.depthmap_arr[x, y + 1]
            d11 = dmap.depthmap_arr[x + 1, y + 1]

            # check if first triangle points have existing indices
            if indices[x, y] > 0 and indices[x + 1, y] > 0 and indices[x, y + 1] > 0:
                # check if the triangle size is valid (to prevent generating triangle
                # connecting child and background)
                if abs(d00 - d10) + abs(d00 - d01) + abs(d10 - d01) < max_diff:
                    c = str(int(indices[x, y]))
                    b = str(int(indices[x + 1, y]))
                    a = str(int(indices[x, y + 1]))
                    # define triangle indices in (world coordinates / texture coordinates)
                    _write_obj_triangle_indices(filehandle, a, b, c)

            # check if second triangle points have existing indices
            if indices[x + 1, y + 1] > 0 and indices[x + 1, y] > 0 and indices[x, y + 1] > 0:
                # check if the triangle size is valid (to prevent generating triangle
                # connecting child and background)
                if abs(d11 - d10) + abs(d11 - d01) + abs(d10 - d01) < max_diff:
                    a = str(int(indices[x + 1, y + 1]))
                    b = str(int(indices[x + 1, y]))
                    c = str(int(indices[x, y + 1]))
                    # define triangle indices in (world coordinates / texture coordinates)
                    _write_obj_triangle_indices(filehandle, a, b, c)


def _write_obj_cube(
        f,
        dmap: Depthmap,
        x: int,
        y: int,
        count: int,
        x_coord: float,
        y_coord: float,
        z_coord: float,
        size: float):

    # cube points
    f.write(f'v {x_coord - size} {y_coord - size} {z_coord - size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord + size} {y_coord - size} {z_coord - size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord - size} {y_coord + size} {z_coord - size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord + size} {y_coord + size} {z_coord - size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord - size} {y_coord - size} {z_coord + size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord + size} {y_coord - size} {z_coord + size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord - size} {y_coord + size} {z_coord + size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord + size} {y_coord + size} {z_coord + size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')

    # front face
    _write_obj_triangle_indices(f, str(count + 2), str(count + 1), str(count + 0))
    _write_obj_triangle_indices(f, str(count + 3), str(count + 2), str(count + 1))

    # back face
    _write_obj_triangle_indices(f, str(count + 4), str(count + 5), str(count + 6))
    _write_obj_triangle_indices(f, str(count + 7), str(count + 6), str(count + 5))

    # left face
    _write_obj_triangle_indices(f, str(count + 2), str(count + 4), str(count + 0))
    _write_obj_triangle_indices(f, str(count + 2), str(count + 4), str(count + 6))

    # right face
    _write_obj_triangle_indices(f, str(count + 1), str(count + 5), str(count + 3))
    _write_obj_triangle_indices(f, str(count + 7), str(count + 5), str(count + 3))

    # top face
    _write_obj_triangle_indices(f, str(count + 6), str(count + 3), str(count + 2))
    _write_obj_triangle_indices(f, str(count + 3), str(count + 6), str(count + 7))

    # bottom face
    _write_obj_triangle_indices(f, str(count + 0), str(count + 1), str(count + 4))
    _write_obj_triangle_indices(f, str(count + 4), str(count + 3), str(count + 1))


def _write_obj_triangle_indices(f, a: str, b: str, c: str):
    f.write(f'f {a}/{a} {b}/{b} {c}/{c}\n')
