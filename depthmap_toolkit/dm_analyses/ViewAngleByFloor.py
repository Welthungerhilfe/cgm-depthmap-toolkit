import numpy as np

from helpers import depthmap_to_normals
from helpers.pc_generator import PCGenerator
from helpers.ransac_plane_extractor import PlaneExtractor


def run(depthmap):
    """
    Get the camera pitch angle by the floor angle in given depth map.

    1. Extracts surface normals from depth map.
    2. Creates point cloud from depth map. Filter invalid values.
    3. Extracts floor plane from point cloud using surface normals.
    4. Return angle between view axis and floor plane - 90
    :param depthmap:    depth map to get camera pitch angle
    :return:            
    """
    normals = depthmap_to_normals.get_pixel_normals(depthmap)

    pc = PCGenerator(depthmap).generate_pixelmap()

    include = np.bitwise_not(np.isnan(depthmap)).squeeze()

    pe = PlaneExtractor(pc, include, normals, iterations=20, threshold=0.02)
    floor_plane, _, _ = pe.get_floor()

    return np.degrees(np.arccos(floor_plane.n[2] / np.linalg.norm(floor_plane.n))) - 90

