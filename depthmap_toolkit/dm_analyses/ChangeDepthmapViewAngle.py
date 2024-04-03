import numpy as np

from helpers import depthmap_to_normals
from helpers.pc_generator import PCGenerator
from helpers.ransac_plane_extractor import PlaneExtractor

from helpers.child_point_recovery import ChildPointRecovery
from helpers.connection_filter import PointCloudConnectionFilter
from helpers.point_cloud_depth_filter import PointCloudDepthFilter
from helpers.point_cloud_alignment import PointCloudAlignment
from helpers.depthmap_from_mesh_generator import DepthmapFromMeshGenerator


def run(dm, pitch, distance):
    """
    Runner for depthmap view angle change.

    :param dm:          source depthmap
    :param pitch:       view angle change of depthmap
    :param distance:    child distance in target depthmP
    :return:            simulated depthmap after view angle change
    """

    # extract normals at pixel positions
    normals = depthmap_to_normals.get_pixel_normals(dm)

    # generate point cloud from depthmap with mask for valid pixel
    pc = PCGenerator(dm).generate_pixelmap()
    bools = np.bitwise_not(np.isnan(dm)).squeeze()

    # extract floor and wall plane from point cloud
    pe = PlaneExtractor(pc, bools, normals, iterations=30, threshold=0.02)
    wall, _, wall_outliers = pe.get_wall()
    floor, floor_points, floor_outliers = pe.get_floor()

    assert floor is not None

    # generate initial child mask
    child = np.logical_and(wall_outliers, floor_outliers)

    # recover points that might be extracted as floor
    pc, child = ChildPointRecovery(pc, floor_points, child).recover()

    # filter outliers by connectivity and depth difference
    pc, child = PointCloudConnectionFilter(pc, child).filter()
    pc, child = PointCloudDepthFilter(pc, child).filter()

    # realign remaining point cloud and extracted floor and wall
    pc, floor, wall = PointCloudAlignment(pc, child, floor, wall).align(distance, pitch)

    # generate a mesh from point cloud and generate a new depthmap by raycasting
    aligned_depthmap = DepthmapFromMeshGenerator(pc, child, floor, wall, bins=5).generate()

    return aligned_depthmap

