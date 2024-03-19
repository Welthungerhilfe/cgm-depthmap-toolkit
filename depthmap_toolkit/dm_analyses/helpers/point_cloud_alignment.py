from copy import deepcopy
import math
import numpy as np


def get_pitch_matrix(pitch):
    """
    Converts a pitch angle to a ratation matrix.
    :param pitch: pitch angle (in degrees)
    :return: rotation matrix
    """
    rot_angle = math.radians(pitch)
    return np.array([
        [1, 0, 0],
        [0, math.cos(rot_angle), -math.sin(rot_angle)],
        [0, math.sin(rot_angle), math.cos(rot_angle)]
    ])


class PointCloudAlignment:
    def __init__(self, pc, filter, floor, wall):
        """
        Constructor pipeline that aligns child point cloud and floor and wall planes.
        :param pc:      point cloud of all points (shape HxWx3, H and W being the height and width of the depth map)
        :param filter:  filter defining points that are part of the child
        :param floor:   floor plane
        :param wall:    wall plane
        """
        self.original_pc = pc
        self.pc = pc[filter]
        self.floor = floor
        self.wall = wall

    def align(self, distance, pitch=0):
        """
        Aligning point cloud and planes with given parameters.
        :param distance:    target distance to the child's average point
        :param pitch:       change of pitch that shall be applied
        :return:            updated child point cloud, updated floor plane, updated wall plane
        """
        # calculate rotation matrix
        rot_mat = get_pitch_matrix(pitch)

        # get average point of child after rotation
        average = rot_mat @ np.average(self.pc, axis=0)

        # get offset for moving everything
        v = np.array([0, 0, distance]) - average

        # copy wall and apply transformation
        moved_wall = deepcopy(self.wall)
        moved_wall.origin = v + (rot_mat @ self.wall.origin)
        moved_wall.n = rot_mat @ self.wall.n

        # copy floor and apply transformation
        moved_floor = deepcopy(self.floor)
        moved_floor.origin = v + (rot_mat @ self.floor.origin)
        moved_floor.n = rot_mat @ self.floor.n

        # apply transformation to point cloud
        original_pc_list = self.original_pc.reshape(-1, 3)
        moved_original_pc = v + np.array([(rot_mat @ original_pc_list[i, :].reshape(3, 1)) for i in range(original_pc_list.shape[0])]).reshape(self.original_pc.shape)

        # return moved_pc, moved_floor, moved_wall
        return moved_original_pc, moved_floor, moved_wall
