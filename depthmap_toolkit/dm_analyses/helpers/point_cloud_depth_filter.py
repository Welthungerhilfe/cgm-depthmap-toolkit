import numpy as np


class PointCloudDepthFilter:
    def __init__(self, pc, filter_list):
        """
        Constructor of point cloud filter based on point depth.
        :param pc:          point cloud to filter
        :param filter_list: pre-filter for child points
        """
        self.pc = pc
        self.filter_list = filter_list

    def filter(self, max_z_diff=0.2):
        """
        Filter point cloud by given depth difference.
        :param max_z_diff:  maximum allowed z difference
        :return:            original point cloud, new filter mask
        """
        median = np.median(self.pc[self.filter_list], axis=0)[2]
        z = self.pc[:, :, 2]

        return self.pc, np.logical_and(np.abs(z - median) < max_z_diff, self.filter_list)
