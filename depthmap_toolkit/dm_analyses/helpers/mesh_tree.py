import numpy as np

from helpers.pc_to_mesh import generate_mesh_from_2d_point_cloud


def to_angles(pc, z):
    """
    Calculate angles of points in point cloud similar to polar coordinates rotation part.
    :param pc:  point cloud's points without third coordinate
    :param z:   third coordinate of point cloud's points
    :return:    angle to z-axis split into x and y part
    """
    return np.degrees(np.arctan(pc / z))


def get_optimal_borders(pc, bins, angular):
    """
    Calculate mesh tree bin borders based on point cloud.
    :param pc:      point cloud used to calculate mesh tree borders
    :param bins:    number of bins per axis
    :param angular: True if the borders intersect coordinate origin, else borders are parallel to axes
    :return:        list with border angles or offsets
    """
    values = pc

    # convert polar angles if angular borders are requested
    if angular:
        values = to_angles(pc[:, :2], pc[:, 2].reshape([pc.shape[0], 1]))

    # get min and max values of coordinates
    max_v = np.max(values, axis=0)
    min_v = np.min(values, axis=0)

    # equal split of area
    return [[min_v[i] + (max_v[i] - min_v[i]) / bins[i] * n for n in range(1, bins[i])] for i in range(2)]


class MeshTree:
    def __init__(self, pc, filter, borders, *, margin=0, angular=False, max_bin_points=-1, max_recursions=0):
        """
        Constructor of Mesh Tree.
        Recursively creates a tree from a given point cloud with given bin borders per axis on each layer until
        recursion anchors are reached. A margin can be given to make the bins overlap.
        :param pc:              point cloud to generate tree from
        :param filter:          indices for points that shall be used
        :param borders:         values of borders
        :param margin:          margin for each bin
        :param angular:         True if split bins depending on angle to z-axis
        :param max_bin_points:  maximum number of points per bin as recursion anchor, inactive if less than 0
        :param max_recursions:  maximum number of recursions
        """
        self.pc = pc
        self.borders = [[float("-inf")] + b + [float("inf")] for b in borders]
        self.margin = margin
        self.angular = angular

        # calculate the points offset by the given margin
        x_l = self.pc[:, :, 0].reshape([-1, 1]) - margin
        x_g = self.pc[:, :, 0].reshape([-1, 1]) + margin
        y_l = self.pc[:, :, 1].reshape([-1, 1]) - margin
        y_g = self.pc[:, :, 1].reshape([-1, 1]) + margin

        # generate lists with upper and lower bin borders
        borders_l_x = np.array(self.borders[0])[1:].reshape((1, len(self.borders[0]) - 1))
        borders_l_y = np.array(self.borders[1])[1:].reshape((1, len(self.borders[1]) - 1))
        borders_g_x = np.array(self.borders[0])[:-1].reshape((1, len(self.borders[0]) - 1))
        borders_g_y = np.array(self.borders[1])[:-1].reshape((1, len(self.borders[1]) - 1))

        z = pc[:, :, 2].reshape([-1, 1])

        # check in which bins the points are present
        x_intervals_l = np.less_equal(to_angles(x_l, z) if angular else x_l, borders_l_x)
        x_intervals_g = np.greater_equal(to_angles(x_g, z) if angular else x_g, borders_g_x)
        y_intervals_l = np.less_equal(to_angles(y_l, z) if angular else y_l, borders_l_y)
        y_intervals_g = np.greater_equal(to_angles(y_g, z) if angular else y_g, borders_g_y)

        x_intervals = (x_intervals_l & x_intervals_g).reshape(self.pc.shape[:2] + x_intervals_g.shape[-1:])
        y_intervals = (y_intervals_l & y_intervals_g).reshape(self.pc.shape[:2] + y_intervals_g.shape[-1:])

        # create filters for each bin
        bins = [len(self.borders[0]) - 1, len(self.borders[1]) - 1]
        self.intervals = np.array([[np.bitwise_and(x_intervals[:, :, x], y_intervals[:, :, y]) for x in range(bins[0])] for y in range(bins[1])])
        self.bin_filters = np.array([np.bitwise_and(i, filter) for i in [self.intervals[*i, :, :] for i, _ in np.ndenumerate(self.intervals[:,:,0,0])]]).reshape(*bins[::-1], *self.pc.shape[:2])

        # get number of points per bin for recursion anchor
        self.bin_points = np.sum(self.bin_filters.reshape(*bins[::-1], -1), axis=2)

        # generate mesh for every bin
        self.intervals = np.array([set(generate_mesh_from_2d_point_cloud(self.pc, f)) for f in [self.bin_filters[*i, :, :] for i, _ in np.ndenumerate(self.bin_filters[:,:,0,0])]]).reshape(bins[1], bins[0])

        self.full_bin_idxs = []

        # check recursion anchor
        if max_bin_points > 0 and max_recursions > 0:
            # get indices of bins that require next recursion step
            self.full_bin_idxs = [idx for idx, bin in np.ndenumerate(self.bin_points) if pc.shape[0] > bin > max_bin_points]

            # do the next recursion step
            full_bin_filters = [self.bin_filters[*idx] for idx in self.full_bin_idxs]
            full_bin_borders = [get_optimal_borders(self.pc[f], bins, angular) for f in full_bin_filters]
            self.full_bins = [MeshTree(self.pc, f, b, margin=margin, angular=angular, max_bin_points=max_bin_points, max_recursions=max_recursions-1) for f, b in zip(full_bin_filters, full_bin_borders)]

    def get_interval_by_point(self, point):
        """
        Get the bin in which the given point would be.

        I.e. the bin the ray would hit during ray casting. Recursively goes through the tree if necessary.
        :param point:   if angular the angles of the ray, else the x- and y-coordinate of a point
        :return:        the requested bin
        """
        # get the bin index
        idx = [[i for i, (l, g) in enumerate(zip(self.borders[j][:-1], self.borders[j][1:])) if l <= p <= g][0] for j, p in enumerate(point)][::-1]

        # check if recursive call is necessary
        if idx in self.full_bin_idxs:
            bin = [n for n, i in self.full_bin_idxs if idx == i][0]
            return self.full_bins[bin].get_interval_by_point(point)

        return self.intervals[*idx]

    #def get_intervals(self):
    #    """
    #    Get list of bins with all points that are included in it.
    #    :return:
    #    """
    #    return [self.pc[self.intervals[i][j]] for i in range(len(self.intervals)) for j in range(len(self.intervals[i]))]
