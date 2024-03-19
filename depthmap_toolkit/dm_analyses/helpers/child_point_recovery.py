import numpy as np

from multiprocessing import Pool

single_thread = True


class ChildPointRecovery:
    def __init__(self, pc, floor_points, filter_list=None, max_extension=0.05):
        """
        Constructor for Child Point Recovery from floor points.

        Pre-computes distances between child points and floor points. Filter floor that is too far away from the child
        points to be relevant.
        :param pc:              points generated from all pixels of a depth image (shape W * H * 3)
        :param floor_points:    indexes of floor points in pc
        :param filter_list:     optional filter to exclude points from child
        :param max_extension:   maximum distance for floor points to be added
        """
        self.original_pc = pc
        filter_list = filter_list if filter_list is not None else np.ones(pc.shape[:2], dtype=bool)
        self.pc = pc[filter_list]
        self.indexes = np.array([idx for idx, _ in np.ndenumerate(pc[:, :, 0])]).reshape(pc.shape[:2] + tuple([2]))[filter_list]
        self.floor_indexes = np.array([idx for idx, _ in np.ndenumerate(pc[:, :, 0])]).reshape(pc.shape[:2] + tuple([2]))[floor_points]
        self.filter = filter_list
        self.floor_filter = np.zeros(self.original_pc.shape[:2], dtype=bool)
        self.point_count = self.pc.shape[0]
        self.floor_points = pc[floor_points]
        self.recovered = False

        # precompute distances between point cloud and floor points
        if not single_thread:
            with Pool(8) as p:
                self.distances = np.array(p.map(self.get_floor_distances, range(self.point_count)))
        else:
            self.distances = np.array([self.get_floor_distances(i) for i in range(self.point_count)])

        # get possible floor points in range of child point cloud and their distances
        floor_in_range = np.min(self.distances, axis=0) <= max_extension
        self.floor_points = self.floor_points[floor_in_range, :]
        self.floor_indexes = self.floor_indexes[floor_in_range, :]
        self.distances = self.distances[:, floor_in_range]

    def get_floor_distances(self, idx):
        """
        Compute the distance between a single point and all floor points.

        :param idx: index of the point to compute the distances for
        :return:    np array with distance of given point to every floor point
        """
        return np.linalg.norm(self.floor_points - self.pc[idx, :], axis=1)

    def get_distances_in_range(self, idx):
        """
        Compute the distances between all relevant points (i.e. child points near floor and vice versa)

        :param idx: index of the point to compute the distances for
        :return:    np array containing distances
        """
        return np.linalg.norm(self.pc_in_range - self.pc_in_range[idx, :], axis=1)

    def recover(self, neighborhood=0.01):
        """
        Recover the floor points that are near child points.
        Adding point depends on max extension and connection criteria.

        :param neighborhood:    maximum distance between points to be considered connected
        :return:                original point cloud
        :return:                filter mask (indexes) of points contained after recovery
        """
        if not self.recovered:
            # get required child points near floor points
            child_in_range = np.min(self.distances, axis=1) <= neighborhood
            self.pc_in_range = np.vstack([self.pc[child_in_range, :], self.floor_points])

            # compute distance for all remaining point pairs
            if not single_thread:
                with Pool(8) as p:
                    distances_in_range = np.array(p.map(self.get_distances_in_range, range(self.pc_in_range.shape[0])))
            else:
                distances_in_range = np.array([self.get_distances_in_range(i) for i in range(self.pc_in_range.shape[0])])

            # initialize extraction of floor points
            in_range = np.array([True] * np.sum(child_in_range) + [False] * self.floor_points.shape[0])
            old_in_range = np.array([False] * self.pc_in_range.shape[0])

            # recursively add floor points based on connectivity
            while np.any(np.not_equal(in_range, old_in_range)):
                old_in_range = np.copy(in_range)
                in_range = (np.min(distances_in_range[:, old_in_range], axis=1) <= neighborhood)

            # crop in range list to only consist information of floor points
            in_range = in_range[np.sum(child_in_range):]

            # update filter mask for floor points that actually are child points
            self.floor_filter[self.floor_indexes[in_range][:, 0], self.floor_indexes[in_range][:, 1]] = True

        return self.original_pc, np.logical_or(self.filter, self.floor_filter)
