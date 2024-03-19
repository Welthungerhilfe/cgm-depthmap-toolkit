import numpy as np

from multiprocessing import Pool


class PointCloudConnectionFilter:
    def __init__(self, pc, filter_list=None, neighborhood=0.04, single_thread=True):
        """
        Constructor for Point Cloud Connection Filtering.
        Filters a given point cloud by a connected component size. I.e. a mask for all points within a connected
        component of a specified size is created. Points are connected if their distance is less than the specified
        neighborhood threshold.

        Pre-computation of distances between points and connected components.

        :param pc:              points generated from all pixels of a depth image (shape W * H * 3)
        :param filter_list:     optional filter to exclude points from neighborhoods
        :param neighborhood:    maximum distance for points to count as connected
        """
        self.pc_original = pc
        self.pc = np.copy(pc)
        filter_list = filter_list if filter_list is not None else np.ones(pc.shape[:2], dtype=bool)
        self.indexes = np.array([idx for idx, _ in np.ndenumerate(pc[:,:,0])]).reshape(pc.shape[:2] + tuple([2]))[filter_list]
        self.pc = self.pc[filter_list]
        self.filter_list = filter_list
        self.point_count = self.pc.shape[0]
        self.neighborhood = neighborhood

        self.areas = []
        self.not_connected = range(self.point_count)

        # precompute point to point distances for every point pair
        if not single_thread:
            with Pool(8) as p:
                self.distances = np.array(p.map(self.get_distances, range(self.point_count)))
        else:
            self.distances = np.array([self.get_distances(i) for i in range(self.point_count)])

        # compute connected areas sorted by size
        while len(self.not_connected) > 0:
            self.find_next_area()
        self.areas = sorted(self.areas, key=lambda x: len(x), reverse=True)

    def get_distances(self, idx):
        """
        Compute the distance between a single point and all others.

        :param idx: index of the point to compute the distance for
        :return:    np array with distances from selected point to every other point
        """
        return np.linalg.norm(self.pc - self.pc[idx, :], axis=1)

    def find_next_area(self):
        """
        Find a single new connected component from the list of points that were not connected yet

        Starts from an arbitrary point to find all other points that are in one connected component with it. Adds the
        found component to the list of all connected components
        :return: None
        """
        area = set()
        # use first not connected point as seed for next area
        new_points = {self.not_connected[0]}
        while len(new_points) > 0:
            area = area.union(new_points)
            self.not_connected = [c for c in self.not_connected if c not in new_points]

            # create temp set to collect all neighbor points of new points that are not in area yet
            temp = set()

            # extent set by neighbors for every point that are not in the area yet
            for n in new_points:
                p = [i for i in self.not_connected if i not in area and self.distances[i, n] <= self.neighborhood]
                temp = temp.union(p)

            # update new points to temp set
            new_points = set(temp)

        # add connected points to list of areas and update not connected points list
        self.areas.append(list(area))
        self.not_connected = [nc for nc in self.not_connected if nc not in area]

    def filter(self, min_connected_points=5):
        """
        Filters the original point cloud to only contain points with at least the specified amount of neighbors.

        :param min_connected_points:    minimum number of neighbors for point to still be contained
        :return:                        original point cloud
        :return:                        filter mask (indexes) of points still contained after filtering
        """
        area_indexes = np.vstack([self.indexes[a, :] for a in self.areas if len(a) > min_connected_points])
        p_filter = np.zeros(self.pc_original.shape[:2], dtype=bool)
        p_filter[area_indexes[:, 0], area_indexes[:, 1]] = True
        return self.pc_original, p_filter
