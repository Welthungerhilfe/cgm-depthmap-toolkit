from copy import deepcopy
import numpy as np


class Model:
    min_samples = None

    def __init__(self):
        self.min_samples = None

    def fit(self, samples):
        """
        Fit this plane to the given point sample
        :param samples:     numpy array containing 3 points
        """
        pass

    def predict(self, X):
        """
        Predict the distances of the given point set
        :param X:   numpy array containing points
        :return:    distance for each given point
        """
        return float("inf")


class PlaneModel(Model):
    """
    Class to describe a mathematical plane
    """
    min_samples = 3

    def __init__(self):
        super().__init__()
        self.min_samples = 3
        self.origin = None
        self.n = None

    def fit(self, samples):
        """
        Fit this plane to the given point sample
        :param samples:     numpy array containing 3 points
        """
        self.origin = samples[0, :]
        v1 = samples[1, :] - self.origin
        v2 = samples[2, :] - self.origin
        #v3 = samples[2, :] - samples[1, :]

        self.n = np.cross(v1, v2)
        self.n = self.n / np.linalg.norm(self.n)

    def predict(self, X):
        """
        Predict the distances of the given point set
        :param X:   numpy array containing points
        :return:    distance for each given point
        """
        if self.n is None or self.origin is None:
            return np.ones(X.shape[:-1]) * float("inf")

        shape = X.shape
        X = X.reshape([-1, 3])
        return np.array([np.dot(self.n, X[i, :] - self.origin) for i in range(X.shape[0])]).reshape(shape[:-1])


def ransac(X, threshold=0.1, num_iterations=50, *, include=None, model_class=PlaneModel, sample_criterions=None,
           expected_normal=None, normal_threshold=(3 ** .5) / 2):
    """
    Executes RANSAC for fitting a (plane) model onto a point set.
    :param X:                   list of points the model shall be fit onto
    :param threshold:           threshold for error to distinguish outliers
    :param num_iterations:      number of repetitions of fitting
    :param include:             additional filter for the points
    :param model_class:         model to fit
    :param sample_criterions:   list of filter to single sample draws
    :param expected_normal:     expected result for normal
    :param normal_threshold:    cosine of the maximum accepted angular deviation for the normal
    :return:
    """
    best_model = None
    best_inlier_count = 0
    best_inliers = np.array([])
    best_outliers = np.array([])

    # init filters with defaults if not defined or too short
    if include is None:
        include = np.ones(X.shape[:-1], dtype=bool)

    if sample_criterions is None:
        sample_criterions = []

    while len(sample_criterions) < PlaneModel.min_samples:
        sample_criterions.append(np.ones(X.shape[:-1], dtype=bool))

    iteration = 0
    backup_iterations = 0

    while iteration < num_iterations and backup_iterations < 2 * num_iterations:
        backup_iterations += 1
        model = model_class()

        # Randomly sample a subset based on given criteria
        sample = []
        for s in range(model_class.min_samples):
            idx = np.random.choice(X[include & sample_criterions[s]].size // 3, 1, replace=False)
            sample.append(X[include & sample_criterions[s]][idx, :])

        sample = np.vstack(sample)

        # Fit a model to the random sample
        model.fit(sample)

        if np.linalg.norm(model.n) == 0:
            continue

        if expected_normal is not None and np.dot(model.n, expected_normal) < normal_threshold:
            continue

        iteration += 1

        # Calculate the error for all points
        y_pred = model.predict(X)
        errors = np.abs(y_pred)

        # Identify inliers based on the threshold
        inliers = errors < threshold

        # Update the best model if the current one is better
        if best_inlier_count < np.sum(inliers):
            best_inlier_count = np.sum(inliers)
            best_model = model
            best_inliers = inliers
            best_outliers = X[np.logical_not(inliers)]

    return best_model, best_inliers, best_outliers


class PlaneExtractor:
    """
    Class for plane extractions from point clouds
    """
    def __init__(self, pc, normals, *, include=None, threshold=0.1, iterations=30):
        """
        :param pc:          point cloud
        :param include:     filter for points that shall be used
        :param normals:     surface normals at point locations (same shape as pc)
        :param threshold:   distance threshold from point to plane in cm to distinguish outliers
        :param iterations:  number of repetitions of plane fitting
        """
        self.pc = deepcopy(pc)
        self.out = deepcopy(pc)

        if include is None:
            self.include = np.ones(normals.shape[:2], dtype='bool')
        else:
            self.include = include

        self.shape = pc.shape[:2]

        self.normals = normals
        self.possible_floor = np.abs(self.normals[:, :, 1]) > 2 ** -.5

        self.threshold = threshold
        self.iterations = iterations

    def get_plane(self, selection):
        """
        Run plane fitting (ransac) on given pixel selection
        :param selection: pixel mask used for plane fitting
        """

        if np.count_nonzero(selection) == 0:
            print('error empty selection provided')
            return

        # use ransac to get optimal fitting plane
        floor_plane, inliers, _ = ransac(self.pc, self.threshold, self.iterations, include=selection,
                                         model_class=PlaneModel, 
                                         expected_normal=None)
                                         
        outliers = np.logical_and(self.include, np.logical_not(inliers))

        return floor_plane, inliers, outliers

        """
        Get floor from point cloud self.pc
        :return: PlaneModel for wall, list of points in plane, list of points out of plane
        """
        # get the surface normals for the lower 15% of the image
        lower_normals = self.normals[int(.85*self.shape[0]):, :, :].reshape(-1, 3)
        lower_normals = lower_normals[np.logical_not(np.any(np.isnan(lower_normals), axis=1)), :]

        # compute the average normal of lower region as estimate for the floor normal
        average_normal = np.average(lower_normals, axis=0)
        average_normal = average_normal / np.linalg.norm(average_normal)
        
            # filter points if angle between surface normal and average normal is greater than 30°
            possible_floor = np.abs(np.dot(self.normals, average_normal)) > 3 ** .5 / 2
            possible_floor[:int(.5 * self.shape[0]), :] = False

            # combine point filters
            include = np.bitwise_and(self.include, possible_floor)

        # define sampling criteria
        # 1st: bottom left corner of image
        # 2nd: bottom right corner of image
        # 3rd: bottom half of image
        crit1 = lambda idx : idx[0] > self.pc.shape[0] * .75 and idx[1] < self.pc.shape[1] * 0.25 and idx[0] % 2 == 0 and idx[1] % 2 == 0
        crit1_np = np.array([crit1(idx) for idx, _ in np.ndenumerate(self.pc[:, :, 0])]).reshape(self.pc.shape[:-1])
        crit2 = lambda idx : idx[0] > self.pc.shape[0] * .75 and idx[1] > self.pc.shape[1] * 0.75 and idx[0] % 2 == 0 and idx[1] % 2 == 0
        crit2_np = np.array([crit2(idx) for idx, _ in np.ndenumerate(self.pc[:, :, 0])]).reshape(self.pc.shape[:-1])
        crit3 = lambda idx : idx[0] > self.pc.shape[0] * .5 and idx[0] % 2 == 0 and idx[1] % 2 == 0
        crit3_np = np.array([crit3(idx) for idx, _ in np.ndenumerate(self.pc[:, :, 0])]).reshape(self.pc.shape[:-1])

        # use ransac to get optimal fitting plane
        floor_plane, inliers, _ = ransac(self.pc, self.threshold, self.iterations, include=include,
                                         model_class=PlaneModel, expected_normal=average_normal,
                                         sample_criterions=[crit1_np, crit2_np, crit3_np])
        outliers = np.logical_and(self.include, np.logical_not(inliers))

        return floor_plane, inliers, outliers

    def get_wall(self):
        """
        Get wall from point cloud self.pc
        :return: PlaneModel for wall, list of points in plane, list of points out of plane
        """
        # get the surface normals for the upper 15% of the image
        upper_normals = self.normals[:int(.15*self.shape[0]), :, :].reshape(-1, 3)
        upper_normals = upper_normals[np.logical_not(np.any(np.isnan(upper_normals), axis=1)), :]

        # compute the average normal of upper region as estimate for the wall normal
        average_normal = np.average(upper_normals, axis=0)
        average_normal = average_normal / np.linalg.norm(average_normal)

             # filter points if angle between surface normal and average normal is greater than 30°
            possible_wall = np.abs(np.dot(self.normals, average_normal)) > 3 ** .5 / 2
            possible_wall[int(.5 * self.shape[0]):, :] = False

            # combine point filters
            include = np.bitwise_and(self.include, possible_wall)

        # define sampling criteria
        # 1st: upper left corner of image
        # 2nd: upper right corner of image
        # 3rd: upper half of image
        crit1 = lambda idx : idx[0] < self.pc.shape[0] * .25 and idx[1] < self.pc.shape[1] * 0.25 and idx[0] % 2 == 0 and idx[1] % 2 == 0
        crit1_np = np.array([crit1(idx) for idx, _ in np.ndenumerate(self.pc[:, :, 0])]).reshape(self.pc.shape[:-1])
        crit2 = lambda idx : idx[0] < self.pc.shape[0] * .25 and idx[1] > self.pc.shape[1] * 0.75 and idx[0] % 2 == 0 and idx[1] % 2 == 0
        crit2_np = np.array([crit2(idx) for idx, _ in np.ndenumerate(self.pc[:, :, 0])]).reshape(self.pc.shape[:-1])
        crit3 = lambda idx : idx[0] < self.pc.shape[0] * .5 and idx[0] % 2 == 0 and idx[1] % 2 == 0
        crit3_np = np.array([crit3(idx) for idx, _ in np.ndenumerate(self.pc[:, :, 0])]).reshape(self.pc.shape[:-1])

        # use ransac to get optimal fitting plane
        wall_plane, inliers, _ = ransac(self.pc, self.threshold, self.iterations, include=include,
                                        model_class=PlaneModel, expected_normal=average_normal,
                                        sample_criterions=[crit1_np, crit2_np, crit3_np])
        outliers = np.logical_and(self.include, np.logical_not(inliers))

        return wall_plane, inliers, outliers
