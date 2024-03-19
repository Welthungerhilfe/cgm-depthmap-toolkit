import numpy as np


def get_pixel_normals(dm, fov=(42.5, 54.8)):
    """
    Calculate the surface normals from a depth map

    Returns 0 for image border normals
    :param dm:      depth map to calculate surface normals from
    :param fov:     field of view of the camera used to record the depth map
    :return:        2d array containing the surface normals at the pixel locations of the depth map
    """
    shape = dm.shape[:2] + (3,)
    pixel_fov = (fov[1] / shape[0], fov[0] / shape[1])

    # initialize normals
    pixel_normals = np.zeros(shape)
    pixel_normals_inner = np.ones((shape[0] - 2, shape[1] - 2, 3))

    # get the in plane spacing at the depth value for each pixel
    x_fov = dm[1:-1, 1:-1] * np.tan(np.radians(pixel_fov[1]))
    y_fov = dm[1:-1, 1:-1] * np.tan(np.radians(pixel_fov[0]))

    # compute the surface directions
    pixel_normals_inner[:, :, 0] = (-(dm[1:-1, 2:] - dm[1:-1, :-2]) / x_fov).squeeze() / 2
    pixel_normals_inner[:, :, 1] = (-(dm[2:, 1:-1] - dm[:-2, 1:-1]) / y_fov).squeeze() / 2
    pixel_normals_inner[:, :, 2] = 1

    # normalize the directions
    norms = np.linalg.norm(pixel_normals_inner, axis=2).reshape((shape[0] - 2, shape[1] - 2, 1))
    pixel_normals_inner = pixel_normals_inner / norms

    # assign the inner normals to the return value
    pixel_normals[1:-1, 1:-1, :] = pixel_normals_inner

    return pixel_normals
