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


class PCGenerator:
    """
    Class that provides depthmap to point cloud conversion
    """
    def __init__(self, depthmap, fov=(42.5, 54.8), pitch=0):
        """
        :param depthmap:    depth map to convert
        :param fov:         field of view angles of recording camera
        :param pitch:       pitch angle of the recording
        """
        self.depthmap = depthmap.squeeze()
        self.img_size = depthmap.shape[:2]
        self.fov = fov
        self.pitch = pitch

    def depth_pixel_to_point(self, value, coords):
        """
        Converts single depth pixel to a point.
        :param value:   pixel value in depth map
        :param coords:  pixel coordinate in depth map
        :return:
        """
        x = coords[1]
        y = coords[0]
        z = value

        fov_x = math.radians(self.fov[0])
        fov_y = math.radians(self.fov[1])

        if fov_x > 0 and fov_y > 0:
            w = self.img_size[1]
            h = self.img_size[0]

            # compute "sensor lense"-distance
            img_d_x = (w / 2) / math.sin(fov_x / 2) * math.sin(math.radians(90) - fov_x / 2)
            img_d_y = (h / 2) / math.sin(fov_y / 2) * math.sin(math.radians(90) - fov_y / 2)

            # compute pixel projection angle
            theta_x = math.atan2((x - w / 2), img_d_x)
            theta_y = math.atan2((y - h / 2), img_d_y)

            # apply angle to distance for x and y
            x = value * math.tan(theta_x)
            y = value * math.tan(theta_y)

        return x, y, z

    def generate(self, pixel_list=None):
        """
        Convert the depth map into a list of points.

        If pixel_list is given only converts the pixels with contained indices .
        :param pixel_list:  pixel indices to be converted (if None, all pixels are converted)
        :return:    list of points converted from the depth map pixels
        """
        if pixel_list is None:
            pc = [self.depth_pixel_to_point(pix, idx) for idx, pix in np.ndenumerate(self.depthmap) if 3 > pix > 0]
        else:
            pc = [self.depth_pixel_to_point(pix, idx) for idx, pix in np.ndenumerate(self.depthmap) if 3 > pix > 0 if idx[::-1] in pixel_list]

        pc = np.array(pc)

        if self.pitch != 0:
            pc = np.array([get_pitch_matrix(self.pitch) @ pc[i, :] for i in range(pc.shape[0])])

        return pc

    def generate_pixelmap(self):
        """
        Convert the depth map into a 2d list of points.

        The shape of the list is equal to the depth map's shape.
        :return:
        """
        shape = list(self.depthmap.shape) + [3]
        pc = np.zeros(shape)

        for idx, pix in np.ndenumerate(self.depthmap):
            if 0 < pix < 3:
                pc[idx[0], idx[1], :] = self.depth_pixel_to_point(pix, idx)

        if self.pitch != 0:
            pc = np.array([get_pitch_matrix(self.pitch) @ pc[idx[0], idx[1], :] for idx, _ in np.ndenumerate(pc[:, :, 0])]).reshape(shape)

        return pc
