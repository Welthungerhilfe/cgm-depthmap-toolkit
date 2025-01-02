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
    def __init__(self, depthmap, camera_parameters=(42.5, 54.8), pitch=0):
        """
        :param depthmap:    depth map to convert
        :param camera_parameters:  
            if 2 values : field of view angles of recording camera
            if 4 values : camera calibration (should be the preferred option)
        :param pitch:       pitch angle of the recording
        """
        self.depthmap = depthmap.squeeze()
        self.img_size = depthmap.shape[:2]
        self.pitch = pitch
        self.fov = (-1,-1)
        self.fy = -1
        self.fx = -1
        self.cy = -1
        self.cx = -1

        # update parameters 
        if len(camera_parameters) == 2:
            self.set_fov(camera_parameters)
        elif len(camera_parameters) == 4:
            self.set_intrinsics(camera_parameters)
        else:
            print('Error: invalid camera parameters provided')
            return        

    def set_fov(self, fov):
        self.fov = fov
        self.fy, self.fx, self.cy, self.cx = self.compute_focal_length_from_fov()

    def set_intrinsics(self, intrinsics):
        """ set intrinsic parameters"""
        fx, fy, cx, cy = intrinsics

        # scale intrinsics to correct depth image size
        height, width = self.img_size
        self.fx = fx * width 
        self.fy = fy * height
        self.cx = cx * width
        self.cy = cy * height

        self.compute_fov_from_calibration()

    def compute_fov_from_calibration(self):
        # Compute horizontal and vertical FOV in degrees
        fov_x = 2 * np.degrees(np.arctan(self.img_size[1] / (2 * self.fx)))
        fov_y = 2 * np.degrees(np.arctan(self.img_size[0] / (2 * self.fy)))

        self.fov = (fov_y, fov_x)
    
    def compute_focal_length_from_fov(self):
        height, width = self.img_size
        # Convert FOV from degrees to radians
        fov_x_rad = np.radians(self.fov[1])
        fov_y_rad = np.radians(self.fov[0])
        
        # Calculate focal lengths for x and y directions
        cx = (width / 2)
        cy = (height / 2)
        fx = cx / np.tan(fov_x_rad / 2)
        fy = cy / np.tan(fov_y_rad / 2)
        
        return fy, fx, cy, cx

    def depth_pixel_to_point_rs(self, value, coords):
        """ 
        deprojects depth pixel to 3d point the same way as in realsense code 
        but without applying a distortion model for now

        :param value: depth map value form outside mus match
        :param coords: pixel coordinates
        :return: x,y,z coordinates for 3d point
        """
        # Calculate the 3D coordinates
        depth = self.depthmap[coords]
        if value != depth:
            print("value does not match internal depthmap")

        y = (coords[0] - self.cy) / self.fy
        x = (coords[1] - self.cx) / self.fx

        #TODO add distortion models rs2_deproject_pixel_to_point

        x = depth * x
        y = depth * y
        z = depth

        # Stack to create an Nx3 array of 3D points
        # points_3d = np.stack((X, Y, Z), axis=-1)
        return x, y, z

    # TODO use internal depthmap of pont cloud
    def depth_pixel_to_point(self, value, coords):
        """
        Converts single depth pixel to a point.
        :param value:   pixel value in depth map
        :param coords:  pixel coordinate in depth map
        :return:
        """
        x = coords[1]
        y = coords[0]
        z = self.depthmap[y,x]
        if value != z:
            print("value does not match internal depthmap")

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
            pc = [self.depth_pixel_to_point_rs(pix, idx) for idx, pix in np.ndenumerate(self.depthmap) ]
        else:
            pc = [self.depth_pixel_to_point_rs(pix, idx) for idx, pix in np.ndenumerate(self.depthmap) if idx[::-1] in pixel_list]

        pc = np.array(pc)

        if self.pitch != 0:
            pc = np.array([get_pitch_matrix(self.pitch) @ pc[i, :] for i in range(pc.shape[0])])

        return pc

    def generate_pixelmap(self, near=0, far=4000):
        """
        Convert the depth map into a 2d list of points.

        The shape of the list is equal to the depth map's shape.
        :param near: shortest distance of considert data
        :param far: longest distance of considert data
        :return: map of 3d points converted from the depth map pixels
        """
        shape = list(self.depthmap.shape) + [3]
        pc = np.zeros(shape)

        for idx, pix in np.ndenumerate(self.depthmap):
            if near < pix < far:
                pc[idx[0], idx[1], :] = self.depth_pixel_to_point_rs(pix, idx)

        if self.pitch != 0:
            pc = np.array([get_pitch_matrix(self.pitch) @ pc[idx[0], idx[1], :] for idx, _ in np.ndenumerate(pc[:, :, 0])]).reshape(shape)

        return pc
