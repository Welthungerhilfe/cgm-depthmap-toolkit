from collections import namedtuple
import logging
import zipfile
import math
import sys
import tempfile
from typing import List, Optional, Tuple
from pathlib import Path

from scipy import ndimage
import numpy as np
from PIL import Image

from cgmml.common.depthmap_toolkit.depthmap_utils import (
    matrix_calculate, IDENTITY_MATRIX_4D, parse_numbers, calculate_boundary, matrix_transform_point)
from cgmml.common.depthmap_toolkit.constants import EXTRACTED_DEPTH_FILE_NAME, MASK_FLOOR, MASK_CHILD, MASK_INVALID

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)

NEIGHBOUR_PIXELS_MAX_DISTANCE_IN_METER = 0.05
FLOOR_THRESHOLD_IN_METER = 0.1
SCREEN_BORDER_MARGIN_PERCENTAGE = 0.05
SCREEN_BORDER_PERCENTAGE = 0.025
TOOLKIT_DIR = Path(__file__).parents[0].absolute()


Segment = namedtuple('Segment', 'id aabb')


def extract_depthmap(depthmap_fpath: str, dest_dir: str) -> Path:
    """Extract depthmap from given file"""
    with zipfile.ZipFile(Path(depthmap_fpath), 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    return Path(dest_dir) / EXTRACTED_DEPTH_FILE_NAME


def smoothen_depthmap_array(image_arr: np.ndarray) -> np.ndarray:
    """Smoothen image array by averaging with direct neighbor pixels.

    Args:
        image_arr: shape (width, height)

    Returns:
        shape (width, height)
    """

    # Apply a convolution
    conv_filter = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    conv_filter = conv_filter / conv_filter.sum()
    smooth_image_arr = ndimage.convolve(image_arr, conv_filter)

    # Create mask which is 0 if any of the pixels in the convolution is 0
    smooth_center = (image_arr == 0.)
    smooth_right = np.zeros(image_arr.shape, dtype=bool)
    smooth_left = np.zeros(image_arr.shape, dtype=bool)
    smooth_up = np.zeros(image_arr.shape, dtype=bool)
    smooth_down = np.zeros(image_arr.shape, dtype=bool)
    smooth_right[1:, :] = smooth_center[:-1, :]
    smooth_left[:-1, :] = smooth_center[1:, :]
    smooth_up[:, 1:] = smooth_center[:, :-1]
    smooth_down[:, :-1] = smooth_center[:, 1:]
    mask = (smooth_center | smooth_right | smooth_left | smooth_up | smooth_down)

    # Apply mask
    smooth_image_arr[mask] = 0.

    return smooth_image_arr


class Depthmap:
    """Depthmap and optional RGB

    Args:
        intrinsic (np.array): Camera intrinsic
        width (int): The longer side of the depthmap (and RGB)
        height (int): The shorter side of the depthmap (and RGB)
        data (bytes): pixel_data
        depth_scale (float): Scalar to scale depthmap pixel to meters
        max_confidence (float): Confidence is amount of IR light reflected
                                (e.g. 0 to 255 in Lenovo, new standard is 0 to 7)
                                This is actually an int.
        device_pose (List[float]): The device pose (= position and rotation)
                              The ZIP-file header contains this pose
                              - `device_pose` is a list representation of this pose
                              - can be used to project into a different space
        rgb_fpath (str): Path to RGB file (e.g. to the jpg)
        rgb_array (np.array): RGB data
        header: raw depthmap header with metainformation
    """

    def __init__(
            self,
            intrinsics: np.ndarray,
            width: int,
            height: int,
            data: Optional[bytes],
            depthmap_arr: Optional[np.array],
            depth_scale: float,
            max_confidence: float,
            device_pose: List[float],
            rgb_fpath: Path,
            rgb_array: np.ndarray,
            header: str = None):
        """Constructor

        Either `data` or `depthmap_arr` has to be defined
        """
        self.width = width
        self.height = height

        sensor = 1
        intrinsics = np.array(intrinsics)
        self.fx = intrinsics[sensor, 0] * width
        self.fy = intrinsics[sensor, 1] * height
        self.cx = intrinsics[sensor, 2] * width
        self.cy = intrinsics[sensor, 3] * height

        self.depth_scale = depth_scale
        self.max_confidence = max_confidence
        self.device_pose = device_pose
        if self.device_pose:
            self.device_pose_arr = np.array(device_pose).reshape(4, 4).T
        else:
            self.device_pose_arr = None
        if header:
            self.header = header

        self.rgb_fpath = rgb_fpath
        self.rgb_array = rgb_array  # shape (width, height, 3)  # (240, 180, 3)

        assert depthmap_arr is not None or data is not None
        self.depthmap_arr = self._parse_depth_data(data) if data else depthmap_arr  # (240, 180)

        assert self.depthmap_arr.shape[:2] == (self.width, self.height)
        if self.rgb_array is not None:
            assert self.rgb_array.shape[:2] == (self.width, self.height)

        # smoothing is only for normals, otherwise there is noise
        self.depthmap_arr_smooth = smoothen_depthmap_array(self.depthmap_arr)

        self.confidence_arr = self._parse_confidence_data(data) if data else None

    @property
    def has_rgb(self) -> bool:
        """Bool that indicates if the object has RGB data"""
        return self.rgb_array is not None

    @classmethod
    def create_from_zip_absolute(cls,
                                 depthmap_fpath: str,
                                 rgb_fpath: str,
                                 calibration_fpath: str) -> 'Depthmap':
        width, height, data, depth_scale, max_confidence, device_pose, header_line = (
            Depthmap.read_depthmap_data(depthmap_fpath))
        rgb_array = Depthmap.read_rgb_data(rgb_fpath, width, height)
        intrinsics = parse_calibration(calibration_fpath)
        depthmap_arr = None

        return cls(intrinsics, width, height, data, depthmap_arr,
                   depth_scale, max_confidence, device_pose,
                   rgb_fpath, rgb_array, header_line)

    @staticmethod
    def read_rgb_data(rgb_fpath, width, height):
        if rgb_fpath:
            pil_im = Image.open(rgb_fpath)
            pil_im = pil_im.rotate(-90, expand=True)
            rgb_height, rgb_width = pil_im.width, pil_im.height  # Weird switch
            assert rgb_width / width == rgb_height / height, f'{rgb_width} / {width} != {rgb_height} / {height}'
            pil_im = pil_im.resize((height, width), Image.ANTIALIAS)
            rgb_array = np.asarray(pil_im)
        else:
            rgb_array = None
        return rgb_array

    @staticmethod
    def read_depthmap_data(depthmap_fpath):
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = extract_depthmap(depthmap_fpath, tmpdirname)
            with open(path, 'rb') as f:
                header_line = f.readline().decode().strip()
                width, height, depth_scale, max_confidence, device_pose = Depthmap.parse_header(header_line)
                data = f.read()
                f.close()
        return width, height, data, depth_scale, max_confidence, device_pose, header_line

    @staticmethod
    def parse_header(header_line: str) -> Tuple:
        header_parts = header_line.split('_')
        res = header_parts[0].split('x')
        width = int(res[0])
        height = int(res[1])
        depth_scale = float(header_parts[1])
        max_confidence = float(header_parts[2])
        if len(header_parts) >= 10:
            position = (float(header_parts[7]), float(header_parts[8]), float(header_parts[9]))
            rotation = (float(header_parts[3]), float(header_parts[4]),
                        float(header_parts[5]), float(header_parts[6]))
            if position == (0., 0., 0.):
                logger.warn(f"device_pose looks wrong: position='{position}'")
                device_pose = None
            else:
                device_pose = matrix_calculate(position, rotation)
        else:
            device_pose = IDENTITY_MATRIX_4D
        return width, height, depth_scale, max_confidence, device_pose

    @classmethod
    def create_from_zip(cls,
                        depthmap_dir: str,
                        depthmap_fname: str,
                        rgb_fname: str,
                        calibration_fpath: str) -> 'Depthmap':

        depthmap_path = Path(depthmap_dir) / 'depth' / depthmap_fname
        rgb_fpath = 0
        if rgb_fname:
            rgb_fpath = Path(depthmap_dir) / 'rgb' / rgb_fname
        return cls.create_from_zip_absolute(depthmap_path, rgb_fpath, calibration_fpath)

    def calculate_normalmap_array(self, points_3d_arr: np.ndarray) -> np.ndarray:
        """Calculate normalmap consisting of normal vectors.

        A normal vector is based on a surface.
        The surface is constructed by a 3D point and it's neighbors.

        points_3d_arr: shape (3, width, height)

        Returns:
            3D points: shape (3, width, height)
        """

        # Get depth of the neighbor pixels
        dim_w = self.width - 1
        dim_h = self.height - 1
        depth_center = points_3d_arr[:, 1:, 1:].reshape(3, dim_w * dim_h)
        depth_x_minus = points_3d_arr[:, 0:-1, 1:].reshape(3, dim_w * dim_h)
        depth_y_minus = points_3d_arr[:, 1:, 0:-1].reshape(3, dim_w * dim_h)

        # Calculate a normal of the triangle
        vector_u = depth_center - depth_x_minus
        vector_v = depth_center - depth_y_minus

        normal = np.cross(vector_u, vector_v, axisa=0, axisb=0, axisc=0)

        normal = normalize(normal)

        normal = normal.reshape(3, dim_w, dim_h)

        # add black border to keep the dimensionality
        output = np.zeros((3, self.width, self.height))
        output[:, 1:, 1:] = normal
        return output

    def convert_2d_to_3d(self, x: float, y: float, depth: float) -> np.ndarray:
        """Convert point in pixels into point in meters

        Args:
            x
            y
            depth

        Returns:
            3D point
        """
        tx = (x - self.cx) * depth / self.fx
        ty = (y - self.cy) * depth / self.fy
        return np.array([tx, ty, depth])

    def convert_2d_to_3d_oriented(self, should_smooth: bool = False) -> np.ndarray:
        """Convert points in pixels into points in meters (and applying rotation)

        Args:
            should_smooth: Flag indicating weather to use a smoothed or an un-smoothed depthmap

        Returns:
            array of 3D points: shape(3, width, height)
        """
        depth = self.depthmap_arr_smooth if should_smooth else self.depthmap_arr  # shape: (width, height)

        xbig = np.expand_dims(np.array(range(self.width)), -1).repeat(self.height, axis=1)  # shape: (width, height)
        ybig = np.expand_dims(np.array(range(self.height)), 0).repeat(self.width, axis=0)  # shape: (width, height)

        # Convert point in pixels into point in meters
        tx = depth * (xbig - self.cx) / self.fx
        ty = depth * (ybig - self.cy) / self.fy
        dim4 = np.ones((self.width, self.height))
        res = np.stack([-tx, -ty, depth, dim4], axis=0)

        # Transformation of point by device pose matrix
        points_4d = res.reshape((4, self.width * self.height))
        output = np.matmul(self.device_pose_arr, points_4d)
        output[0:2, :] = output[0:2, :] / abs(output[3, :])
        output = output.reshape((4, self.width, self.height))
        res = output[0:-1]

        # Invert y axis
        res[1, :, :] = -res[1, :, :]
        return res

    def is_child_fully_visible(self, mask: np.array) -> bool:

        # Get the boundary of child and of valid data
        margin = max(self.width, self.height) * SCREEN_BORDER_MARGIN_PERCENTAGE
        child_aabb = calculate_boundary(mask == MASK_CHILD)
        valid_aabb = calculate_boundary(self.depthmap_arr > 0)

        # Check if the child boundary is inside valid boundary
        if child_aabb[0] < valid_aabb[0] + margin:
            return False
        if child_aabb[1] < valid_aabb[1] + margin:
            return False
        if child_aabb[2] > valid_aabb[2] - margin:
            return False
        if child_aabb[3] > valid_aabb[3] - margin:
            return False
        return True

    def segment_child(self, floor: float) -> np.ndarray:
        """Segment the child from the background

        Args:
            floor: At which y-value the floor is

        Returns:
            np.ndarray (int): Each pixel has either
                * -1, -2, -3 (according to number of objects in the scene)
                * MASK_FLOOR = 1
                * MASK_CHILD = 2
                * MASK_INVALID = 3
        """
        mask, segments = self.detect_objects(floor)

        # Select the most focused segment
        closest = sys.maxsize
        focus = -1
        for segment in segments:
            a = segment.aabb[0] - int(self.width / 2)
            b = segment.aabb[1] - int(self.height / 2)
            c = segment.aabb[2] - int(self.width / 2)
            d = segment.aabb[3] - int(self.height / 2)
            distance = a * a + b * b + c * c + d * d
            if closest > distance:
                closest = distance
                focus = segment.id

        mask = np.where(mask == focus, MASK_CHILD, mask)

        assert not (mask == 0).any()  # 0 pixels in output_mask are not allowed

        return mask

    def detect_floor(self, floor: float) -> np.ndarray:
        mask = np.zeros((self.width, self.height))
        assert self.depthmap_arr_smooth.shape == (self.width, self.height)
        mask[self.depthmap_arr_smooth == 0] = MASK_INVALID

        points_3d_arr = self.convert_2d_to_3d_oriented(should_smooth=True)
        cond = (points_3d_arr[1, :, :] - floor) < FLOOR_THRESHOLD_IN_METER
        mask[cond] = MASK_FLOOR
        return mask

    def detect_objects(self, floor: float) -> Tuple[np.array, List[Segment]]:
        """Detect objects/children using seed algorithm

        Can likely not be used without for-loops over x,y

        Args:
            floor: Value of y-coordinate where the floor is

        Returns:
            mask (np.array): binary mask
            List[Segment]: a list of segments
        """
        current_id = -1
        segments = []
        dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        mask = self.detect_floor(floor)
        points_3d_arr = self.convert_2d_to_3d_oriented(should_smooth=False)
        for x in range(self.width):
            for y in range(self.height):
                if mask[x, y] != 0:
                    continue
                pixel = [x, y]
                stack = [pixel]
                while len(stack) > 0:

                    # Get a next pixel from the stack
                    pixel = stack.pop()
                    depth_center = self.depthmap_arr[pixel[0], pixel[1]]

                    # Add neighbor points (if there is no floor and they are connected)
                    if mask[pixel[0], pixel[1]] == 0:
                        if points_3d_arr[1, pixel[0], pixel[1]] - floor > FLOOR_THRESHOLD_IN_METER:
                            for direction in dirs:
                                pixel_dir = [pixel[0] + direction[0], pixel[1] + direction[1]]
                                depth_dir = self.depthmap_arr[pixel_dir[0], pixel_dir[1]]
                                if depth_dir > 0 and abs(
                                        depth_dir - depth_center) < NEIGHBOUR_PIXELS_MAX_DISTANCE_IN_METER:
                                    stack.append(pixel_dir)

                    # Update the mask
                    mask[pixel[0], pixel[1]] = current_id

                # Check if the object size is valid
                aabb = calculate_boundary(mask == current_id)
                object_size_pixels = max(aabb[2] - aabb[0], aabb[3] - aabb[1])
                if object_size_pixels > self.width / 4:
                    segments.append(Segment(current_id, aabb))
                current_id = current_id - 1

        return mask, segments

    def get_distance_of_child_from_camera(self, mask: np.ndarray) -> float:
        """Find the child's distance to the camera using depthmap.

        Args:
            mask: segmentation mask with MASK_FLOOR and MASK_INVALID

        Returns:
            distance in meters
        """
        cond1 = mask != MASK_FLOOR
        cond2 = mask != MASK_INVALID
        depths = self.depthmap_arr[cond1 & cond2]
        if depths.any():
            return min(depths)
        return sys.maxsize

    def get_angle_between_camera_and_floor(self) -> float:
        """Calculate an angle between camera and floor based on device pose

        The angle is often a negative values because the phone is pointing down.

        Angle examples:
        angle=-90deg: The phone's camera is fully facing the floor
        angle=0deg: The horizon is in the center
        angle=90deg: The phone's camera is facing straight up to the sky.
        """
        forward = matrix_transform_point([0, 0, 1], self.device_pose_arr)
        camera = matrix_transform_point([0, 0, 0], self.device_pose_arr)
        return math.degrees(math.asin(camera[1] - forward[1]))

    def get_camera_direction_angle(self) -> float:
        """Calculate an angle of camera direction

        This angle is related to compass directions, it is 0 when the camera started
        """
        forward = matrix_transform_point([0, 0, 1], self.device_pose_arr)
        camera = matrix_transform_point([0, 0, 0], self.device_pose_arr)
        return math.degrees(math.atan2(camera[0] - forward[0], camera[2] - forward[2])) - 90.

    def get_floor_level(self) -> float:
        """Calculate an altitude of the floor in the world coordinates"""

        # Get normal vectors
        mask = np.zeros((self.width, self.height))
        assert self.depthmap_arr_smooth.shape == (self.width, self.height)
        mask[self.depthmap_arr_smooth == 0] = MASK_INVALID
        points_3d_arr = self.convert_2d_to_3d_oriented(should_smooth=True)
        normal = self.calculate_normalmap_array(points_3d_arr)

        cond = np.abs(normal[1, :, :]) > 0.5
        selection_of_points = points_3d_arr[1, :, :][cond]
        median = np.median(selection_of_points)
        return median

    def get_highest_point(self, mask: np.ndarray) -> np.ndarray:
        points_3d_arr = self.convert_2d_to_3d_oriented()
        y_array = np.copy(points_3d_arr[1, :, :])
        y_array[mask != MASK_CHILD] = -np.inf
        idx_highest_child_point = np.unravel_index(np.argmax(y_array, axis=None), y_array.shape)
        highest_point = points_3d_arr[:, idx_highest_child_point[0], idx_highest_child_point[1]]
        return highest_point

    def resize(self, new_width: int, new_height: int):
        """Rescale calibration and depthmap"""

        # Rescale calibration
        scale_x = float(new_width) / float(self.width)
        scale_y = float(new_height) / float(self.height)
        self.cx = (self.cx - float(self.width) * 0.5) * scale_x + float(new_width) * 0.5
        self.cy = (self.cy - float(self.height) * 0.5) * scale_y + float(new_height) * 0.5
        self.fx *= scale_x
        self.fy *= scale_y

        # Mapping from new coordinates to original coordinates (e.g. xbig[new_width / 2] = self.width / 2)
        xbig = np.expand_dims(np.array(range(int(new_width))), -1).repeat(int(new_height), axis=1) / scale_x
        ybig = np.expand_dims(np.array(range(int(new_height))), 0).repeat(int(new_width), axis=0) / scale_y
        xbig[xbig + 1 >= self.width - 1] = self.width - 2
        ybig[ybig + 1 >= self.height - 1] = self.height - 2

        # Get depth information to interpolate
        d00 = self.depthmap_arr[xbig.astype(int), ybig.astype(int)]
        d10 = self.depthmap_arr[xbig.astype(int) + 1, ybig.astype(int)]
        d01 = self.depthmap_arr[xbig.astype(int), ybig.astype(int) + 1]
        d11 = self.depthmap_arr[xbig.astype(int) + 1, ybig.astype(int) + 1]

        # Bilinear interpolation of the depth data
        mix_x = xbig - xbig.astype(int)
        mix_y = ybig - ybig.astype(int)
        a00 = (1. - mix_x) * (1. - mix_y)
        a10 = mix_x * (1. - mix_y)
        a01 = (1. - mix_x) * mix_y
        a11 = mix_x * mix_y
        self.depthmap_arr = (d00 * a00 + d11 * a11 + d01 * a01 + d10 * a10) / (a00 + a10 + a01 + a11)

        # Mask depth interpolations which are obviously not a connected surface
        self.depthmap_arr[abs(d00 - d10) > NEIGHBOUR_PIXELS_MAX_DISTANCE_IN_METER] = 0
        self.depthmap_arr[abs(d00 - d01) > NEIGHBOUR_PIXELS_MAX_DISTANCE_IN_METER] = 0
        self.depthmap_arr[abs(d00 - d11) > NEIGHBOUR_PIXELS_MAX_DISTANCE_IN_METER] = 0
        self.depthmap_arr[abs(d10 - d01) > NEIGHBOUR_PIXELS_MAX_DISTANCE_IN_METER] = 0
        self.depthmap_arr[abs(d10 - d11) > NEIGHBOUR_PIXELS_MAX_DISTANCE_IN_METER] = 0
        self.depthmap_arr[abs(d01 - d11) > NEIGHBOUR_PIXELS_MAX_DISTANCE_IN_METER] = 0

        # Apply new resolution
        self.width = int(new_width)
        self.height = int(new_height)
        self.confidence_arr = None
        self.depthmap_arr_smooth = smoothen_depthmap_array(self.depthmap_arr)

    def _parse_confidence_data(self, data) -> np.ndarray:
        """Parse depthmap confidence

        Returns:
            2D array of floats
        """
        output = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                output[x, y] = self._parse_confidence(data, x, y)
        return output

    def _parse_confidence(self, data: bytes, tx: int, ty) -> float:
        """Get confidence of the point in scale 0-1"""
        index = self.height - int(ty) - 1
        return data[(index * self.width + int(tx)) * 3 + 2] / self.max_confidence

    def _parse_depth_data(self, data) -> np.ndarray:
        output = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                output[x, y] = self._parse_depth(data, x, y)
        return output

    def _parse_depth(self, data: bytes, tx: int, ty: int) -> float:
        """Get depth of the point in meters"""
        if tx < 1 or ty < 1 or tx >= self.width or ty >= self.height:
            return 0.
        index = self.height - int(ty) - 1
        depth = data[(index * self.width + int(tx)) * 3 + 0] << 8
        depth += data[(index * self.width + int(tx)) * 3 + 1]
        depth *= self.depth_scale
        return depth


def parse_calibration(filepath: str) -> List[List[float]]:
    """Parse calibration file
    filepath: The content of a calibration file looks like this:
        Color camera intrinsic:
        0.6786797 0.90489584 0.49585155 0.5035042
        Depth camera intrinsic:
        0.6786797 0.90489584 0.49585155 0.5035042
    """
    with open(filepath, 'r') as f:
        calibration = []
        for _ in range(2):
            f.readline().strip()
            line_with_numbers = f.readline()
            intrinsic = parse_numbers(line_with_numbers)
            calibration.append(intrinsic)
    return calibration


def is_google_tango_resolution(width, height):
    """Check for special case for Google Tango devices with different rotation"""
    return width == 180 and height == 135


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Ensure the normal has a length of one

    This way of normalizing is commonly used for normals.
    It achieves that normals are of size 1.

    Args:
        vectors (np.array): Multiple vectors (e.g. could be normals)

    Returns:
        This achieves: abs(x) + abs(y) + abs(z) = 1
    """
    length = abs(vectors[0]) + abs(vectors[1]) + abs(vectors[2])
    length[length == 0] = 1
    return vectors / length
