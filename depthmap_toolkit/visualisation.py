import logging

import numpy as np

from cgmml.common.depthmap_toolkit.constants import MASK_CHILD, MASK_FLOOR
from cgmml.common.depthmap_toolkit.depthmap import Depthmap
from cgmml.common.depthmap_toolkit.depthmap_utils import calculate_boundary, get_smoothed_pixel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)

CHILD_HEAD_HEIGHT_IN_METERS = 0.3
PATTERN_LENGTH_IN_METERS = 0.2


def blur_face(data: np.ndarray, highest_point: np.ndarray, dmap: Depthmap, radius: float) -> np.ndarray:
    """Faceblur of the detected standing child.

    It uses the highest point of the child and blur all pixels in distance less than radius.

    Args:
        data: existing canvas to blur
        highest_point: 3D point. The surroundings of this point will be blurred.
        dmap: depthmap
        radius: radius around highest_point to blur

    Returns:
        Canvas like data with face blurred.
    """
    output = np.copy(data)
    points_3d_arr = dmap.convert_2d_to_3d_oriented()

    # blur RGB data around face
    for x in range(dmap.width):
        for y in range(dmap.height):

            # count distance from the highest child point
            depth = dmap.depthmap_arr[x, y]
            if not depth:
                continue
            point = points_3d_arr[:, x, y]

            vector = point - highest_point
            distance = abs(vector[0]) + abs(vector[1]) + abs(vector[2])
            if distance >= radius:
                continue

            # Gausian blur
            output[x, y] = get_smoothed_pixel(data, x, y, 10)

    return output


def draw_boundary(output: np.ndarray, aabb: np.array, color: np.array):
    output[aabb[0]:aabb[2], aabb[1], :] = color
    output[aabb[0]:aabb[2], aabb[3], :] = color
    output[aabb[0], aabb[1]:aabb[3], :] = color
    output[aabb[2], aabb[1]:aabb[3], :] = color


def render_confidence(dmap: Depthmap):
    confidence = dmap.confidence_arr
    confidence[confidence == 0.] = 1.
    return np.stack([confidence, confidence, confidence], axis=2)


def render_depth(dmap: Depthmap, use_smooth=False) -> np.ndarray:
    """Render depthmap into a 2D image.

    We assume here that all values in dmap.depthmap_arr are positive.

    A distance of 0m is visualized in white.
    A distance of 2m is visualized in black.

    flashlight analogy: close-to-cam data is white
    """
    if use_smooth:
        dmap_arr = np.minimum(dmap.depthmap_arr_smooth / 2., 1.)
    else:
        dmap_arr = np.minimum(dmap.depthmap_arr / 2., 1.)

    cond = (dmap_arr != 0.)
    dmap_arr[cond] = 1. - dmap_arr[cond]
    return np.stack([dmap_arr, dmap_arr, dmap_arr], axis=2)


def render_normal(dmap: Depthmap) -> np.ndarray:
    """Render normal vectors

    How normal vector are visualized:
    When a vector has (x,y,z)=(1,0,0), this will show in red color.
    When a vector has (x,y,z)=(0,1,0), this will show in green color (e.g. floor).
    When a vector has (x,y,z)=(0,0,1), this will show in blue color.
    """

    points_3d_arr = dmap.convert_2d_to_3d_oriented(should_smooth=True)
    normal = dmap.calculate_normalmap_array(points_3d_arr)

    # We can't see negative values, so we take the absolute value
    normal = abs(normal)  # shape: (3, width, height)

    return np.moveaxis(normal, 0, -1)


def render_rgb(dmap: Depthmap) -> np.ndarray:
    return dmap.rgb_array / 255  # shape (width, height, 3)


def render_segmentation(floor: float,
                        mask: np.ndarray,
                        dmap: Depthmap) -> np.ndarray:

    # segmentation
    red = [1, 0, 0]
    blue = [0, 0, 1]
    yellow = [1, 1, 0]
    output = np.zeros((dmap.width, dmap.height, 3))
    output[mask == MASK_CHILD] = yellow
    output[mask == MASK_FLOOR] = blue
    output[mask < 0] = red

    # pattern mapping
    points_3d_arr = dmap.convert_2d_to_3d_oriented(should_smooth=True)
    elevation = points_3d_arr[1, :, :] - floor
    horizontal = (elevation % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
    vertical_x = (points_3d_arr[0, :, :] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
    vertical_z = (points_3d_arr[2, :, :] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
    vertical = (vertical_x + vertical_z) / 2.0
    output[:, :, 0] *= horizontal
    output[:, :, 1] *= horizontal
    output[:, :, 2] *= vertical

    # Fog effect
    fog = dmap.depthmap_arr * dmap.depthmap_arr
    fog[fog == 0] = 1
    output[:, :, 0] /= fog
    output[:, :, 1] /= fog
    output[:, :, 2] /= fog

    # Ensure pixel clipping
    np.clip(output, 0., 1., output)

    # Show the boundary of the child
    color = [1, 0, 1]  # purple
    if dmap.is_child_fully_visible(mask):
        color = [0, 1, 0]  # green
    aabb = calculate_boundary(mask == MASK_CHILD)
    draw_boundary(output, aabb, color)

    return output


def render_plot(dmap: Depthmap) -> np.ndarray:
    # detect floor and child
    floor: float = dmap.get_floor_level()
    mask = dmap.segment_child(floor)  # dmap.detect_floor(floor)

    # prepare plots
    output_plots = [
        render_depth(dmap),
        render_normal(dmap),
        render_segmentation(floor, mask, dmap),
        render_confidence(dmap),
    ]
    if dmap.has_rgb:
        highest_point: np.ndarray = dmap.get_highest_point(mask)
        output_rgb = render_rgb(dmap)
        output_rgb = blur_face(output_rgb, highest_point, dmap, CHILD_HEAD_HEIGHT_IN_METERS)
        output_plots.append(output_rgb)

    return np.concatenate(output_plots, axis=1)


def render_plot_debug(dmap: Depthmap) -> np.ndarray:
    output_plots = [
        render_depth(dmap),
        render_normal(dmap),
        render_confidence(dmap),
    ]
    if dmap.has_rgb:
        output_plots.append(render_rgb(dmap))
    return np.concatenate(output_plots, axis=1)
