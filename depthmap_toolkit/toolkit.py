import shutil
import sys
from os import walk
import logging
from pathlib import Path
import functools
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy import ndimage

from cgmml.common.depthmap_toolkit.depthmap import Depthmap
from cgmml.common.depthmap_toolkit.exporter import export_obj, export_extrapolated_obj, export_ply
from cgmml.common.depthmap_toolkit.visualisation import render_plot

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)

REPO_DIR = Path(__file__).resolve().parents[3]
EXPORT_DIR = REPO_DIR / 'data' / 'export'

LAST_CLICK_COORD = [0, 0, 0]
IDX_CUR_DMAP = None
DMAP = None


def onclick(event):
    global DMAP
    global LAST_CLICK_COORD
    if event.xdata is not None and event.ydata is not None:
        x = int(event.ydata)
        y = int(event.xdata)
        if x > 1 and y > 1 and x < DMAP.width - 2 and y < DMAP.height - 2:
            depth = DMAP.depthmap_arr[x, y]
            if not depth:
                logger.info('no valid data')
                return
            res = DMAP.convert_2d_to_3d(x, y, depth)
            diff: list = [LAST_CLICK_COORD[0] - res[0], LAST_CLICK_COORD[1] - res[1], LAST_CLICK_COORD[2] - res[2]]
            dst: float = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            logger.info(f'x={res[0]:.3f}, y={res[1]:.3f}, depth={res[2]:.3f}, diff={dst:.3f}')
            LAST_CLICK_COORD[0] = res[0]
            LAST_CLICK_COORD[1] = res[1]
            LAST_CLICK_COORD[2] = res[2]


def export_extrapolated_object(event):
    global DMAP
    floor = DMAP.get_floor_level()
    export_extrapolated_obj(EXPORT_DIR / f'output{IDX_CUR_DMAP}_extrapolated.obj', DMAP, floor)


def export_textured_object(event):
    global DMAP
    floor = DMAP.get_floor_level()
    export_obj(EXPORT_DIR / f'output{IDX_CUR_DMAP}_textured.obj', DMAP, floor, True)


def export_pointcloud(event):
    global DMAP
    floor = DMAP.get_floor_level()
    export_ply(EXPORT_DIR / f'output{IDX_CUR_DMAP}.ply', DMAP, floor)


def next_click(event, calibration_fpath: str, depthmap_dir: str):
    global IDX_CUR_DMAP
    IDX_CUR_DMAP = IDX_CUR_DMAP + 1
    if (IDX_CUR_DMAP == size):
        IDX_CUR_DMAP = 0
    show(depthmap_dir, calibration_fpath)


def prev_click(event, calibration_fpath: str, depthmap_dir: str):
    global IDX_CUR_DMAP
    IDX_CUR_DMAP = IDX_CUR_DMAP - 1
    if (IDX_CUR_DMAP == -1):
        IDX_CUR_DMAP = size - 1
    show(depthmap_dir, calibration_fpath)


def show(depthmap_dir: str, calibration_file: str, original_orientation=False):
    global DMAP
    fig.canvas.manager.set_window_title(depth_filenames[IDX_CUR_DMAP])
    rgb_filename = rgb_filenames[IDX_CUR_DMAP] if rgb_filenames else 0
    DMAP = Depthmap.create_from_zip(depthmap_dir,
                                    depth_filenames[IDX_CUR_DMAP],
                                    rgb_filename,
                                    calibration_file)

    angle = DMAP.get_angle_between_camera_and_floor()
    logging.info('angle between camera and floor is %f', angle)

    output = render_plot(DMAP)
    if original_orientation:
        output = ndimage.rotate(output, 90)
    plt.imshow(output)
    plot_names = ['depth', 'normals', 'child/background segmentation', 'confidence']
    if DMAP.has_rgb:
        plot_names.append('rgb')
    plot_title = '\n'.join([f'{i}: {plot_name}' for i, plot_name in enumerate(plot_names)])
    plt.title(plot_title)
    plt.show()


def is_legit_file(fpath: Union[str, Path]) -> bool:
    """Find non-hidden files"""
    if Path(fpath).name.startswith('.'):
        return False
    return True


def assemble_filenames(input_dir: Path):
    """Inspect input dir for files and return a sorted list of those files"""
    all_filenames = []
    for (_dirpath, _dirnames, filenames) in walk(input_dir):
        fns = list(filter(is_legit_file, filenames))
        all_filenames.extend(fns)
    all_filenames.sort()
    return all_filenames


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('You did not enter depthmap_dir folder and calibration file path')
        print('E.g.: python toolkit.py depthmap_dir calibration_fpath')
        sys.exit(1)

    depthmap_dir = sys.argv[1]
    calibration_fpath = sys.argv[2]

    depth_dir = Path(depthmap_dir) / 'depth'
    rgb_dir = Path(depthmap_dir) / 'rgb'
    assert depth_dir.exists(), depthmap_dir
    if not rgb_dir.exists():
        logging.warn("RGB directory doesn't exist. Working with depth data only")

    depth_filenames = assemble_filenames(depth_dir)
    rgb_filenames = assemble_filenames(rgb_dir)

    # Re-create export folder
    try:
        shutil.rmtree(EXPORT_DIR)
    except BaseException:
        print('no previous data to delete')
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Show viewer
    IDX_CUR_DMAP = 0
    size = len(depth_filenames)
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', functools.partial(onclick))
    bprev = Button(plt.axes([0.0, 0.0, 0.1, 0.075]), '<<', color='gray')
    bprev.on_clicked(functools.partial(prev_click, calibration_fpath=calibration_fpath, depthmap_dir=depthmap_dir))
    bnext = Button(plt.axes([0.9, 0.0, 0.1, 0.075]), '>>', color='gray')
    bnext.on_clicked(functools.partial(next_click, calibration_fpath=calibration_fpath, depthmap_dir=depthmap_dir))
    bexport_ply = Button(plt.axes([0.125, 0.0, 0.25, 0.05]), 'Export pointcloud', color='gray')
    bexport_ply.on_clicked(functools.partial(export_pointcloud))
    bexport_textured_obj = Button(plt.axes([0.375, 0.0, 0.25, 0.05]), 'Export textured mesh', color='gray')
    bexport_textured_obj.on_clicked(functools.partial(export_textured_object))
    bexport_extrapolated_obj = Button(plt.axes([0.625, 0.0, 0.25, 0.05]), 'Export Poisson mesh', color='gray')
    bexport_extrapolated_obj.on_clicked(functools.partial(export_extrapolated_object))
    background = Button(plt.axes([0.0, 0.0, 1.0, 1.0]), '', color='white')
    show(depthmap_dir, calibration_fpath)
