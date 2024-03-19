import math
import time

import numpy as np

from multiprocessing import Pool

from helpers.mesh_tree import MeshTree

debug = True


class Line:
    def __init__(self, angles):
        """
        Constructor of a Line
        :param angles:
        """
        self.angles = angles
        self.d = np.array([math.tan(math.radians(a)) for a in angles] + [1])
        self.d /= np.linalg.norm(self.d)
        self.o = np.zeros(3)

    def evaluate(self, point):
        return 0


class DepthmapFromMeshGenerator:
    def __init__(self, pc, filter, floor, wall, fov=(42.5, 54.8), *, bins=4, ray_margin=0.01, max_mt_depth=5, max_mt_bin_size=100):
        """
        Constructor for Mesh to Depthmap Cenverter.

        Pre-generates a tree struct to be faster during ray casting (similar to Octree).
        :param pc:                  point cloud to generate mesh from
        :param filter:              indices for points that are part of the child
        :param floor:               Plane that describes the floor
        :param wall:                Plane that describes the wall behind the child
        :param fov:                 Field ov view that shall be emulated
        :param bins:                number of bins that should be generated per layer of the tree struct
        :param ray_margin:          overlap of the single bins (makes sure all triangles can be recovered from a single bin)
        :param max_mt_depth:        maximum depth for the tree struct
        :param max_mt_bin_size:     tree struct generate will interrupt early if fewer elements are contained in bin
        """
        self.pc = pc
        self.filter = filter
        self.floor = floor
        self.wall = wall

        self.fov = fov
        self.ray_margin = ray_margin

        # compute bin borders based on fov
        x_borders = [n * fov[0] / bins - fov[0] / 2 for n in range(1, bins)]
        y_borders = [n * fov[1] / bins - fov[1] / 2 for n in range(1, bins)]

        # generate mesh tree (i.e. struct containing point bins based on borders)
        self.mt = MeshTree(pc, filter, [x_borders, y_borders], margin=ray_margin, angular=True, max_recursions=max_mt_depth, max_bin_points=max_mt_bin_size)
        return

    def generate(self, img_size=(180, 240), *, rays=1):
        """
        Generate depth map from pre-generated mesh tree and given plains.

        Uses ray casting on mesh to generate depthmap.
        :param img_size:    size of resulting depthmap
        :param rays:        sqrt of number of rays per pixel during ray casting
        :return:            depthmap generated from mesh and plains
        """
        w, h = img_size

        self.fov_x_1 = self.fov[0] / w
        self.fov_y_1 = self.fov[1] / h

        fov_x_1 = self.fov[0] / w
        fov_y_1 = self.fov[1] / h

        self.rays = rays

        self.img_size = img_size
        self.directions = [[(x - w / 2), (y - h / 2)] for y in range(h) for x in range(w)]

        start = time.time()
        if not debug:
            with Pool(8) as p:
                result = np.array(p.map(self.get_pixel_value, range(len(self.directions)))).reshape(img_size[::-1])
        else:
            result = np.array([self.get_pixel_value(i) for i in range(len(self.directions))]).reshape(img_size[::-1])

        print(f'Generating depthmap took {time.time() - start} seconds.')

        return result

    def get_pixel_value(self, idx):
        """
        Gets depth value for a single pixel.
        :param idx: 2d index of the pixel
        :return:    depth value for pixel
        """
        idx = self.directions[idx]

        # get ray offsets for all rays through the pixel
        ray_offsets = [r / 2 / self.rays for r in range(1, 2 * self.rays, 2)]

        # get all absolute directions of rays
        directions = [[(idx[0] + x) * self.fov_x_1, (idx[1] + y) * self.fov_y_1] for y in ray_offsets for x in ray_offsets]

        # compute the first intersection with the mesh for each ray
        results = np.array([self.get_depth_value_along_line(d) for d in directions])

        # return the nearest intersection of all rays
        # (averaging would not be optimal because some rays could miss the child and only intersect the background)
        return np.min(results)

    def get_depth_value_along_line(self, angles):
        """
        Intersects a single ray with the mesh and returns depth value of intersection.

        There should always be an intersection if the ray is not parallel to all planes.
        :param angles:  angle of the ray
        :return:        depth value of first intersection
        """
        if type(angles) is int:
            angles = self.directions[angles]
        line = Line(angles)
        point = None
        dist = self.get_nearest_intersection_along_line(line)

        if dist == float("inf"):
            result = self.get_nearest_line_plane_intersection(line)[2]
        else:
            result = min(dist, self.get_nearest_line_plane_intersection(line)[2])

        return result

    def get_nearest_intersection_along_line(self, line):
        """
        Get the distance between origin of given line and nearest intersection of given line with mesh tree.
        :param line:    line to intersect with own mesh tree
        :return:        distance to nearest intersection point
        """
        triangles = self.mt.get_interval_by_point(line.angles)

        intersections = [t.intersect_ray(line.o, line.d) for t in triangles]
        intersections.append(float("inf"))

        return min(i for i in intersections if i > 0)

    def get_nearest_line_plane_intersection(self, line):
        """
        Get the distance between origin of given line and nearest intersection of given line with wall or floor plane.
        :param line:    line to intersect with own wall and floor plane
        :return:        distance to nearest intersection point
        """
        wall_t = (self.wall.n @ self.wall.origin - self.wall.n.dot(line.o)) / self.wall.n.dot(line.d / np.linalg.norm(line.d))
        floor_t = (self.floor.n @ self.floor.origin - self.floor.n.dot(line.o)) / self.floor.n.dot(line.d / np.linalg.norm(line.d))

        best_t = floor_t if 0 < floor_t < wall_t else wall_t

        return line.o + (line.d / np.linalg.norm(line.d)) * best_t
