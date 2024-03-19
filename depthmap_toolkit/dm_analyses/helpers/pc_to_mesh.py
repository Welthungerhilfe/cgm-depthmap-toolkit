import numpy as np


class Face:
    def __init__(self, points):
        """
        Constructor for a triangular face representation.
        :param points: points of the face
        """
        points = points[::-1]
        self.points = np.array(points)
        self.edges = np.array([points[1] - points[0], points[2] - points[1], points[0] - points[2]])
        self.normal = np.cross(self.edges[0, :], - self.edges[2, :])
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.D = -np.dot(self.normal, self.points[0])

    def intersect_ray(self, origin, direction):
        """
        Calculate if the given ray intersects the face.
        :param origin:      origin of the ray
        :param direction:   direction of the ray
        :return:            distance between intersection and origin of ray or inf if no intersection
        """
        t = - (np.dot(self.normal, origin) + self.D) / np.dot(self.normal, direction)
        p = origin + t * direction
        if self.check_point(p):
            return p[2]
        return float("inf")

    def check_point(self, point):
        """
        Check if the given point on the faces plane is within the edges of the face.
        :param point:   point to check
        :return:        True if the point is within the edges of the face
        """
        return np.min(np.dot(np.cross(self.edges, point - self.points), self.normal)) >= 0


class Point:
    def __init__(self, id, position=(0, 0, 0)):
        """
        Constructor of 3d point with 2d identifier for generation of triangles.
        :param id:          2d identifier of point (i.e. pixel coordinate of depth map pixel the point originates from)
        :param position:    3d coordinates of the point
        """
        self.id = id
        self.position = np.array(position)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def get_triangles(self, point_set):
        """
        Get triangles if neighbors existing in point set.

        Neighbor points are points that are created from pixels in the 8-neighborhood of this points origin pixels.
        Generation pattern is designed to ensure that every triangle will be generated exactly once if every
        get_triangles is called for every point in point_set.
        :param point_set:   point set to use for triangle generation
        :return:            list of triangles with existing neighboring points
        """

        # create points to fast search point set for neighbors (i.e. points with id the neighbors would have)
        p00 = Point((self.id[0] - 1, self.id[1] - 1))
        p01 = Point((self.id[0] - 1, self.id[1]))
        p10 = Point((self.id[0], self.id[1] - 1))
        p21 = Point((self.id[0] + 1, self.id[1]))
        p12 = Point((self.id[0], self.id[1] + 1))
        p22 = Point((self.id[0] + 1, self.id[1] + 1))

        results = []

        # create triangles if necessary
        # showing triangles in different cases
        # o - this point; x - neighbors
        if p22 in point_set:
            p22 = [p for p in point_set if p == p22][0]
            if p12 in point_set:
                # o-x
                #  \|
                #   x
                p12 = [p for p in point_set if p == p12][0]
                results.append(Face([self.position, p12.position, p22.position]))
            if p21 in point_set:
                # o
                # |\
                # x-x
                p21 = [p for p in point_set if p == p21][0]
                results.append(Face([self.position, p22.position, p21.position]))
        elif p12 in point_set and p21 in point_set:
            # o-x
            # |/
            # x
            p12 = [p for p in point_set if p == p12][0]
            p21 = [p for p in point_set if p == p21][0]
            results.append(Face([self.position, p12.position, p21.position]))
        if p00 not in point_set:
            if p01 in point_set and p10 in point_set:
                #   x
                #  /|
                # x-o
                p01 = [p for p in point_set if p == p01][0]
                p10 = [p for p in point_set if p == p10][0]
                results.append(Face([self.position, p01.position, p10.position]))
        return results


def generate_mesh_from_2d_point_cloud(pc, filter):
    """
    Generate a triangle mesh from given filtered point cloud.
    :param pc:      unfiltered point cloud (has to be shaped HxWx3 with H and W being height and width of origin depth map)
    :param filter:  filter telling which points shall be used
    :return:        list of triangles from filtered point cloud
    """
    points = {Point(idx, pc[idx]) for idx, p in np.ndenumerate(pc[:, :, 0]) if filter[idx]}
    triangles = [t for p in list(points) for t in p.get_triangles(points)]
    return triangles
