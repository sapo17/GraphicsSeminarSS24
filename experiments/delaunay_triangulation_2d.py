"""
We follow the implementation by
Jose M. Espadero (http://github.com/jmespadero/pyDelaunay2D), which was based on
the implementation from Ayron Catteau (http://github.com/ayron/delaunay). 
"""

import numpy as np
from math import sqrt


class Delaunay2D:
    """
    Computes a Delaunay Triangulation in 2D using Bowyer-Watson algorithm.
    """

    def __init__(self, center: tuple = (0, 0), radius: int = 9999) -> None:
        """
        Initialize and create a new frame to contain the triangulation.

        Args:
            center (tuple, optional): Optional position for the center of the
            frame. Defaults to (0, 0).
            radius (int, optional): Optional distance from corners to the
            center. Defaults to 9999.
        """

        center = np.asarray(center)

        # create coordinates for the corners of the frame
        self.coords = [
            center + radius * np.array((-1, -1)),
            center + radius * np.array((+1, -1)),
            center + radius * np.array((+1, +1)),
            center + radius * np.array((-1, +1)),
        ]

        # create two dicts to store triangle neighbors and circumcircles
        self.triangles = {}
        self.circles = {}

        # create two counter-clockwise triangles for the frame
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # compute circumcenters and circumradius for each triangle
        for t in self.triangles:
            self.circles[t] = self.eval_circumcenter(t)

    def eval_circumcenter(self, tri: list) -> tuple[float, ...]:
        """
        Compute circumcenter and circumradius of a triangle in 2D.

        Args:
            tri (list): List of triangle vertices (i.e., triangle).

        Returns:
            tuple[float, ...]: Returns the circumcenter and circumradius of a
            triangle.
        """
        points_1 = np.asarray([self.coords[v] for v in tri])
        points_2 = np.dot(points_1, points_1.T)

        A = np.bmat([[2 * points_2, [[1], [1], [1]]], [[[1, 1, 1, 0]]]])
        b = np.hstack((np.sum(points_1 * points_1, axis=1), [1]))
        x = np.linalg.solve(A, b)

        bary_coords = x[:-1]
        center = np.dot(bary_coords, points_1)
        radius = np.sum(np.square(points_1[0] - center))  # squared distance
        return (center, radius)

    def is_point_in_circle_fast(self, tri: list, p: np.ndarray) -> bool:
        """
        Check if point p is inside of precomputed circumcircle of the triangle
        tri.

        Args:
            tri (list): List of triangle vertices (i.e., triangle).
            p (np.ndarray): A point in 2D.

        Returns:
            bool: true if point p is inside of precomputed circumcircle of the
            triangle tri.
        """
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    def is_point_in_circle_robust(self, tri: list, p: np.ndarray) -> bool:
        """
        Check if point p is inside of circumcircle around the triangle tri.

        Args:
            tri (list): List of triangle vertices (i.e., triangle).
            p (np.ndarray): A point in 2D.

        Returns:
            bool: Returns true if p is inside circumcirle of tri, otherwise
            false.
        """
        m1 = np.asarray([self.coords[v] - p for v in tri])
        m2 = np.sum(np.square(m1), axis=1).reshape((3, 1))
        m = np.hstack((m1, m2))  # 3x3 matrix check
        return np.linalg.det(m) <= 0

    def add_point(self, p: np.ndarray):
        """
        Add a point to the current Delaunay Triangulation, and refine it using
        Bowyer-Watson.

        Args:
            p (np.ndarray): A point in 2D.
        """
        p = np.asarray(p)
        idx = len(self.coords)
        self.coords.append(p)

        # search the triangle(s) whose circumcircle contains p
        bad_triangles = []
        for tri in self.triangles:
            if self.is_point_in_circle_fast(tri, p):
                bad_triangles.append(tri)

        # find the counter-clockwise boundary (star shape) of the bad triangles,
        # expressed as list of edges (point pairs) and the opposite triangle to
        # each edge
        boundary = []
        tri = bad_triangles[0]
        edge = 0

        while True:  # get the opposite triangle of edge
            # check if edge of triangle tri is on the boundary
            tri_opposite = self.triangles[tri][edge]
            if tri_opposite not in bad_triangles:
                # insert edge and external triangle into boundary list
                boundary.append(
                    (tri[(edge + 1) % 3], tri[(edge - 1) % 3], tri_opposite)
                )

                # move to the next counter-clockwise edge in this triangle
                edge = (edge + 1) % 3

                # check if boundary is a closed loop
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                # move to next counter-clockwise edge in opposite triangle
                edge = (self.triangles[tri_opposite].index(tri) + 1) % 3
                tri = tri_opposite

        # remove triangles too near of point p of our solution
        for tri in bad_triangles:
            del self.triangles[tri]
            del self.circles[tri]

        # retriangle the hole left by bad triangles
        new_triangles = []
        for e0, e1, tri_opposite in boundary:
            # create a new triangle using point p and edge extremes
            tri = (idx, e0, e1)

            # store circumcenter and circumradius of the triangle
            self.circles[tri] = self.eval_circumcenter(tri)

            # set opposite triangle of the edge as neighbor of triangle
            self.triangles[tri] = [tri_opposite, None, None]

            # try to set triangle as neighbor of the opposite triangle
            if tri_opposite:
                # search the neighbor of triangle opposite that use edge(e1, e0)
                for i, neigh in enumerate(self.triangles[tri_opposite]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                            # change link to use our new triangle
                            self.triangles[tri_opposite][i] = tri

            # add triangle to a temporal list
            new_triangles.append(tri)

        # link the new triangles each other
        n = len(new_triangles)
        for i, tri in enumerate(new_triangles):
            self.triangles[tri][1] = new_triangles[(i + 1) % n]  # next
            self.triangles[tri][2] = new_triangles[(i - 1) % n]  # previous

    def export_triangles(self) -> list:
        """
        Export the current list of Delaunay Triangles.

        Returns:
            list: current list of Delaunay Triangles.
        """
        # filter out triangles with any vertex in the extended bounding box
        return [
            (a - 4, b - 4, c - 4)
            for (a, b, c) in self.triangles
            if a > 3 and b > 3 and c > 3
        ]

    def export_circles(self) -> list:
        """
        Export the circumcirles as a list of (center, radius).

        Returns:
            list: circumcirles as a list of (center, radius).
        """
        # filter out triangles with any vertex in the extended bounding box and
        # sqrt of radius
        return [
            (self.circles[(a, b, c)][0], sqrt(self.circles[(a, b, c)][1]))
            for (a, b, c) in self.triangles
            if a > 3 and b > 3 and c > 3
        ]

    def export_delaunay_triangles(self) -> tuple:
        """
        Export the current set of Delaunay coordinates and triangles.

        Returns:
            tuple: current set of Delaunay coordinates and triangles.
        """
        # filter out coordinates in the extended bounding box
        coord = self.coords[4:]

        # filter out triangles with any vertex in the extended bounding box
        tris = [
            (a - 4, b - 4, c - 4)
            for (a, b, c) in self.triangles
            if a > 3 and b > 3 and c > 3
        ]
        return coord, tris

    def export_extended_delaunay_triangles(self) -> tuple:
        """
        Export the Extended Delaunay Triangulation with the frame vertex.

        Returns:
            tuple: Extended Delaunay Triangulation with the frame vertex.
        """
        return self.coords, list(self.triangles)

    def export_voronoi_regions(self) -> tuple:
        """
        Export coordinates and regions of Voronoi diagram as indexed data.

        Returns:
            tuple: coordinates and regions of Voronoi diagram as indexed data.
        """
        use_vertex = {i: [] for i in range(len(self.coords))}
        vor_coords = []
        idx = {}

        # build a list of coordinates and one index per triangle / region
        for tri_idx, (a, b, c) in enumerate(sorted(self.triangles)):
            vor_coords.append(self.circles[(a, b, c)][0])

            # insert triangle, rotating it so the key is the "last" vertex
            use_vertex[a] += [(b, c, a)]
            use_vertex[b] += [(c, a, b)]
            use_vertex[c] += [(a, b, c)]

            # set tri_idx as the index to use with this triangle
            idx[(a, b, c)] = tri_idx
            idx[(c, a, b)] = tri_idx
            idx[(b, c, a)] = tri_idx

        # initialize regions per coordinate dictionary
        regions = {}

        # sort each region in a coherent order, and substitute each triangle by
        # its index
        for i in range(4, len(self.coords)):
            v = use_vertex[i][0][0]  # vertex of a triangle
            r = []

            for _ in range(len(use_vertex[i])):
                # search the triangle beginning with vertex v
                tri = [t for t in use_vertex[i] if t[0] == v][0]
                r.append(idx[tri])  # add the index of this triangle to region
                v = tri[1]  # choose the next vertex to search

            regions[i - 4] = r  # store region

        return vor_coords, regions
