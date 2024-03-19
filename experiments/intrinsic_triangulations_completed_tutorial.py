"""
This file contains the implementation for the tutorial by [Sharp et al. 2021](https://doi.org/10.1145/3450508.3464592).

@inproceedings{10.1145/3450508.3464592,
author = {Sharp, Nicholas and Gillespie, Mark and Crane, Keenan},
title = {Geometry processing with intrinsic triangulations},
year = {2021},
isbn = {9781450383615},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3450508.3464592},
doi = {10.1145/3450508.3464592},
booktitle = {ACM SIGGRAPH 2021 Courses},
articleno = {6},
numpages = {1},
location = {Virtual Event, USA},
series = {SIGGRAPH '21}
}
"""

import numpy as np

##############################################################
### Mesh management and traversal helpers
##############################################################


def next_side(fs):
    """
    For a given side s of a triangle, returns the next side t. (This method serves mainly to make code more readable.)

    :param fs: A face side (f,s)
    :returns: The next face side in the same triangle (f, sn)
    """
    return (fs[0], (fs[1] + 1) % 3)


def other(G, fs):
    """
    For a given face-side fs, returns the neighboring face-side in some other triangle.

    :param G: |F|x3x2 gluing map G,
    :param fs: a face-side (f,s)
    :returns: The neighboring face-side (f_opp,s_opp)
    """
    # Assuming you have a face-side fs, one can use the gluing-map to get the
    # neighboring face-side. The index number of the face fs is how we access
    # the corresponding face-side.
    return tuple(G[fs])


def n_faces(F):
    """
    Return the number of faces in the triangulation.

    :param F: |F|x3 array of face-vertex indices
    :returns: |F|
    """
    return F.shape[0]


def n_verts(F):
    """
    Return the number of vertices in the triangulation.

    Note that for simplicity this function recovers the number of vertices from
    the face listing only. As a consequence it is _not_ constant-time, and
    should not be called in a tight loop.

    :param F: |F|x3 array of face-vertex indices
    :returns: |F|
    """
    return np.amax(F) + 1


##############################################################
### Geometric subroutines
##############################################################


def face_area(l, f):
    """
    Computes the area of the face f from edge lengths

    :param l: |F|x3 array of face-side edge lengths
    :param f: An integer index specifying the face
    :returns: The area of the face.
    """
    # gathering edge lengths
    l_a = l[f, 0]
    l_b = l[f, 1]
    l_c = l[f, 2]

    # use Heron's formula
    s = 0.5 * (l_a + l_b + l_c)
    d = s * (s - l_a) * (s - l_b) * (s - l_c)
    return np.sqrt(d)


def surface_area(F, l):
    """
    Compute the surface area of a triangulation.

    :param F: A |F|x3 vertex-face adjacency list F
    :param l: F |F|x3 edge-lengths array, giving the length of each face-side
    :returns: The surface area
    """
    result = 0.0
    for f in range(n_faces(F)):
        result += face_area(l, f)
    return result


def opposite_corner_angle(l, fs):
    """
    Computes triangle corner angle opposite the face-side fs.

    :param l: A |F|x3 array of face-side edge lengths
    :param fs: An face-side (f,s)
    :returns: The corner angle, in radians
    """
    # using law of cosines: relate cosine of one angle to all three edge lengths
    l_a = l[fs]
    l_b = l[next_side(fs)]
    l_c = l[next_side(next_side(fs))]

    d = (l_b**2 + l_c**2 - l_a**2) / (2 * l_b * l_c)
    return np.arccos(d)


def diagonal_length(G, l, fs):
    """
    Computes the length of the opposite diagonal of the diamond formed by the
    triangle containing fs, and the neighboring triangle adjacent to fs.

    This is the new edge length needed when flipping the edge fs.

    :param G: |F|x3x2 gluing map
    :param l: |F|x3 array of face-side edge lengths
    :param fs: A face-side (f,s)
    :returns: The diagonal length
    """
    # using law of cosines: compared to `opposite_corner_angle()`, this time we
    # use the angle to compute the edge (diagonal) length

    # gather lengths and angles
    fs_opposite = other(G, fs)
    u = l[next_side(next_side(fs))]
    v = l[next_side(fs_opposite)]
    theta_a = opposite_corner_angle(l, next_side(fs))
    theta_b = opposite_corner_angle(l, next_side(next_side((fs_opposite))))

    d = u**2 + v**2 - 2 * u * v * np.cos(theta_a + theta_b)
    return np.sqrt(d)


def is_delaunay(G, l, fs):
    """
    Test if the edge given by face-side fs satisfies the intrinsic Delaunay property.

    :param G: |F|x3x2 gluing map G,
    :param l: |F|x3 array of face-side edge lengths
    :param fs: A face-side (f,s)
    :returns: True if the edge is Delaunay
    """
    # An edge satisfies intrinsic Delaunay property iff, theta_a + theta_b <= pi
    # where theta_a is the opposite angle of a shared edge, and theta_b is the
    # opposite angle on the corresponding face
    fs_opposite = other(G, fs)
    theta_a = opposite_corner_angle(l, fs)
    theta_b = opposite_corner_angle(l, fs_opposite)

    # test against PI - eps to conservatively pass in cases where the sum
    # approximates to PI. This ensures the algorithm terminates even in the case
    # of a co-circular diamond, in the presence of floating-point errors.
    EPS = 1e-5
    return theta_a + theta_b <= np.pi + EPS


##############################################################
### Construct initial data
##############################################################


def build_edge_lengths(V, F):
    """
    Compute edge lengths for the triangulation.

    Note that we store a length per face-side, which means that each edge
    length appears twice. This is just to make our code simpler.

    :param V: |V|x3 array of vertex positions
    :param F: |F|x3 array of face-vertex indices
    :returns: The |F|x3 array of face-side lengths
    """
    # we store our edge lengths not just a list of one value per edge, but
    # rather as list of one value per face sides. So each side of each face we
    # store the length of that face side in an Fx3 array.

    # allocate an empty Fx3 array to fill
    l = np.empty((n_faces(F), 3))
    for f in range(n_faces(F)):  # iterate over triangles
        for s in range(3):  # iterate over each side
            # get two endpoints (i, j) of this side
            i = F[f, s]
            j = F[next_side((f, s))]

            # measure the length of the side
            length = np.linalg.norm(V[j] - V[i])
            l[f, s] = length

    return l


def sort_rows(A):
    """
    Sorts rows lexicographically, i.e., comparing the first column first, then using subsequent columns to break ties.

    :param A: A 2D array
    :returns: A sorted array with the same dimensions as A
    """
    return A[np.lexsort(np.rot90(A))]


def glue_together(G, fs1, fs2):
    """
    Glues together the two specified face sides.  Using this routine (rather
    than manipulating G directly) just helps to ensure that a basic invariant
    of G is always preserved: if a is glued to b, then b is glued to a.

    The gluing map G is updated in-place.

    :param G: |F|x3x2 gluing map
    :param fs1: a face-side (f1,s1)
    :param fs2: another face-side (f2,s2)
    """
    G[fs1] = fs2
    G[fs2] = fs1


def build_gluing_map(F):
    """
    Builds the gluing map for a triangle mesh.

    :param F: |F|x3 vertex-face adjacency list F describing a manifold, oriented triangle mesh without boundary.
    :returns: |F|x3x2 gluing map G, which for each side of each face stores the
    face-side it is glued to.  In particular, G[f,s] is a pair (f',s') such
    that (f,s) and (f',s') are glued together.
    """
    # 1. build matrix with row per face-side
    # Each row has 4 columns: (1) smaller of the two vertex indices for that
    # edge (2) larger of the two vertex indices for that edge (3) index of the
    # face (4) index of the side
    n_sides = 3 * n_faces(F)
    S = np.empty([n_sides, 4], dtype=np.int64)

    for f in range(n_faces(F)):  # iterate over triangles
        for s in range(3):  # iterate over three sides
            # get the two endpoints (i, j) of this side
            i = F[f, s]
            j = F[next_side((f, s))]

            # create the quadruple
            S[f * 3 + s] = (min(i, j), max(i, j), f, s)

    # 2. Sort rows lexicographically. The face side that we want to glue
    # together end up as adjacent rows, because they have the same vertex
    # indices along the edge.
    S = sort_rows(S)

    # 3. Glue adjacent rows
    G = np.empty([n_faces(F), 3, 2], dtype=np.int64)
    for p in range(0, n_sides, 2):
        # extra sanity check to fail nicely if a mesh with boundary
        # or nonmanifold mesh is given as input
        if S[p + 0, 0] != S[p + 1, 0] or S[p + 0, 1] != S[p + 1, 1]:
            raise ValueError(
                "Problem building glue map. Is input closed & manifold?"
            )

        fs0 = tuple(S[p + 0, 2:4])
        fs1 = tuple(S[p + 1, 2:4])
        glue_together(G, fs0, fs1)

    # A sanity check test
    validate_gluing_map(G)

    return G


def validate_gluing_map(G):
    """
    Performs sanity checks on the connectivity of the gluing map. Throws an
    exception if anything is wrong.

    :param G: |F|x3x2 gluing map G
    """

    for f in range(n_faces(G)):
        for s in range(3):

            fs = (f, s)
            fs_other = other(G, fs)

            if fs == fs_other:
                raise ValueError(
                    "gluing map points face-side to itself {}".format(fs)
                )

            if fs != other(G, fs_other):
                raise ValueError(
                    "gluing map is not involution (applying it twice does not return the original face-side) {} -- {} -- {}".format(
                        fs, fs_other, other(G, fs_other)
                    )
                )


##############################################################
### Intrinsic Delaunay and edge flipping
##############################################################


def flip_edge(F, G, l, s0):
    """
    Performs an intrinsic edge flip on the edge given by face-side s0. The
    arrays F, G, and l are updated in-place.

    This routine _does not_ check if the edge is flippable. Conveniently, in
    the particular case of flipping to Delaunay, non-Delaunay edges can always
    be flipped.

    :param F: |F|x3 vertex-face adjacency list F
    :param G: |F|x3x2 gluing map G
    :param l: |F|x3 edge-lengths array, giving the length of each face-side
    :param s0: A face-side of the edge that we want to flip

    :returns: The new identity of the side fs_a
    """
    # get the neighboring face-side
    s1 = other(G, s0)

    # get the 3 sides of each face
    s2, s3 = next_side(s0), next_side(next_side(s0))
    s4, s5 = next_side(s1), next_side(next_side(s1))

    # get opposite face sides, the sides glued to each edge of the diamond
    s6, s7, s8, s9 = other(G, s2), other(G, s3), other(G, s4), other(G, s5)

    # get vertex indices for the vertices of the diamond
    v0, v1, v2, v3 = F[s0], F[s2], F[s3], F[s5]

    # get the two faces from our face sides
    f0, f1 = s0[0], s1[0]

    # get the original lengths of the outside edges of the diamond
    l2, l3, l4, l5 = l[s2], l[s3], l[s4], l[s5]

    # compute the length of the new edge
    new_length = diagonal_length(G, l, s0)

    # update the adjacency list F (i.e., new faces)
    F[f0] = (v3, v2, v0)
    F[f1] = (v2, v3, v1)

    def relabel(s):
        """Usually this does nothing, but in a Delta-complex one of the
        neighbors of these faces might be one of the faces themselves!
        In that case, this re-labels the neighbors according to the updated
        labels after the edge flip.

         Args:
             s (_type_): face side

         Returns:
             _type_: relabel neighbor according to the updated labels after
             edge flip
        """
        if s == s2:
            return (f1, 2)
        if s == s3:
            return (f0, 1)
        if s == s4:
            return (f0, 2)
        if s == s5:
            return (f1, 1)
        return s

    # Additional step in case of a delta-complex. There is a self
    # edge in our triangulation, which can happen in our intrinsic
    # triangulations.
    s6, s7, s8, s9 = relabel(s6), relabel(s7), relabel(s8), relabel(s9)

    # update the gluing map G. The new diagonal requires different gluing.
    glue_together(G, (f0, 0), (f1, 0))
    glue_together(G, (f0, 1), s7)
    glue_together(G, (f0, 2), s8)
    glue_together(G, (f1, 1), s9)
    glue_together(G, (f1, 2), s6)

    # update edge lengths
    # note that even the edges we did not flip have been relabeled
    l[f0] = (new_length, l3, l4)
    l[f1] = (new_length, l5, l2)

    # return the face side we just flipped
    return f0, 0


def flip_to_delaunay(F, G, l):
    """
    Flip edges in the triangulation until it satisifes the intrinsic Delaunay criterion.

    For simplicity, we will implement this algorithm in terms of face-sides, checking if
    each face-side satisifes the criterion. Technically, this means we are testing each
    edge twice, which is unecessary, but makes our implementation simpler.

    The arrays F,G,l are modified in-place.

    :param F: |F|x3 vertex-face adjacency list F
    :param G: |F|x3x2 gluing map G
    :param l: |F|x3 edge-lengths array, giving the length of each face-side
    """
    from collections import deque

    # a queue of face-sides to test for Delaunay criterion
    to_process = deque()

    # initially add all face-sides for processing
    for f in range(n_faces(F)):  # iterate over triangles
        for s in range(3):  # iterate over the three sides
            to_process.append((f, s))

    # while the queue is not empty
    while to_process:
        # get the next-face side in the queue
        fs = to_process.pop()

        # check if it satisfies the Delaunay criterion
        if not is_delaunay(G, l, fs):
            # flip the edge, flipping a non Delaunay edge always makes it one
            fs = flip_edge(F, G, l, fs)

            # enqueue neighbors for processing, as they may have become
            # non-Delaunay
            neighbors = [
                next_side(fs),
                next_side(next_side(fs)),
                next_side(other(G, fs)),
                next_side(next_side(other(G, fs))),
            ]
            for n in neighbors:
                to_process.append(n)


def check_delaunay(F, G, l):
    """
    Check if a triangulation satisifies the intrinsic Delaunay property.

    :param F: |F|x3 vertex-face adjacency list F
    :param G: |F|x3x2 gluing map G
    :param l: |F|x3 edge-lengths array, giving the length of each face-side
    :returns: True if the triangulation is intrinsic Delaunay.
    """
    for f in range(n_faces(F)):
        for s in range(3):
            if not is_delaunay(G, l, (f, s)):
                return False
    return True


def print_info(F, G, l):
    """
    Print some info about a mesh

    :param F: |F|x3 vertex-face adjacency list F
    :param G: |F|x3x2 gluing map G
    :param l: |F|x3 edge-lengths array, giving the length of each face-side
    """

    print("  n_verts = {}".format(n_verts(F)))
    print("  n_faces = {}".format(n_faces(F)))
    print("  surface area = {}".format(surface_area(F, l)))
    print("  is Delaunay = {}".format(check_delaunay(F, G, l)))


### Can: Commenting out from here

# ##############################################################
# ### Run the code: flip to intrinsic Delaunay
# ##############################################################

# # Some test data: a simple shape with 5 vertices
# V = np.array(
#     [[0, 5.0, 0], [0, 1, -3.0], [-4.0, 0, 0], [0, 1, 3.0], [4.0, 0, 0]]
# )
# F = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 4, 2], [2, 4, 3]])
# source_vert = 0

# # Use these lines to load any triangle mesh you would like.
# # .obj, .ply, and .off formats are supported
# # (install with python -m pip install potpourri3d)
# # import potpourri3d as pp3d

# # uncomment these lines to run on the meshes included with the tutorial
# # (note that they additionally set a good source vertex for the later example)
# # (V, F), source_vert = pp3d.read_mesh("example_data/terrain8k.obj"), 2894
# # (V, F), source_vert = pp3d.read_mesh("example_data/pegasus.obj"), 1669
# # (V, F), source_vert = pp3d.read_mesh("example_data/rocketship.ply"), 26403

# # use this line to run on your own mesh of interest
# # V, F = pp3d.read_mesh("path/to/your/mesh.obj")

# # initialize the glue map and edge lengths arrays from the input data
# G = build_gluing_map(F)
# l = build_edge_lengths(V, F)

# print("Initial mesh:")
# print_info(F, G, l)

# # make a copy (so we preserve the original mesh), and flip to Delaunay
# F_delaunay = F.copy()
# G_delaunay = G.copy()
# l_delaunay = l.copy()
# flip_to_delaunay(F_delaunay, G_delaunay, l_delaunay)

# print("After Delaunay flips:")
# print_info(F_delaunay, G_delaunay, l_delaunay)


# ##############################################################
# ### Example application: Heat Method for Distance
# ##############################################################

# This section contains a simple self-contained implementation of the Heat
# Method, a PDE-based method to compute geodesic distance along a surface from
# a specified source point.

# For reference, see "The Heat Method for Distance Computation", by Crane,
# Weischedel, Wardetzky (2017).

# This algorithm makes use of the Laplace matrix, and we will see that applying
# the Delaunay edge flipping routine from above automatically improves results on
# low-quality triangulations.

# we will use Scipy for sparse matrix operations
import scipy
import scipy.sparse
import scipy.sparse.linalg


def build_cotan_laplacian(F, l):
    """
    Build the cotan-Laplace matrix for a triangulation.

    :param F: |F|x3 vertex-face adjacency list F
    :param l: |F|x3 edge-lengths array, giving the length of each face-side
    :returns: The Laplace matrix, as a sparse, |V|x|V| real scipy matrix
    """
    # build VxV sparse matrix where the off diagonal entries are given by
    # cotan of corner angles and the diagonal entries are sum of the off
    # diagonal entries in the row

    # initialize empty sparse matrix
    N = n_verts(F)
    # lil_matrix: structure for constructing sparse matrices incrementally
    L = scipy.sparse.lil_matrix((N, N))

    # walk through faces of our triangulation
    for f in range(n_faces(F)):
        for s in range(3):
            i = F[f, s]  # vertex i
            j = F[f, (s + 1) % 3]  # vertex j
            opposite_theta = opposite_corner_angle(l, (f, s))
            opposite_cotan = 1.0 / np.tan(opposite_theta)
            cotan_weight = 0.5 * opposite_cotan

            # add the corresponding cotangent weight entries
            L[i, j] -= cotan_weight
            L[j, i] -= cotan_weight
            L[i, i] += cotan_weight
            L[j, j] += cotan_weight

    # convert to a compressed sparse row matrix
    return L.tocsr()


def build_lumped_mass(F, l):
    """
    Build the lumped mass matrix for a triangulation, which associates an area with each vertex.

    :param F: |F|x3 vertex-face adjacency list F
    :param l: |F|x3 edge-lengths array, giving the length of each face-side
    :returns: The mass matrix, as a sparse, |V|x|V| real scipy matrix (which
    happens to be a diagonal matrix)
    """
    # each vertex has 1/3  the area of the adjacent faces associated with it
    # as a diagonal matrix

    # initialize empty sparse matrix
    N = n_verts(F)
    M = scipy.sparse.lil_matrix((N, N))

    # construct the matrix by summing contributions from each triangle
    for f in range(n_faces(F)):
        area = face_area(l, f)
        for s in range(3):
            i = F[f, s]  # get vertex i
            M[i, i] += area / 3.0

    # convert to a compressed sparse row matrix
    return M.tocsr()


def edge_in_face_basis(l, fs):
    """
    We associate a 2D-coordinate system with each face in a triangulation.
    Given a face-side, this routine returns the vector of the corresponding
    edge in that face.

    :param l: |F|x3 edge-lengths array, giving the length of each face-side
    :returns: The edge vector, as little length-2 numpy array
    """
    # usually we might think of a gradient of a function on a triangle mesh
    # as a 3D vector in each triangle. But in the intrinsic setting because
    # our triangles do not sit in space, it does not makes sense.
    # In the intrinsic setting, we represent gradient vectors in triangles in a
    # little local 2D coordinate system, that we define for each intrinsic
    # triangle. We define this coordinate system by just logically laying out
    # each triangle in the plane. One axis of our coordinate system will be
    # aligned with the zeroth edge of our triangle.

    f, s = fs
    theta = opposite_corner_angle(l, (f, 1))

    # construct local positions for each of the triangle vertices
    local_vert_positions = np.array(
        [
            [0, 0],  # first vertex origin
            [l[f, 0], 0],  # second vertex along the axis
            [np.cos(theta) * l[f, 2], np.sin(theta) * l[f, 2]],  # third vertex
        ]
    )

    # the edge vector is the difference of vertex positions in the local frame
    edge_vec = local_vert_positions[(s + 1) % 3] - local_vert_positions[s]
    return edge_vec


def evaluate_gradient_at_faces(F, l, x):
    """
    Given a scalar function at vertices, compute its gradient in each face.
    The gradients are defined in a tangent-coordinate system in each face as
    specified by edge_in_face_basis().

    :param F: |F|x3 vertex-face adjacency list F
    :param l: |F|x3 edge-lengths array, giving the length of each face-side
    :param x: A scalar function as a length-|V| numpy array, holding one value per vertex
    :returns: |F|x2 array of gradient vectors per-face (defined in the local 2D basis for each face)
    """
    grads = np.empty((n_faces(F), 2))

    # evaluate gradients independently in each triangle given a scalar function
    # at vertices represented in the local basis of each face
    for f in range(n_faces(F)):
        # gradient of a piecewise-linear function in a triangle
        face_grad = np.array([0.0, 0.0])
        for s in range(3):
            i = F[f, s]  # ith vertex
            edge_vec = edge_in_face_basis(l, next_side((f, s)))
            edge_vec_rot = np.array([-edge_vec[1], edge_vec[0]])
            face_grad += x[i] * edge_vec_rot

        area = face_area(l, f)
        face_grad /= 2.0 * area
        grads[f] = face_grad

    return grads


def evaluate_divergence_at_vertices(F, l, v):
    """
    Given a vector field defined by a collection of gradient vectors at the
    faces of a triangulation, evaluate the divergence of the vector field as
    a scalar value at vertices.

    :param F: |F|x3 vertex-face adjacency list F
    :param l: |F|x3 edge-lengths array, giving the length of each face-side
    :param v: |F|x2 array of vectors per-face (defined in the local 2D basis for each face)
    :returns: The divergences as a length-|V| numpy array
    """
    divs = np.zeros((n_verts(F)))

    # uses piecewise linear scalar functions in triangle
    # evaluate divergence as a summation over contributions from each face
    for f in range(n_faces(F)):
        grad_vec = v[f]

        # add the contribution of each triangle to the divergence at the
        # adjacent vertices
        for s in range(3):
            i = F[f, s]  # ith vertex
            j = F[f, (s + 1) % 3]  # vth vertex

            edge_vec = edge_in_face_basis(l, (f, s))
            opposite_theta = opposite_corner_angle(l, (f, s))
            opposite_cotan = 1.0 / np.tan(opposite_theta)
            cotan_weight = 0.5 * opposite_cotan
            div_contribution = cotan_weight * np.dot(edge_vec, grad_vec)

            divs[i] += div_contribution
            divs[j] -= div_contribution

    return divs


def heat_method_distance_from_vertex(F, l, source_vert):
    """
    Use the Heat Method to compute geodesic distance along a surface from
    a source vertex.

    For reference, see "The Heat Method for Distance Computation", by Crane,
    Weischedel, Wardetzky (2017).

    The heat method uses the Laplace matrix as one of its main ingredients, so
    if the triangulation is intrinsic Delaunay, accuracy will be improved.

    :param F: |F|x3 vertex-face adjacency list F
    :param l: |F|x3 edge-lengths array, giving the length of each face-side
    :param source_vert: The index of a vertex to use as the source.
    :returns: The distances from the source vertex, as a length-|V| numpy array
    """
    # build matrices
    L = build_cotan_laplacian(F, l)
    M = build_lumped_mass(F, l)

    # compute mean edge length h
    mean_edge_length = np.mean(l)
    short_time = mean_edge_length**2

    # build the heat operator
    H = (M + short_time * L)

    # build the initial conditions
    init_RHS = np.zeros(n_verts(F))
    init_RHS[source_vert] = 1.0

    # solve the linear system to evaluate heat flow
    heat = scipy.sparse.linalg.spsolve(H, init_RHS)

    # compute gradients and normalize
    grads = evaluate_gradient_at_faces(F, l, heat)
    # normalize in each face
    grads = grads / np.linalg.norm(grads, axis=1, keepdims=True)

    # solve for the function which has those gradients
    div = evaluate_divergence_at_vertices(F, l, grads)
    dist = scipy.sparse.linalg.spsolve(
        L + scipy.sparse.eye(L.shape[0]) * 1e-6, div
    )

    # shift so the source has distance 0
    dist -= dist[source_vert]

    return dist


# ##############################################################
# ### Run the code: compute distance
# ##############################################################

# # Remember: choose what mesh to run on in the in the flipping example above

# # Compute distance using the heat method, both before and after flipping
# print("computing distance on original triangulation...")
# dists_before = heat_method_distance_from_vertex(F, l, source_vert)
# print("computing distance on Delaunay triangulation...")
# dists_after = heat_method_distance_from_vertex(
#     F_delaunay, l_delaunay, source_vert
# )

# # Visualize the geodesic distances
# # (click the 'enable' checkbox on the left sidebar to inspect the distances)
# print("Visualizing in Polyscope window")
# import polyscope as ps

# ps.init()
# ps_mesh = ps.register_surface_mesh("test mesh", V, F)
# ps_mesh.add_distance_quantity(
#     "distance on initial triangulation", dists_before, enabled=True
# )
# ps_mesh.add_distance_quantity("distance after Delaunay flips", dists_after)
# ps.show()
