import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, voronoi_plot_2d, Voronoi
import numpy as np


def plot_voronoi_square_lattice(size):
    # Generate points for the square lattice
    points = np.array([[i, j] for i in range(size) for j in range(size)])

    # Define colors
    gnbu_colormap = plt.colormaps["GnBu"]
    point_color = gnbu_colormap(0.6)  # Adjusted point color
    voronoi_color = gnbu_colormap(0.4)

    # Compute Voronoi diagram
    vor = Voronoi(points)

    # Plot without axes and title
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    voronoi_plot_2d(
        vor,
        ax=ax,
        show_vertices=False,
        line_colors=voronoi_color,
        line_width=2,
        line_alpha=0.6,
        show_points=False,  # Hide default points
    )
    ax.scatter(
        points[:, 0], points[:, 1], color=point_color, label="Site"
    )  # Plot with new color
    ax.plot(
        [],
        [],
        color=voronoi_color,
        linewidth=2,
        alpha=0.6,
        label="Voronoi Region",
    )
    plt.tight_layout()
    plt.axis("off")
    plt.legend(loc="lower right", framealpha=1.0)


plot_voronoi_square_lattice(5)


# ---


from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Generate points for a 3x3x3 cubic lattice
def generate_cubic_lattice_points(size):
    points = np.array(
        [
            [x, y, z]
            for x in range(size)
            for y in range(size)
            for z in range(size)
        ]
    )
    return points


def plot_cube_lattice(points, elev=30, azim=30):
    fig = plt.figure(figsize=(7, 7), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    gnbu_colormap = plt.colormaps["GnBu"]
    point_color = gnbu_colormap(0.6)  # Adjusted point color for lattice points
    voronoi_color = gnbu_colormap(0.4)  # Color for connecting lines
    lattice_grid_color = gnbu_colormap(0.2)
    edge_color = voronoi_color  # Color for cube edges and vertices

    # Plotting all points in the specified point color for lattice points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=[point_color],
        label="Sites of cube lattice",
    )

    # Connect each lattice point only to its direct adjacent neighbors
    for point in points:
        for d in range(3):  # Iterate over dimensions
            for offset in [-1, 1]:  # Positive and negative directions
                neighbor = point.copy()
                neighbor[d] += offset
                if np.all(neighbor >= 0) and np.all(
                    neighbor < 3
                ):  # Check bounds
                    ax.plot(
                        [point[0], neighbor[0]],
                        [point[1], neighbor[1]],
                        [point[2], neighbor[2]],
                        linestyle="dotted",
                        color=lattice_grid_color,
                        alpha=0.5,
                    )

    # Define the cube at the center with adjusted transparency
    cube_vertices = np.array(
        [
            [0.5, 0.5, 0.5],
            [1.5, 0.5, 0.5],
            [1.5, 1.5, 0.5],
            [0.5, 1.5, 0.5],
            [0.5, 0.5, 1.5],
            [1.5, 0.5, 1.5],
            [1.5, 1.5, 1.5],
            [0.5, 1.5, 1.5],
        ]
    )

    # Define the cube's faces
    cube_faces = [
        [cube_vertices[i] for i in [0, 1, 2, 3]],
        [cube_vertices[i] for i in [4, 5, 6, 7]],
        [cube_vertices[i] for i in [0, 1, 5, 4]],
        [cube_vertices[i] for i in [2, 3, 7, 6]],
        [cube_vertices[i] for i in [1, 2, 6, 5]],
        [cube_vertices[i] for i in [4, 7, 3, 0]],
    ]
    ax.scatter(
        cube_vertices[:, 0],
        cube_vertices[:, 1],
        cube_vertices[:, 2],
        c=[edge_color],
        label="Voronoi Region of the site at the center",
    )

    voronoi_color_rgba = voronoi_color[:3] + (
        0.05,
    )  # Adjusted transparency for cube faces

    # Construct and add the cube with transparency and edge color
    cube = Poly3DCollection(
        cube_faces,
        facecolors=[voronoi_color_rgba],
        linewidths=1,
        edgecolors=edge_color,
    )
    ax.add_collection3d(cube)
    ax.view_init(elev=elev, azim=azim)

    # Remove axis and adjust layout
    ax.set_axis_off()
    plt.legend()
    plt.tight_layout(pad=0)
    plt.show()


# Plot the cubic lattice points with a central transparent unit cube
cubic_points = generate_cubic_lattice_points(3)
plot_cube_lattice(cubic_points, elev=10, azim=25)

import trimesh

# Load the OBJ files
file_path1 = "models/cube.obj"
file_path2 = "models/truncated_octahedron.obj"

cube = trimesh.load(file_path1)
octahedron = trimesh.load(file_path2)

# Define the colors from the GnBu colormap
gnbu_colormap = plt.colormaps["GnBu"]
lattice_grid_color = gnbu_colormap(0.2)  # Color for cube edges
voronoi_color = gnbu_colormap(0.4)  # Color for octahedron
point_color = gnbu_colormap(0.5)

# Extracting the unique edges for the octahedron
edges = octahedron.edges_unique
edge_vertices = octahedron.vertices[edges]


# Define a function to draw the octahedron faces without drawing each triangle
def draw_polygons(ax, vertices, faces, color, alpha=0.05):
    poly3d = [[vertices[vertex] for vertex in face] for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            poly3d,
            facecolors=color,
            linewidths=1,
            edgecolors=voronoi_color,
            alpha=alpha,
        )
    )


# Set up the plot
fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection="3d")

ax.plot_trisurf(
    cube.vertices[:, 0],
    cube.vertices[:, 1],
    cube.vertices[:, 2],
    triangles=cube.faces,
    color=lattice_grid_color,
    edgecolor=lattice_grid_color,
    linestyle='dotted',
    alpha=0
)

# Draw the polygons for the octahedron
draw_polygons(ax, octahedron.vertices, octahedron.faces, voronoi_color)

ax.scatter(0, 0, 0, color=point_color, s=50)

# Plot the vertices
ax.scatter(
    octahedron.vertices[:, 0],
    octahedron.vertices[:, 1],
    octahedron.vertices[:, 2],
    color=voronoi_color,
    label="Voronoi Region of the site at the center"
)

ax.scatter(
    cube.vertices[:, 0],
    cube.vertices[:, 1],
    cube.vertices[:, 2],
    color=point_color,
)

plt.legend()

# Setting the axis off and adjusting the layout
ax.axis("off")
plt.tight_layout()
plt.show()
