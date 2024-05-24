manifold  can be thought of as a surface or shape that is continuous and without any breaks or disjointed parts. It locally (around every small neighborhood of each point) resembles flat Euclidean space,

Manifold:
Very rough: "nice" space in geometry, like arc length parametrized curves
It is simplifying assumption how our space looks which simplifies analysis and computation.
When we zoom in and look local local neighborhood at any point on the surface, it looks like R^n.
(Use figure on video Lecture2B Keenan Crane, 04:05)

Manifold connectivity: 
(http://15462.courses.cs.cmu.edu/fall2020/lecture/meshes/slide_025#:~:text=keenan-,Great%20questions,-%2D%2D%2Dthis%20is%20a) Manifold connectivity is the "fans but not fins" condition: every edge must be contained in two polygons (or one on the boundary); every vertex must be contained in a single loop of polygons (or a strip of polygons on the boundary). This condition only has to do with how mesh elements are connected up; the vertex positions are totally irrelevant.
 each edge is shared by at most two faces
To see if a triangle mesh is a manifold, we do:
1. every edge must be contained in exactly two triangles. Or, if you have an edge on the boundary of your domain (i.e., surface stops), then one triangle contains that edge.
2. Every vertex in a manifold must be contained in a single fan (or loop) of triangles. Or, given boundary vertex, it is contained in a single fan of triangles.

(Find formal definition either in Sharp 21 or Krane Discrete Diff geometry Lecture 2.)
(Use figure on video Lecture2B Keenan Crane, 13:37)



right intuition for a manifold is: if you pick any point in your shape, and zoom in far enough, you should be able to draw a regular coordinate grid with two distinct directions. (earth from space vs. from a specific location)

Manifold connectivity:

A manifold polygon mesh has fans but not fins.
Two simple conditions can be checked:
1. Every edge is contained in only two polygons (No fins).
2. Every vertex in a manifold should be contained in a single fan (or loop) of polygons.

