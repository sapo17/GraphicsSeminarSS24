### What is Delaunay Triangulation?

Set of triangles made from a discrete set of vertices so that no vertex lies inside the circumcircle of any triangle in the set.

A circumcircle of a triangle is a circle passing through all its vertices.

![Alt text](image.png)

### Computing Delaunay Triangulations

One way is Bowyer-Watson algorithm.

1. Find a triangle containing all our points. Call this "super-triangle".
2. Select a point and look for triangles whose circumcircle contain our point (initially super-triangle).
3. Connect super-triangle vertices to our point to make a new set of triangles.
4. Repeat these steps for all other points.


See [also](https://www.youtube.com/watch?v=GctAunEuHt4&ab_channel=SCIco).
