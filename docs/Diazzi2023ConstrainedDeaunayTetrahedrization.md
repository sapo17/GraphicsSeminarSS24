## Constrained Delaunay Tetrahedrization

This file contains notes from Constrained Delaunay Tetrahedrization by Diazzi et al.

Each sentence corresponds to a paragraph in the paper. With each sentence (max. 3 sentences) we try to summarize the paraph with our words.

## Introduction

Constrained Delaunay Tetrahedrization (CDT) can be used for solving partial differential equations (PDEs) [Si 2008], calculation of shape thickness [Cabbiddu and **Attene** 2017], and algorithms that need explicit discretization of a volume.
Diazzi et al. mention some boundary-approximating tetrahedral meshing algorithms that do not obey the boundary constraints without loss of accuracy. In contrast, they propose their algorithm which produces a boundary-preserving CDT.

CDT is one of the triangulations that fulfills the Delaunay criterion to the fullest extent [Alexa 2020].
In 2D, non-intersecting and non-degenerate segments always allow a CDT. 
However, the 3D case is not possible without augmenting the input Piecewise-Linear Complex (PLC) and providing additional Steiner points [Murphy et al 2001, Shewchuck 2002, Si and Gaertner 2005].

Diazzi et al. mention that this approach leads to numerous Steiner points.
They remind the work of [Si and Gaertner 2005] that led to fewer Steiner points.
*tetgen* provides a floating-point implementation of this algorithm.
However, it fails to produce a CDT on around 8.5 of the valid models in the *Thingi10k* dataset.

Diazzi et al. mention that the failure may partly stem from the numerical inaccuracies due to the *tetgen*'s floating-point implementation.
They add that the failure also originates from an incorrect assumption of [Si and Gaertner 2005]'s theory in the implementation.
The authors validate this using an exact-number implementation (CORE library [Burnikel et al 1996; Karamcheti et al. 1999]).

In summary, they propose a novel algorithm for computing the CDT of a valid PLC on HW accelerated floating-point computation without giving up on robustness.
This is achieved by (1) avoiding the tacit assumption that all local cavities formed to insert PLC can be sufficiently expanded.
This leads to a key property of their algorithm without using irrational coordinates for Steiner points.
(2) An exact and efficient implementation of their algorithm using indirect floating-point predicates [**Attene 2020**].

## Related Work

[Hu et al. 2018, 2020] provide a good summary using tetrahedrazation of a volumetric region.

### Delaunay Meshing

Incremental insertion algorithms [Bowyer 1981, Watson 1981] allow the calculation of Delaunay tetrahedrization (DT).
The Delaunay criterion dictates that the circumsphere that bounds the tetrahedron does not contain any point of the input set.
Among robust and efficient methods [Fabri and Pion 2009], DT can also be implemented in a parallelized fashion[Marot et al. 2019].

#### Constrained Delaunay

In 2D, the implementation of constrained Delauney triangulation is straightforward.
The 3D case raises the necessity of Steiner points due to the nature of some polyhedra.
Over the years, there were different methods to tackle the 3D case [Joe 1991, George et al. 1991, Weatherill and Hassan 1994, Guan et al. 2006, Shewchuk 2002].
[Si and Gaertner 2005] proposed an approach that was influenced by the work of [Shewchuck 2002].
[Shewchuck 2002] presented an algorithm that splits all of the edges that neighbors acute vertices.
This led to a provably accurate CDT that protected acute vertices.
To this day, this approach is the state of the art for calculating CDTs in 3D.

#### Conforming Delaunay

Needless to say, algorithms can be developed that use fewer Steiner points.
But, this approach might result in elements that do not fulfill Delaunay criterion.
Conforming tetrahedrizations (e.g., [Cohen-Steiner et al. 2002, Murphy et al. 2001]) introduce many Steiner points and tetrahedras.
But, they produce DT with *improved* boundary subfaces and subedges.
\cmtsch{Not sure if this is an accurate description of what Diazzi et al. meant here}
[Alexa 2020] investigated the possibility of using weights for input vertices with the expectation that resulting DT would need fewer or no Steiner points.
Even if those weights exist, Diazzi et al. conclude that the computational complexity makes them impractical.

#### Delaunay refinement

Tetrahedras can be improved by introducing new vertices at the centers of circumscribing spheres (e.g., [Jamin et al 2015, Ruppert 1995, Shwchuk 1998]).
These approaches ensure termination but may allow slivers in the resulting tetrahedra.
Slivers refers to tetrahedras with close to degenerate volume, but appropriate radius-edge-length ratios.
There are various approaches to address slivers [Alliez et al. 2005, Du and Wang 2003, Tournois et al. 2009, Alexa 2019].
Diazzi et al. compare their approach against CGAL.

### Non-Delaunay Tetrahedral meshing

There are grid-based methods for non-Delaunay tetrahedrization.
These either surround a uniform grid or an adaptive octree around an object.
Diazzi et al. also mention other methods by [Bridson and Doran 2014; Bronson et al.2013; Doran et al.2013; Labelle and Shewchuk 2007, Molino et al. 2003].
But, in essence, they argue that the element quality becomes questionable, especially near the surface.

---
Note: Content will now only be written onto the Overleaf document.















