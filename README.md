# GVG
A generalized Voronoi diagram or graph (GVG)
is a generalization of the point source voronoi diagram,
or the set of points that are equidistant to two or more 
points in a given domain (see [Wiki](https://en.wikipedia.org/wiki/Voronoi_diagram)).

## A few notes
This is a brute force calculation that iterates over every point in the 
free space and calculates its distance to every point on the boundary set
of points. It's slow. There's also a thresholding issue that can be tailored
by editing `eps' in the gvd main file.
