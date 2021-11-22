import cv2
import numpy as np
import numpy.ma
import os
import sys
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Iterable
from collections import deque

# Extracts the (row, column) indices from a flattened array of indices
def getIdxInImage(flat):
    return np.unravel_index(flat, (N, M), order='C')


# Returns the clearance of a point q when input q has dimensions 1 x # boundary pts
def getClearance(q):
    return np.amin(q)


# Returns the index of the clearance point when q has dimensions 1 x # boundary pts
def getClearanceIdx(q):
    return np.argmin(q)


# Borrowed this from an online resource. Adapted for python from C++
# https://answers.opencv.org/question/66758/how-would-you-get-all-the-points-on-a-line/
# This function just interpolates points between a line between (x0,y0) and (x1,y1)
def linePoints(x0, y0, x1, y1):
    # Empty array
    points_of_line = []

    dx = abs(x1 - x0)
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    dy = abs(y1 - y0)
    if y0 < y1:
        sy = 1
    else:
        sy = -1

    if dx > dy:
        err = dx / 2
    else:
        err = -dy / 2

    while True:
        points_of_line.append(np.array([x0, y0]))

        if x0 == x1 and y0 == y1:
            break

        e2 = err
        if e2 > -dx:
            err -= dy
            x0 += sx
        if e2 < dy:
            err += dx
            y0 += sy

    return points_of_line


# Borrowed this function from online as well
# https://www.python.org/doc/essays/graphs/
def find_shortest_path(graph, start_node, end_node):
    dist = {start_node: [start_node]}
    q = deque([start_node])
    while len(q):
        at = q.popleft()
        for next_node in graph[at]:
            if next_node not in dist:
                dist[next_node] = [dist[at], next_node]
                q.append(next_node)
    return dist.get(end_node)


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


file = sys.argv[1]
start_x_in = int(sys.argv[2])
start_x = start_x_in
start_y_in = int(sys.argv[3])
start_y = start_y_in
goal_x_in = int(sys.argv[4])
goal_x = goal_x_in
goal_y_in = int(sys.argv[5])
goal_y = goal_y_in

# file = 'env2_200x200_50dpi.png'


file_parts = os.path.splitext(os.path.basename(file))
file_name = file_parts[0]

img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

# Make sure that all pixels are either 255 or 0
N, M = img.shape
if (start_x >= M or start_x < 0 or start_y < 0 or start_y >= N
        or goal_x < 0 or goal_x >= M or goal_y >= N or goal_y < 0):
    print("Valid start points must be inside the environment")
    print("Try selecting a point x in [0," + str(M - 1) + "] and y in [0," + str(N - 1) + "]")
    quit()
# It is assumed that the start and goals are with respect to an
# origin at the bottom left
# First switch to make sure that their entered points make sense in
# row, column notation where (row <=> y and column <=> x)
# start_x = 50
# start_y = 20
start_x, start_y = start_y, start_x
# goal_x = 160
# goal_y = 160
goal_x, goal_y = goal_y, goal_x
# Now scale the starting and goal points to cast
# points into origin at top left
goal_x = N - goal_x
start_x = N - start_x

for i in range(N):
    for j in range(M):
        if img[i][j] < 128:
            img[i][j] = 0
        else:
            img[i][j] = 255

# Compute the boundary and interior points
bnd_pts = np.zeros([N, M])
int_pts = np.zeros([N, M])

for i in range(1, N - 1):
    for j in range(1, M - 1):
        if img[i][j] == 0:
            if ((img[i - 1][j] == 255) or (img[i + 1][j] == 255) or
                    (img[i][j - 1] == 255) or (img[i][j + 1] == 255)):
                bnd_pts[i][j] = 1
            else:
                int_pts[i][j] = 1

#cv2.imshow('image', int_pts)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Make sure the border is boundary
bnd_pts[0, :] = 1
bnd_pts[:, 0] = 1
bnd_pts[N - 1, :] = 1
bnd_pts[:, M - 1] = 1


if bnd_pts[start_x, start_y] == 1:
    print("Start point selected is in an obstacle.\nNO PATH EXISTS!")
    quit()
elif bnd_pts[goal_x, goal_y] == 1:
    print("Goal point selected is in an obstacle.\nNO PATH EXISTS!")
    quit()

#cv2.imshow('image', bnd_pts)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Now compute the free region
c_free = np.zeros([N, M])
for i in range(N):
    for j in range(M):
        if (bnd_pts[i][j] != 1) and (int_pts[i][j] != 1):
            c_free[i][j] = 1

# Remove any redundant boundary points
# for i in range(1, N - 1):
#    for j in range(1, M - 1):
#        if bnd_pts[i][j] == 1:
#            if bnd_pts[i + 1][j] == 1 or bnd_pts[i - 1][j] == 1:
#                bnd_pts[i][j] = 0

#cv2.imshow('image', c_free)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# get all the points from c_free as a flattened array
c_flat = c_free.flatten('C')
bnd_flat = bnd_pts.flatten('C')

# Get all of the indices from these as tuples
idx = np.arange(0, N * M, 1)

c_mask = numpy.ma.make_mask(c_flat)
bnd_mask = numpy.ma.make_mask(bnd_flat)

c_idx = idx[c_mask]
b_idx = idx[bnd_mask]

# b_idx = b_idx[::2]

nBndPts = len(b_idx)
nCpts = len(c_idx)

c_free_xy = map(getIdxInImage, c_idx)
c_free_xy = np.array(list(c_free_xy))

bnd_xy = map(getIdxInImage, b_idx)
bnd_xy = np.array(list(bnd_xy))

obstacles = -1 * (c_free - 1)

#cv2.imshow('image', obstacles)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

obstacles_flat = obstacles.flatten('C')
obstacles_mask = np.ma.make_mask(obstacles_flat)
obstacles_idx = idx[obstacles_mask]
obstacle_xy = np.array(list(map(getIdxInImage, obstacles_idx)))
obstacle_xy = StandardScaler().fit_transform(obstacle_xy)

bnd_in_obstacle = np.zeros([len(obstacles_idx)])
for i in range(len(obstacles_idx)):
    for j in range(len(b_idx)):
        if obstacles_idx[i] == b_idx[j]:
            bnd_in_obstacle[i] = 1

db = DBSCAN(eps=0.1, min_samples=1).fit(obstacle_xy)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

bnd_cluster_label = labels[np.ma.make_mask(bnd_in_obstacle)]

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

# if distance matrix exists in file, load it, else recompute
dist_mtrx_file = file_name + ".csv"
if os.path.exists(dist_mtrx_file):
    dist_mtrx = np.genfromtxt(dist_mtrx_file, delimiter=',')
else:
    dist_mtrx = np.zeros([nCpts, nBndPts])
    cnt_c_free = 0
    gvd = np.zeros([N, M])
    for c in c_free_xy:
        cnt_b_free = 0
        for b in bnd_xy:
            dist_mtrx[cnt_c_free][cnt_b_free] = np.linalg.norm(c - b)
            cnt_b_free += 1

        cnt_c_free += 1

    np.savetxt(dist_mtrx_file, dist_mtrx, delimiter=',')

# Compute the clearance vector
clearance = np.array(list(map(getClearance, dist_mtrx[:])))
clearance_idx = np.array(list(map(getClearanceIdx, dist_mtrx[:])))

# Here's the tricky part. If there is an obstacle that clings
# to the border of the image, then it will have a label of '0'
# effectively guaranteeing spurious points around it
# so we do a check
nBorderPixels = 2 * N + 2 * M - 4
obstacle_on_border = False
first_cluster = labels[labels == 0]
if len(first_cluster) > nBorderPixels:
    obstacle_on_border = True
    print("There's an obstacle on the border!")

gvd_file = file_name + "_gvd" + ".csv"

if os.path.exists(gvd_file):
    gvd = np.genfromtxt(gvd_file, delimiter=',')
else:
    gvd = np.zeros([N, M])
    eps = 1.9
    eps_border = 0.01
    for i in range(nCpts):
        cnt = 1
        for j in range(nBndPts):
            if j != clearance_idx[i]:  # We need to exclude the one that gives us the clearance
                if bnd_cluster_label[clearance_idx[i]] == 0 and ~obstacle_on_border:
                    if abs(dist_mtrx[i][j] - clearance[i]) < eps_border:
                        # Check to see if it belongs to the same cluster
                        # Cluster zero is by default the boundary
                        cnt += 1
                        if cnt > 1:
                            (x_free, y_free) = c_free_xy[i]
                            gvd[x_free, y_free] = 1
                            break
                elif bnd_cluster_label[clearance_idx[i]] == 0 and obstacle_on_border:
                    (x_cl, y_cl) = getIdxInImage(clearance_idx[i])
                    if (x_cl == N - 1 or x_cl == 0) and (y_cl == M - 1 or y_cl == 0):
                        if abs(dist_mtrx[i][j] - clearance[i]) < eps_border:
                            # Check to see if it belongs to the same cluster
                            # Cluster zero is by default the boundary
                            cnt += 1
                            if cnt > 1:
                                (x_free, y_free) = c_free_xy[i]
                                gvd[x_free, y_free] = 1
                                break
                elif bnd_cluster_label[j] != bnd_cluster_label[clearance_idx[i]]:
                    if abs(dist_mtrx[i][j] - clearance[i]) < eps:
                        cnt += 1
                        if cnt > 1:
                            (x_free, y_free) = c_free_xy[i]
                            gvd[x_free, y_free] = 1
                            break

    np.savetxt(gvd_file, gvd, delimiter=',')

cv2.imshow('GVG', gvd)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Now that we have the GVG, we need to find a valid path
gvd_flat = gvd.flatten('C')
gvd_idx = idx[np.ma.make_mask(gvd_flat)]
gvd_in_image = np.array(list(map(getIdxInImage, gvd_idx)))
nGvdPnts = len(gvd_idx)

start = np.array([start_x, start_y])
if gvd[start_x, start_y] == 1:
    print("Start point exists on GVG already!")
else:
    print("Start point not on GVG. Finding closest point...")
    dist_s_to_gvd = np.zeros([nGvdPnts])
    for i in range(nGvdPnts):
        dist_s_to_gvd[i] = np.linalg.norm(start - gvd_in_image[i])

    # Closest point
    min_dist_to_gvd_idx = np.argmin(dist_s_to_gvd)

    # Now we turn on all of the pixels between start and end
    cx, cy = gvd_in_image[min_dist_to_gvd_idx]
    points_to_fill = linePoints(start_x, start_y, cx, cy)
    for i in range(len(points_to_fill)):
        if (bnd_pts[points_to_fill[i][0], points_to_fill[i][1]] == 1 or
                int_pts[points_to_fill[i][0], points_to_fill[i][1]] == 1):
            print("Impossible to reach GVG from start point. No path exists!")
            quit()
        else:
            gvd[points_to_fill[i][0], points_to_fill[i][1]] = 1

goal = np.array([goal_x, goal_y])
if gvd[goal_x, goal_y] == 1:
    print("Goal point exists on GVG already!")
else:
    print("Goal point not on GVG. Finding closest point...")
    dist_s_to_gvd = np.zeros([nGvdPnts])
    for i in range(nGvdPnts):
        dist_s_to_gvd[i] = np.linalg.norm(goal - gvd_in_image[i])

    # Closest point
    min_dist_to_gvd_idx = np.argmin(dist_s_to_gvd)

    # Now we turn on all of the pixels between start and end
    cx, cy = gvd_in_image[min_dist_to_gvd_idx]
    points_to_fill = linePoints(cx, cy, goal_x, goal_y)
    for i in range(len(points_to_fill)):
        if (bnd_pts[points_to_fill[i][0], points_to_fill[i][1]] == 1 or
                int_pts[points_to_fill[i][0], points_to_fill[i][1]] == 1):
            print("Impossible to reach goal point from GVG. No path exists!")
            quit()
        else:
            gvd[points_to_fill[i][0], points_to_fill[i][1]] = 1

# Construct the dictionary containing key:value pairs of all accessible gvd points
gvd_flat = gvd.flatten('C')
gvd_idx = idx[np.ma.make_mask(gvd_flat)]
gvd_in_image = np.array(list(map(getIdxInImage, gvd_idx)))
nGvdPnts = len(gvd_idx)
graph = {}
for i in range(nGvdPnts):
    x, y = gvd_in_image[i]
    accessible_set = []
    for j in range(nGvdPnts):
        if i != j:
            x1, y1 = gvd_in_image[j]
            if ((x + 1 == x1 and y == y1) or (x - 1 == x1 and y == y1) or
                    (x == x1 and y + 1 == y1) or (x == x1 and y - 1 == y1) or
                    (x + 1 == x1 and y + 1 == y1) or (x - 1 == x1 and y + 1 == y1) or
                    (x + 1 == x1 and y - 1 == y1) or (x - 1 == x1 and y - 1 == y1) or
                    (x + 2 == x1 and y == y1) or (x - 2 == x1 and y == y1) or
                    (x == x1 and y + 2 == y1) or (x == x1 and y - 2 == y1) or
                    (x + 1 == x1 and y + 2 == y1) or (x + 1 == x1 and y - 2 == y1) or
                    (x - 1 == x1 and y + 2 == y1) or (x - 1 == x1 and y - 2 == y1) or
                    (x + 2 == x1 and y + 1 == y1) or (x - 2 == x1 and y + 1 == y1) or
                    (x - 2 == x1 and y - 1 == y1) or (x - 2 == x1 and y - 1 == y1) or
                    (x + 3 == x1 and y == y1) or (x - 3 == x1 and y == y1) or
                    (x == x1 and y + 3 == y1) or (x == x1 and y - 3 == y1)):
                accessible_set.append(j)

    graph[i] = accessible_set

start_idx = 0
for i in range(nGvdPnts):
    x, y = gvd_in_image[i]
    if x == start_x and y == start_y:
        start_idx = i
        break

goal_idx = 0
for i in range(nGvdPnts):
    x, y = gvd_in_image[i]
    if x == goal_x and y == goal_y:
        goal_idx = i
        break

path_to_goal = find_shortest_path(graph, start_idx, goal_idx)
path_to_goal = list(flatten(path_to_goal))
path_idx = gvd_idx[path_to_goal]
path_mtrx = np.zeros([N, M])
for i in range(len(path_to_goal)):
    (x, y) = getIdxInImage(path_idx[i])
    path_mtrx[x, y] = 1

# Create an BGR image to display
# Blue color
path_color = [0, 0, 255]
# Black tracks are the GVG
gvg_color = [0, 0, 0]
# Start is green
start_color = [0, 255, 0]
# Goal is red
goal_color = [0, 0, 255]
# Obstacles are black (0, 0, 0)
obstacle_color = [0, 0, 0]
# Free space is white (255, 255, 255)
free_space_color = [255, 255, 255]
bnd_pts = bnd_pts + int_pts
diagram = np.zeros([N, M, 3])
for i in range(N):
    for j in range(M):
        for cl in range(3):
            if i == start_x and j == start_y:
                diagram[i, j, cl] = start_color[cl]
            elif i == goal_x and j == goal_y:
                diagram[i, j, cl] = goal_color[cl]
            elif path_mtrx[i, j] == 1:
                diagram[i, j, cl] = path_color[cl]
            elif gvd[i, j] == 1:
                diagram[i, j, cl] = gvg_color[cl]
            elif c_free[i, j] == 1:
                diagram[i, j, cl] = free_space_color[cl]
            elif bnd_pts[i, j] == 1:
                diagram[i, j, cl] = obstacle_color[cl]

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 0.25
# Blue color in BGR
color = (0, 0, 0)
# Line thickness of 2 px
thickness = 1

# org
org = (50, 50)

# Using cv2.putText() method
image = cv2.putText(diagram, 'Start', (start_y + 1, start_x + 1), font,
                    fontScale, color, thickness)
image = cv2.putText(diagram, 'Goal', (goal_y - 1, goal_x + 1), font,
                    fontScale, color, thickness)

window_name = "GVG for " + file_name
cv2.imshow(window_name, diagram)
cv2.waitKey(0)
cv2.destroyAllWindows()

input_pos_str = str(start_x_in) + "_" + str(start_y_in) + "_" + str(goal_x_in) + "_" + str(goal_y_in)
cv2.imwrite(file_name + "_" + input_pos_str + "_with_path.png", diagram)
