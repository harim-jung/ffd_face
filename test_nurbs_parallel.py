import numpy as np
from bernstein_ffd.test_nurbs_ffd_face_parallel import *
# import bernstein_ffd.ffd.deform_parallel as deform_parallel


"""reference meshes"""

# the  face just below the nose
# face# 15997 : vert# (8084 8204 8203)
# face vert 0 : vert# 8084
# position [61.493000 63.313400 -32.137001]
# normal [-0.141119 -4.286755 4.582139]
# face vert 1 : vert# 8204
# position [62.125301 63.311699 -32.132500]
# normal [0.385106 -4.480810 4.356285]
# face vert 2 : vert# 8203
# position [62.130501 63.812500 -31.638201]
# normal [0.303801 -4.853062 3.926230]

# %%


"""Augmented LP reference mesh (only rigid part wo pose) (HELEN_HELEN_3036412907_2_0_1_wo_pose.ply)"""
plydata = PlyData.read('train.configs/HELEN_HELEN_3036412907_2_0_1_wo_pose.ply')
v = plydata['vertex']

vert = np.zeros((3, 35709))
for i, vt in enumerate(v):
 vert[:, i] = np.array(list(vt))

reference_mesh = vert

faces = tri_  # (76073, 3)

landmarks = keypoints_

"""find B and P"""

# %%

# define nurbs surface: we use the notation used in paper https://asmedigitalcollection.asme.org/computingengineering/article-abstract/8/2/024001/465778/Freeform-Deformation-Versus-B-Spline?redirectedFrom=fulltext

# extra parameters:
#  a,b,c: a+1, b+1, c+1 are the numbers of control points in the x, y, z direcitons
#  p,m,n: ( 1< p <= a, 1< m <= b, 1< n <= c)  the degrees of the basis functionns in u,v,w; default: p,m,n = 3
#  knot vectors:
#  U = (u0, u1,...,uq), q = a + p+1,  V = (v0, u1,...,ur), r = b + m+1, W = (w0, w1,...,ws), s = c +n+1
#  ui = 0 if 0<= i <= p; i - p if p < i <= (q-p-1); q - 2p if (q-p-1) < i <= q

# xyz = R(s,t,u) = (R_1(s,t,u), R_2(s,t,u), R_3(s,t,u) ) =
#      = sum_{i=0}^{nx}sum_{j=0}^{ny}sum_{k=0}^{nz} N_{i,3}(s) N_{j,3}(t) N_{k,3}(u) ( p_{ijk}[0], p_{ijk}[1],p_{ijk}[2] )
#     = ( sum_{i=0}^{nx}sum_{j=0}^{ny}sum_{k=0}^{nz} N_{i,3}(s) N_{j,3}(t) N_{k,3}(u)  p_{ijk}[0],
#         sum_{i=0}^{nx}sum_{j=0}^{ny}sum_{k=0}^{nz} N_{i,3}(s) N_{j,3}(t) N_{k,3}(u)  p_{ijk}[1],
#          sum_{i=0}^{nx}sum_{j=0}^{ny}sum_{k=0}^{nz} N_{i,3}(s) N_{j,3}(t) N_{k,3}(u)  p_{ijk}[2] )


# import pdb

# pdb.set_trace()
# stu_origin, stu_axes = deform.get_stu_params(reference_mesh.T)

# import pdb; pdb.set_trace()
a = 8  # a+1 is the number of contol points in the x direction
b = 11
c = 6

# define U,V,W, and P

#  knot vectors:
#  U = (u0, u1,...,uq ), q = a + p+1,  V = (v0, u1,...,ur), r = b + m+1, W = (w0, w1,...,ws), s = c +n+1
#  ui = 0 if 0<= i <= p; i - p if p < i <= (q-p-1); q - 2p if (q-p-1) < i <= q

p = m = n = 3  # the degree of basis functions

q = a + p + 1
r = b + m + 1
s = c + n + 1

U = np.zeros(shape=(q + 1,), dtype=np.float)
V = np.zeros(shape=(r + 1,), dtype=np.float)
W = np.zeros(shape=(s + 1,), dtype=np.float)

for i in range(q + 1):  # 0,1,2,....q: the u knot index

 if i <= p:
  U[i] = 0
 else:
  if i <= (q - p - 1):
   U[i] = i - p
  else:
   U[i] = q - 2 * p

for i in range(r + 1):
 if i <= m:
  V[i] = 0
 else:
  if i <= (r - m - 1):
   V[i] = i - m
  else:
   V[i] = r - 2 * m

for i in range(s + 1):
 if i <= n:
  W[i] = 0
 else:
  if i <= (s - n - 1):
   W[i] = i - n
  else:
   W[i] = s - 2 * n

print('U=', U)
print('V=', V)
print('W=', W)

dims = (a, b, c)

stu_origin, stu_axes = deform_parallel.get_stu_params(reference_mesh.T)#[landmarks])

#stu_axes contains the directions and the extents of the bounding box of the face

# control_points_dist = np.array(
#     [
#     [  1.0/a * i for i in range(a+1)  ],
#     [   1.0/b * i for i in range(b+1)   ] ,
#     [   1.0/c * i for i in range(c+1)   ]
#     ]
# )

# 8, 21, 4 (considering lips)
# control_points_dist = np.array(
#     [
#     [0, 0.14, 0.29, 0.4 , 0.52, 0.63, 0.75, 0.87, 1],
#     [0, 0.1, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.5, 0.65, 0.8, 0.9, 1],
#     [1.0/c * i for i in range(c+1)]
#     ]
# )

# 8, 11, 6 (considering lips)
x_cp_ratio = get_control_points_dist_dim(2, 5, 2, axis="x", middle="mouth_large")
y_cp_ratio = get_control_points_dist_dim(2, 5, 5, axis="y", middle="mouth_large")
z_cp_ratio = np.round(np.array([1.0/c * i for i in range(c+1)]), decimals=2)
control_points_dist = np.array(
    [
    list(x_cp_ratio),
    list(y_cp_ratio),
    list(z_cp_ratio)
    ]
)
# [list([0.0, 0.12, 0.24, 0.37, 0.51, 0.64, 0.77, 0.88, 1.0])
#  list([0.0, 0.08, 0.16, 0.22, 0.27, 0.32, 0.38, 0.5, 0.63, 0.75, 0.88, 1.0])
#  list([0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0])
#  ]

print(control_points_dist)

# # 13, 21, 4 (considering lips, more points on x-axis)
# control_points_dist = np.array(
#     [
#     [0, 0.14, 0.29, 0.34, 0.39, 0.44, 0.49, 0.54, 0.6, 0.65, 0.7, 0.75, 0.87, 1],
#     [0, 0.1, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.5, 0.65, 0.8, 0.9, 1],
#     [1.0/c * i for i in range(c+1)]
#     ]
# )

# 16, 23, 4 (considering eyes and lips)
# control_points_dist = np.array(
#     [
#     [0, 0.15, 0.21, 0.26, 0.3, 0.35, 0.39, 0.44, 0.49, 0.54, 0.61, 0.65, 0.7, 0.74, 0.79, 0.9, 1],
#     [0, 0.1, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.5, 0.64, 0.66, 0.68, 0.8, 0.9, 1],
#     [1.0/c * i for i in range(c+1)]
#     ]
# )


#a_control_points = [ stu_axes[0] * i  for i in [0,  0.2, 0.3, 0.5, 0.7, 0.9,1.0  ] ]
#b_control_points = [ stu_axes[1] * i  for i in [0,  0.2, 0.3, 0.5, 0.7, 0.9,1.0  ] ]
#c_control_points =  [ stu_axes[2] * i  for i in [0,  0.2, 0.3, 0.5, 0.7, 0.9,1.0  ] ]

#P_lattice = deform_parallel.get_control_points_nurbs(dims, stu_origin, stu_axes)

P_lattice = deform_parallel.get_control_points_nurbs_nonuniform(dims, stu_origin, stu_axes, control_points_dist)

# print('control points in 3d lattice form=', P_lattice)

# print('xyz.shape=', reference_mesh.T[landmarks].shape)

# print('xyz.dtype=', reference_mesh.T.dtype)

# np.float is an alias for python float type. np.float32 and np.float64 are numpy specific 32 and 64-bit float types.
# np.double == np.float64
# z = 2.0 's type is float64

# np.double(reference_mesh.T[landmarks])

# https://forums.fast.ai/t/use-of-parallel-function-in-fastai-core/35704/6
# parallel( func, arr:Collection, max_works:int=None)
# call func on every element of arr in parallel using max_workers
# func must accept two values: both the value of the array element and the index of the array element e.g. def my_func(value, index)


deform_matrix, control_points = test_face_ffd_nurbs(reference_mesh.T.astype(np.double), faces, U, V, W,
                                                    P_lattice, sample_indices=None)
# deform_matrix, control_points = test_face_ffd_nurbs(reference_mesh.T[landmarks].astype(np.double), faces, U, V, W,
#                                                     P_lattice, sample_indices=None)

# deform_matrix = dic["b"]  # (38365, 216)

# control_points = dic["p"]  # (216, 3)
cp_num = control_points.shape[0]

print('deform_matrix=', deform_matrix)
print('control_points=', control_points)
print('num of control points=', cp_num)

print('deform_matrix.shape=', deform_matrix.shape)
print('control_points.shape=', control_points.shape)

reconstructed_vertices = deform_matrix @ control_points

BF = open("train.configs/nurbs_deform_matrix_8_11_6.pkl", "wb")
np.save(BF, deform_matrix)
BF.close()

PF = open("train.configs/nurbs_control_points_8_11_6.pkl", "wb")
np.save(PF, control_points)
PF.close()

# deform_matrix: M x N
# P_lattice:  n1 x n2 x n3 x 3

# mesh = np.zeros(shape=(deform_matrix.shape[0], 3), dtype=np.double)

# for l in range(deform_matrix.shape[0]):
#  for i in range(P_lattice.shape[0]):  # i=0...a
#   for j in range(P_lattice.shape[1]):  # j=0...b
#    for k in range(P_lattice.shape[2]):  # k = 0...c

#     # print("N1(i, 3, u, U)= N({0}, 3, {1},U)= {2}".format(i,u, N1(i, 3, u, U) ) )
#     # print("N2(j, 3, v, V)= N({0}, 3, {1},V)= {2}".format(j, v, N2(j, 3, v, V) ) )
#     # print("N3(k, 3, w, W)= N({0}, 3, {1},W)= {2}".format(k, w, N3(k, 3, w, W) ) )
#     # print("N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W)= {0}".format( N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W) ) )

#     mesh[l, :] += deform_matrix[
#                    l, i * (P_lattice.shape[1] * P_lattice.shape[2]) + j * P_lattice.shape[2] + k] * P_lattice[i, j, k, :]

# for l in range(deform_matrix.shape[0]):
# for l in range(reference_mesh.T[landmarks].shape[0]):
#  print('the original mesh: i={0}: xyz = {1}'.format(l, reference_mesh.T[landmarks][l]))
#  print('the reconstructed mesh (Sum): i={0}: xyz2 = {1}'.format(l, mesh[l]))
#  print('the b@p: i={0}: xyz2 = {1}\n'.format(l, (deform_matrix @ control_points)[l]))

