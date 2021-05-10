# import numpy as np
# from bernstein_ffd.test_nurbs_ffd_face import *


# import pdb

# # pdb.set_trace()
# # stu_origin, stu_axes = deform.get_stu_params(reference_mesh.T)

# # import pdb; pdb.set_trace()
# a = 6  # a+1 is the number of contol points in the x direction
# b = 6
# c = 6

# # define U,V,W, and P

# #  knot vectors:
# #  U = (u0, u1,...,uq ), q = a + p+1,  V = (v0, u1,...,ur), r = b + m+1, W = (w0, w1,...,ws), s = c +n+1
# #  ui = 0 if 0<= i <= p; i - p if p < i <= (q-p-1); q - 2p if (q-p-1) < i <= q

# p = m = n = 3  # the degree of basis functions

# q = a + p + 1
# r = b + m + 1
# s = c + n + 1

# U = np.zeros(shape=(q + 1,), dtype=np.float)
# V = np.zeros(shape=(r + 1,), dtype=np.float)
# W = np.zeros(shape=(s + 1,), dtype=np.float)

# for i in range(q + 1):  # 0,1,2,....q: the u knot index

#     if i <= p:
#         U[i] = 0
#     else:
#         if i <= (q - p - 1):
#             U[i] = i - p
#         else:
#             U[i] = q - 2 * p

# for i in range(r + 1):
#     if i <= m:
#         V[i] = 0
#     else:
#         if i <= (r - m - 1):
#             V[i] = i - m
#         else:
#             V[i] = r - 2 * m

# for i in range(s + 1):
#     if i <= n:
#         W[i] = 0
#     else:
#         if i <= (s - n - 1):
#             W[i] = i - n
#         else:
#             W[i] = s - 2 * n

# print('U=', U)
# print('V=', V)
# print('W=', W)

# dims = (a, b, c)

# stu_origin, stu_axes = deform.get_stu_params(reference_mesh.T[landmarks])

# P_lattice = deform.get_control_points_nurbs(dims, stu_origin, stu_axes)

# print('control points in 3d lattice form=', P_lattice)

# print('xyz.shape=', reference_mesh.T[landmarks].shape)

# print('xyz.dtype=', reference_mesh.T.dtype )

# #np.float is an alias for python float type. np.float32 and np.float64 are numpy specific 32 and 64-bit float types.
# # np.double == np.float64
# # z = 2.0 's type is float64

# # np.double(reference_mesh.T[landmarks])
# deform_matrix, control_points = test_face_ffd_nurbs(reference_mesh.T[landmarks].astype(np.double), faces,  U, V, W, P_lattice, sample_indices = None)

# #deform_matrix = dic["b"]  # (38365, 216)

# #control_points = dic["p"]  # (216, 3)
# cp_num = control_points.shape[0]

# print('deform_matrix=', deform_matrix)
# print('control_points=', control_points)
# print('num of control points=', cp_num)

# print('deform_matrix.shape=', deform_matrix.shape)
# print('control_points.shape=', control_points.shape)

# BF = open("train.configs/nurbs_deform_matrix_lm.pkl", "wb")
# np.save(BF, deform_matrix)
# BF.close()

# PF = open("train.configs/nurbs_control_points_lm.pkl", "wb")
# np.save(PF, control_points)
# PF.close()


# reconstructed_vertices = deform_matrix @ control_points

# # deform_matrix: M x N
# # P_lattice:  n1 x n2 x n3 x 3

# mesh = np.zeros( shape=( deform_matrix.shape[0], 3), dtype=np.float)

# for l in range( deform_matrix.shape[0]):
#   for i in range(P_lattice.shape[0]):  # i=0...a
#       for j in range(P_lattice.shape[1]):  # j=0...b
#          for k in range(P_lattice.shape[2]):  # k = 0...c

#             # print("N1(i, 3, u, U)= N({0}, 3, {1},U)= {2}".format(i,u, N1(i, 3, u, U) ) )
#             # print("N2(j, 3, v, V)= N({0}, 3, {1},V)= {2}".format(j, v, N2(j, 3, v, V) ) )
#             # print("N3(k, 3, w, W)= N({0}, 3, {1},W)= {2}".format(k, w, N3(k, 3, w, W) ) )
#             # print("N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W)= {0}".format( N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W) ) )

#             mesh[l,:] += deform_matrix[l, i * (P_lattice.shape[1] * P_lattice.shape[2]) + j * P_lattice.shape[2] + k] * P_lattice[i,j,k,:]

# #for l in range(deform_matrix.shape[0]):
# for l in range( reference_mesh.T[landmarks].shape[0] ):
#   print('the original mesh: i={0}: xyz = {1}'.format(l, reference_mesh.T[landmarks][l]))
#   print('the reconstructed mesh (Sum): i={0}: xyz2 = {1}'.format(l, mesh[l]))
#   print('the b@p: i={0}: xyz2 = {1}\n'.format(l, (deform_matrix@control_points)[l] ) )


# import numpy as np
# from bernstein_ffd.test_nurbs_ffd_face import *
# import bernstein_ffd.ffd.deform_parallel

# import pdb

# # pdb.set_trace()
# # stu_origin, stu_axes = deform.get_stu_params(reference_mesh.T)

# # import pdb; pdb.set_trace()
# # 6, 6, 6
# a = 6  # a+1 is the number of contol points in the x direction
# b = 19
# c = 4

# # define U,V,W, and P

# #  knot vectors:
# #  U = (u0, u1,...,uq ), q = a + p+1,  V = (v0, u1,...,ur), r = b + m+1, W = (w0, w1,...,ws), s = c +n+1
# #  ui = 0 if 0<= i <= p; i - p if p < i <= (q-p-1); q - 2p if (q-p-1) < i <= q

# p = m = n = 3  # the degree of basis functions

# q = a + p + 1
# r = b + m + 1
# s = c + n + 1

# U = np.zeros(shape=(q + 1,), dtype=np.float)
# V = np.zeros(shape=(r + 1,), dtype=np.float)
# W = np.zeros(shape=(s + 1,), dtype=np.float)

# for i in range(q + 1):  # 0,1,2,....q: the u knot index

#     if i <= p:
#         U[i] = 0
#     else:
#         if i <= (q - p - 1):
#             U[i] = i - p
#         else:
#             U[i] = q - 2 * p

# for i in range(r + 1):
#     if i <= m:
#         V[i] = 0
#     else:
#         if i <= (r - m - 1):
#             V[i] = i - m
#         else:
#             V[i] = r - 2 * m

# for i in range(s + 1):
#     if i <= n:
#         W[i] = 0
#     else:
#         if i <= (s - n - 1):
#             W[i] = i - n
#         else:
#             W[i] = s - 2 * n

# print('U=', U)
# print('V=', V)
# print('W=', W)

# dims = (a, b, c)

# stu_origin, stu_axes = deform.get_stu_params(reference_mesh.T)

# P_lattice = deform.get_control_points_nurbs(dims, stu_origin, stu_axes)

# print('control points in 3d lattice form=', P_lattice)

# print('xyz.shape=', reference_mesh.T.shape)

# print('xyz.dtype=', reference_mesh.T.dtype)

# #np.float is an alias for python float type. np.float32 and np.float64 are numpy specific 32 and 64-bit float types.
# # np.double == np.float64
# # z = 2.0 's type is float64

# # np.double(reference_mesh.T[landmarks])
# deform_matrix, control_points = test_face_ffd_nurbs(reference_mesh.T.astype(np.double), faces,  U, V, W, P_lattice, sample_indices = None)

# #deform_matrix = dic["b"]  # (38365, 216)

# #control_points = dic["p"]  # (216, 3)
# cp_num = control_points.shape[0]

# print('deform_matrix=', deform_matrix)
# print('control_points=', control_points)
# print('num of control points=', cp_num)

# print('deform_matrix.shape=', deform_matrix.shape)
# print('control_points.shape=', control_points.shape)

# BF = open("train.configs/nurbs_deform_matrix_6_19_4.pkl", "wb")
# np.save(BF, deform_matrix)
# BF.close()

# PF = open("train.configs/nurbs_control_points_6_19_4.pkl", "wb")
# np.save(PF, control_points)
# PF.close()

# reconstructed_vertices = deform_matrix @ control_points

# # deform_matrix: M x N
# # P_lattice:  n1 x n2 x n3 x 3

# mesh = np.zeros(shape=( deform_matrix.shape[0], 3), dtype=np.float)

# for l in range(deform_matrix.shape[0]):
#   for i in range(P_lattice.shape[0]):  # i=0...a
#       for j in range(P_lattice.shape[1]):  # j=0...b
#          for k in range(P_lattice.shape[2]):  # k = 0...c

#             # print("N1(i, 3, u, U)= N({0}, 3, {1},U)= {2}".format(i,u, N1(i, 3, u, U) ) )
#             # print("N2(j, 3, v, V)= N({0}, 3, {1},V)= {2}".format(j, v, N2(j, 3, v, V) ) )
#             # print("N3(k, 3, w, W)= N({0}, 3, {1},W)= {2}".format(k, w, N3(k, 3, w, W) ) )
#             # print("N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W)= {0}".format( N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W) ) )

#             mesh[l,:] += deform_matrix[l, i * (P_lattice.shape[1] * P_lattice.shape[2]) + j * P_lattice.shape[2] + k] * P_lattice[i,j,k,:]

# # for l in range(reference_mesh.T.shape[0]):
# #   print('the original mesh: i={0}: xyz = {1}'.format(l, reference_mesh.T[l]))
# #   print('the reconstructed mesh (Sum): i={0}: xyz2 = {1}'.format(l, mesh[l]))
# #   print('the b@p: i={0}: xyz2 = {1}\n'.format(l, (deform_matrix@control_points)[l] ) )



import numpy as np
from bernstein_ffd.test_nurbs_ffd_face import *
import bernstein_ffd.ffd.deform_parallel

import pdb

# pdb.set_trace()
# stu_origin, stu_axes = deform.get_stu_params(reference_mesh.T)

# import pdb; pdb.set_trace()
# 6, 6, 6
a = 6  # a+1 is the number of contol points in the x direction
b = 6
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

stu_origin, stu_axes = deform_parallel.get_stu_params(reference_mesh.T)

P_lattice = deform_parallel.get_control_points_nurbs(dims, stu_origin, stu_axes)

print('control points in 3d lattice form=', P_lattice)

print('xyz.shape=', reference_mesh.T.shape)

print('xyz.dtype=', reference_mesh.T.dtype)

#np.float is an alias for python float type. np.float32 and np.float64 are numpy specific 32 and 64-bit float types.
# np.double == np.float64
# z = 2.0 's type is float64

# np.double(reference_mesh.T[landmarks])
deform_matrix, control_points = test_face_ffd_nurbs(reference_mesh.T.astype(np.double), faces,  U, V, W, P_lattice, sample_indices = None)

#deform_matrix = dic["b"]  # (38365, 216)

#control_points = dic["p"]  # (216, 3)
cp_num = control_points.shape[0]

print('deform_matrix=', deform_matrix)
print('control_points=', control_points)
print('num of control points=', cp_num)

print('deform_matrix.shape=', deform_matrix.shape)
print('control_points.shape=', control_points.shape)

# BF = open("train.configs/nurbs_deform_matrix_6_19_4.pkl", "wb")
# np.save(BF, deform_matrix)
# BF.close()

# PF = open("train.configs/nurbs_control_points_6_19_4.pkl", "wb")
# np.save(PF, control_points)
# PF.close()

reconstructed_vertices = deform_matrix @ control_points

# deform_matrix: M x N
# P_lattice:  n1 x n2 x n3 x 3

mesh = np.zeros(shape=( deform_matrix.shape[0], 3), dtype=np.float)

for l in range(deform_matrix.shape[0]):
  for i in range(P_lattice.shape[0]):  # i=0...a
      for j in range(P_lattice.shape[1]):  # j=0...b
         for k in range(P_lattice.shape[2]):  # k = 0...c

            # print("N1(i, 3, u, U)= N({0}, 3, {1},U)= {2}".format(i,u, N1(i, 3, u, U) ) )
            # print("N2(j, 3, v, V)= N({0}, 3, {1},V)= {2}".format(j, v, N2(j, 3, v, V) ) )
            # print("N3(k, 3, w, W)= N({0}, 3, {1},W)= {2}".format(k, w, N3(k, 3, w, W) ) )
            # print("N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W)= {0}".format( N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W) ) )

            mesh[l,:] += deform_matrix[l, i * (P_lattice.shape[1] * P_lattice.shape[2]) + j * P_lattice.shape[2] + k] * P_lattice[i,j,k,:]

# for l in range(reference_mesh.T.shape[0]):
#   print('the original mesh: i={0}: xyz = {1}'.format(l, reference_mesh.T[l]))
#   print('the reconstructed mesh (Sum): i={0}: xyz2 = {1}'.format(l, mesh[l]))
#   print('the b@p: i={0}: xyz2 = {1}\n'.format(l, (deform_matrix@control_points)[l] ) )


