import numpy as np
import bernstein_ffd.ffd.util as util
from bernstein_ffd.ffd.bernstein import bernstein_poly, trivariate_bernstein

import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import optimize



def xyz_to_stu(xyz, origin, STU_axes):
    if STU_axes.shape == (3,):
        STU_axes = np.diag(STU_axes)
        # raise ValueError(
        #     'stu_axes should have shape (3,), got %s' % str(stu_axes.shape))
    # s, t, u = np.diag(stu_axes)
    assert(STU_axes.shape == (3, 3))
    S, T, U = STU_axes
    TU = np.cross(T, U)
    SU = np.cross(S, U)
    ST = np.cross(S, T)

    diff = xyz - origin

    # TODO: vectorize? np.dot(diff, [tu, su, st]) / ...
    stu = np.stack([
        np.dot(diff, TU) / np.dot(S, TU),
        np.dot(diff, SU) / np.dot(T, SU),
        np.dot(diff, ST) / np.dot(U, ST)
    ], axis=-1)
    return stu




def N(i,p, u, U):

    if (i == 0):
        if ( U[i] <= u && u > U[i+1]):
            return 1.0
        else
            return 0.0
    else:
        if ( u - U[i]) == 0 &&  ( U[i+p] - U[i]) == 0:
            coeff1 = 0.0
        else:
            coeff1 =  ( u - U[i]) / ( U[i+p] - U[i])

        if    ( U[i+p+1] - u) == 0 &&   ( U[i+p+1] - U[i+1]) == 0
            coeff2 = 0.0
        else:
            coeff2  =   ( U[i+p+1] - u) / ( U[i+p+1] - U[i+1])\


        return  coeff1  * N(i, p-1, u, U) +  coeff2 * N(i+1, p-1, u, U)

def fun( x, F, U, V, W, P, N):  # G = 1; x = [u,v,w]; F=[x,y,z]: (a+1) x (b+1) x (c+1)  are control points

    u = x[0]
    v = x[1]
    w = x[2]

    R = 0.0

    for i in range(P.shape[0]):  # i =0.... a
      for j in range(P.shape[1]):  # j =0.... b
          for k in range(P.shape[2]):  # k =0.... c

              R += P[i, j, k] * N(i, 3, u, U) * N(j, 3, v, V) * N(k, 3, w, W)  # Use cubic basis functions

    return R - F

# refer to https://dspace5.zcu.cz/bitstream/11025/852/1/Prochazkova.pdf for the recursive formula for the derivative of N
def dN( i,p, u, U):

    if p ==0 && ( U[i+p] - U[i]) == 0:
        coeff1 = 0.0
    else:
        coeff1 = p /  ( U[i+p] - U[i])

    if p ==0 &&  (U[i+p+1] - U[i+1]) == 0:
        coeff2 = 0.0
    else:
        coeff2 = p /  ( U[i+p] - U[i+1])

    return coeff1 * N(i,p-1,u,U) + coeff2 * N(i+1,p-1,u,U )

def jac( x, F, U, V, W, P,N ): # jacobain matrix 3 x 3

    u = x[0]
    v = x[1]
    w = x[2]
    J = np.zeors( shape = (3,3), dtype=np.float)

    # derivative with respect to u
    # dR(u,v,w)/du = sum_{i = 0 ^ {a}sum_{j = 0} ^ {b}sum_{k = 0} ^ {c}dN(i, 3, u, U) / du N(j, 3, v, V) N(k, 3, w, W) P_ijk

    gradOfXYZwrtuvw = np.zeros( shape=(3,3), dtype=np.float)

    for jj in range(3):

      for i in range(P.shape[0]):  # i =0.... a
        for j in range(P.shape[1]):  # j =0.... b
          for k in range(P.shape[2]):  # k =0.... c
              J[:,jj] = dN(i,3,u, U) * N(j,3,v,V) * N(k,3,w,W) * P[i,j,k]


    return J

#xyz_to_stu_nurbs(p, xyz,  U, V, W, P)
def xyz_to_stu_nurbs(xyz, U,V,W,P,N):  #  xyz: vertices of a mesh

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


    # Invert the equation  xyz = R(s,t,u) above.

    # cf. def fun(x,F, U, V, W, P, N):

    stu = np.zeros( shape = (U.shape[0] * V.shape[0] * W.shape[2], 2),dtype=np.float  )


    for i in range( xyz.shape[0] ):
        F = xyz[i]

        x0 = np.array([ ( U[0] + U[ U.shape[0] ]) /2.0,  (V[0] + V[ V.shape[0] ]) /2.0,  (W[0] + W[ W.shape[0] ] ) /2.0)     # initial guess
        sol = optimize.root(fun, x0, (F, U,V,W,P,N), method='hybr', jac=jac, tol=None)
        stu[i] = sol.x

    return stu



def stu_to_xyz_nurbs(stu_points, U,V,W,P,N):

    xyz = np.zeros( shape = stu_points.shape, dtype = np.float)
    for i in range(stu_points.shape[0]):

      uvw = stu_points[i]

      R = 0.0

      for i in range(P.shape[0]):  # i =0.... a
        for j in range(P.shape[1]):  # j =0.... b
            for k in range(P.shape[2]):  # k =0.... c

                R += P[i, j, k] * N(i, 3, u, U) * N(j, 3, v, V) * N(k, 3, w, W)  # Use cubic basis functions

      xyz[i] = R

    return xyz




def stu_to_xyz(stu_points, stu_origin, stu_axes):
    if stu_axes.shape != (3,):
        raise NotImplementedError()
    return stu_origin + stu_points*stu_axes


def get_stu_control_points(dims):
    stu_lattice = util.mesh3d(
        *(np.linspace(0, 1, d+1) for d in dims), dtype=np.float32)
    stu_points = np.reshape(stu_lattice, (-1, 3))
    return stu_points



def get_stu_control_points_nurbs(dims, STU_axes):
    stu_lattice = util.mesh3d(
        *(np.linspace(0, STU_axes[i], dims[i]+1 ) for i in range(3) ), dtype=np.float32)
    stu_points = np.reshape(stu_lattice, (-1, 3))
    return  stu_points


def get_control_points(dims, STU_origin, STU_axes):
    stu_points = get_stu_control_points(dims)
    xyz_points = stu_to_xyz(stu_points, STU_origin, STU_axes)
    return xyz_points

#get_control_points_nurbs(dims, stu_origin, stu_axes)
def get_control_points_nurbs(dims, STU_origin, STU_axes):

    stu_points = get_stu_control_points_nurbs(dims,  STU_axes)

    xyz_points = STU_origin + stu_points
    return xyz_points

def get_stu_deformation_matrix(stu, dims):
    v = util.mesh3d(
        *(np.arange(0, d+1, dtype=np.int32) for d in dims),
        dtype=np.int32)
    v = np.reshape(v, (-1, 3))

    weights = bernstein_poly(
        n=np.array(dims, dtype=np.int32),
        v=v,
        stu=np.expand_dims(stu, axis=-2))

    b = np.prod(weights, axis=-1)
    return b

def get_stu_deformation_matrix_nurbs(stu, dims, U,V,W, N):
    #v = util.mesh3d(
    #    *(np.arange(0, d+1, dtype=np.int32) for d in dims),
    #    dtype=np.int32) #V: (a+1) x (b+1) x (c+1)

    #v = np.reshape(v, (-1, 3)) # N x 3 = (a+1) x (b+1) x (c+1)

    #weights = nurbs_weight_matrix(
    #    n=np.array(dims, dtype=np.int32),
    #    v=v,
    #    stu=np.expand_dims(stu, axis=-2))

    weights = np.zeros( shape=( stu.shape[0], v.shape[0]*v.shape[1] * v.shape[2]))

    for l in range( stu.shape[0]):
        for i in range( v.shape[0]):
            for j in range( v.shape[1]):
               for k in range( v.shape[2]):
                  u = stu[l][0]
                  v = stu[l][1]
                  w = stu[l][2]
                  weights[l, i * v.shape[1] *  v.shape[2] + j * v.shape[2] + k ] = N(i,3,u,U) * N(j,3,v,V) * N(k,3,w.W)


   return weights




def get_deformation_matrix(xyz, dims, stu_origin, stu_axes):
    stu = xyz_to_stu(xyz, stu_origin, stu_axes)
    return get_stu_deformation_matrix(stu, dims)

def get_deformation_matrix_nurbs(xyz, dims,  U, V, W, P,N):

    stu = xyz_to_stu_nurbs(xyz,  U, V, W, P,N) #xyz_to_stu_nurbs(xyz, U,V,W,P):

    return get_stu_deformation_matrix_nurbs(stu, dims, U,V,W, N) #  get_stu_deformation_matrix_nurbs(stu, dims, U,V,W, N):


def get_reference_ffd_param(vertices, dims, stu_origin=None, stu_axes=None):
    if stu_origin is None or stu_axes is None:
        if not (stu_origin is None and stu_axes is None):
            raise ValueError(
                'Either both or neither of stu_origin/stu_axes must be None')
        stu_origin, stu_axes = get_stu_params(vertices)
    b = get_deformation_matrix(vertices, dims, stu_origin, stu_axes)
    p = get_control_points(dims, stu_origin, stu_axes)
    return b, p



#get_reference_ffd_param_nurbs(xyz, dims, U, V, W, P)
def get_reference_ffd_param_nurbs(vertices, dims, U, V, W, P ):


    b = get_deformation_matrix_nurbs(vertices, dims, U, V, W,P,N)

    return b, P


def deform_mesh(xyz, lattice):
    return trivariate_bernstein(lattice, xyz)


def get_stu_params(vertices):
    minimum, maximum = util.extent(vertices, axis=0)
    stu_origin = minimum
    # stu_axes = np.diag(maximum - minimum)
    stu_axes = maximum - minimum
    return stu_origin, stu_axes
