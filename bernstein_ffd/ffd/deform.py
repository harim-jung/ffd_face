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
    assert (STU_axes.shape == (3, 3))
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

#https://math.stackexchange.com/questions/413563/potential-division-by-zero-in-the-construction-of-nurbs-basis-functions-how-to
def N(i, p, u, U):

    if p == 0:
        if U[i] <= u and u < U[i + 1]:
            return 1.0
        else:
            return 0.0 # U[i] == U[i+1]
    else:

        if (U[i + p] - U[i]) == 0:  # the case of division by zero
            w1 = 0.0
        else:
            w1 = (u - U[i]) / (U[i + p] - U[i])

        if (U[i + p + 1] - U[i + 1]) == 0:  # the case of division by zero
            w2 = 0.0

        else:
            w2 = (U[i + p + 1] - u) / (U[i + p + 1] - U[i + 1])

        return w1 * N(i, p - 1, u, U) + w2 * N(i + 1, p - 1, u, U)




def N1(i, p, u, U):

    if p == 0:
        if U[i] <= u and u < U[i + 1]:
            return 1.0
        else:
            return 0.0  # U[i] == U[i+1]
    else:

        if (U[i + p] - U[i]) == 0:  # the case of division by zero
            w1 = 0.0
        else:
            w1 = (u - U[i]) / (U[i + p] - U[i])
            #if (w1 < 0):
               # print('w1: N1(i,p,u,U)= N1({0}, {1}, {2}, {3})'.format(i,p,u,U) )
               # print("(u- U[{0}]) / (U[{1}] - U[{2}]) = ( {3} - {4} )/ ( {5} - {6} )  < 0"
               #          .format(i, i + p,  i,  u, U[i], U[i+p], U[i] )  )

        if (U[i + p + 1] - U[i + 1]) == 0:  # the case of division by zero
            w2 = 0.0

        else:
            w2 = (U[i + p + 1] - u) / (U[i + p + 1] - U[i + 1])
            #if (w2 < 0):
                #print('w2: N1(i,p,u,U)= N1({0}, {1}, {2}, {3})'.format(i, p, u, U))
                #print(" (U[{0}] - u) / (U[{1}] - U[{2}]) = ( {3} - {4} )/ ( {5} - {6} )  < 0"
                #      .format(i+p+1,    i + p+1, i+1, U[i + p+1], u, U[i+p+1], U[i+1] ) )


        #print("N1(i, p-1, u, U) = N1({0}, {1}, {2}, U) = {3}".format( i,p-1, u,  N1(i, p-1, u, U) ) )
        #print("N1(i+1, p-1, u, U) = N1({0}, {1}, {2}, U) = {3}".format(i+1, p - 1, u, N1(i+1, p-1, u, U)  ))
        assert N1(i, p-1, u, U) >= 0, "N1(i, p-1, u, U) should be  non-negative"
        assert N1(i+1, p-1, u, U) >= 0, "N1(i+1, p-1, u, V) should be non-negative"

        #The previous property shows that Ni-p,p(u), Ni-p+1,p(u), Ni-p+2,p(u), ..., and Ni,p(u) are non-zero on [ui, ui+1).
        # (1) Basis function Ni,p(u) is non-zero on [ui, ui+p+1). Or, equivalently, Ni,p(u) is non-zero on p+1 knot spans [ui, ui+1), [ui+1, ui+2), ..., [ui+p, ui+p+1).

        # This one states that the sum of these p+1 basis functions is 1.
        # (2) On any knot span [ui, ui+1), at most p+1 degree p basis functions are non-zero, namely: Ni-p,p(u), Ni-p+1,p(u), Ni-p+2,p(u), ..., Ni-1,p(u) and Ni,p(u),



        return w1 * N1(i, p - 1, u, U) + w2 * N1(i + 1, p - 1, u, U)


def N2(i, p, u, U):
    if p == 0:
        if U[i] <= u and u < U[i + 1]:
            return 1.0
        else:
            return 0.0  # U[i] == U[i+1]
    else:

        if (U[i + p] - U[i]) == 0:  # the case of division by zero
            w1 = 0.0
        else:
            w1 = (u - U[i]) / (U[i + p] - U[i])
            #if (w1 < 0):
             #   print('w1: N2(i,p,v,V)= N2({0}, {1}, {2}, {3})'.format(i, p, u, U))
             #   print(" (v- V[{0}]) / (V[{1}] - V[{2}]) = ( {3} - {4} )/ ( {5} - {6} )  < 0"
             #         .format(i, i + p, i, u, U[i], U[i + p], U[i]))

        if (U[i + p + 1] - U[i + 1]) == 0:  # the case of division by zero
            w2 = 0.0

        else:
            w2 = (U[i + p + 1] - u) / (U[i + p + 1] - U[i + 1])
            #if (w2 < 0):
            #    print('w2: N2(i,p,v,V)= N2({0}, {1}, {2}, {3})'.format(i, p, u, U))
            #    print(" (V[{0}] - v) / (V[{1}] - V[{2}]) = ( {3} - {4} )/ ( {5} - {6} )  < 0"
            #          .format(i + p + 1, i + p + 1, i + 1, U[i + p + 1], u, U[i + p + 1], U[i + 1]))

        assert N2(i, p - 1, u, U) >= 0, "N2(i, p-1, u, U) should be non-negative"
        assert N2(i + 1, p - 1, u, U) >= 0, "N2(i+1, p-1, u, V) should be non-negative"

        return w1 * N2(i, p - 1, u, U) + w2 * N2(i + 1, p - 1, u, U)


def N3(i, p, u, U):
    if p == 0:
        if U[i] <= u and u < U[i + 1]:
            return 1.0
        else:
            return 0.0  # U[i] == U[i+1]
    else:

        if (U[i + p] - U[i]) == 0:  # the case of division by zero
            w1 = 0.0
        else:
            w1 = (u - U[i]) / (U[i + p] - U[i])
            #if (w1 < 0):
               # print('w1: N3(i,p,w,W)= N3({0}, {1}, {2}, {3})'.format(i, p, u, U))
               # print(" (w- W[{0}]) / (W[{1}] - W[{2}]) = ( {3} - {4} )/ ( {5} - {6} )  < 0"
               #       .format(i, i + p, i, u, U[i], U[i + p], U[i]))

        if (U[i + p + 1] - U[i + 1]) == 0:  # the case of division by zero
            w2 = 0.0

        else:
            w2 = (U[i + p + 1] - u) / (U[i + p + 1] - U[i + 1])
            #if (w2 < 0):
                #print('w2: N3(i,p,w,W)= N3({0}, {1}, {2}, {3})'.format(i, p, u, U))
                #print(" (W[{0}] - w) / (W[{1}] - W[{2}]) = ( {3} - {4} )/ ( {5} - {6} )  < 0"
                #      .format(i + p + 1, i + p + 1, i + 1, U[i + p + 1], u, U[i + p + 1], U[i + 1]))

        assert N3(i, p - 1, u, U) >= 0, "N3(i, p-1, u, U) should be non-negative"
        assert N3(i + 1, p - 1, u, U) >= 0, "N3(i+1, p-1, u, V) should be non-negative"

        return w1 * N3(i, p - 1, u, U) + w2 * N3(i + 1, p - 1, u, U)



def fun(x, F, U, V, W, P, N):  # G = 1; x = [u,v,w]; F=[x,y,z]: (a+1) x (b+1) x (c+1)  are control points

    u = x[0]
    v = x[1]
    w = x[2]

    R = [0.0, 0.0, 0.0]

    for i in range(P.shape[0]):  # i =0.... a
        for j in range(P.shape[1]):  # j =0.... b
            for k in range(P.shape[2]):  # k =0.... c

                R += P[i, j, k] * N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W)  # Use cubic basis functions

    return R - F

#https://math.stackexchange.com/questions/1486833/how-to-deduce-the-recursive-derivative-formula-of-b-spline-basis
# refer to https://dspace5.zcu.cz/bitstream/11025/852/1/Prochazkova.pdf for the recursive formula for the derivative of N
def dN1(i, p, u, U):
    if  (U[i + p] - U[i]) == 0: # the case of division by zero
        w1 = 0.0
    else:
        w1 = p / (U[i + p] - U[i])

    if (U[i + p + 1] - U[i + 1]) == 0:  # the case of division by zero
        w2 = 0.0
    else:
        w2 = p / (U[i + p +1] - U[i + 1])

    return w1 * N1(i, p - 1, u, U) -  w2 * N1(i + 1, p - 1, u, U)

def dN2(i, p, u, U):
    if  (U[i + p] - U[i]) == 0: # the case of division by zero
        w1 = 0.0
    else:
        w1 = p / (U[i + p] - U[i])

    if (U[i + p + 1] - U[i + 1]) == 0:  # the case of division by zero
        w2 = 0.0
    else:
        w2 = p / (U[i + p +1] - U[i + 1])

    return w1 * N2(i, p - 1, u, U) -  w2 * N2(i + 1, p - 1, u, U)

def dN3(i, p, u, U):
    if  (U[i + p] - U[i]) == 0: # the case of division by zero
        w1 = 0.0
    else:
        w1 = p / (U[i + p] - U[i])

    if (U[i + p + 1] - U[i + 1]) == 0:  # the case of division by zero
        w2 = 0.0
    else:
        w2 = p / (U[i + p +1] - U[i + 1])

    return w1 * N3(i, p - 1, u, U) -  w2 * N3(i + 1, p - 1, u, U)


def jac(x, F, U, V, W, P, N):  # jacobain matrix 3 x 3

    u = x[0]
    v = x[1]
    w = x[2]
    J = np.zeros(shape=(3, 3), dtype=np.float)

    # derivative with respect to u
    # dR(u,v,w)/du = sum_{i = 0 ^ {a}sum_{j = 0} ^ {b}sum_{k = 0} ^ {c}dN(i, 3, u, U) / du N(j, 3, v, V) N(k, 3, w, W) P_ijk


    for i in range(P.shape[0]):  # i =0.... a
        for j in range(P.shape[1]):  # j =0.... b
             for k in range(P.shape[2]):  # k =0.... c

                    J[:, 0] = dN1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W) * P[i, j, k]
                    J[:, 1] = N1(i, 3, u, U) * dN2(j, 3, v, V) * N3(k, 3, w, W) * P[i, j, k]
                    J[:, 2] = N1(i, 3, u, U) * N2(j, 3, v, V) * dN3(k, 3, w, W) * P[i, j, k]

    return J


def initial_guess_for_nonlinear_equations(xyz, l, U,V,W):

    # transform each of xyz to uvw by scaling xyz to the range of uvw specified in U,V,W

    xyz_minimum, xyz_maximum = util.extent(xyz, axis=0)

    xyz_range = xyz_maximum - xyz_minimum


    u_max_index = U.shape[0] - 1
    v_max_index = V.shape[0] - 1
    w_max_index = W.shape[0] - 1
    u_range = U[ u_max_index] - 0
    v_range = V[ v_max_index] - 0
    w_range = W[ w_max_index] - 0

    uvw_range = np.array( [ u_range, v_range, w_range] )
    #uvw = xyz * ( uvw_range / xyz_range )

    # xyz: uvw = xyz_range: uvw_range
    # print('xyz[l]=xyz[{0}={1}'.format(l, xyz[l]) )
    uvw = (xyz[l] - xyz_minimum)  * ( uvw_range / xyz_range )

    return uvw

def xyz_to_uvw_nurbs(xyz, U, V, W, P_lattice, N):  # xyz: vertices of a mesh

    # define nurbs surface: we use the notation used in paper https://asmedigitalcollection.asme.org/computingengineering/article-abstract/8/2/024001/465778/Freeform-Deformation-Versus-B-Spline?redirectedFrom=fulltext

    # extra parameters:
    #  a,b,c: a+1, b+1, c+1 are the numbers of control points in the x, y, z direcitons
    #  p,m,n: ( 1< p <= a, 1< m <= b, 1< n <= c)  the degrees of the basis functionns in u,v,w; default: p,m,n = 3
    #  knot vectors:
    #  U = (u0, u1,...,uq), q = a + p+1,  V = (v0, u1,...,ur), r = b + m+1, W = (w0, w1,...,ws), s = c +n+1
    #  ui = 0 if 0<= i <= p; i - p if p < i <= (q-p-1); q - 2p if (q-p-1) < i <= q

    # xyz = R(u,v,w)
    #      = sum_{i=0}^{a}sum_{j=0}^{b}sum_{k=0}^{c} N_{i,3}(u) N_{j,3}(v) N_{k,3}(w)  p_{ijk}


    # Invert the equation  xyz = R(u,v,w) above: Find (u,v,w) for each (x,y,z) on a given mesh.

    # cf. def fun(x,F, U, V, W, P, N):

    print('xyz.shape=', xyz.shape) #xyz.shape= (35709, 3)
    print('xyz=', xyz)
    uvw = np.zeros(shape=xyz.shape, dtype=np.float)

    for l in range(xyz.shape[0]):
    #for l in range(1):

        #print('xyz[l]=', xyz[l])
        x0 = initial_guess_for_nonlinear_equations(xyz, l, U, V, W)
        #xyz_guess = uvw_to_xyz_nurbs(np.array([x0] ), U, V, W, P_lattice, N)

        #print('initial guess = x0=', x0)
        #print('xyz-guess = ', xyz_guess)

        F = xyz[l]

        args = (F, U, V, W, P_lattice, N)

        #print('x0=', x0)

        sol = optimize.root(fun, x0, args, method='hybr', jac=jac, tol=None)
        uvw[l] = sol.x

    return uvw


def xyz_to_uvw_nurbs_debug(xyz,l, U, V, W, P_lattice, N):  # xyz: vertices of a mesh

    # define nurbs surface: we use the notation used in paper https://asmedigitalcollection.asme.org/computingengineering/article-abstract/8/2/024001/465778/Freeform-Deformation-Versus-B-Spline?redirectedFrom=fulltext

    # extra parameters:
    #  a,b,c: a+1, b+1, c+1 are the numbers of control points in the x, y, z direcitons
    #  p,m,n: ( 1< p <= a, 1< m <= b, 1< n <= c)  the degrees of the basis functionns in u,v,w; default: p,m,n = 3
    #  knot vectors:
    #  U = (u0, u1,...,uq), q = a + p+1,  V = (v0, u1,...,ur), r = b + m+1, W = (w0, w1,...,ws), s = c +n+1
    #  ui = 0 if 0<= i <= p; i - p if p < i <= (q-p-1); q - 2p if (q-p-1) < i <= q

    # xyz = R(u,v,w)
    #      = sum_{i=0}^{a}sum_{j=0}^{b}sum_{k=0}^{c} N_{i,3}(u) N_{j,3}(v) N_{k,3}(w)  p_{ijk}


    # Invert the equation  xyz = R(u,v,w) above: Find (u,v,w) for each (x,y,z) on a given mesh.

    # cf. def fun(x,F, U, V, W, P, N):

    print('xyz.shape=', xyz.shape) #xyz.shape= (35709, 3)
    print('xyz=', xyz)

    uvw = np.zeros(shape=xyz.shape, dtype=np.float)

    #for l in range(xyz.shape[0]):

    print('xyz[l]=', xyz[l])
    x0 = initial_guess_for_nonlinear_equations(xyz, l, U, V, W)
        #xyz_guess = uvw_to_xyz_nurbs(np.array([x0] ), U, V, W, P_lattice, N)


    print('initial guess = x0=', x0)
        #print('xyz-guess = ', xyz_guess)

    F = xyz[l]

    args = (F, U, V, W, P_lattice, N)

        #print('x0=', x0)

    sol = optimize.root(fun, x0, args, method='hybr', jac=jac, tol=None)
    uvw[l] = sol.x

    return uvw




def uvw_to_xyz_nurbs(uvw_points, U, V, W, P_lattice, N):
    xyz = np.zeros(shape=uvw_points.shape, dtype=np.float)

    for l in range(uvw_points.shape[0]):
        uvw = uvw_points[l]


        #print( "the {0} th  uvw {1}=".format(l, uvw) )
        u = uvw[0]
        v = uvw[1]
        w = uvw[2]

        R = 0.0

        for i in range(P_lattice.shape[0]):  # i =0.... a
            for j in range(P_lattice.shape[1]):  # j =0.... b
                for k in range(P_lattice.shape[2]):  # k =0.... c

                    #print("N(i, 3, u, U)= N({0}, 3, {1},U)= {2}".format(i,u, N(i, 3, u, U) ) )
                    #print("N(j, 3, v, V)= N({0}, 3, {1},V)= {2}".format(j, v, N(j, 3, v, V) ) )
                    #print("N(k, 3, w, W)= N({0}, 3, {1},W)= {2}".format(k, w, N(k, 3, w, W) ) )

                    #https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-property.html

                    assert N1(i, 3, u, U) >=0, "N1(i, 3, u, U) should be non-negative"
                    assert N2(j, 3, v, V) >= 0, "N2(j, 3, v, V) should be non-negative"
                    assert N3(k, 3, w, W) >= 0, "N2(k, 3, w, W) should be non-negative"

                    #Local Support -- Ni,p(u) is a non-zero polynomial on [ui,ui+p+1)
                    R += P_lattice[i, j, k] * N1(i, 3, u, U) * N1(j, 3, v, V) * N3(k, 3, w, W)  # Use cubic basis functions
                    #print("intermediate xyz[{0}] = R ={1} ".format(l, R) )

        #print("final  xyz[{0}] = R = {1}".format(l, R))
        xyz[l] = R

    return xyz


def stu_to_xyz(stu_points, stu_origin, stu_axes):
    if stu_axes.shape != (3,):
        raise NotImplementedError()
    return stu_origin + stu_points * stu_axes


def get_stu_control_points(dims):
    stu_lattice = util.mesh3d(
        *(np.linspace(0, 1, d + 1) for d in dims), dtype=np.float32)
    stu_points = np.reshape(stu_lattice, (-1, 3))
    return stu_points


def get_stu_control_points_nurbs(dims, STU_axes):
    stu_lattice = util.mesh3d(
        *(np.linspace(0, STU_axes[i], dims[i] + 1) for i in range(3)), dtype=np.float32)

    #stu_points = np.reshape(stu_lattice, (-1, 3))
    return stu_lattice


def get_control_points(dims, STU_origin, STU_axes):
    stu_points = get_stu_control_points(dims)
    xyz_points = stu_to_xyz(stu_points, STU_origin, STU_axes)
    return xyz_points


# get_control_points_nurbs(dims, stu_origin, stu_axes)
def get_control_points_nurbs(dims, STU_origin, STU_axes):

    stu_lattice = get_stu_control_points_nurbs(dims, STU_axes)

    xyz_lattice = STU_origin + stu_lattice
    return xyz_lattice


def get_stu_deformation_matrix(stu, dims):
    v = util.mesh3d(
        *(np.arange(0, d + 1, dtype=np.int32) for d in dims),
        dtype=np.int32)
    v = np.reshape(v, (-1, 3))

    weights = bernstein_poly(
        n=np.array(dims, dtype=np.int32),
        v=v,
        stu=np.expand_dims(stu, axis=-2))

    b = np.prod(weights, axis=-1)
    return b

#return get_uvw_deformation_matrix_nurbs(uvw,  U, V, W,P_lattice, N)
def get_uvw_deformation_matrix_nurbs(uvw, U, V, W, P_lattice, N):
    # v = util.mesh3d(
    #    *(np.arange(0, d+1, dtype=np.int32) for d in dims),
    #    dtype=np.int32) #V: (a+1) x (b+1) x (c+1)

    # v = np.reshape(v, (-1, 3)) # N x 3 = (a+1) x (b+1) x (c+1)

    # weights = nurbs_weight_matrix(
    #    n=np.array(dims, dtype=np.int32),
    #    v=v,
    #    stu=np.expand_dims(stu, axis=-2))

    weights = np.zeros( shape= ( uvw.shape[0], P_lattice.shape[0] * P_lattice.shape[1] * P_lattice.shape[2]) )

    for l in range(uvw.shape[0]):


        u = uvw[l][0]
        v = uvw[l][1]
        w = uvw[l][2]

        for i in range( P_lattice.shape[0] ): # i=0...a
            for j in range( P_lattice.shape[1]  ): # j=0...b
                for k in range( P_lattice.shape[2] ): # k = 0...c

                    #print("N1(i, 3, u, U)= N({0}, 3, {1},U)= {2}".format(i,u, N1(i, 3, u, U) ) )
                    #print("N2(j, 3, v, V)= N({0}, 3, {1},V)= {2}".format(j, v, N2(j, 3, v, V) ) )
                    #print("N3(k, 3, w, W)= N({0}, 3, {1},W)= {2}".format(k, w, N3(k, 3, w, W) ) )
                    #print("N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W)= {0}".format( N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W) ) )

                    weights[ l, i * (P_lattice.shape[1] * P_lattice.shape[2]) + j * P_lattice.shape[2] + k]\
                        = N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W)

    p =  np.reshape(P_lattice, (-1, 3) )
    return weights, p


def get_deformation_matrix(xyz, dims, stu_origin, stu_axes):
    stu = xyz_to_stu(xyz, stu_origin, stu_axes)
    return get_stu_deformation_matrix(stu, dims)


def get_deformation_matrix_nurbs(xyz,  U, V, W, P_lattice, N):

    uvw = xyz_to_uvw_nurbs(xyz, U, V, W, P_lattice, N) # for debugging use only xyz[0]

    #uvw = xyz_to_uvw_nurbs_debug(xyz, 0, U, V, W, P_lattice, N)  # for debugging use only xyz[0]
    # uvw: [1]
    #print('the first  uvw=\n', uvw[0] )

    #xyz2 = uvw_to_xyz_nurbs(uvw[0:1], U, V, W, P_lattice, N)

    #print('the first  xyz=\n', xyz[0:1])
    #print('the first  xyz2=\n', xyz2)

    return get_uvw_deformation_matrix_nurbs(uvw,  U, V, W, P_lattice,N)


def get_reference_ffd_param(vertices, dims, stu_origin=None, stu_axes=None):
    if stu_origin is None or stu_axes is None:
        if not (stu_origin is None and stu_axes is None):
            raise ValueError(
                'Either both or neither of stu_origin/stu_axes must be None')
        stu_origin, stu_axes = get_stu_params(vertices)

    print('xyz.shape=', vertices.shape) #xyz.shape= (35709, 3)
    b = get_deformation_matrix(vertices, dims, stu_origin, stu_axes)
    p = get_control_points(dims, stu_origin, stu_axes)
    return b, p



def get_reference_ffd_param_nurbs(vertices, U, V, W, P_lattice):
    b,p = get_deformation_matrix_nurbs(vertices,  U, V, W, P_lattice, N)
    return b, p


def deform_mesh(xyz, lattice):
    return trivariate_bernstein(lattice, xyz)


def get_stu_params(vertices):
    minimum, maximum = util.extent(vertices, axis=0)
    stu_origin = minimum
    # stu_axes = np.diag(maximum - minimum)
    stu_axes = maximum - minimum
    return stu_origin, stu_axes
