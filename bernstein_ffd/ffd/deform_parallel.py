import numpy as np
import bernstein_ffd.ffd.util as util
from bernstein_ffd.ffd.bernstein import bernstein_poly, trivariate_bernstein

import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import optimize
from fastai.vision import *


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


# https://math.stackexchange.com/questions/413563/potential-division-by-zero-in-the-construction-of-nurbs-basis-functions-how-to
def N(i, p, u, U):
    # N1, 3(u) is non - zero on [u1, u2), [u2, u3), [u3, u4) and [u4, u5).
    # Or, equivalently, it is non - zero on[u1, u5).

    # In general, Basis function Ni,p(u) is non-zero on [ui, ui+p+1).
    # Or, equivalently, Ni,p(u) is non-zero on p+1 knot spans [ui, ui+1), [ui+1, ui+2), ..., [ui+p, ui+p+1).

    # In general, Basis function Ni,p(u) is non-zero on [ui, ui+p+1).
    # Or, equivalently, Ni,p(u) is non-zero on p+1 knot spans [ui, ui+1), [ui+1, ui+2), ..., [ui+p, ui+p+1).

    # u is within U[0:-1]; N(i,p,u,U) is not zero

    if p == 0:
        if U[i] <= u and u < U[i + 1]:
            return 1.0
        else:
            return 0.0  # U[i] == U[i+1]

    else:
        if (U[i + p] - U[i]) == 0:
            w1 = 0.0
        else:
            w1 = (u - U[i]) / (U[i + p] - U[i])

        if (U[i + p + 1] - U[i + 1]) == 0:
            w2 = 0.0

        else:
            w2 = (U[i + p + 1] - u) / (U[i + p + 1] - U[i + 1])

            # Ni, p-1(u) is non - zero on[ui, ui + (p-1) + 1).
            # Ni+1, p-1(u) is non - zero on[ui+1, ui+1 + (p-1) + 1).
        return w1 * N(i, p - 1, u, U) + w2 * N(i + 1, p - 1, u, U)


def within(u, U):
    if (u >= U[0] and u < U[-1]):
        return True;


def funf(x, F, U, V, W, P, N):  # G = 1; x = [u,v,w]; F=[x,y,z]: (a+1) x (b+1) x (c+1)  are control points

    u = x[0]
    v = x[1]
    w = x[2]

    R = [0.0, 0.0, 0.0]

    for i in range(P.shape[0]):  # i =0.... a
        for j in range(P.shape[1]):  # j =0.... b
            for k in range(P.shape[2]):  # k =0.... c

                # N1, 3(u) is non - zero on [u1, u2), [u2, u3), [u3, u4) and [u4, u5).
                # Or, equivalently, it is non - zero on[u1, u5).
                # In general, Basis function Ni,p(u) is non-zero on [ui, ui+p+1).
                # Or, equivalently, Ni,p(u) is non-zero on p+1 knot spans [ui, ui+1), [ui+1, ui+2), ..., [ui+p, ui+p+1).
                if within(u, U[i:(i + 3 + 1) + 1]) and within(v, V[j:(j + 3 + 1) + 1]) and within(w,
                                                                                                  W[k:(k + 3 + 1) + 1]):
                    R += P[i, j, k] * N(i, 3, u, U) * N(j, 3, v, V) * N(k, 3, w, W)

    return R - F


# https://math.stackexchange.com/questions/1486833/how-to-deduce-the-recursive-derivative-formula-of-b-spline-basis
# refer to https://dspace5.zcu.cz/bitstream/11025/852/1/Prochazkova.pdf for the recursive formula for the derivative of N

def dN(i, p, u, U):
    if (U[i + p] - U[i]) == 0:  # the case of division by zero
        w1 = 0.0
    else:
        w1 = p / (U[i + p] - U[i])

    if (U[i + p + 1] - U[i + 1]) == 0:  # the case of division by zero
        w2 = 0.0
    else:
        w2 = p / (U[i + p + 1] - U[i + 1])

    return w1 * N(i, p - 1, u, U) - w2 * N(i + 1, p - 1, u, U)


def jacf(x, F, U, V, W, P, N):  # jacobain matrix 3 x 3

    u = x[0]
    v = x[1]
    w = x[2]
    J = np.zeros(shape=(3, 3), dtype=np.double)

    # derivative with respect to u
    # dR(u,v,w)/du = sum_{i = 0 ^ {a}sum_{j = 0} ^ {b}sum_{k = 0} ^ {c}dN(i, 3, u, U) / du N(j, 3, v, V) N(k, 3, w, W) P_ijk

    # print("shape of P=", P.shape)
    for i in range(P.shape[0]):  # i =0.... a
        for j in range(P.shape[1]):  # j =0.... b
            for k in range(P.shape[2]):  # k =0.... c

                if within(u, U[i:(i + 3 + 1) + 1]) and within(v, V[j:(j + 3 + 1) + 1]) \
                        and within(w, W[k:(k + 3 + 1) + 1]):
                    J[:, 0] += dN(i, 3, u, U) * N(j, 3, v, V) * N(k, 3, w, W) * P[i, j, k, :]
                    J[:, 1] += N(i, 3, u, U) * dN(j, 3, v, V) * N(k, 3, w, W) * P[i, j, k, :]
                    J[:, 2] += N(i, 3, u, U) * N(j, 3, v, V) * dN(k, 3, w, W) * P[i, j, k, :]

    return J


def initial_guess_for_nonlinear_equations(xyz, U, V, W):
    # transform each of xyz to uvw by scaling xyz to the range of uvw specified in U,V,W

    uvw_ = np.zeros(shape=xyz.shape, dtype=np.double)

    xyz_minimum, xyz_maximum = util.extent(xyz, axis=0)

    xyz_range = xyz_maximum - xyz_minimum

    u_range = U[-1] - U[0]
    v_range = V[-1] - V[0]
    w_range = W[-1] - W[0]

    uvw_range = np.array([u_range, v_range, w_range])
    # uvw = xyz * ( uvw_range / xyz_range )

    # xyz: uvw = xyz_range: uvw_range
    # print('xyz[l]=xyz[{0}={1}'.format(l, xyz[l]) )
    uvw_[:] = (xyz[:] - xyz_minimum) / xyz_range * uvw_range  # uvm: 68 x 3

    print('uvw=', uvw_)

    u_min_index = np.argmin(uvw_[:, 0])
    v_min_index = np.argmin(uvw_[:, 1])
    w_min_index = np.argmin(uvw_[:, 2])

    u_max_index = np.argmax(uvw_[:, 0])
    v_max_index = np.argmax(uvw_[:, 1])
    w_max_index = np.argmax(uvw_[:, 2])

    print('u_max_index={0}: {1}'.format(u_max_index, uvw_[u_max_index]))
    print('v_max_index={0}: {1}'.format(v_max_index, uvw_[v_max_index]))
    print('w_max_index={0}: {1}'.format(w_max_index, uvw_[w_max_index]))

    print('xyz_range=', xyz_range)
    print('uvw_range=', uvw_range)

    uvw_[:] = uvw_[:] * 0.999

    # uvw[u_min_index, 0] += 0.1
    # uvw[v_min_index, 1] += 0.1
    # uvw[w_min_index, 2] += 0.1

    # uvw[u_max_index, 0] -=  0.1
    # uvw[v_max_index, 1] -=  0.1
    # uvw[w_max_index, 2] -=  0.1

    print('new u_min_index={0}: {1}'.format(u_min_index, uvw_[u_min_index]))
    print('new v_min_index={0}: {1}'.format(v_min_index, uvw_[v_min_index]))
    print('new w_min_index={0}: {1}'.format(w_min_index, uvw_[w_min_index]))

    print('new u_max_index={0}: {1}'.format(u_max_index, uvw_[u_max_index]))
    print('new v_max_index={0}: {1}'.format(v_max_index, uvw_[v_max_index]))
    print('new w_max_index={0}: {1}'.format(w_max_index, uvw_[w_max_index]))

    return uvw_


# Find_root_nurbs_ffd_each(uvw_par, uvw0, uvw_min, uvw_max, U, V, W, P_lattice, N)
#uvw_par = np.zeros(shape=xyz.shape, dtype=np.double)  # global variable set
class Find_root_nurbs_ffd_each(object):
    def __init__(self, xyz_shape, uvw0, uvw_min, uvw_max, U, V, W, P_lattice, N):

        self.xyz_shape = xyz_shape
        self.uvw0 = uvw0
        self.uvw_min = uvw_min
        self.uvw_max = uvw_max
        self.U = U
        self.V = V
        self.W = W
        self.P_lattice = P_lattice
        self.N = N

        self.uvw_par = np.array(shape = xyz_shape, dtype = np.double)

    def __call__(self, xyz_l, l):
        # def find_root_nurbs_ffd( xyz_i,  i):

        print('find_root_nurbs_ffd_each:')
        print("original xyz[{0}] = {1}".format(l, xyz_l))
        print('initial guess uvw0[{0}]= {1}\n'.format(l, self.uvw0[l]))
        xyz2 = uvw_to_xyz_nurbs_one(self.uvw0[l], self.U, self.V, self.W, self.P_lattice, self.N)
        print('xyz-guess [{0}]= {1}'.format(l, xyz2))

        F = xyz_l

        args = (
        F, self.U, self.V, self.W, self.P_lattice, self.N)  # How are these variables bound? By "parallel" function?

        sol = optimize.root(funf, self.uvw0[l], args, method='hybr', jac=jacf, tol=None)

        # x, info, ier, mesg = optimize.fsolve(fun, x0, args, fprime=jac)
        # uvw_, info, ier, mesg = optimize.fsolve(fun, uvw0[l], args, fprime=jac, full_output = 1 )
        # options = {'maxfev': 1000}

        # sol = optimize.root(fun, x0, args, method='hybr') # options=options)

        # compute the error of the equations:

        xyz2 = uvw_to_xyz_nurbs_one(sol.x, self.U, self.V, self.W, self.P_lattice, self.N)
        error = np.linalg.norm(xyz2 - F)

        n = 0  # The number of trials for new initial guess



        while (error > 1.0):
            n += 1
            print("l ={0}, n={1}: The error = {2} is greater than 1.0. Retry with a new initial guess.".format(l, n,
                                                                                                                  error))

            deviation = np.random.uniform(low=-1.0, high=1.0, size=(3,))
            uvw0_new = self.uvw0[l] + deviation

            uvw0_new_arr = np.clip(uvw0_new, self.uvw_min, self.uvw_max) * 0.999

            print("l ={0}, n ={1}:  Retry with a new initial guess = {2}.".format(l, n, uvw0_new_arr))

            sol = optimize.root(funf, uvw0_new_arr, args, method='hybr', jac=jacf, tol=None)

            xyz2 = uvw_to_xyz_nurbs_one(sol.x, self.U, self.V, self.W, self.P_lattice, self.N)
            error = np.linalg.norm(xyz2 - F)

        #print("At l ={0}, n={1}: The  error = {2} is less than 1.0, and succeeds.\n".format(l, n, error))

        self.uvw_par[l] = sol.x
        print('SUCCESS: find_root_nurbs_ffd_each: l ={0}, n={1}: uvw_par[{2}]={3}, error = {4}'.format(l,n, l, self.uvw_par[l], error))
        print("xyz's:  l={0}, n ={1}:  original xyz={2}: computed xyz={3}\n".format(l, n,  xyz_l, xyz2) )


# parallel version

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

    uvw_min = np.array([U[0], V[0], W[0]])
    uvw_max = np.array([U[-1], V[-1], W[-1]])
    # print('xyz.shape=', xyz.shape) #xyz.shape= (35709, 3)
    # print('xyz=', xyz)

    # global uvw_par
    # If a variable is assigned a value anywhere within the function’s body,
    # it’s assumed to be a local unless explicitly declared as global.

    #uvw_par = np.zeros(shape=xyz.shape, dtype=np.double)  # global variable set
    # The global variable uvw is used in find_root_nurbs_ffd()

    uvw0 = initial_guess_for_nonlinear_equations(xyz, U, V, W)

    # xyz_guess = uvw_to_xyz_nurbs(uvw0, U, V, W, P_lattice, N)
    # class partial:
    # """New function with partial application of the given arguments
    #   and keywords.
    # """

    #find_root_nurbs_ffd_each = Find_root_nurbs_ffd_each(uvw_par, uvw0, uvw_min, uvw_max, U, V, W, P_lattice, N)
    find_root_nurbs_ffd_each = Find_root_nurbs_ffd_each(xyz.shape, uvw0, uvw_min, uvw_max, U, V, W, P_lattice, N)

    parallel(find_root_nurbs_ffd_each,
             xyz)  # U, W, P_lattice, N are needed by find_root_nurbs_ffd(), how are they passed into it??

    print('find_root_nurbs_ffd_each=', find_root_nurbs_ffd_each)

    print('find_root_nurbs_ffd_each.uvw_par=', find_root_nurbs_ffd_each.uvw_par)
    for l in range(xyz.shape[0]):
        print('xyz_to_uvw_nurbs: landmark uvw {0}: uvw = {1}'.format(l, find_root_nurbs_ffd_each.uvw_par[l]))
    return find_root_nurbs_ffd_each.uvw_par


# Uvw_to_xyz_nurbs_each (xyz_par,  U, V, W, P_lattice, N)
class Uvw_to_xyz_nurbs_each(object):

    def __init__(self, xyz_par_shape, U, V, W, P_lattice, N):


        self.U = U
        self.V = V
        self.W = W
        self.P_lattice = P_lattice
        self.N = N

        self.xyz_par = np.array( shape = xyz_par_shape, dtype = np.double)

    def __call__(self, uvw_l, l):
        # print( "the {0} th  uvw {1}=".format(l, uvw) )
        u = uvw_l[0]
        v = uvw_l[1]
        w = uvw_l[2]

        R = np.array([0.0, 0.0, 0.0])

        for i in range(self.P_lattice.shape[0]):  # i =0.... a
            for j in range(self.P_lattice.shape[1]):  # j =0.... b
                for k in range(self.P_lattice.shape[2]):  # k =0.... c

                    # print("N(i, 3, u, U)= N({0}, 3, {1},U)= {2}".format(i,u, N(i, 3, u, U) ) )
                    # print("N(j, 3, v, V)= N({0}, 3, {1},V)= {2}".format(j, v, N(j, 3, v, V) ) )
                    # print("N(k, 3, w, W)= N({0}, 3, {1},W)= {2}".format(k, w, N(k, 3, w, W) ) )

                    # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-property.html

                    assert self.N(i, 3, u, self.U) >= 0, "N1(i, 3, u, U) should be non-negative"
                    assert self.N(j, 3, v, self.V) >= 0, "N2(j, 3, v, V) should be non-negative"
                    assert self.N(k, 3, w, self.W) >= 0, "N2(k, 3, w, W) should be non-negative"

                    if within(u, self.U[i:(i + 3 + 1) + 1]) and within(v, self.V[j:(j + 3 + 1) + 1])\
                               and within(w, self.W[k:(k + 3 + 1) + 1]):
                        # Local Support -- Ni,p(u) is a non-zero polynomial on [ui,ui+p+1)
                        R += self.P_lattice[i, j, k] * self.N(i, 3, u, self.U) * self.N(j, 3, v, self.V) * self.N(k, 3,
                                                                                                                  w,
                                                                                                                  self.W)  # Use cubic basis functions
                    # R += P_lattice[i, j, k] * N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W)  # Use cubic basis functions
                    # print("intermediate xyz[{0}] = R ={1} ".format(l, R) )

        print("computed  xyz[{0}] ( uvw_l = {1} ) = {2}".format(l, uvw_l,  R))

        self.xyz_par[l] = R


# xyz2 = uvw_to_xyz_nurbs_one( sol.x, U, V, W, P_lattice, N)

def uvw_to_xyz_nurbs_one(uvw_l, U, V, W, P_lattice, N):
    # print( "the {0} th  uvw {1}=".format(l, uvw) )
    u = uvw_l[0]
    v = uvw_l[1]
    w = uvw_l[2]

    R = np.array([0.0, 0.0, 0.0])

    for i in range(P_lattice.shape[0]):  # i =0.... a
        for j in range(P_lattice.shape[1]):  # j =0.... b
            for k in range(P_lattice.shape[2]):  # k =0.... c

                # print("N(i, 3, u, U)= N({0}, 3, {1},U)= {2}".format(i,u, N(i, 3, u, U) ) )
                # print("N(j, 3, v, V)= N({0}, 3, {1},V)= {2}".format(j, v, N(j, 3, v, V) ) )
                # print("N(k, 3, w, W)= N({0}, 3, {1},W)= {2}".format(k, w, N(k, 3, w, W) ) )

                # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-property.html

                assert N(i, 3, u, U) >= 0, "N1(i, 3, u, U) should be non-negative"
                assert N(j, 3, v, V) >= 0, "N2(j, 3, v, V) should be non-negative"
                assert N(k, 3, w, W) >= 0, "N2(k, 3, w, W) should be non-negative"

                if within(u, U[i:(i + 3 + 1) + 1]) and within(v, V[j:(j + 3 + 1) + 1]) and within(w,
                                                                                                  W[k:(k + 3 + 1) + 1]):
                    # Local Support -- Ni,p(u) is a non-zero polynomial on [ui,ui+p+1)
                    R += P_lattice[i, j, k] * N(i, 3, u, U) * N(j, 3, v, V) * N(k, 3, w, W)  # Use cubic basis functions
                    # R += P_lattice[i, j, k] * N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W)  # Use cubic basis functions
                    # print("intermediate xyz[{0}] = R ={1} ".format(l, R) )

    # print("final  xyz[{0}] = R = {1}".format(l, R))

    return R


# The code looks like this:
# parallel(partial(resize_one, path=p, size=size), il.items)
# which I want to change to:
# for anitem in il.items:
# resize_one(path=p,size=size)
# but resize_one requires ‘fn’ and ‘i’ parameters, which are not passed in the current code and I cannot see where they are magically coming from?

# Thanks. ChrisP.

# # parallel version
# xyz_par = uvw_to_xyz_nurbs(uvw_par, U, V, W, P_lattice, N)  # parallel version
def uvw_to_xyz_nurbs(uvw_points, U, V, W, P_lattice, N):
    # global xyz_par
    #xyz_par = np.zeros(shape=uvw_points.shape, dtype=np.double)
    xyz_par_shape = uvw_points.shape
    uvw_to_xyz_nurbs_each = Uvw_to_xyz_nurbs_each(xyz_par_shape, U, V, W, P_lattice, N)
    parallel(uvw_to_xyz_nurbs_each, uvw_points)

    # parallel( func, arr ) ==> [func,o,i) for i,o in enumerate(arr)]

    return uvw_to_xyz_nurbs_each.xyz_par


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
        *(np.linspace(0, STU_axes[i], dims[i] + 1) for i in range(3)),
        dtype=np.float32)  # stu_lattice : shape = 6 x 6 x 6 x3

    # stu_points = np.reshape(stu_lattice, (-1, 3))
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


# Get_uvw_deformation_matrix_nurb_each_row(weights_par, U, V, W, P_lattice, N)
class Get_uvw_deformation_matrix_nurb_each_row(object):

    def __init__(self, weights_par_shape, U, V, W, P_lattice, N):

        self.U = U
        self.V = V
        self.W = W
        self.P_lattice = P_lattice
        self.N = N

        self.weights_par = np.array( np.zeros(shape= weights_par_shape, dtype=np.double),
                                       dtype = np.double)

    def __call__(self, uvw_l, l):

        u = uvw_l[0]
        v = uvw_l[1]
        w = uvw_l[2]

        for i in range(self.P_lattice.shape[0]):  # i=0...a
            for j in range(self.P_lattice.shape[1]):  # j=0...b
                for k in range(self.P_lattice.shape[2]):  # k = 0...c

                    # print("N1(i, 3, u, U)= N({0}, 3, {1},U)= {2}".format(i,u, N1(i, 3, u, U) ) )
                    # print("N2(j, 3, v, V)= N({0}, 3, {1},V)= {2}".format(j, v, N2(j, 3, v, V) ) )
                    # print("N3(k, 3, w, W)= N({0}, 3, {1},W)= {2}".format(k, w, N3(k, 3, w, W) ) )
                    # print("N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W)= {0}".format( N1(i, 3, u, U) * N2(j, 3, v, V) * N3(k, 3, w, W) ) )

                    if within(u, self.U[i:(i + 3 + 1) + 1]) and within(v, self.V[j:(j + 3 + 1) + 1]) and within(w,
                                                                                                                self.W[
                                                                                                                k:(
                                                                                                                          k + 3 + 1) + 1]):
                        self.weights_par[
                            l, i * (self.P_lattice.shape[1] * self.P_lattice.shape[2]) + j * self.P_lattice.shape[
                                2] + k] \
                            = self.N(i, 3, u, self.U) * self.N(j, 3, v, self.V) * self.N(k, 3, w, self.W)


# return get_uvw_deformation_matrix_nurbs(uvw,  U, V, W,P_lattice, N)
# #parallel version

def get_uvw_deformation_matrix_nurbs(uvw_par, U, V, W, P_lattice, N):
    # v = util.mesh3d(
    #    *(np.arange(0, d+1, dtype=np.int32) for d in dims),
    #    dtype=np.int32) #V: (a+1) x (b+1) x (c+1)

    # v = np.reshape(v, (-1, 3)) # N x 3 = (a+1) x (b+1) x (c+1)

    # weights = nurbs_weight_matrix(
    #    n=np.array(dims, dtype=np.int32),
    #    v=v,
    #    stu=np.expand_dims(stu, axis=-2))

    weights_par_shape = (uvw_par.shape[0], P_lattice.shape[0] * P_lattice.shape[1] * P_lattice.shape[2])

    get_uvw_deformation_matrix_nurb_each_row = Get_uvw_deformation_matrix_nurb_each_row(weights_par_shape, U, V, W, P_lattice,
                                                                                        N)

    parallel(get_uvw_deformation_matrix_nurb_each_row, uvw_par)
    # print('uvw.shape =', uvw.shape)
    # for l in range(uvw.shape[0]):

    p = np.reshape(P_lattice, (-1, 3))  # N x 3
    return get_uvw_deformation_matrix_nurb_each_row.weights_par, p  # weights: M x N


def get_deformation_matrix(xyz, dims, stu_origin, stu_axes):
    stu = xyz_to_stu(xyz, stu_origin, stu_axes)
    return get_stu_deformation_matrix(stu, dims)


def get_deformation_matrix_nurbs(xyz, U, V, W, P_lattice, N):
    # import pdb
    # pdb.set_trace()
    print('parallel version:::::::::::::::::::::::::::::::::::')

    print('parallel version:::::::::::::::::::::::::::::::::::')

    print('parallel version:::::::::::::::::::::::::::::::::::')

    uvw_par = xyz_to_uvw_nurbs(xyz, U, V, W, P_lattice, N)  # parallel version


    xyz_par = uvw_to_xyz_nurbs(uvw_par, U, V, W, P_lattice, N)  # parallel version

    b, p = get_uvw_deformation_matrix_nurbs(uvw_par, U, V, W, P_lattice, N)  # parallel version


    for i in range(xyz.shape[0]):
        print('landmark uvw {0}: uvw = {1}'.format(i, uvw_par[i]))
        print('the landmark {0}:  origin xyz={1}'.format(i, xyz[i]))
        print('the landmark {0}:  nurbs ffd xyz (sum)={1}\n'.format(i, xyz_par[i]))
        print('b@p [{0}]  = xyz = {1}\n'.format(i, (b @ p)[i]), )

    return b, p


def get_reference_ffd_param(vertices, dims, stu_origin=None, stu_axes=None):
    if stu_origin is None or stu_axes is None:
        if not (stu_origin is None and stu_axes is None):
            raise ValueError(
                'Either both or neither of stu_origin/stu_axes must be None')
        stu_origin, stu_axes = get_stu_params(vertices)

    # print('xyz.shape=', vertices.shape) #xyz.shape= (35709, 3)
    b = get_deformation_matrix(vertices, dims, stu_origin, stu_axes)
    p = get_control_points(dims, stu_origin, stu_axes)
    return b, p


def get_reference_ffd_param_nurbs(vertices, U, V, W, P_lattice):
    b, p = get_deformation_matrix_nurbs(vertices, U, V, W, P_lattice, N)

    # print("b=\n")
    # print(b)
    return b, p


def deform_mesh(xyz, lattice):
    return trivariate_bernstein(lattice, xyz)


def get_stu_params(vertices):
    minimum, maximum = util.extent(vertices, axis=0)
    stu_origin = minimum
    # stu_axes = np.diag(maximum - minimum)
    stu_axes = maximum - minimum
    return stu_origin, stu_axes
