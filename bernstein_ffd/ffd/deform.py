import numpy as np
import bernstein_ffd.ffd.util as util
from bernstein_ffd.ffd.bernstein import bernstein_poly, trivariate_bernstein


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


def stu_to_xyz_in_3d_lattice(stu_points_3d_lattice, stu_origin, stu_axes): # stu_to_xyz_control_points
    if stu_axes.shape != (3,):
        raise NotImplementedError()
    return stu_origin + stu_points_3d_lattice * stu_axes


def get_stu_control_points(dims):
    #stu_control_lattice = util.mesh3d(
    #    *(np.arange(0, d + 1, dtype=np.int32) for d in dims),  # mesh3d: (nx+1) x (ny+1) x (nz +1) control points
    #    dtype=np.int32)  # v: (nx+1) x (ny+1) x (nz +1) x 3 = stu point (i,j,k) indices

    stu_lattice = util.mesh3d(
        *(np.linspace(0, 1, d+1) for d in dims), dtype=np.float32)

    # stu_lattice: (nx +1) x (ny+1) x (nz+1) x 3: (s,t,u)

    #stu_points = np.reshape(stu_lattice, (-1, 3)) # N x 3, where N =  (nx +1) x (ny+1) x (nz+1)

    return stu_lattice



def get_control_points_in_3d_lattice(dims, STU_origin, STU_axes):
    stu_points_in_3d_lattice = get_stu_control_points(dims)

    xyz_points_in_3d_lattice = stu_to_xyz_in_3d_lattice(stu_points_in_3d_lattice, STU_origin, STU_axes)
    # ==> stu_origin + stu_points * stu_axes

    return xyz_points_in_3d_lattice


def get_stu_deformation_matrix(mesh_stu, dims):
    v = util.mesh3d(
        *(np.arange(0, d+1, dtype=np.int32) for d in dims), # mesh3d: (nx+1) x (ny+1) x (nz +1) control points
        dtype=np.int32) # v: (nx+1) x (ny+1) x (nz +1) x 3 = stu point (i,j,k) indices
    v = np.reshape(v, (-1, 3)) #  N x 3, where N = (nx+1) x (ny+1) x (nz +1)

    weights = bernstein_poly(
        n=np.array(dims, dtype=np.int32), #dims = (6,6,6) => array( [6,6,6])
        v=v,
        stu=np.expand_dims(mesh_stu, axis=-2)) # vecorr (2,) => matrix (2,1) or (2,1)

    #coeff = comb(n, v) N : 3, v: N x 3 comb: N x 3
    #weights = coeff * ((1 - stu) ** (n - v)) * (stu ** v)

    b = np.prod(weights, axis=-1) # product of array elements over a given axis (the last axis), axis=0: across rows, axis=1: across columns
    return b


def get_deformation_matrix(mesh_xyz, dims, stu_origin, stu_axes):
    mesh_stu = xyz_to_stu(mesh_xyz, stu_origin, stu_axes)
    return get_stu_deformation_matrix(mesh_stu, dims)


def get_reference_ffd_param(vertices, dims, stu_origin=None, stu_axes=None):
    if stu_origin is None or stu_axes is None:
        if not (stu_origin is None and stu_axes is None):
            raise ValueError(
                'Either both or neither of stu_origin/stu_axes must be None')
        stu_origin, stu_axes = get_stu_params(vertices)
    b = get_deformation_matrix(vertices, dims, stu_origin, stu_axes)

    control_lattice, p = get_control_points_in_3d_lattice(dims, stu_origin, stu_axes)
    return b, control_lattice, p


def deform_mesh(xyz, lattice):
    return trivariate_bernstein(lattice, xyz)


def get_stu_params(vertices):
    minimum, maximum = util.extent(vertices, axis=0)
    stu_origin = minimum
    # stu_axes = np.diag(maximum - minimum)
    stu_axes = maximum - minimum
    return stu_origin, stu_axes
