#!/usr/bin/env python3
# coding: utf-8


"""
Modified from https://raw.githubusercontent.com/YadiraF/PRNet/master/utils/render.py
"""

__author__ = 'cleardusk'

import numpy as np
from .cython import mesh_core_cython


cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


def is_point_in_tri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: [u, v] or [x, y]
        tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[:, 2] - tp[:, 0]
    v1 = tp[:, 1] - tp[:, 0]
    v2 = point - tp[:, 0]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00 * dot11 - dot01 * dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)


def render_colors(vertices, colors, tri, h, w, c=3):
    """ render mesh by z buffer
    Args:
        vertices: 3 x nver
        colors: 3 x nver
        tri: 3 x ntri
        h: height
        w: width
    """
    # initial
    image = np.zeros((h, w, c))

    depth_buffer = np.zeros([h, w]) - 999999.
    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[2, tri[0, :]] + vertices[2, tri[1, :]] + vertices[2, tri[2, :]]) / 3.
    tri_tex = (colors[:, tri[0, :]] + colors[:, tri[1, :]] + colors[:, tri[2, :]]) / 3.

    for i in range(tri.shape[1]):
        tri_idx = tri[:, i]  # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[0, tri_idx]))), 0)
        umax = min(int(np.floor(np.max(vertices[0, tri_idx]))), w - 1)

        vmin = max(int(np.ceil(np.min(vertices[1, tri_idx]))), 0)
        vmax = min(int(np.floor(np.max(vertices[1, tri_idx]))), h - 1)

        if umax < umin or vmax < vmin:
            continue

        for u in range(umin, umax + 1):
            for v in range(vmin, vmax + 1):
                if tri_depth[i] > depth_buffer[v, u] and is_point_in_tri([u, v], vertices[:2, tri_idx]):
                    depth_buffer[v, u] = tri_depth[i]
                    image[v, u, :] = tri_tex[:, i]
    return image


def crender_colors(vertices, triangles, colors, h, w, c=3, BG=None):
    """ render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        colors: [nver, 3]
        h: height
        w: width
        c: channel
        BG: background image
    Returns:
        image: [h, w, c]. rendered image./rendering.
    """

    if BG is None:
        image = np.zeros((h, w, c), dtype=np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG.astype(np.float32).copy(order='C')
    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.

    # to C order
    vertices = vertices.astype(np.float32).copy(order='C')
    triangles = triangles.astype(np.int32).copy(order='C')
    colors = colors.astype(np.float32).copy(order='C')

    mesh_core_cython.render_colors_core(
        image, vertices, triangles,
        colors,
        depth_buffer,
        vertices.shape[0], triangles.shape[0],
        h, w, c
    )
    return image


def ncc(vertices):
    # matrix version
    v_min = np.min(vertices, axis=1).reshape(-1, 1)
    v_max = np.max(vertices, axis=1).reshape(-1, 1)
    ncc_vertices = (vertices - v_min) / (v_max - v_min)

    return ncc_vertices


def cpncc(img, vertices_lst, tri):
    """cython version for PNCC render: original paper"""
    h, w = img.shape[:2]
    c = 3

    pnccs_img = np.zeros((h, w, c))
    # pncc_code = np.full((3, 53215), 0.6)
    pncc_code = np.full((3, 38365), 0.6)
    for i in range(len(vertices_lst)):
        vertices = vertices_lst[i]
        pncc_img = crender_colors(vertices.T, tri.T, pncc_code.T, h, w, c)
        pnccs_img[pncc_img > 0] = pncc_img[pncc_img > 0]

    pnccs_img = pnccs_img.squeeze() * 255
    return pnccs_img


def cpncc_v2(img, vertices_lst, tri):
    """cython version for PNCC render"""
    h, w = img.shape[:2]
    c = 3

    pnccs_img = np.zeros((h, w, c))
    for i in range(len(vertices_lst)):
        vertices = vertices_lst[i]
        ncc_vertices = ncc(vertices)
        pncc_img = crender_colors(vertices.T, tri.T, ncc_vertices.T, h, w, c)
        pnccs_img[pncc_img > 0] = pncc_img[pncc_img > 0]

    pnccs_img = pnccs_img.squeeze() * 255
    return pnccs_img


def main():
    pass


if __name__ == '__main__':
    main()
