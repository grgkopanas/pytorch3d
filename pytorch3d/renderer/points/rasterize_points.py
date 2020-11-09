# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional
import torch
import time
from pytorch3d import _C
from pytorch3d.renderer.mesh.rasterize_meshes import pix_to_ndc


# TODO(jcjohns): Support non-square images
def rasterize_points(
    points,
    features,
    sigmas,
    inv_cov,
    max_radius,
    image_height: int = 256,
    image_width: int = 256,
    points_per_pixel: int = 8,
    zfar: float = -0.5,
    znear: float = -0.5,
    gamma: float = None
):

    return _RasterizePoints.apply(
        points,
        features,
        sigmas,
        inv_cov,
        max_radius,
        image_height,
        image_width,
        points_per_pixel,
        zfar,
        znear,
        gamma
    )


class _RasterizePoints(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points,  # (P, 3)
        colors,  # (P, C)
        sigmas,  # (P, 1)
        inv_cov,
        max_radius,
        image_height: int = 256,
        image_width: int = 256,
        points_per_pixel: int = 8,
        zfar: float = -0.5,
        znear: float = -0.5,
        gamma: float = None
    ):
        # TODO: Add better error handling for when there are more than
        # max_points_per_bin in any bin.
        args = (
            points,
            colors,
            sigmas,
            inv_cov,
            max_radius,
            image_height,
            image_width,
            points_per_pixel,
            zfar,
            znear,
            gamma
        )
        idx, color, k_idxs, depth, mask = _C.rasterize_points(*args)
        ctx.znear = znear
        ctx.zfar = zfar
        ctx.gamma = gamma
        ctx.max_radius = max_radius
        ctx.save_for_backward(points, colors, sigmas, idx, k_idxs)
        return idx, color, depth, mask

    @staticmethod
    def backward(ctx, grad_idx, grad_out_color, grad_depth, grad_mask):
        grad_points = None
        grad_colors = None
        grad_sigmas = None
        grad_max_radius = None
        grad_image_height = None
        grad_image_width = None
        grad_points_per_pixel = None
        grad_bin_size = None
        grad_zfar = None
        grad_znear = None
        grad_gamma = None
        znear = ctx.znear
        zfar = ctx.zfar
        gamma = ctx.gamma
        max_radius = ctx.max_radius
        points, colors, sigmas, idx, k_idxs = ctx.saved_tensors
        args = (points, colors, sigmas, max_radius, idx, k_idxs, znear, zfar, gamma, grad_out_color)
        grad_points, grad_colors, grad_sigmas = _C.rasterize_points_backward(*args)
        grads = (
            grad_points,
            grad_colors,
            grad_sigmas,
            grad_max_radius,
            grad_image_height,
            grad_image_width,
            grad_points_per_pixel,
            grad_bin_size,
            grad_zfar,
            grad_znear,
            grad_gamma
        )
        return grads


def rasterize_points_python(
    pointclouds,
    image_size: int = 256,
    radius: float = 0.01,
    points_per_pixel: int = 8,
):
    N = len(pointclouds)
    S, K = image_size, points_per_pixel
    device = pointclouds.device

    points_packed = pointclouds.points_packed()
    cloud_to_packed_first_idx = pointclouds.cloud_to_packed_first_idx()
    num_points_per_cloud = pointclouds.num_points_per_cloud()

    # Intialize output tensors.
    point_idxs = torch.full(
        (N, S, S, K), fill_value=-1, dtype=torch.int32, device=device
    )
    zbuf = torch.full(
        (N, S, S, K), fill_value=-1, dtype=torch.float32, device=device
    )
    pix_dists = torch.full(
        (N, S, S, K), fill_value=-1, dtype=torch.float32, device=device
    )

    # NDC is from [-1, 1]. Get pixel size using specified image size.
    radius2 = radius * radius

    # Iterate through the batch of point clouds.
    for n in range(N):
        point_start_idx = cloud_to_packed_first_idx[n]
        point_stop_idx = point_start_idx + num_points_per_cloud[n]

        # Iterate through the horizontal lines of the image from top to bottom.
        for yi in range(S):
            # Y coordinate of one end of the image. Reverse the ordering
            # of yi so that +Y is pointing up in the image.
            yfix = S - 1 - yi
            yf = pix_to_ndc(yfix, S)

            # Iterate through pixels on this horizontal line, left to right.
            for xi in range(S):
                # X coordinate of one end of the image. Reverse the ordering
                # of xi so that +X is pointing to the left in the image.
                xfix = S - 1 - xi
                xf = pix_to_ndc(xfix, S)

                top_k_points = []
                # Check whether each point in the batch affects this pixel.
                for p in range(point_start_idx, point_stop_idx):
                    px, py, pz = points_packed[p, :]
                    if pz < 0:
                        continue
                    dx = px - xf
                    dy = py - yf
                    dist2 = dx * dx + dy * dy
                    if dist2 < radius2:
                        top_k_points.append((pz, p, dist2))
                        top_k_points.sort()
                        if len(top_k_points) > K:
                            top_k_points = top_k_points[:K]
                for k, (pz, p, dist2) in enumerate(top_k_points):
                    zbuf[n, yi, xi, k] = pz
                    point_idxs[n, yi, xi, k] = p
                    pix_dists[n, yi, xi, k] = dist2
    return point_idxs, zbuf, pix_dists

