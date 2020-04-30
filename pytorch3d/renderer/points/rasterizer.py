#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import NamedTuple, Optional
import torch
import torch.nn as nn
import time

from ..cameras import get_world_to_view_transform
from .rasterize_points import rasterize_points


# Class to store the outputs of point rasterization
class PointFragments(NamedTuple):
    idx: torch.Tensor
    zbuf: torch.Tensor
    dists: torch.Tensor


# Class to store the point rasterization params with defaults
class PointsRasterizationSettings(NamedTuple):
    image_height: int = 256
    image_width: int =256
    radius: float = 0.01
    points_per_pixel: int = 8
    bin_size: Optional[int] = None
    max_points_per_bin: Optional[int] = None
    zfar: float = None


class PointsRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of pointclouds.
    """

    def __init__(self, cameras, camera_gk, raster_settings=None):
        """
        cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-screen
                transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = PointsRasterizationSettings()

        self.cameras = cameras
        self.camera_gk = camera_gk
        self.raster_settings = raster_settings

    def transform(self, point_clouds, hom_cloud=None, profile=False, **kwargs) -> torch.Tensor:
        """
        Args:
            point_clouds: a set of point clouds

        Returns:
            points_screen: the points with the vertex positions in screen
            space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        if hom_cloud is not None:
            full_proj_transform = self.camera_gk.full_proj_transform.get_matrix()
            pts_projected = hom_cloud.bmm(full_proj_transform)
            pts_projected_normalised = pts_projected/pts_projected[..., 3:]

            view_transform = self.camera_gk.world_view_transform.get_matrix()
            points_viewspace = hom_cloud.bmm(view_transform)
            points_viewspace = points_viewspace/points_viewspace[..., 3:]

            pts_projected_normalised[..., 2] = points_viewspace[..., 2]
            return pts_projected_normalised


        cameras = kwargs.get("cameras", self.cameras)
        pts_world = point_clouds.points_padded()
        pts_world_packed = point_clouds.points_packed()
        pts_screen = cameras.transform_points(pts_world, **kwargs)

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Remove this line when the convention for the z coordinate in
        # the rasterizer is decided. i.e. retain z in view space or transform
        # to a different range.
        view_transform = get_world_to_view_transform(R=cameras.R, T=cameras.T)
        verts_view = view_transform.transform_points(pts_world)

        pts_screen[..., 2] = verts_view[..., 2]

        # Offset points of input pointcloud to reuse cached padded/packed calculations.

        pad_to_packed_idx = point_clouds.padded_to_packed_idx()
        pts_screen_packed = pts_screen.view(-1, 3)[pad_to_packed_idx, :]
        pts_packed_offset = pts_screen_packed - pts_world_packed
        point_clouds = point_clouds.offset(pts_packed_offset)
        return point_clouds

    def forward(self, point_clouds, hom_point_cloud=None, profile=False, **kwargs) -> PointFragments:
        """
        Args:
            point_clouds: a set of point clouds with coordinates in world space.
        Returns:
            PointFragments: Rasterization outputs as a named tuple.
        """
        if hom_point_cloud is not None:
            points_screen = self.transform(point_clouds, hom_point_cloud, **kwargs)
        if hom_point_cloud is None:
            point_clouds = self.transform(point_clouds, hom_point_cloud, **kwargs)
            points_screen = None

        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        idx, zbuf, dists2 = rasterize_points(
            point_clouds,
            points_screen,
            image_height=raster_settings.image_height,
            image_width=raster_settings.image_width,
            radius=raster_settings.radius,
            points_per_pixel=raster_settings.points_per_pixel,
            bin_size=raster_settings.bin_size,
            max_points_per_bin=raster_settings.max_points_per_bin,
            zfar=raster_settings.zfar
        )
        return PointFragments(idx=idx, zbuf=zbuf, dists=dists2)
