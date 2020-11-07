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
    points_per_pixel: int = 8
    zfar: float = None
    znear: float = None
    gamma: float = None


class PointsRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of pointclouds.
    """

    def __init__(self, raster_settings):
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
        self.raster_settings = raster_settings


    def forward(self, points_screen, features, sigmas, max_radius) -> PointFragments:
        """
        Args:
            point_clouds: a set of point clouds with coordinates in world space.
        Returns:
            PointFragments: Rasterization outputs as a named tuple.
        """
        raster_settings = self.raster_settings
        idx, color, depth, mask = rasterize_points(
            points_screen,
            features,
            sigmas,
            max_radius,
            image_height=raster_settings.image_height,
            image_width=raster_settings.image_width,
            points_per_pixel=raster_settings.points_per_pixel,
            zfar=raster_settings.zfar,
            znear=raster_settings.znear,
            gamma=raster_settings.gamma
        )
        return color, depth, mask
