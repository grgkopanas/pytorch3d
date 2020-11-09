// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>

// ****************************************************************************
// *                          NAIVE RASTERIZATION                             *
// ****************************************************************************


#ifdef WITH_CUDA

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizePointsGKCuda(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    const torch::Tensor& sigmas,
    const torch::Tensor& inv_cov,
    const int max_radius,
    const int image_height,
    const int image_width,
    const int points_per_pixel,
    const float zfar,
    const float znear,
    const float gamma);
#endif



// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizePointsBackwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    const torch::Tensor& sigmas,
    const int max_radius,
    const torch::Tensor& idxs,
    const torch::Tensor& k_idxs,
    const float znear,
    const float zfar,
    const float gamma,
    const torch::Tensor& grad_out_color);
#endif
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  idxs: int32 Tensor of shape (N, H, W, K) (from forward pass)
//  grad_zbuf: float32 Tensor of shape (N, H, W, K) giving upstream gradient
//             d(loss)/d(zbuf) of the distances from each pixel to its nearest
//             points.
//  grad_dists: Tensor of shape (N, H, W, K) giving upstream gradient
//              d(loss)/d(dists) of the dists tensor returned by the forward
//              pass.
//
// Returns:
//  grad_points: float32 Tensor of shape (N, P, 3) giving downstream gradients
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizePointsBackward(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    const torch::Tensor& sigmas,
    const int max_radius,
    const torch::Tensor& idxs,
    const torch::Tensor& k_idxs,
    const float znear,
    const float zfar,
    const float gamma,
    const torch::Tensor& grad_out_color) {
  if (points.type().is_cuda()) {
#ifdef WITH_CUDA
    return RasterizePointsBackwardCuda(points, colors, sigmas, max_radius,
                                       idxs, k_idxs, znear, zfar,
                                       gamma, grad_out_color);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("No CPU support");
  }
}

// ****************************************************************************
// *                         MAIN ENTRY POINT                                 *
// ****************************************************************************

// This is the main entry point for the forward pass of the point rasterizer;
// it uses either naive or coarse-to-fine rasterization based on bin_size.
//
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
//                          points_packed of the first point in each pointcloud
//                          in the batch where N is the batch size.
//  num_points_per_cloud: LongTensor of shape (N) giving the number of points
//                        for each pointcloud in the batch.
//  radius: Radius of each point (in NDC units)
//  image_size:  (S) Size of the image to return (in pixels)
//  points_per_pixel: (K) The number of points to return for each pixel
//  bin_size: Bin size (in pixels) for coarse-to-fine rasterization. Setting
//            bin_size=0 uses naive rasterization instead.
//  max_points_per_bin: The maximum number of points allowed to fall into each
//                      bin when using coarse-to-fine rasterization.
//
// Returns:
//  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
//        closest K points along the z-axis for each pixel, padded with -1 for
//        pixels hit by fewer than K points. The indices refer to points in
//        points packed i.e a tensor of shape (P, 3) representing the flattened
//        points for all pointclouds in the batch.
//  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each of each
//        closest point for each pixel
//  dists: float32 Tensor of shape (N, S, S, K) giving squared Euclidean
//         distance in the (NDC) x/y plane between each pixel and its K closest
//         points along the z axis.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizePoints(
    const torch::Tensor& points,
    const torch::Tensor& colors,
    const torch::Tensor& sigmas,
    const torch::Tensor& inv_cov,
    const int max_radius,
    const int image_height,
    const int image_width,
    const int points_per_pixel,
    const float zfar,
    const float znear,
    const float gamma)
{
    return RasterizePointsGKCuda(
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
        gamma);
}

