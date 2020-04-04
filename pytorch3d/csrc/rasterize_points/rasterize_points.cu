// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <tuple>
#include "rasterize_points/bitmask.cuh"
#include "rasterize_points/rasterization_utils.cuh"

namespace {
// A little structure for holding details about a pixel.
struct Pix {
  float z; // Depth of the reference point.
  int32_t idx; // Index of the reference point.
  float dist2; // Euclidean distance square to the reference point.
};

__device__ inline bool operator<(const Pix& a, const Pix& b) {
  return a.z < b.z;
}

// This function checks if a pixel given by xy location pxy lies within the
// point with index p and batch index n. One of the inputs is a list (q)
// which contains Pixel structs with the indices of the points which intersect
// with this pixel sorted by closest z distance. If the pixel pxy lies in the
// point, the list (q) is updated and re-orderered in place. In addition
// the auxillary variables q_size, q_max_z and q_max_idx are also modified.
// This code is shared between RasterizePointsNaiveCudaKernel and
// RasterizePointsFineCudaKernel.
template <typename PointQ>
__device__ void CheckPixelInsidePoint(
    const float* points, // (P, 3)
    const int p_idx,
    int& q_size,
    float& q_max_z,
    int& q_max_idx,
    PointQ& q,
    const float radius2,
    const float xf,
    const float yf,
    const int K) {
  const float px = points[p_idx * 3 + 0];
  const float py = points[p_idx * 3 + 1];
  const float pz = points[p_idx * 3 + 2];
  if (pz < 0)
    return; // Don't render points behind the camera
  const float dx = xf - px;
  const float dy = yf - py;
  const float dist2 = dx * dx + dy * dy;
  if (dist2 < radius2) {
    if (q_size < K) {
      // Just insert it
      q[q_size] = {pz, p_idx, dist2};
      if (pz > q_max_z) {
        q_max_z = pz;
        q_max_idx = q_size;
      }
      q_size++;
    } else if (pz < q_max_z) {
      // Overwrite the old max, and find the new max
      q[q_max_idx] = {pz, p_idx, dist2};
      q_max_z = pz;
      for (int i = 0; i < K; i++) {
        if (q[i].z > q_max_z) {
          q_max_z = q[i].z;
          q_max_idx = i;
        }
      }
    }
  }
}
} // namespace
// ****************************************************************************
// *                          NAIVE RASTERIZATION                             *
// ****************************************************************************

__global__ void RasterizePointsNaiveCudaKernel(
    const float* points, // (P, 3)
    const int64_t* cloud_to_packed_first_idx, // (N)
    const int64_t* num_points_per_cloud, // (N)
    const float radius,
    const int N,
    const int H,
    const int W,
    const int K,
    int32_t* point_idxs, // (N, H, W, K)
    float* zbuf, // (N, H, W, K)
    float* pix_dists) { // (N, H, W, K)
  // Simple version: One thread per output pixel
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const float radius2 = radius * radius;
  for (int i = tid; i < N * H * W; i += num_threads) {
    // Convert linear index to 3D index
    const int n = i / (H * W); // Batch index
    const int pix_idx = i % (H * W);

    // Reverse ordering of X and Y axes.
    const int yi = pix_idx / W;
    const int xi = pix_idx % W;

    const float xf = PixToNdc(xi, W);
    const float yf = PixToNdc(yi, H);

    // For keeping track of the K closest points we want a data structure
    // that (1) gives O(1) access to the closest point for easy comparisons,
    // and (2) allows insertion of new elements. In the CPU version we use
    // std::priority_queue; then (2) is O(log K). We can't use STL
    // containers in CUDA; we could roll our own max heap in an array, but
    // that would likely have a lot of warp divergence so we do something
    // simpler instead: keep the elements in an unsorted array, but keep
    // track of the max value and the index of the max value. Then (1) is
    // still O(1) time, while (2) is O(K) with a clean loop. Since K <= 8
    // this should be fast enough for our purposes.
    // TODO(jcjohns) Abstract this out into a standalone data structure
    Pix q[kMaxPointsPerPixel];
    int q_size = 0;
    float q_max_z = -1000;
    int q_max_idx = -1;

    // Using the batch index of the thread get the start and stop
    // indices for the points.
    const int64_t point_start_idx = cloud_to_packed_first_idx[n];
    const int64_t point_stop_idx = point_start_idx + num_points_per_cloud[n];

    for (int p_idx = point_start_idx; p_idx < point_stop_idx; ++p_idx) {
      CheckPixelInsidePoint(
          points, p_idx, q_size, q_max_z, q_max_idx, q, radius2, xf, yf, K);
    }
    BubbleSort(q, q_size);
    int idx = n * H * W * K + pix_idx * K;
    for (int k = 0; k < q_size; ++k) {
      point_idxs[idx + k] = q[k].idx;
      zbuf[idx + k] = q[k].z;
      pix_dists[idx + k] = q[k].dist2;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizePointsNaiveCuda(
    const torch::Tensor& points, // (P. 3)
    const torch::Tensor& cloud_to_packed_first_idx, // (N)
    const torch::Tensor& num_points_per_cloud, // (N)
    const int image_height,
    const int image_width,
    const float radius,
    const int points_per_pixel) {
  if (points.ndimension() != 2 || points.size(1) != 3) {
    AT_ERROR("points must have dimensions (num_points, 3)");
  }
  if (num_points_per_cloud.size(0) != cloud_to_packed_first_idx.size(0)) {
    AT_ERROR(
        "num_points_per_cloud must have same size first dimension as cloud_to_packed_first_idx");
  }

  const int N = num_points_per_cloud.size(0); // batch size.
  const int H = image_height;
  const int W = image_width;
  const int K = points_per_pixel;

  if (K > kMaxPointsPerPixel) {
    std::stringstream ss;
    ss << "Must have points_per_pixel <= " << kMaxPointsPerPixel;
    AT_ERROR(ss.str());
  }

  auto int_opts = points.options().dtype(torch::kInt32);
  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor point_idxs = torch::full({N, H, W, K}, -1, int_opts);
  torch::Tensor zbuf = torch::full({N, H, W, K}, -1, float_opts);
  torch::Tensor pix_dists = torch::full({N, H, W, K}, -1, float_opts);

  const size_t blocks = 1024;
  const size_t threads = 64;
  RasterizePointsNaiveCudaKernel<<<blocks, threads>>>(
      points.contiguous().data<float>(),
      cloud_to_packed_first_idx.contiguous().data<int64_t>(),
      num_points_per_cloud.contiguous().data<int64_t>(),
      radius,
      N,
      H,
      W,
      K,
      point_idxs.contiguous().data<int32_t>(),
      zbuf.contiguous().data<float>(),
      pix_dists.contiguous().data<float>());
  return std::make_tuple(point_idxs, zbuf, pix_dists);
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************
// TODO(T55115174) Add more documentation for backward kernel.
__global__ void RasterizePointsBackwardCudaKernel(
    const float* points, // (P, 3)
    const int32_t* idxs, // (N, H, W, K)
    const int N,
    const int P,
    const int H,
    const int W,
    const int K,
    const float* grad_zbuf, // (N, H, W, K)
    const float* grad_dists, // (N, H, W, K)
    float* grad_points) { // (P, 3)
  // Parallelized over each of K points per pixel, for each pixel in images of
  // size H * W, for each image in the batch of size N.
  int num_threads = gridDim.x * blockDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < N * H * W * K; i += num_threads) {
    // const int n = i / (H * W * K); // batch index (not needed).
    const int yxk = i % (H * W * K);
    const int yi = yxk / (W * K);
    const int xk = yxk % (W * K);
    const int xi = xk / K;
    // k = xk % K (We don't actually need k, but this would be it.)
    // Reverse ordering of X and Y axes.
    const int yidx = yi;
    const int xidx = xi;

    const float xf = PixToNdc(xidx, W);
    const float yf = PixToNdc(yidx, H);

    const int p = idxs[i];
    if (p < 0)
      continue;
    const float grad_dist2 = grad_dists[i];
    const int p_ind = p * 3; // index into packed points tensor
    const float px = points[p_ind + 0];
    const float py = points[p_ind + 1];
    const float dx = px - xf;
    const float dy = py - yf;
    const float grad_px = 2.0f * grad_dist2 * dx;
    const float grad_py = 2.0f * grad_dist2 * dy;
    const float grad_pz = grad_zbuf[i];
    atomicAdd(grad_points + p_ind + 0, grad_px);
    atomicAdd(grad_points + p_ind + 1, grad_py);
    atomicAdd(grad_points + p_ind + 2, grad_pz);
  }
}

torch::Tensor RasterizePointsBackwardCuda(
    const torch::Tensor& points, // (N, P, 3)
    const torch::Tensor& idxs, // (N, H, W, K)
    const torch::Tensor& grad_zbuf, // (N, H, W, K)
    const torch::Tensor& grad_dists) { // (N, H, W, K)
  const int P = points.size(0);
  const int N = idxs.size(0);
  const int H = idxs.size(1);
  const int W = idxs.size(2);
  const int K = idxs.size(3);

  torch::Tensor grad_points = torch::zeros({P, 3}, points.options());
  const size_t blocks = 1024;
  const size_t threads = 64;

  RasterizePointsBackwardCudaKernel<<<blocks, threads>>>(
      points.contiguous().data<float>(),
      idxs.contiguous().data<int32_t>(),
      N,
      P,
      H,
      W,
      K,
      grad_zbuf.contiguous().data<float>(),
      grad_dists.contiguous().data<float>(),
      grad_points.contiguous().data<float>());

  return grad_points;
}
