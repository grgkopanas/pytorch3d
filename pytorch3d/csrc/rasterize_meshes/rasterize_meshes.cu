// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <float.h>
#include <math.h>
#include <thrust/tuple.h>
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include "float_math.cuh"
#include "geometry_utils.cuh"
#include "rasterize_points/bitmask.cuh"
#include "rasterize_points/rasterization_utils.cuh"

namespace {
// A structure for holding details about a pixel.
struct Pixel {
  float z;
  int64_t idx;
  float dist;
  float3 bary;
};

__device__ bool operator<(const Pixel& a, const Pixel& b) {
  return a.z < b.z;
}

__device__ float FloatMin3(const float p1, const float p2, const float p3) {
  return fminf(p1, fminf(p2, p3));
}

__device__ float FloatMax3(const float p1, const float p2, const float p3) {
  return fmaxf(p1, fmaxf(p2, p3));
}

// Get the xyz coordinates of the three vertices for the face given by the
// index face_idx into face_verts.
__device__ thrust::tuple<float3, float3, float3> GetSingleFaceVerts(
    const float* face_verts,
    int face_idx) {
  const float x0 = face_verts[face_idx * 9 + 0];
  const float y0 = face_verts[face_idx * 9 + 1];
  const float z0 = face_verts[face_idx * 9 + 2];
  const float x1 = face_verts[face_idx * 9 + 3];
  const float y1 = face_verts[face_idx * 9 + 4];
  const float z1 = face_verts[face_idx * 9 + 5];
  const float x2 = face_verts[face_idx * 9 + 6];
  const float y2 = face_verts[face_idx * 9 + 7];
  const float z2 = face_verts[face_idx * 9 + 8];

  const float3 v0xyz = make_float3(x0, y0, z0);
  const float3 v1xyz = make_float3(x1, y1, z1);
  const float3 v2xyz = make_float3(x2, y2, z2);

  return thrust::make_tuple(v0xyz, v1xyz, v2xyz);
}

// Get the min/max x/y/z values for the face given by vertices v0, v1, v2.
__device__ thrust::tuple<float2, float2, float2>
GetFaceBoundingBox(float3 v0, float3 v1, float3 v2) {
  const float xmin = FloatMin3(v0.x, v1.x, v2.x);
  const float ymin = FloatMin3(v0.y, v1.y, v2.y);
  const float zmin = FloatMin3(v0.z, v1.z, v2.z);
  const float xmax = FloatMax3(v0.x, v1.x, v2.x);
  const float ymax = FloatMax3(v0.y, v1.y, v2.y);
  const float zmax = FloatMax3(v0.z, v1.z, v2.z);

  return thrust::make_tuple(
      make_float2(xmin, xmax),
      make_float2(ymin, ymax),
      make_float2(zmin, zmax));
}

// Check if the point (px, py) lies outside the face bounding box face_bbox.
// Return true if the point is outside.
__device__ bool CheckPointOutsideBoundingBox(
    float3 v0,
    float3 v1,
    float3 v2,
    float blur_radius,
    float2 pxy) {
  const auto bbox = GetFaceBoundingBox(v0, v1, v2);
  const float2 xlims = thrust::get<0>(bbox);
  const float2 ylims = thrust::get<1>(bbox);
  const float2 zlims = thrust::get<2>(bbox);

  const float x_min = xlims.x - blur_radius;
  const float y_min = ylims.x - blur_radius;
  const float x_max = xlims.y + blur_radius;
  const float y_max = ylims.y + blur_radius;

  // Check if the current point is oustside the triangle bounding box.
  return (pxy.x > x_max || pxy.x < x_min || pxy.y > y_max || pxy.y < y_min);
}

// This function checks if a pixel given by xy location pxy lies within the
// face with index face_idx in face_verts. One of the inputs is a list (q)
// which contains Pixel structs with the indices of the faces which intersect
// with this pixel sorted by closest z distance. If the point pxy lies in the
// face, the list (q) is updated and re-orderered in place. In addition
// the auxillary variables q_size, q_max_z and q_max_idx are also modified.
// This code is shared between RasterizeMeshesNaiveCudaKernel and
// RasterizeMeshesFineCudaKernel.
template <typename FaceQ>
__device__ void CheckPixelInsideFace(
    const float* face_verts, // (F, 3, 3)
    const int face_idx,
    int& q_size,
    float& q_max_z,
    int& q_max_idx,
    FaceQ& q,
    const float blur_radius,
    const float2 pxy, // Coordinates of the pixel
    const int K,
    const bool perspective_correct) {
  const auto v012 = GetSingleFaceVerts(face_verts, face_idx);
  const float3 v0 = thrust::get<0>(v012);
  const float3 v1 = thrust::get<1>(v012);
  const float3 v2 = thrust::get<2>(v012);

  // Only need xy for barycentric coordinates and distance calculations.
  const float2 v0xy = make_float2(v0.x, v0.y);
  const float2 v1xy = make_float2(v1.x, v1.y);
  const float2 v2xy = make_float2(v2.x, v2.y);

  // Perform checks and skip if:
  // 1. the face is behind the camera
  // 2. the face has very small face area
  // 3. the pixel is outside the face bbox
  const float zmax = FloatMax3(v0.z, v1.z, v2.z);
  const float zmin = FloatMin3(v0.z, v1.z, v2.z);

  const bool outside_bbox = CheckPointOutsideBoundingBox(
      v0, v1, v2, sqrt(blur_radius), pxy); // use sqrt of blur for bbox
  const float face_area = EdgeFunctionForward(v0xy, v1xy, v2xy);
  const bool zero_face_area =
      (face_area <= kEpsilon && face_area >= -1.0f * kEpsilon);

  if (zmin < 0.1 || outside_bbox || zero_face_area) {
    return;
  }

  // Calculate barycentric coords and euclidean dist to triangle.
  const float3 p_bary0 = BarycentricCoordsForward(pxy, v0xy, v1xy, v2xy);
  const float3 p_bary = !perspective_correct
      ? p_bary0
      : BarycentricPerspectiveCorrectionForward(p_bary0, v0.z, v1.z, v2.z);

  const float pz = p_bary.x * v0.z + p_bary.y * v1.z + p_bary.z * v2.z;
  if (pz < 0) {
    return; // Face is behind the image plane.
  }

  // Get abs squared distance
  const float dist = PointTriangleDistanceForward(pxy, v0xy, v1xy, v2xy);

  // Use the bary coordinates to determine if the point is inside the face.
  const bool inside = p_bary.x > 0.0f && p_bary.y > 0.0f && p_bary.z > 0.0f;
  const float signed_dist = inside ? -dist : dist;

  // Check if pixel is outside blur region
  if (!inside && dist >= blur_radius) {
    return;
  }

  if (q_size < K) {
    // Just insert it.
    q[q_size] = {pz, face_idx, signed_dist, p_bary};
    if (pz > q_max_z) {
      q_max_z = pz;
      q_max_idx = q_size;
    }
    q_size++;
  } else if (pz < q_max_z) {
    // Overwrite the old max, and find the new max.
    q[q_max_idx] = {pz, face_idx, signed_dist, p_bary};
    q_max_z = pz;
    for (int i = 0; i < K; i++) {
      if (q[i].z > q_max_z) {
        q_max_z = q[i].z;
        q_max_idx = i;
      }
    }
  }
}
} // namespace

// ****************************************************************************
// *                          NAIVE RASTERIZATION                      *
// ****************************************************************************
__global__ void RasterizeMeshesNaiveCudaKernel(
    const float* face_verts,
    const int64_t* mesh_to_face_first_idx,
    const int64_t* num_faces_per_mesh,
    const float blur_radius,
    const bool perspective_correct,
    const int N,
    const int H,
    const int W,
    const int K,
    int64_t* face_idxs,
    float* zbuf,
    float* pix_dists,
    float* bary) {
  // Simple version: One thread per output pixel
  int num_threads = gridDim.x * blockDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = tid; i < N * H * W; i += num_threads) {
    // Convert linear index to 3D index
    const int n = i / (H * W); // batch index.
    const int pix_idx = i % (H * W);

    // Reverse ordering of X and Y axes
    const int yi = pix_idx / W;
    const int xi = pix_idx % W;

    // screen coordinates to ndc coordiantes of pixel.
    const float xf = PixToNdc(xi, W);
    const float yf = PixToNdc(yi, H);
    const float2 pxy = make_float2(xf, yf);

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
    Pixel q[kMaxPointsPerPixel];
    int q_size = 0;
    float q_max_z = -1000;
    int q_max_idx = -1;

    // Using the batch index of the thread get the start and stop
    // indices for the faces.
    const int64_t face_start_idx = mesh_to_face_first_idx[n];
    const int64_t face_stop_idx = face_start_idx + num_faces_per_mesh[n];

    // Loop through the faces in the mesh.
    for (int f = face_start_idx; f < face_stop_idx; ++f) {
      // Check if the pixel pxy is inside the face bounding box and if it is,
      // update q, q_size, q_max_z and q_max_idx in place.
      CheckPixelInsideFace(
          face_verts,
          f,
          q_size,
          q_max_z,
          q_max_idx,
          q,
          blur_radius,
          pxy,
          K,
          perspective_correct);
    }

    // TODO: make sorting an option as only top k is needed, not sorted values.
    BubbleSort(q, q_size);
    int idx = n * H * W * K + pix_idx * K;
    for (int k = 0; k < q_size; ++k) {
      face_idxs[idx + k] = q[k].idx;
      zbuf[idx + k] = q[k].z;
      pix_dists[idx + k] = q[k].dist;
      bary[(idx + k) * 3 + 0] = q[k].bary.x;
      bary[(idx + k) * 3 + 1] = q[k].bary.y;
      bary[(idx + k) * 3 + 2] = q[k].bary.z;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeMeshesNaiveCuda(
    const torch::Tensor& face_verts,
    const torch::Tensor& mesh_to_faces_packed_first_idx,
    const torch::Tensor& num_faces_per_mesh,
    const int image_height,
    const int image_width,
    const float blur_radius,
    const int num_closest,
    const bool perspective_correct) {
  if (face_verts.ndimension() != 3 || face_verts.size(1) != 3 ||
      face_verts.size(2) != 3) {
    AT_ERROR("face_verts must have dimensions (num_faces, 3, 3)");
  }
  if (num_faces_per_mesh.size(0) != mesh_to_faces_packed_first_idx.size(0)) {
    AT_ERROR(
        "num_faces_per_mesh must have save size first dimension as mesh_to_faces_packed_first_idx");
  }

  if (num_closest > kMaxPointsPerPixel) {
    std::stringstream ss;
    ss << "Must have points_per_pixel <= " << kMaxPointsPerPixel;
    AT_ERROR(ss.str());
  }

  const int N = num_faces_per_mesh.size(0); // batch size.
  const int H = image_height; // Assume square images.
  const int W = image_width;
  const int K = num_closest;

  auto long_opts = face_verts.options().dtype(torch::kInt64);
  auto float_opts = face_verts.options().dtype(torch::kFloat32);

  torch::Tensor face_idxs = torch::full({N, H, W, K}, -1, long_opts);
  torch::Tensor zbuf = torch::full({N, H, W, K}, -1, float_opts);
  torch::Tensor pix_dists = torch::full({N, H, W, K}, -1, float_opts);
  torch::Tensor bary = torch::full({N, H, W, K, 3}, -1, float_opts);

  const size_t blocks = 1024;
  const size_t threads = 64;

  RasterizeMeshesNaiveCudaKernel<<<blocks, threads>>>(
      face_verts.contiguous().data<float>(),
      mesh_to_faces_packed_first_idx.contiguous().data<int64_t>(),
      num_faces_per_mesh.contiguous().data<int64_t>(),
      blur_radius,
      perspective_correct,
      N,
      H,
      W,
      K,
      face_idxs.contiguous().data<int64_t>(),
      zbuf.contiguous().data<float>(),
      pix_dists.contiguous().data<float>(),
      bary.contiguous().data<float>());

  return std::make_tuple(face_idxs, zbuf, bary, pix_dists);
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************
// TODO: benchmark parallelizing over faces_verts instead of over pixels.
__global__ void RasterizeMeshesBackwardCudaKernel(
    const float* face_verts, // (F, 3, 3)
    const int64_t* pix_to_face, // (N, H, W, K)
    const bool perspective_correct,
    const int N,
    const int H,
    const int W,
    const int K,
    const float* grad_zbuf, // (N, H, W, K)
    const float* grad_bary, // (N, H, W, K, 3)
    const float* grad_dists, // (N, H, W, K)
    float* grad_face_verts) { // (F, 3, 3)

  // Parallelize over each pixel in images of
  // size H * W, for each image in the batch of size N.
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int t_i = tid; t_i < N * H * W; t_i += num_threads) {
    // Convert linear index to 3D index
    const int n = t_i / (H * W); // batch index.
    const int pix_idx = t_i % (H * W);

    // Reverse ordering of X and Y axes.
    const int yi = pix_idx / W;
    const int xi = pix_idx % W;

    const float xf = PixToNdc(xi, W);
    const float yf = PixToNdc(yi, H);
    const float2 pxy = make_float2(xf, yf);

    // Loop over all the faces for this pixel.
    for (int k = 0; k < K; k++) {
      // Index into (N, H, W, K, :) grad tensors
      // pixel index + top k index
      int i = n * H * W * K + pix_idx * K + k;

      const int f = pix_to_face[i];
      if (f < 0) {
        continue; // padded face.
      }
      // Get xyz coordinates of the three face vertices.
      const auto v012 = GetSingleFaceVerts(face_verts, f);
      const float3 v0 = thrust::get<0>(v012);
      const float3 v1 = thrust::get<1>(v012);
      const float3 v2 = thrust::get<2>(v012);

      // Only neex xy for barycentric coordinate and distance calculations.
      const float2 v0xy = make_float2(v0.x, v0.y);
      const float2 v1xy = make_float2(v1.x, v1.y);
      const float2 v2xy = make_float2(v2.x, v2.y);

      // Get upstream gradients for the face.
      const float grad_dist_upstream = grad_dists[i];
      const float grad_zbuf_upstream = grad_zbuf[i];
      const float grad_bary_upstream_w0 = grad_bary[i * 3 + 0];
      const float grad_bary_upstream_w1 = grad_bary[i * 3 + 1];
      const float grad_bary_upstream_w2 = grad_bary[i * 3 + 2];
      const float3 grad_bary_upstream = make_float3(
          grad_bary_upstream_w0, grad_bary_upstream_w1, grad_bary_upstream_w2);

      const float3 bary0 = BarycentricCoordsForward(pxy, v0xy, v1xy, v2xy);
      const float3 bary = !perspective_correct
          ? bary0
          : BarycentricPerspectiveCorrectionForward(bary0, v0.z, v1.z, v2.z);
      const bool inside = bary.x > 0.0f && bary.y > 0.0f && bary.z > 0.0f;
      const float sign = inside ? -1.0f : 1.0f;

      // TODO(T52813608) Add support for non-square images.
      auto grad_dist_f = PointTriangleDistanceBackward(
          pxy, v0xy, v1xy, v2xy, sign * grad_dist_upstream);
      const float2 ddist_d_v0 = thrust::get<1>(grad_dist_f);
      const float2 ddist_d_v1 = thrust::get<2>(grad_dist_f);
      const float2 ddist_d_v2 = thrust::get<3>(grad_dist_f);

      // Upstream gradient for barycentric coords from zbuf calculation:
      // zbuf = bary_w0 * z0 + bary_w1 * z1 + bary_w2 * z2
      // Therefore
      // d_zbuf/d_bary_w0 = z0
      // d_zbuf/d_bary_w1 = z1
      // d_zbuf/d_bary_w2 = z2
      const float3 d_zbuf_d_bary = make_float3(v0.z, v1.z, v2.z);

      // Total upstream barycentric gradients are the sum of
      // external upstream gradients and contribution from zbuf.
      const float3 grad_bary_f_sum =
          (grad_bary_upstream + grad_zbuf_upstream * d_zbuf_d_bary);
      float3 grad_bary0 = grad_bary_f_sum;
      float dz0_persp = 0.0f, dz1_persp = 0.0f, dz2_persp = 0.0f;
      if (perspective_correct) {
        auto perspective_grads = BarycentricPerspectiveCorrectionBackward(
            bary0, v0.z, v1.z, v2.z, grad_bary_f_sum);
        grad_bary0 = thrust::get<0>(perspective_grads);
        dz0_persp = thrust::get<1>(perspective_grads);
        dz1_persp = thrust::get<2>(perspective_grads);
        dz2_persp = thrust::get<3>(perspective_grads);
      }
      auto grad_bary_f =
          BarycentricCoordsBackward(pxy, v0xy, v1xy, v2xy, grad_bary0);
      const float2 dbary_d_v0 = thrust::get<1>(grad_bary_f);
      const float2 dbary_d_v1 = thrust::get<2>(grad_bary_f);
      const float2 dbary_d_v2 = thrust::get<3>(grad_bary_f);

      atomicAdd(grad_face_verts + f * 9 + 0, dbary_d_v0.x + ddist_d_v0.x);
      atomicAdd(grad_face_verts + f * 9 + 1, dbary_d_v0.y + ddist_d_v0.y);
      atomicAdd(
          grad_face_verts + f * 9 + 2, grad_zbuf_upstream * bary.x + dz0_persp);
      atomicAdd(grad_face_verts + f * 9 + 3, dbary_d_v1.x + ddist_d_v1.x);
      atomicAdd(grad_face_verts + f * 9 + 4, dbary_d_v1.y + ddist_d_v1.y);
      atomicAdd(
          grad_face_verts + f * 9 + 5, grad_zbuf_upstream * bary.y + dz1_persp);
      atomicAdd(grad_face_verts + f * 9 + 6, dbary_d_v2.x + ddist_d_v2.x);
      atomicAdd(grad_face_verts + f * 9 + 7, dbary_d_v2.y + ddist_d_v2.y);
      atomicAdd(
          grad_face_verts + f * 9 + 8, grad_zbuf_upstream * bary.z + dz2_persp);
    }
  }
}

torch::Tensor RasterizeMeshesBackwardCuda(
    const torch::Tensor& face_verts, // (F, 3, 3)
    const torch::Tensor& pix_to_face, // (N, H, W, K)
    const torch::Tensor& grad_zbuf, // (N, H, W, K)
    const torch::Tensor& grad_bary, // (N, H, W, K, 3)
    const torch::Tensor& grad_dists, // (N, H, W, K)
    const bool perspective_correct) {
  const int F = face_verts.size(0);
  const int N = pix_to_face.size(0);
  const int H = pix_to_face.size(1);
  const int W = pix_to_face.size(2);
  const int K = pix_to_face.size(3);

  torch::Tensor grad_face_verts = torch::zeros({F, 3, 3}, face_verts.options());
  const size_t blocks = 1024;
  const size_t threads = 64;

  RasterizeMeshesBackwardCudaKernel<<<blocks, threads>>>(
      face_verts.contiguous().data<float>(),
      pix_to_face.contiguous().data<int64_t>(),
      perspective_correct,
      N,
      H,
      W,
      K,
      grad_zbuf.contiguous().data<float>(),
      grad_bary.contiguous().data<float>(),
      grad_dists.contiguous().data<float>(),
      grad_face_verts.contiguous().data<float>());

  return grad_face_verts;
}
