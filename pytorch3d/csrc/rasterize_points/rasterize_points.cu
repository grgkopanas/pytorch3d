// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
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
    const int points_per_pixel,
    const float zfar) {
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
  torch::Tensor zbuf = torch::full({N, H, W, K}, 2.0*zfar, float_opts);
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
// *                          GK RASTERIZATION                             *
// ****************************************************************************

__global__ void OrderPointsGKCudaKernel(
    int32_t* point_idxs, // (N, H, W, K)
    float* zbuf, // (N, H, W, K)
    float* pix_dists,
    uint32_t* k_idxs,
    int H,
    int W,
    int K)
{
  // Simple version: One thread per output pixel
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = tid; i < H * W; i += num_threads) {
    // Convert linear index to 3D index
    const int pix_idx = i % (H * W);

    const int yi = pix_idx / W;
    const int xi = pix_idx % W;

    int idx = 0 * H * W * K + yi * W * K + xi * K + 0;
    Pix q[kMaxPointsPerPixel];
    int k = min(k_idxs[yi*W + xi], K);
    for (int i=0; i<k; i++) {
        q[i].idx = point_idxs[idx + i];
        q[i].z = zbuf[idx + i];
        q[i].dist2 = pix_dists[idx + i];
    }
    BubbleSort(q, k);
    for (int i=0; i<k; i++) {
        point_idxs[idx + i] = q[i].idx;
        zbuf[idx + i] = q[i].z;
        pix_dists[idx + i] = q[i].dist2;
    }
  }
}

__global__ void RasterizePointsGKCudaKernel(
    const float* points, // (P, 3)
    const float* colors,
    const int P,
    const int64_t* cloud_to_packed_first_idx, // (N)
    const int64_t* num_points_per_cloud, // (N)
    const float radius,
    const int N,
    const int C,
    const int H,
    const int W,
    float* accum_product,
    float* accum_weights,
    const float znear,
    const float zfar,
    const float sigma,
    const float gamma)
{
    // Simple version: One thread per output pixel
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const float radius2 = radius * radius;
    // TODO gkopanas more than 1 batches?
    for (int i = tid; i < P ; i += num_threads) {
        const float px_ndc = points[i * 3 + 0];
        const float py_ndc = points[i * 3 + 1];
        const float pz = points[i * 3 + 2];
        if (pz < 0) {
            continue;
        }
        const float color_r = colors[i*3 + 0];
        const float color_g = colors[i*3 + 1];
        const float color_b = colors[i*3 + 2];

        const float px = NdcToPix(px_ndc, W);
        const float py = NdcToPix(py_ndc, H);
        const float radius_pixels_x = radius*W/2.0;
        const float radius_pixels_y = radius*H/2.0;

        //printf("i %d tid %d px_ndc %f py_ndc %f px %f py %f ry %f rx %f\n", i, tid, px_ndc, py_ndc, px, py, radius_pixels_y, radius_pixels_x);
        int y_start = int(py - radius_pixels_y);
        int y_finish = int(py + radius_pixels_y);
        int x_start = int(px - radius_pixels_x);
        int x_finish = int(px + radius_pixels_x);
        //if (y_finish < 0 || y_start > H-1 || x_finish < 0 && x_start > W-1)
        //    continue;
        //printf("i %d tid %d y_start %d y_finish %d x_start %d x_finish %d\n", i, tid, y_start, y_finish, x_start, x_finish);
        for (int y_idx = y_start; y_idx < y_finish + 1; y_idx++) {
            for (int x_idx = x_start; x_idx <  x_finish + 1; x_idx++) {
                if (y_idx < 0 || y_idx > H - 1 || x_idx < 0 || x_idx > W - 1)
                    continue;
                float dx = PixToNdc(x_idx, W) - px_ndc;
                float dy = PixToNdc(y_idx, H) - py_ndc;
                float dist2 = dx*dx + dy*dy;
                if (dist2 > radius2)
                    // This doesn't create divergence since all threads will skip in sync.
                    continue;
                //float dist = sqrtf(dist2);
                float z_exponent = ((zfar - pz)/(zfar - znear))/gamma;
                float gauss_exponent = -dist2/(2*sigma*sigma);
                float w = expf(gauss_exponent + z_exponent);

                int idx_accum_weights = 0*H*W + y_idx*W + x_idx;
                atomicAdd(&(accum_weights[idx_accum_weights]), w);

                int idx_product = 0*C*H*W + 0*H*W + y_idx*W + x_idx;
                atomicAdd(&(accum_product[idx_product]), w*color_r);
                idx_product += H*W;
                atomicAdd(&(accum_product[idx_product]), w*color_g);
                idx_product += H*W;
                atomicAdd(&(accum_product[idx_product]), w*color_b);
                //printf("i %d y_idx %d x_idx %d w %f dist %f sigma %f z %f\n", i, y_idx, x_idx, w, dist, sigma, z_exponent);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizePointsGKCuda(
    const torch::Tensor& points, // (P, 3)
    const torch::Tensor& colors,
    const torch::Tensor& cloud_to_packed_first_idx, // (N)
    const torch::Tensor& num_points_per_cloud, // (N)
    const int image_height,
    const int image_width,
    const float radius,
    const int points_per_pixel,
    const float znear,
    const float zfar,
    const float sigma,
    const float gamma) {

  if (points.ndimension() != 2 || points.size(1) != 3) {
    AT_ERROR("points must have dimensions (num_points, 3)");
  }
  if (points.size(0) != colors.size(0)) {
    AT_ERROR("points and features must have the same dimensions");
  }
  if (num_points_per_cloud.size(0) != cloud_to_packed_first_idx.size(0)) {
    AT_ERROR(
        "num_points_per_cloud must have same size first dimension as cloud_to_packed_first_idx");
  }

  const int P = points.size(0);
  const int C = colors.size(1);
  const int N = num_points_per_cloud.size(0); // batch size.
  const int H = image_height;
  const int W = image_width;


  auto int_opts = points.options().dtype(torch::kInt32);
  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor accum_product = torch::full({N, C, H, W}, 0.0, float_opts);
  torch::Tensor accum_weights = torch::full({N, 1, H, W}, 0.0, float_opts);


  const size_t blocks = 1024;
  const size_t threads = 64 ;
  //printf("This is the new rast\n");
  RasterizePointsGKCudaKernel<<<blocks, threads>>>(
      points.contiguous().data<float>(),
      colors.contiguous().data<float>(),
      P,
      cloud_to_packed_first_idx.contiguous().data<int64_t>(),
      num_points_per_cloud.contiguous().data<int64_t>(),
      radius,
      N,
      C,
      H,
      W,
      accum_product.contiguous().data<float>(),
      accum_weights.contiguous().data<float>(),
      znear,
      zfar,
      sigma,
      gamma);
   return std::make_tuple(accum_product/(accum_weights + 0.0000001), accum_product, accum_weights);
}


// ****************************************************************************
// *                          COARSE RASTERIZATION                            *
// ****************************************************************************

__global__ void RasterizePointsCoarseCudaKernel(
    const float* points, // (P, 3)
    const int64_t* cloud_to_packed_first_idx, // (N)
    const int64_t* num_points_per_cloud, // (N)
    const float radius,
    const int N,
    const int P,
    const int image_height,
    const int image_width,
    const int bin_size,
    const int chunk_size,
    const int max_points_per_bin,
    int* points_per_bin,
    int* bin_points) {
  extern __shared__ char sbuf[];
  const int M = max_points_per_bin;
  const int H = image_height;
  const int W = image_width;
  const int num_bins_h = 1 + (H - 1) / bin_size; // Integer divide round up
  const int num_bins_w = 1 + (W - 1) / bin_size;

  const float half_pix_y = 1.0f / H; // Size of half a pixel in NDC units
  const float half_pix_x = 1.0f / W; // Size of half a pixel in NDC units

  // This is a boolean array of shape (num_bins, num_bins, chunk_size)
  // stored in shared memory that will track whether each point in the chunk
  // falls into each bin of the image.
  BitMask binmask((unsigned int*)sbuf, num_bins_h, num_bins_w, chunk_size);

  // Have each block handle a chunk of points and build a 3D bitmask in
  // shared memory to mark which points hit which bins.  In this first phase,
  // each thread processes one point at a time. After processing the chunk,
  // one thread is assigned per bin, and the thread counts and writes the
  // points for the bin out to global memory.
  const int chunks_per_batch = 1 + (P - 1) / chunk_size;
  const int num_chunks = N * chunks_per_batch;
  for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
    const int batch_idx = chunk / chunks_per_batch;
    const int chunk_idx = chunk % chunks_per_batch;
    const int point_start_idx = chunk_idx * chunk_size;

    binmask.block_clear();

    // Using the batch index of the thread get the start and stop
    // indices for the points.
    const int64_t cloud_point_start_idx = cloud_to_packed_first_idx[batch_idx];
    const int64_t cloud_point_stop_idx =
        cloud_point_start_idx + num_points_per_cloud[batch_idx];

    // Have each thread handle a different point within the chunk
    for (int p = threadIdx.x; p < chunk_size; p += blockDim.x) {
      const int p_idx = point_start_idx + p;

      // Check if point index corresponds to the cloud in the batch given by
      // batch_idx.
      if (p_idx >= cloud_point_stop_idx || p_idx < cloud_point_start_idx) {
        continue;
      }

      const float px = points[p_idx * 3 + 0];
      const float py = points[p_idx * 3 + 1];
      const float pz = points[p_idx * 3 + 2];
      if (pz < 0)
        continue; // Don't render points behind the camera.
      const float px0 = px - radius;
      const float px1 = px + radius;
      const float py0 = py - radius;
      const float py1 = py + radius;

      // Brute-force search over all bins; TODO something smarter?
      // For example we could compute the exact bin where the point falls,
      // then check neighboring bins. This way we wouldn't have to check
      // all bins (however then we might have more warp divergence?)
      for (int by = 0; by < num_bins_h; ++by) {
        // Get y extent for the bin. PixToNdc gives us the location of
        // the center of each pixel, so we need to add/subtract a half
        // pixel to get the true extent of the bin.
        // Reverse ordering of Y axis so that +Y is upwards in the image.
        const int yidx = by + 1;
        const float bin_y_max = PixToNdc(yidx * bin_size - 1, H) + half_pix_y;
        const float bin_y_min = PixToNdc((yidx - 1) * bin_size, H) - half_pix_y;
        //printf("H: %d half_pix_y %f by: %i yidx: %i insidePixToNDC_max %i insidePixToNDC_min %i bin_y_max %f bin_y_min %f\n", H, half_pix_y, by, yidx, yidx * bin_size - 1, (yidx - 1) * bin_size, bin_y_max, bin_y_min);
        const bool y_overlap = (py0 <= bin_y_max) && (bin_y_min <= py1);
        if (!y_overlap) {
          continue;
        }
        for (int bx = 0; bx < num_bins_w; ++bx) {
          // Get x extent for the bin; again we need to adjust the
          // output of PixToNdc by half a pixel.
          // Reverse ordering of x axis so that +X is left.
          const int xidx = bx + 1;
          const float bin_x_max = PixToNdc(xidx * bin_size - 1, W) + half_pix_x;
          const float bin_x_min = PixToNdc((xidx - 1) * bin_size, W) - half_pix_x;
          const bool x_overlap = (px0 <= bin_x_max) && (bin_x_min <= px1);
          if (x_overlap) {
            binmask.set(by, bx, p);
          }
        }
      }
    }
    __syncthreads();
    // Now we have processed every point in the current chunk. We need to
    // count the number of points in each bin so we can write the indices
    // out to global memory. We have each thread handle a different bin.
    for (int byx = threadIdx.x; byx < num_bins_h * num_bins_w; byx += blockDim.x) {
      const int by = byx / num_bins_w;
      const int bx = byx % num_bins_w;
      const int count = binmask.count(by, bx);
      const int points_per_bin_idx =
          batch_idx * num_bins_h * num_bins_w + by * num_bins_w + bx;

      // This atomically increments the (global) number of points found
      // in the current bin, and gets the previous value of the counter;
      // this effectively allocates space in the bin_points array for the
      // points in the current chunk that fall into this bin.
      const int start = atomicAdd(points_per_bin + points_per_bin_idx, count);

      // Now loop over the binmask and write the active bits for this bin
      // out to bin_points.
      int next_idx = batch_idx * num_bins_h * num_bins_w * M + by * num_bins_w * M +
          bx * M + start;
      for (int p = 0; p < chunk_size; ++p) {
        if (binmask.get(by, bx, p)) {
          // TODO: Throw an error if next_idx >= M -- this means that
          // we got more than max_points_per_bin in this bin
          // TODO: check if atomicAdd is needed in line 265.
          bin_points[next_idx] = point_start_idx + p;
          next_idx++;
        }
      }
    }
    __syncthreads();
  }
}

torch::Tensor RasterizePointsCoarseCuda(
    const torch::Tensor& points, // (P, 3)
    const torch::Tensor& cloud_to_packed_first_idx, // (N)
    const torch::Tensor& num_points_per_cloud, // (N)
    const int image_height,
    const int image_width,
    const float radius,
    const int bin_size,
    const int max_points_per_bin) {
  const int P = points.size(0);
  const int N = num_points_per_cloud.size(0);
  const int num_bins_h = 1 + (image_height - 1) / bin_size; // divide round up
  const int num_bins_w = 1 + (image_width - 1) / bin_size;
  const int M = max_points_per_bin;
  if (points.ndimension() != 2 || points.size(1) != 3) {
    AT_ERROR("points must have dimensions (num_points, 3)");
  }
  if (num_bins_h*num_bins_w >= 22*22) {
    // Make sure we do not use too much shared memory.
    std::stringstream ss;
    ss << "Got " << num_bins_h << "*" << num_bins_w << "; that's too many!";
    AT_ERROR(ss.str());
  }
  auto opts = points.options().dtype(torch::kInt32);
  torch::Tensor points_per_bin = torch::zeros({N, num_bins_h, num_bins_w}, opts);
  torch::Tensor bin_points = torch::full({N, num_bins_h, num_bins_w, M}, -1, opts);
  const int chunk_size = 512;
  const size_t shared_size = num_bins_h * num_bins_w * chunk_size / 8;
  const size_t blocks = 64;
  const size_t threads = 512;
  RasterizePointsCoarseCudaKernel<<<blocks, threads, shared_size>>>(
      points.contiguous().data_ptr<float>(),
      cloud_to_packed_first_idx.contiguous().data_ptr<int64_t>(),
      num_points_per_cloud.contiguous().data_ptr<int64_t>(),
      radius,
      N,
      P,
      image_height,
      image_width,
      bin_size,
      chunk_size,
      M,
      points_per_bin.contiguous().data_ptr<int32_t>(),
      bin_points.contiguous().data_ptr<int32_t>());
  return bin_points;
}

// ****************************************************************************
// *                            FINE RASTERIZATION                            *
// ****************************************************************************

__global__ void RasterizePointsFineCudaKernel(
    const float* points, // (P, 3)
    const int32_t* bin_points, // (N, B, B, T)
    const float radius,
    const int bin_size,
    const int N,
    const int B_h,
    const int B_w,
    const int M,
    const int H,
    const int W,
    const int K,
    int32_t* point_idxs, // (N, S, S, K)
    float* zbuf, // (N, S, S, K)
    float* pix_dists) { // (N, S, S, K)
  // This can be more than S^2 if S is not dividable by bin_size.
  const int num_pixels = N * B_h * B_w * bin_size * bin_size;
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const float radius2 = radius * radius;

  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    // Convert linear index into bin and pixel indices. We make the within
    // block pixel ids move the fastest, so that adjacent threads will fall
    // into the same bin; this should give them coalesced memory reads when
    // they read from points and bin_points.
    int i = pid;
    const int n = i / (B_h * B_w * bin_size * bin_size);
    i %= B_h * B_w * bin_size * bin_size;
    const int by = i / (B_w * bin_size * bin_size);
    i %= B_w * bin_size * bin_size;
    const int bx = i / (bin_size * bin_size);
    i %= bin_size * bin_size;
    const int yi = i / bin_size + by * bin_size;
    const int xi = i % bin_size + bx * bin_size;

    if (yi >= H || xi >= W)
      continue;

    // Reverse ordering of the X and Y axis so that
    // in the image +Y is pointing up and +X is pointing left.
    const int yidx = yi;
    const int xidx = xi;

    const float xf = PixToNdc(xidx, W);
    const float yf = PixToNdc(yidx, H);

    // This part looks like the naive rasterization kernel, except we use
    // bin_points to only look at a subset of points already known to fall
    // in this bin. TODO abstract out this logic into some data structure
    // that is shared by both kernels?
    Pix q[kMaxPointsPerPixel];
    int q_size = 0;
    float q_max_z = -1000;
    int q_max_idx = -1;
    for (int m = 0; m < M; ++m) {
      const int p = bin_points[n * B_h * B_w * M + by * B_w * M + bx * M + m];
      if (p < 0) {
        // bin_points uses -1 as a sentinal value
        continue;
      }
      CheckPixelInsidePoint(
          points, p, q_size, q_max_z, q_max_idx, q, radius2, xf, yf, K);
    }
    // Now we've looked at all the points for this bin, so we can write
    // output for the current pixel.
    BubbleSort(q, q_size);
    const int pix_idx = n * H * W * K + yi * W * K + xi * K;
    for (int k = 0; k < q_size; ++k) {
      point_idxs[pix_idx + k] = q[k].idx;
      zbuf[pix_idx + k] = q[k].z;
      pix_dists[pix_idx + k] = q[k].dist2;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsFineCuda(
    const torch::Tensor& points, // (P, 3)
    const torch::Tensor& bin_points,
    const int image_height,
    const int image_width,
    const float radius,
    const int bin_size,
    const int points_per_pixel,
    float zfar) {
  const int N = bin_points.size(0);
  const int B_h = bin_points.size(1);
  const int B_w = bin_points.size(2);
  const int M = bin_points.size(3);
  const int H = image_height;
  const int W = image_width;
  const int K = points_per_pixel;
  if (K > kMaxPointsPerPixel) {
    AT_ERROR("Must have num_closest <= 8");
  }
  auto int_opts = points.options().dtype(torch::kInt32);
  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor point_idxs = torch::full({N, H, W, K}, -1, int_opts);
  torch::Tensor zbuf = torch::full({N, H, W, K}, 2*zfar, float_opts);
  torch::Tensor pix_dists = torch::full({N, H, W, K}, -1, float_opts);

  const size_t blocks = 1024;
  const size_t threads = 64;
  RasterizePointsFineCudaKernel<<<blocks, threads>>>(
      points.contiguous().data_ptr<float>(),
      bin_points.contiguous().data_ptr<int32_t>(),
      radius,
      bin_size,
      N,
      B_h,
      B_w,
      M,
      H,
      W,
      K,
      point_idxs.contiguous().data_ptr<int32_t>(),
      zbuf.contiguous().data_ptr<float>(),
      pix_dists.contiguous().data_ptr<float>());

  return std::make_tuple(point_idxs, zbuf, pix_dists);
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************
// TODO(T55115174) Add more documentation for backward kernel.
__global__ void RasterizePointsBackwardCudaKernel(
    const float* points, // (P, 3)
    const float* colors, // (P, C)
    const float radius,
    const float znear,
    const float zfar,
    const float sigma,
    const float gamma,
    const int N,
    const int P,
    const int C,
    const int H,
    const int W,
    const float* accum_product, // (N, C, H, W)
    const float* accum_weights, // (N, 1, H, W)
    const float* grad_out_color, // (N, C, H, W)
    float *grad_points) // (P, 3)
{
    // Simple version: One thread per output pixel
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const float radius2 = radius * radius;
    // TODO gkopanas more than 1 batches?
    for (int i = tid; i < P ; i += num_threads) {
        const float px_ndc = points[i * 3 + 0];
        const float py_ndc = points[i * 3 + 1];
        const float pz = points[i * 3 + 2];
        if (pz < 0) {
            continue;
        }
        const float px = NdcToPix(px_ndc, W);
        const float py = NdcToPix(py_ndc, H);
        const float radius_pixels_x = radius*W/2.0;
        const float radius_pixels_y = radius*H/2.0;

        //printf("i %d tid %d px_ndc %f py_ndc %f px %f py %f ry %f rx %f\n", i, tid, px_ndc, py_ndc, px, py, radius_pixels_y, radius_pixels_x);
        int y_start = int(py - radius_pixels_y);
        int y_finish = int(py + radius_pixels_y);
        int x_start = int(px - radius_pixels_x);
        int x_finish = int(px + radius_pixels_x);
        //if (y_finish < 0 || y_start > H-1 || x_finish < 0 && x_start > W-1)
        //    continue;
        //printf("i %d tid %d y_start %d y_finish %d x_start %d x_finish %d\n", i, tid, y_start, y_finish, x_start, x_finish);
        for (int y_idx = y_start; y_idx < y_finish + 1; y_idx++) {
            for (int x_idx = x_start; x_idx <  x_finish + 1; x_idx++) {
                if (y_idx < 0 || y_idx > H - 1 || x_idx < 0 || x_idx > W - 1)
                    continue;
                float dx = PixToNdc(x_idx, W) - px_ndc;
                float dy = PixToNdc(y_idx, H) - py_ndc;
                float dist2 = dx*dx + dy*dy;
                if (dist2 > radius2)
                    continue;
                //float dist = sqrtf(dist2);
                float z_exponent = ((zfar - pz)/(zfar - znear))/gamma;
                float gauss_exponent = -dist2/(2*sigma*sigma);
                float w = expf(gauss_exponent + z_exponent);

                float sum_weights = accum_weights[0*H*W + y_idx*W + x_idx];
                float sum_weights_2 = sum_weights*sum_weights + 0.0000001;
                float dcp_dx = 0;
                float dcp_dy = 0;
                float dcp_dz = 0;
                for (int f = 0; f < C; f++) {
                    float grad_out_color_f = grad_out_color[0*C*H*W + f*H*W + y_idx*W + x_idx] ;
                    float accum_product_f = accum_product[0*C*H*W + f*H*W + y_idx*W + x_idx];
                    float point_f = colors[i*C + f];
                    //xy
                    float coef_xy = -1.0/(sigma*sigma);
                    float dw_dx = coef_xy*dx*w;
                    float dw_dy = coef_xy*dy*w;
                    //z
                    float dw_dz = -w/(gamma*(zfar-znear));

                    dcp_dx += ((-accum_product_f*dw_dx + point_f*dw_dx*sum_weights)/sum_weights_2)*grad_out_color_f;
                    dcp_dy += ((-accum_product_f*dw_dy + point_f*dw_dy*sum_weights)/sum_weights_2)*grad_out_color_f;
                    dcp_dz += ((-accum_product_f*dw_dz + point_f*dw_dz*sum_weights)/sum_weights_2)*grad_out_color_f;
                }
                grad_points[i*3 + 0] -= dcp_dx;
                grad_points[i*3 + 1] -= dcp_dy;
                grad_points[i*3 + 2] -= dcp_dz;
            }
        }
    }
}

torch::Tensor RasterizePointsBackwardCuda(
    const torch::Tensor& points, // (P, 3)
    const torch::Tensor& colors, // (P, 3)
    const float radius,
    const float znear,
    const float zfar,
    const float sigma,
    const float gamma,
    const torch::Tensor& accum_product, // (N, C, H, W)
    const torch::Tensor& accum_weights, // (N, 1, H, W)
    const torch::Tensor& grad_out_color) // (N, C, H, W)
{
  const int P = points.size(0);
  const int N = accum_product.size(0);
  const int C = accum_product.size(1);
  const int H = accum_product.size(2);
  const int W = accum_product.size(3);

  torch::Tensor grad_points = torch::zeros({P, 3}, points.options());
  const size_t blocks = 1024;
  const size_t threads = 64;

  RasterizePointsBackwardCudaKernel<<<blocks, threads>>>(
      points.contiguous().data<float>(),
      colors.contiguous().data<float>(),
      radius,
      znear,
      zfar,
      sigma,
      gamma,
      N,
      P,
      C,
      H,
      W,
      accum_product.contiguous().data<float>(),
      accum_weights.contiguous().data<float>(),
      grad_out_color.contiguous().data<float>(),
      grad_points.contiguous().data<float>());

  return grad_points;
}
