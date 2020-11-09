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
  float alpha; // Alpha blending weight
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
// *                          GK RASTERIZATION                             *
// ****************************************************************************

__global__ void OrderPointsGKCudaKernel(
    int32_t* point_idxs, // (N, H, W, K)
    uint32_t* k_idxs,
    const float* points,
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
    int32_t q[kMaxPointsPerPixel];
    int k = min(k_idxs[yi*W + xi], K);
    for (int i=0; i<k; i++) {
        q[i] = point_idxs[idx + i];
    }
    BubbleSort2(q, points, k);
    for (int i=0; i<k; i++) {
        point_idxs[idx + i] = q[i];
    }
  }
}

__global__ void BlendPointsGKCudaKernel(
    const float* points, // (P, 3)
    int32_t* point_idx, // (N, H, W, K)
    const float* colors, // (P, C)
    const int32_t* k_idxs, // (N, H, W)
    const float* sigmas, // (P, 1)
    const float* inv_cov, // (P, 4)
    const int max_radius,
    const float gamma,
    const int N,
    const int H,
    const int W,
    const int C,
    const int K,
    const float zfar,
    const float znear,
    float* color, // (N, 3, H, W)
    float* mask, // (N, 1, H, W)
    float* depth) // (N, 1, H, W)
{
    const int radius2 = max_radius*max_radius;
    // One thread per output pixel
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < H * W; i += num_threads) {
        // Convert linear index to 3D index
        const int pix_idx = i % (H * W);

        const int yi = pix_idx / W;
        const int xi = pix_idx % W;

        Pix gathered_points[kMaxPointPerPixelLocal];
        int y_start = yi - max_radius;
        int y_finish = yi + max_radius;
        int x_start = xi - max_radius;
        int x_finish = xi + max_radius;

        int gathered_points_idx = 0;
        int gathered_points_idx_max = -1;
        float gathered_points_z_max = -1000000.0;
        for (int y_idx = y_start; y_idx < y_finish + 1; y_idx++) {
            for (int x_idx = x_start; x_idx <  x_finish + 1; x_idx++) {
                if (y_idx < 0 || y_idx > H - 1 || x_idx < 0 || x_idx > W - 1)
                    continue;
                int k = k_idxs[y_idx*W + x_idx];
                int idx = 0 * H * W * K + y_idx * W * K + x_idx * K + 0;
                for (int i=0; i<k; i++) {
                    int p_idx = point_idx[idx + i];
                    float px_ndc = points[p_idx*3 + 0];
                    float py_ndc = points[p_idx*3 + 1];
                    float pz     = points[p_idx*3 + 2];
                    if (pz < 0)
                        // Don't render points behind the camera.
                        continue;
                    float dx = NdcToPix(px_ndc, W) - xi;
                    float dy = NdcToPix(py_ndc, H) - yi;
                    float dist2 = dx*dx + dy*dy;
                    // Trim it to a circle
                    if (dist2 > radius2)
                        continue;

                    float a = inv_cov[p_idx*4 + 0];
                    float b = inv_cov[p_idx*4 + 1];
                    float c = inv_cov[p_idx*4 + 2];
                    float d = inv_cov[p_idx*4 + 3];

                    float g_w = exp((-1.0/2.0)*(a*dx*dx + 2*b*dx*dy + d*dy*dy));
                    //float g_w = exp((-1.0/2.0)*((dx*dx)/(s_x*s_x) + (dy*dy)/(s_y*s_y)));
                    float alpha = pow(g_w, gamma);
                    if (alpha < 1/255.0)
                        continue;

                    // If more than kMaxPointPerPixelLocal we need to compare against the max z
                    // if we are closer we replace our selves and search for the max again
                    if (gathered_points_idx > kMaxPointPerPixelLocal - 1) {
                        if (pz < gathered_points_z_max) {
                            gathered_points[gathered_points_idx_max].idx = p_idx;
                            gathered_points[gathered_points_idx_max].dist2 = dist2;
                            gathered_points[gathered_points_idx_max].alpha = alpha;
                            gathered_points[gathered_points_idx_max].z = pz;

                            gathered_points_z_max = -1.0;
                            for (int j=0; j<gathered_points_idx; j++) {
                                if (gathered_points[j].z > gathered_points_z_max) {
                                    gathered_points_idx_max = j;
                                    gathered_points_z_max = gathered_points[j].z;
                                }
                            }
                        }
                    }
                    else {
                        if (pz > gathered_points_z_max) {
                            gathered_points_idx_max = i;
                            gathered_points_z_max = pz;
                        }
                        gathered_points[gathered_points_idx].idx = p_idx;
                        gathered_points[gathered_points_idx_max].dist2 = dist2;
                        gathered_points[gathered_points_idx].alpha = alpha;
                        gathered_points[gathered_points_idx].z = pz;
                        gathered_points_idx++;
                    }
                }
            }
        }
        BubbleSort(gathered_points, gathered_points_idx);
        int idx = 0 * H * W * K + yi * W * K + xi * K + 0;

        float cum_alpha = 1.0;
        /* TODO: Adding iteratively to global memory can be slow, but dynamic allocation doesnt work. */
        //float result[3] = {0.0, 0.0, 0.0};
        //float* result = new float[C]();
        //float *result = (float *)malloc(3*sizeof(float));
        float max_alpha = 0.0;
        float best_depth = -1.0;
        for (int k=0; k<gathered_points_idx; k++) {
            float alpha = gathered_points[k].alpha;
            for (int ch=0; ch<C; ch++) {
                color[ch*H*W + yi*W + xi] += colors[gathered_points[k].idx*C + ch] * cum_alpha * alpha;
            }
            if (cum_alpha * alpha > max_alpha) {
                max_alpha = cum_alpha*alpha;
                best_depth = (zfar - gathered_points[k].z)/(zfar-znear);
            }
            cum_alpha = cum_alpha * (1 - alpha);
            if (cum_alpha<0.001) {
                break;
            }
        }
        mask[yi*W + xi] = 1.0 - cum_alpha;
        depth[yi*W + xi] = best_depth;
        //for (int ch=0; ch<C; ch++) {
        //    color[ch*H*W + yi*W + xi] = result[ch];
        //}
    }
}

__global__ void RasterizePointsGKCudaKernel(
    const float* points, // (P, 3)
    const int P,
    uint32_t* k_idxs, // (N, H, W)
    const int N,
    const int H,
    const int W,
    const int K,
    int32_t* point_idxs) // (N, H, W, K)
{
    // Simple version: One thread per output pixel
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // TODO gkopanas more than 1 batches?
    for (int i = tid; i < P ; i += num_threads) {
        const float px_ndc = points[i * 3 + 0];
        const float py_ndc = points[i * 3 + 1];
        const float pz = points[i * 3 + 2];

        const float px = NdcToPix(px_ndc, W);
        const float py = NdcToPix(py_ndc, H);

        const int px_rounded = int(px + 0.5);
        const int py_rounded = int(py + 0.5);
        if (py_rounded < 0 || py_rounded > H - 1 || px_rounded < 0 || px_rounded > W - 1)
            continue;

        int k_idx = atomicInc(&(k_idxs[0*H*W + py_rounded*W + px_rounded]), K + 1);
        if (k_idx == K) {
            printf("Pixel y:%d x:%d exceeded point projection limit\n", py_rounded, px_rounded);
            //assert(0);
        }

        int idx = 0 * H * W * K + py_rounded * W * K + px_rounded * K + k_idx;
        point_idxs[idx] = i;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizePointsGKCuda(
    const torch::Tensor& points, // (P, 3)
    const torch::Tensor& colors, // (P, C)
    const torch::Tensor& sigmas, // (P, 1)
    const torch::Tensor& inv_cov, // (P, 4)
    const int max_radius,
    const int image_height,
    const int image_width,
    const int points_per_pixel,
    const float zfar,
    const float znear,
    const float gamma) {

  if (points.ndimension() != 2 || points.size(1) != 3) {
    AT_ERROR("points must have dimensions (num_points, 3)");
  }

  const int P = points.size(0);
  const int C = colors.size(1);
  const int N = 1; // batch size hard-coded
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
  torch::Tensor point_idxs = torch::full({N, H, W, kMaxPointsPerPixel}, -1, int_opts);
  torch::Tensor k_idxs = torch::full({N, H, W}, 0, int_opts);
  torch::Tensor out_color = torch::full({N, C, H, W}, 0.0, float_opts);
  torch::Tensor depth = torch::full({N, 1, H, W}, -1.0, float_opts);
  torch::Tensor mask = torch::full({N, 1, H, W}, 0.0, float_opts);

  const size_t blocks = 1024;
  const size_t threads = 64;

  RasterizePointsGKCudaKernel<<<blocks, threads>>>(
      points.contiguous().data<float>(),
      P,
      (unsigned int *)k_idxs.data<int32_t>(),
      N,
      H,
      W,
      kMaxPointsPerPixel,
      point_idxs.contiguous().data<int32_t>());

  cudaDeviceSynchronize();

  BlendPointsGKCudaKernel<<<blocks, threads>>>(
      points.contiguous().data<float>(),
      point_idxs.contiguous().data<int32_t>(),
      colors.contiguous().data<float>(),
      k_idxs.data<int32_t>(),
      sigmas.data<float>(),
      inv_cov.data<float>(),
      max_radius,
      gamma,
      N,
      H,
      W,
      C,
      kMaxPointsPerPixel,
      zfar,
      znear,
      out_color.contiguous().data<float>(),
      mask.contiguous().data<float>(),
      depth.contiguous().data<float>());

  //point_idxs = point_idxs.narrow(-1, 0, K);

  return std::make_tuple(point_idxs, out_color, k_idxs, depth, mask);
}


// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************
// TODO(T55115174) Add more documentation for backward kernel.
__global__ void RasterizePointsBackwardCudaKernel(
        const float *points, // (P, 3)
        const float *colors, // (P, C)
        const float *sigmas, // (P, 1)
        const int max_radius,
        const int32_t *idxs, // (N, H, W, K)
        const int32_t *k_idxs,
        const int N,
        const int P,
        const int C,
        const int H,
        const int W,
        const int K,
        const float znear,
        const float zfar,
        const float gamma,
        float* grad_out_color,
        float* grad_points,
        float* grad_colors,
        float* grad_sigmas) {
    int radius = max_radius;

    const int radius2 = radius*radius;
    // One thread per output pixel
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < H * W; i += num_threads) {
        // Convert linear index to 3D index
        const int pix_idx = i % (H * W);

        const int yi = pix_idx / W;
        const int xi = pix_idx % W;

        Pix gathered_points[kMaxPointPerPixelLocal];
        int y_start = yi - radius;
        int y_finish = yi + radius;
        int x_start = xi - radius;
        int x_finish = xi + radius;

        int gathered_points_idx = 0;
        int gathered_points_idx_max = -1;
        float gathered_points_z_max = -1000000.0;
        for (int y_idx = y_start; y_idx < y_finish + 1; y_idx++) {
            for (int x_idx = x_start; x_idx <  x_finish + 1; x_idx++) {
                if (y_idx < 0 || y_idx > H - 1 || x_idx < 0 || x_idx > W - 1)
                    continue;
                int k = k_idxs[y_idx*W + x_idx];
                int idx = 0 * H * W * K + y_idx * W * K + x_idx * K + 0;
                for (int i=0; i<k; i++) {
                    int p_idx = idxs[idx + i];
                    float px_ndc = points[p_idx*3 + 0];
                    float py_ndc = points[p_idx*3 + 1];
                    float pz     = points[p_idx*3 + 2];
                    if (pz < 0)
                        // Don't render points behind the camera.
                        continue;
                    float dx = NdcToPix(px_ndc, W) - xi;
                    float dy = NdcToPix(py_ndc, H) - yi;
                    float dist2 = dx*dx + dy*dy;
                    if (dist2 > radius2)
                        continue;

                    float sigma = sigmas[p_idx];
                    float g_w = exp(-dist2/(2*sigma*sigma));
                    float alpha = pow(g_w, gamma);
                    if (alpha < 1/255.0)
                        continue;

                    if (gathered_points_idx > kMaxPointPerPixelLocal - 1) {
                        if (pz < gathered_points_z_max) {
                            gathered_points[gathered_points_idx_max].idx = p_idx;
                            gathered_points[gathered_points_idx_max].dist2 = dist2;
                            gathered_points[gathered_points_idx_max].alpha = alpha;
                            gathered_points[gathered_points_idx_max].z = pz;

                            gathered_points_z_max = -1.0;
                            for (int j=0; j<gathered_points_idx; j++) {
                                if (gathered_points[j].z > gathered_points_z_max) {
                                    gathered_points_idx_max = j;
                                    gathered_points_z_max = gathered_points[j].z;
                                }
                            }
                        }
                    }
                    else {
                        if (pz > gathered_points_z_max) {
                            gathered_points_idx_max = i;
                            gathered_points_z_max = pz;
                        }
                        gathered_points[gathered_points_idx].idx = p_idx;
                        gathered_points[gathered_points_idx_max].dist2 = dist2;
                        gathered_points[gathered_points_idx].alpha = alpha;
                        gathered_points[gathered_points_idx].z = pz;
                        gathered_points_idx++;
                    }
                }
            }
        }
        BubbleSort(gathered_points, gathered_points_idx);
        // Now we have all points(dists2, idx, z) in-order of z for pixel yi,xi
        // for each color every point needs to go the the grad_buffer and add it's contribution to
        // it's index.
        int idx = 0 * H * W * K + yi * W * K + xi * K + 0;
        float w[kMaxPointPerPixelLocal];
        float alpha_cum[kMaxPointPerPixelLocal];
        float cum_alpha = 1.0;
        float result=0.0;
        int num_points_contribute = 0;

        for (int k=0; k<gathered_points_idx; k++) {
            float alpha = gathered_points[k].alpha;
            w[k] = alpha;
            alpha_cum[k] = cum_alpha;
            cum_alpha = cum_alpha * (1 - alpha);
            num_points_contribute = k+1;
            if (cum_alpha<0.001) {
                break;
            }
        }
        for (int ch=0; ch<C; ch++) {
            float grad_out_color_f  = grad_out_color[ ch*H*W + yi*W + xi];
            for (int k=0; k < num_points_contribute; k++) {
                float c_k = colors[gathered_points[k].idx*C + ch];
                float sigma = sigmas[gathered_points[k].idx];
                /* This inner loop can be optimized out */
                float accum_prod_1 = 1.0;
                for (int j=0; j<k; j++) {
                    accum_prod_1 *= (1 - w[j]);
                }

                float accum_sum = 0;
                for (int u=k+1; u < num_points_contribute; u++) {
                    float c_u = colors[gathered_points[u].idx*C + ch];
                    float accum_prod_2 = 1.0;
                    for (int j=0; j < u; j++) {
                        if (j==k) continue;
                        accum_prod_2 *= (1 - w[j]);
                    }
                    accum_sum += c_u*w[u]*accum_prod_2;
                }
                float d_bN_w = c_k*accum_prod_1 - accum_sum;
                float d_wk_x = -(gamma*2*(points[gathered_points[k].idx*3 + 0] - PixToNdc(xi, W))*w[k])/(2*sigma*sigma);
                //float d_wk_x = 0.0;
                float d_wk_y = -(gamma*2*(points[gathered_points[k].idx*3 + 1] - PixToNdc(yi, H))*w[k])/(2*sigma*sigma);
                float d_wk_z = 0.0;
                float d_wk_sigma = w[k]*gathered_points[k].dist2/pow(sigma, 3);
                float d_bN_c = w[k]*alpha_cum[k];
                //if (gathered_points[k].idx == 0) {
                //    printf("\t%d\t%d\t%f\t%f\t%f\t%f\n", xi, yi, d_bN_w, d_wk_x, d_wk_y,  grad_out_color_f);
                //}
                atomicAdd(&(grad_points[gathered_points[k].idx*3 + 0]), d_bN_w*d_wk_x*grad_out_color_f);
                atomicAdd(&(grad_points[gathered_points[k].idx*3 + 1]), d_bN_w*d_wk_y*grad_out_color_f);
                atomicAdd(&(grad_points[gathered_points[k].idx*3 + 2]), d_bN_w*d_wk_z*grad_out_color_f);
                atomicAdd(&(grad_sigmas[gathered_points[k].idx]), d_bN_w*d_wk_sigma*grad_out_color_f);
                atomicAdd(&(grad_colors[gathered_points[k].idx*C + ch]), d_bN_c*grad_out_color_f);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizePointsBackwardCuda(
    const torch::Tensor& points, // (P, 3)
    const torch::Tensor& colors, // (P, C)
    const torch::Tensor& sigmas, // (P, 1)
    const int max_radius,
    const torch::Tensor& idxs, // (N, H, W, K)
    const torch::Tensor& k_idxs,
    const float znear,
    const float zfar,
    const float gamma,
    const torch::Tensor& grad_out_color) {
  const int P = points.size(0);
  const int C = colors.size(1);
  const int N = idxs.size(0);
  const int H = idxs.size(1);
  const int W = idxs.size(2);
  const int K = idxs.size(3);

  torch::Tensor grad_points = torch::zeros({P, 3}, points.options());
  torch::Tensor grad_colors = torch::zeros({P, C}, points.options());
  torch::Tensor grad_sigmas = torch::zeros({P, 1}, points.options());
  const size_t blocks = 1024;
  const size_t threads = 64;
  RasterizePointsBackwardCudaKernel<<<blocks, threads>>>(
      points.contiguous().data<float>(),
      colors.contiguous().data<float>(),
      sigmas.contiguous().data<float>(),
      max_radius,
      idxs.contiguous().data<int32_t>(),
      k_idxs.contiguous().data<int32_t>(),
      N,
      P,
      C,
      H,
      W,
      K,
      znear,
      zfar,
      gamma,
      grad_out_color.contiguous().data<float>(),
      grad_points.contiguous().data<float>(),
      grad_colors.contiguous().data<float>(),
      grad_sigmas.contiguous().data<float>());

  return std::make_tuple(grad_points, grad_colors, grad_sigmas);
}
