#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/maca_kernels/reduce_utils.cuh>
#include <c10/util/ArrayRef.h>

#include <iostream>


namespace at::native {

static inline std::ostream& operator<<(std::ostream& out, dim3 dim) {
  if (dim.y == 1 && dim.z == 1) {
    out << dim.x;
  } else {
    out << "[" << dim.x << "," << dim.y << "," << dim.z << "]";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const ReduceConfigCUDA& config) {
  out << "ReduceConfig(";
  out << "element_size_bytes=" << config.element_size_bytes << ", ";
  out << "num_inputs=" << config.num_inputs << ", ";
  out << "num_outputs=" << config.num_outputs << ", ";
  out << "step_input=" << config.step_input << ", ";
  out << "step_output=" << config.step_output << ", ";
  out << "ctas_per_output=" << config.ctas_per_output << ", ";
  out << "input_mult=[";
  for (int i = 0; i < 3; i++) {
    if (i != 0) {
      out << ",";
    }
    out << config.input_mult[i];
  }
  out << "], ";
  out << "output_mult=[";
  for (int i = 0; i < 2; i++) {
    if (i != 0) {
      out << ",";
    }
    out << config.output_mult[i];
  }
  out << "], ";
  out << "vectorize_input=" << config.vectorize_input << ", ";
  out << "output_vec_size=" << config.output_vec_size << ", ";
  out << "input_vec_size=" << config.input_vec_size << ", ";
  out << "block_width=" << config.block_width << ", ";
  out << "block_height=" << config.block_height << ", ";
  out << "num_threads=" << config.num_threads << ", ";
  out << "values_per_thread=" << config.values_per_thread() << ", ";
  out << "block=" << config.block() << ", ";
  out << "grid=" << config.grid() << ", ";
  out << "global_memory_size=" << config.global_memory_size();
  out << ")";
  return out;
}

std::ostream& operator<<(std::ostream& out, const ReduceConfigMaca& config) {
  out << "ReduceConfigMaca(";
  out << "element_size_bytes=" << config.element_size_bytes << ", ";
  out << "num_inputs=" << config.num_inputs << ", ";
  out << "num_outputs=" << config.num_outputs << ", ";
  out << "step_input=" << config.step_input << ", ";
  out << "step_output=" << config.step_output << ", ";
  out << "ctas_per_output=" << config.ctas_per_output << ", ";
  out << "input_mult=[";
  for (int i = 0; i < 3; i++) {
    if (i != 0) {
      out << ",";
    }
    out << config.input_mult[i];
  }
  out << "], ";
  out << "output_mult=[";
  for (int i = 0; i < 2; i++) {
    if (i != 0) {
      out << ",";
    }
    out << config.output_mult[i];
  }
  out << "], ";
  out << "vectorize_input=" << config.vectorize_input << ", ";
  out << "output_vec_size=" << config.output_vec_size << ", ";
  out << "input_vec_size=" << config.input_vec_size << ", ";
  out << "reduce_type=" << config.reduce_type << ", ";
  out << "block_width=" << config.block_width << ", ";
  out << "block_height=" << config.block_height << ", ";
  out << "num_threads=" << config.num_threads << ", ";
  out << "values_per_thread=" << config.values_per_thread() << ", ";
  out << "block=" << config.block() << ", ";
  out << "grid=" << config.grid() << ", ";
  out << "global_memory_size=" << config.global_memory_size() << ",";
  out << "shared_memory_size=" << config.shared_memory_size();
  out << ")";
  return out;
}

__global__ __forceinline__ void semaphores_reset_impl(int* semaphores, int semaphore_size) {
  constexpr size_t vec_size = 4;
  using vec_t = aligned_vector<int, vec_size>;
  vec_t *semaphores_vec = reinterpret_cast<vec_t *>(semaphores);
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = blockDim.x * gridDim.x;
  int n_vec_to_read = semaphore_size / vec_size;
  for (int i = thread_id; i < n_vec_to_read; i+=step) {
    semaphores_vec[i] = {0, 0, 0, 0};
  }

  size_t remain_offset = n_vec_to_read * vec_size;
  size_t remain_size = semaphore_size - remain_offset;
  for (int i = thread_id; i < remain_size; i+=step) {
    semaphores[remain_offset + i] = 0;
  }
}

void semaphores_reset(int* semaphores, int semaphore_bytes) {
  constexpr size_t vec_size = 4;
  constexpr size_t max_warp_num = 8;
  constexpr size_t max_warp_size = C10_WARP_SIZE * vec_size;
  constexpr size_t max_block_size = max_warp_size * max_warp_num;
  unsigned int semaphore_size = semaphore_bytes / sizeof(int);
  unsigned int grid_dim_x = (semaphore_size + max_block_size - 1) / max_block_size;
  unsigned int num_warps = max_warp_size;
  if (grid_dim_x == 1) {
    num_warps = semaphore_size / max_warp_size;
    num_warps = num_warps == 0 ? 1 : num_warps;
  }

  unsigned int  block_dim_x = C10_WARP_SIZE * num_warps;
  dim3 block = {block_dim_x, 1, 1};
  dim3 grid = {grid_dim_x, 1, 1};
  auto stream = at::cuda::getCurrentCUDAStream();
  semaphores_reset_impl<<<grid, block, 0, stream>>>(semaphores, semaphore_size);
}

}  // namespace at::native
