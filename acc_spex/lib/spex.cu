#include <torch/script.h>
#include <iostream>
#include <cuda_pipeline.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

#define FULL_MASK 0xffffffff

template <class T>
__host__ __device__ T *shared_array(std::size_t n_elements, void *&ptr,
                                    std::size_t *space = nullptr) noexcept
{
    const std::uintptr_t inptr = reinterpret_cast<uintptr_t>(ptr);
    const std::uintptr_t end = inptr + n_elements * sizeof(T);
    if (space)
        *space += static_cast<std::size_t>(end - inptr);
    ptr = reinterpret_cast<void *>(end);
    return reinterpret_cast<T *>(inptr);
}

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim)
{
    return (x + bdim - 1) / bdim;
}

template <typename scalar_t>
__global__ void spex_kernel(const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> radial_features,
                            const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> angular_features,
                            const torch::PackedTensorAccessor64<int, 1, torch::RestrictPtrTraits> neighbour_species,
                            const torch::PackedTensorAccessor64<int, 1, torch::RestrictPtrTraits> receiver_list,
                            const torch::PackedTensorAccessor64<int, 2, torch::RestrictPtrTraits> neighbour_indices_start,
                            const torch::PackedTensorAccessor64<int, 2, torch::RestrictPtrTraits> neighbour_indices_end,
                            const int64_t nnodes,
                            const int64_t nspecies,
                            scalar_t *output)
{

    extern __shared__ char buffer[];
    void *sptr = buffer;

    size_t offset = 0;

    const int nradial = radial_features.size(0);
    const int nangular = angular_features.size(0);

    scalar_t *buffer_node_output = shared_array<scalar_t>(nradial * nangular, sptr, &offset);
    scalar_t *buffer_radial = shared_array<scalar_t>(blockDim.x * nradial, sptr, &offset);
    scalar_t *buffer_angular = shared_array<scalar_t>(blockDim.x * nangular, sptr, &offset);
    int *buffer_valid_edge = shared_array<int>(blockDim.x, sptr, &offset);
    int *buffer_edge = shared_array<int>(blockDim.x, sptr, &offset);

    // edges should be ordered by (species, edge_idx). neighbour_indices_start/end count the index starts and ends for each species type
    // e.g for 1 atom with 10 neighbours of two species types:

    // edge_index: 0 1 2 3 4 5 6 7 8 9 : 10 edges
    // species_type: 0 0 0 0 1 1 1 1 1 : 2 species types
    // receiver_list: 0 0 0 0 0 0 0 0 0 0 : 0-th node

    // neighbour_indices_start[0][0] = 0
    // neighbour_indices_end[0][0] = 4
    // neighbour_indices_start[1][0] = 5
    // neighbour_indices_end[1][0] = 9
    
    for (int species = 0; species < nspecies; species++)
    {

        int edge_start = neighbour_indices_start[species][blockIdx.x];
        int edge_end = neighbour_indices_end[species][blockIdx.x];

        int node_index = receiver_list[edge_start];

        int nedges = edge_end - edge_start;

        // check if this node has neighbours of given element type
        if (nedges == 0)
        {
            continue;
        }

        // compute number of iterations we need to loop over nedges
        int ne_s = find_integer_divisor(nedges, blockDim.x);

        // clear buffers

        for (int tid = threadIdx.y * blockDim.x + threadIdx.x; tid < blockDim.x * nradial; tid += blockDim.x * blockDim.y)
        {
            buffer_radial[tid] = 0.0;
        }

        for (int tid = threadIdx.y * blockDim.x + threadIdx.x; tid < blockDim.x * nangular; tid += blockDim.x * blockDim.y)
        {
            buffer_angular[tid] = 0.0;
        }

        for (int tid = threadIdx.y * blockDim.x + threadIdx.x; tid < nradial * nangular; tid += blockDim.x * blockDim.y)
        {
            buffer_node_output[tid] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < ne_s; i++)
        {

            // only one warp needs to set this, since each warp hnadles a different angular index.
            if (threadIdx.y == 0)
            {
                buffer_edge[threadIdx.x] = i * blockDim.x + threadIdx.x;
                buffer_valid_edge[threadIdx.x] = buffer_edge[threadIdx.x] < nedges;
            }

            __syncthreads();

            // load buffers

            for (int j = threadIdx.y; j < nradial; j += blockDim.y)
            {
                scalar_t r = 0.0;
                if (buffer_valid_edge[threadIdx.x])
                {
                    r = radial_features[j][buffer_edge[threadIdx.x]];
                }

                buffer_radial[j * blockDim.x + threadIdx.x] = r;
            }

            for (int j = threadIdx.y; j < nangular; j += blockDim.y)
            {
                scalar_t ang = 0.0;
                if (buffer_valid_edge[threadIdx.x])
                {
                    ang = angular_features[j][buffer_edge[threadIdx.x]];
                }

                buffer_angular[j * blockDim.x + threadIdx.x] = ang;
            }

            __syncthreads();

            // do computation, and reduce over edges on-the-fly
            for (int r_idx = 0; r_idx < nradial; r_idx++)
            {
                for (int a_idx = threadIdx.y; a_idx < nangular; a_idx += blockDim.y)
                {

                    int tmp = buffer_radial[r_idx * blockDim.x + threadIdx.x] * buffer_angular[a_idx * blockDim.x + threadIdx.x];

                    // compute how many threads are involved in the warp reduction first, then use that for the offset.
                    unsigned int valid_mask = __match_any_sync(__activemask(), buffer_valid_edge[threadIdx.x]);

                    for (int offset = __popc(valid_mask) / 2; offset > 0; offset /= 2) // shuffle reduce to include highest lane discovered in valid_mask
                    {
                        tmp += __shfl_down_sync(valid_mask, tmp, offset);
                    }

                    if (threadIdx.x == 0)
                    {
                        buffer_node_output[species * (nradial + nangular) + r_idx * nangular + a_idx] += tmp;
                    }
                }
            }
        }

        // write to global memory

        __syncthreads();

        for (int tid = threadIdx.y * blockDim.x + threadIdx.x; tid < nradial * nangular; tid += blockDim.x * blockDim.y)
        {
            output[node_index * (nradial * nspecies * nangular) + species * nradial * nangular + tid] = buffer_node_output[tid];
        }
    }
}

torch::Tensor spex_gpu(torch::Tensor radial_features,
                       torch::Tensor angular_features,
                       torch::Tensor neighbour_species,
                       torch::Tensor receiver_list,
                       torch::Tensor neighbour_list_start,
                       torch::Tensor neighbour_list_end,
                       int64_t nnodes,
                       int64_t nspecies)
{

    const int64_t nradial = radial_features.size(0);
    const int64_t nangular = angular_features.size(0);

    torch::Tensor output = torch::empty({nnodes, nspecies, nradial, nangular},
                                        torch::TensorOptions()
                                            .dtype(radial_features.dtype())
                                            .device(radial_features.device()));

    dim3 block_dim(nnodes);

    dim3 grid_dim(32, min(nangular, (long)4), 1);

    AT_DISPATCH_FLOATING_TYPES(
        radial_features.type(), "spex_gpu", ([&]
                                             {
                size_t shared_size = 0;
                void* sptr = nullptr;
                
                shared_array<scalar_t>(nradial * nangular, sptr, &shared_size);
                shared_array<scalar_t>(grid_dim.x * nradial, sptr, &shared_size);
                shared_array<scalar_t>(grid_dim.x * nangular, sptr, &shared_size);
                shared_array<int32_t>(grid_dim.x, sptr, &shared_size);
                shared_array<int32_t>(grid_dim.x, sptr, &shared_size);

                spex_kernel<scalar_t><<<block_dim, grid_dim, shared_size>>>(
                    radial_features.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    angular_features.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    neighbour_species.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    neighbour_list_start.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>(),
                    neighbour_list_end.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>(),
                    nnodes,
                    nspecies,
                    output.data_ptr<scalar_t>()); }));

    cudaDeviceSynchronize();

    return output;
}

#define NEIGHBOUR_NEDGES_PER_BLOCK 512

/*
This function takes a sorted input sender_list, which maps each edge to a node by index, and finds the positions of first occurences

This is required by the CUDA code so we can send all calculations per-node to a single block.

the function loads NEIGHBOUR_NEDGES_PER_BLOCK + 1 elements into shared memory, and then loops through the buffer twice. Once for even boundaries, once for odd boundaries.
*/

__global__ void calculate_neighbours_kernel(const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> sender_list,
                                            torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> edge_indices)
{
    extern __shared__ char buffer[];
    size_t offset = 0;
    int32_t *smem = reinterpret_cast<int32_t *>(buffer + offset);

    int32_t block_start = blockIdx.x * NEIGHBOUR_NEDGES_PER_BLOCK;

    int32_t nedges = sender_list.size(0);

    // load all elements of senderlist needed by block into shared memory
    for (int32_t i = threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1; i += blockDim.x)
    {
        int32_t idx = block_start + i;

        if (idx < nedges)
        {
            smem[i] = sender_list[idx];
        }
    }

    __syncthreads();

    // deal with even boundaries
    for (int32_t i = 2 * threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK; i += 2 * blockDim.x)
    {
        int32_t idx = block_start + i;

        if (idx + 1 < nedges)
        {
            int32_t loc1 = smem[i];
            int32_t loc2 = smem[i + 1];

            if (loc1 != loc2)
            {
                edge_indices[loc2] = idx + 1;
            }
        }
    }

    // deal with odd boundaries
    for (int32_t i = 2 * threadIdx.x + 1; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1; i += 2 * blockDim.x)
    {
        int32_t idx = block_start + i;

        if (idx + 1 < nedges)
        {
            int32_t loc1 = smem[i];
            int32_t loc2 = smem[i + 1];

            if (loc1 != loc2)
            {
                edge_indices[loc2] = idx + 1;
            }
        }
    }

    // deal with 0th element specifically, so we dont need to use torch::zeros
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        edge_indices[0] = 0;
    }
}

torch::Tensor calculate_neighbours_gpu(torch::Tensor sender_list, int64_t natoms, int64_t nthreadx)
{
    torch::Tensor output_indices = torch::empty(natoms,
                                                torch::TensorOptions()
                                                    .dtype(sender_list.dtype())
                                                    .device(sender_list.device()));

    int32_t nbx = find_integer_divisor(sender_list.size(0), NEIGHBOUR_NEDGES_PER_BLOCK);

    dim3 block_dim(nbx);

    dim3 grid_dim(nthreadx, 1, 1);

    size_t total_buff_size = 0;

    total_buff_size += (NEIGHBOUR_NEDGES_PER_BLOCK + 1) * sizeof(int32_t);

    calculate_neighbours_kernel<<<block_dim, grid_dim, total_buff_size>>>(

        sender_list.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        output_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return output_indices;
}

TORCH_LIBRARY(spex_cu, m)
{
    m.def("spex", &spex_gpu);
    m.def("calculate_neighbours", &calculate_neighbours_gpu);
}
