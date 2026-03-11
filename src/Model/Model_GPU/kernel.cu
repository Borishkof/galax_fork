#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
    const int block_size = 128; 

    __shared__ float3 shared_pos[block_size];
    __shared__ float  shared_mass[block_size];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 acc = {0.0f, 0.0f, 0.0f};
    float3 my_pos;

    if(i < n_particles) {
        my_pos = positionsGPU[i];
    }

    // On parcourt toutes les particules j par tiles
    for (int tile = 0; tile < n_particles; tile += block_size)
    {
        // Chaque thread du bloc
        int target_j = tile + threadIdx.x;
        if (target_j < n_particles) {
            shared_pos[threadIdx.x]  = positionsGPU[target_j];
            shared_mass[threadIdx.x] = massesGPU[target_j];
        } else {
            shared_pos[threadIdx.x]  = {0,0,0};
            shared_mass[threadIdx.x] = 0.0f;
        }

        // Bloc chargé
        __syncthreads();

        // CALCUL dans le tile
        if (i < n_particles) {
            for (int j = 0; j < block_size; j++) {
                if (i != (tile + j)) {
                    float3 pos_j = shared_pos[j];
                    float diffx = pos_j.x - my_pos.x;
                    float diffy = pos_j.y - my_pos.y;
                    float diffz = pos_j.z - my_pos.z;

                    float dist_sq = diffx * diffx + diffy * diffy + diffz * diffz;
                    
                    float dij;
                    if (dist_sq < 1.0f) {
                        dij = 10.0f;
                    } else {
                        // Utilisation de rsqrtf pour la performance
                        float inv_dist = rsqrtf(dist_sq);
                        dij = 10.0f * (inv_dist * inv_dist * inv_dist);
                    }

					dij *= shared_mass[j];

                    acc.x += diffx * dij;
                    acc.y += diffy * dij;
                    acc.z += diffz * dij;
                }
            }
        }
        __syncthreads();
    }

    if(i < n_particles) {
        accelerationsGPU[i] = acc;
    }
}

__global__ void maj_pos(float3* positionsGPU, float3* velocitiesGPU, 
                        float3* accelerationsGPU, int n_particles)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
    velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
    velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
    positionsGPU[i].x  += velocitiesGPU[i].x * 0.1f;
    positionsGPU[i].y  += velocitiesGPU[i].y * 0.1f;
    positionsGPU[i].z  += velocitiesGPU[i].z * 0.1f;
}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, 
                        float3* accelerationsGPU, float* massesGPU, int n_particles)
{
    int nthreads = 128;
    int nblocks  = (n_particles + nthreads - 1) / nthreads;

    compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
    maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}

#endif // GALAX_MODEL_GPU