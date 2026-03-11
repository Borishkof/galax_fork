#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3* positionsGPU, float3* velocitiesGPU, 
                             float3* accelerationsGPU, float* massesGPU, int n_particles)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Position de i chargée une fois en registre
    float3 pi = (i < n_particles) ? positionsGPU[i] : make_float3(0,0,0);
    float3 acc = make_float3(0.0f, 0.0f, 0.0f);

    // Tile partagée par tout le bloc
    __shared__ float3 shared_pos [128];
    __shared__ float  shared_mass[128];

    for (int tile = 0; tile < (n_particles + blockDim.x - 1) / blockDim.x; tile++)
    {
        // Chaque thread charge UN élément j dans la shared memory
        int j = tile * blockDim.x + threadIdx.x;
        shared_pos [threadIdx.x] = (j < n_particles) ? positionsGPU[j] : make_float3(0,0,0);
        shared_mass[threadIdx.x] = (j < n_particles) ? massesGPU[j]    : 0.0f;
        __syncthreads();  // attendre que le tile soit chargé

        // Calculer les interactions avec les j du tile
        #pragma unroll
        for (int k = 0; k < blockDim.x; k++)
        {
            float dx = shared_pos[k].x - pi.x;
            float dy = shared_pos[k].y - pi.y;
            float dz = shared_pos[k].z - pi.z;

            float d2 = dx*dx + dy*dy + dz*dz;

            float force;
            if (d2 < 1.0f || d2 == 0.0f)
                force = (d2 == 0.0f) ? 0.0f : 10.0f;
            else {
                float inv = rsqrtf(d2);       // instruction native GPU, plus rapide que sqrt
                force = 10.0f * inv*inv*inv;
            }

            float fm = force * shared_mass[k];
            acc.x += dx * fm;
            acc.y += dy * fm;
            acc.z += dz * fm;
        }
        __syncthreads();  // avant de réécrire le tile
    }

    if (i < n_particles)
    {
        accelerationsGPU[i].x = acc.x;
        accelerationsGPU[i].y = acc.y;
        accelerationsGPU[i].z = acc.z;
    }
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n_particles)
		return;

	velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
	velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
	velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
	positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
	positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
	positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
