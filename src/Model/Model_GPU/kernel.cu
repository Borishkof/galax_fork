#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"

__global__ void compute_acc(float3* positionsGPU, float3* velocitiesGPU, 
                             float3* accelerationsGPU, float* massesGPU, int n_particles)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles)
		return;

    float ai_x = 0.f, ai_y = 0.f, ai_z = 0.f;

    for (int j = 0; j < n_particles; j++)
    {
        float dx = positionsGPU[j].x - positionsGPU[i].x;
        float dy = positionsGPU[j].y - positionsGPU[i].y;
        float dz = positionsGPU[j].z - positionsGPU[i].z;

        float d2 = dx*dx + dy*dy + dz*dz;

        float force;
        if (d2 <  1.f)
			force = 10.f;
        else
			force = 10.f * rsqrtf(d2*d2*d2);

        float fm = force * massesGPU[j];
        ai_x += dx * fm;
        ai_y += dy * fm;
        ai_z += dz * fm;
    }

    accelerationsGPU[i].x = ai_x;
    accelerationsGPU[i].y = ai_y;
    accelerationsGPU[i].z = ai_z;
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