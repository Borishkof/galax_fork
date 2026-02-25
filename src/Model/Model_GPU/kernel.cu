#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > n_particles)
		return;


	for (int j = 0; j < n_particles; j++)
		{
			const float diffx = positionsGPU.x[j] - positionsGPU.x[i];
			const float diffy = positionsGPU.y[j] - positionsGPU.y[i];
			const float diffz = positionsGPU.z[j] - positionsGPU.z[i];

			float dij = diffx * diffx + diffy * diffy + diffz * diffz;

			if (dij < 1.0)
			{
				dij = 10.0;
			}
			else
			{
				dij = std::sqrt(dij);
				dij = 10.0 / (dij * dij * dij);
			}

			accelerationsGPU.x[i] += diffx * dij * massesGPU[j];
			accelerationsGPU.y[i] += diffy * dij * massesGPU[j];
			accelerationsGPU.z[i] += diffz * dij * massesGPU[j];
		}
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n_particles)
		return;

	velocitiesGPU.x[i] += accelerationsGPU.x[i] * 2.0f;
	velocitiesGPU.y[i] += accelerationsGPU.y[i] * 2.0f;
	velocitiesGPU.z[i] += accelerationsGPU.z[i] * 2.0f;
	positionsGPU.x[i] += velocitiesGPU.x   [i] * 0.1f;
	positionsGPU.y[i] += velocitiesGPU.y   [i] * 0.1f;
	positionsGPU.z[i] += velocitiesGPU.z   [i] * 0.1f;

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
