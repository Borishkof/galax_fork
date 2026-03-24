#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
//using b_type = xs::batch<float, xs::avx2>;
using arch_type = xs::default_arch;
using b_type    = xs::batch<float, arch_type>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
/*::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);



    #pragma omp parallel for
    for (int i = 0; i < n_particles; i += b_type::size)
    {
        b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        b_type raccx_i(0.0f);
        b_type raccy_i(0.0f);
        b_type raccz_i(0.0f);

        for (int j = 0; j < n_particles; j++)  // j scalaire
        {
            b_type rposx_j(particles.x[j]);
            b_type rposy_j(particles.y[j]);
            b_type rposz_j(particles.z[j]);
            b_type rmasse_j(initstate.masses[j]);

            auto diffx = rposx_j - rposx_i;
            auto diffy = rposy_j - rposy_i;
            auto diffz = rposz_j - rposz_i;

            auto dij = diffx * diffx + diffy * diffy + diffz * diffz;

            auto inv_sqrt = xsimd::rsqrt(dij);
            auto dij_else = 10.0f * inv_sqrt * inv_sqrt * inv_sqrt;

            dij = xsimd::select(dij < b_type(1.0f), b_type(10.0f), dij_else);
            dij *= rmasse_j;

            raccx_i += diffx * dij;
            raccy_i += diffy * dij;
            raccz_i += diffz * dij;
        }

        raccx_i.store_unaligned(&accelerationsx[i]);
        raccy_i.store_unaligned(&accelerationsy[i]);
        raccz_i.store_unaligned(&accelerationsz[i]);
    }

    #pragma omp parallel for
    for (int i = 0; i < n_particles; i += b_type::size){

        // load registers body i
        b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

        // load velocities i
        b_type rvelox_i = b_type::load_unaligned(&velocitiesx[i]);
        b_type rveloy_i = b_type::load_unaligned(&velocitiesy[i]);
        b_type rveloz_i = b_type::load_unaligned(&velocitiesz[i]);

        rvelox_i += raccx_i * 2.0f;
        rveloy_i += raccy_i * 2.0f;
        rveloz_i += raccz_i * 2.0f;

        rvelox_i.store_unaligned(&velocitiesx[i]);
        rveloy_i.store_unaligned(&velocitiesy[i]);
        rveloz_i.store_unaligned(&velocitiesz[i]);

        rposx_i += rvelox_i * 0.1f;
        rposy_i += rveloy_i * 0.1f;
        rposz_i += rveloz_i * 0.1f;

        rposx_i.store_unaligned(&particles.x[i]);
        rposy_i.store_unaligned(&particles.y[i]);
        rposz_i.store_unaligned(&particles.z[i]);
    }
}*/

::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);



    #pragma omp parallel for
    for (int i = 0; i < n_particles; i += b_type::size)
    {
        b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        b_type raccx_i(0.0f);
        b_type raccy_i(0.0f);
        b_type raccz_i(0.0f);

		for (int j = 0; j < n_particles; j++)
		{
            b_type rposx_j(particles.x[j]);
            b_type rposy_j(particles.y[j]);
            b_type rposz_j(particles.z[j]);
            b_type rmasse_j(initstate.masses[j]);

            auto diffx = rposx_j - rposx_i;
            auto diffy = rposy_j - rposy_i;
            auto diffz = rposz_j - rposz_i;

            auto dij = diffx * diffx + diffy * diffy + diffz * diffz;

            auto inv_sqrt = xsimd::rsqrt(dij);
            auto dij_else = 10.0f * inv_sqrt * inv_sqrt * inv_sqrt;

            dij = xsimd::select(dij < b_type(1.0f), b_type(10.0f), dij_else);
            dij *= rmasse_j;

            raccx_i += diffx * dij;
            raccy_i += diffy * dij;
            raccz_i += diffz * dij;
		}

        raccx_i.store_unaligned(&accelerationsx[i]);
        raccy_i.store_unaligned(&accelerationsy[i]);
        raccz_i.store_unaligned(&accelerationsz[i]);
	}

    #pragma omp parallel for
    for (int i = 0; i < n_particles; i += b_type::size){

        // load registers body i
        b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

        // load velocities i
        b_type rvelox_i = b_type::load_unaligned(&velocitiesx[i]);
        b_type rveloy_i = b_type::load_unaligned(&velocitiesy[i]);
        b_type rveloz_i = b_type::load_unaligned(&velocitiesz[i]);

        rvelox_i += raccx_i * 2.0f;
        rveloy_i += raccy_i * 2.0f;
        rveloz_i += raccz_i * 2.0f;

        rvelox_i.store_unaligned(&velocitiesx[i]);
        rveloy_i.store_unaligned(&velocitiesy[i]);
        rveloz_i.store_unaligned(&velocitiesz[i]);

        rposx_i += rvelox_i * 0.1f;
        rposy_i += rveloy_i * 0.1f;
        rposz_i += rveloz_i * 0.1f;

        rposx_i.store_unaligned(&particles.x[i]);
        rposy_i.store_unaligned(&particles.y[i]);
        rposz_i.store_unaligned(&particles.z[i]);
    }
}

#endif // GALAX_MODEL_CPU_FAST
