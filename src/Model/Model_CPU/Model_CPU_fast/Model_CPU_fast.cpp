#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);



    // OMP  version
    #pragma omp parallel for
    for (int i = 0; i < n_particles; i += b_type::size)
    {
        // load registers body i
        b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

        for (int j = 0; j < n_particles; j += b_type::size)
        {
            // load registers body j
            b_type rposx_j = b_type::load_unaligned(&particles.x[j]);
            b_type rposy_j = b_type::load_unaligned(&particles.y[j]);
            b_type rposz_j = b_type::load_unaligned(&particles.z[j]);
            b_type rmasse_j = b_type::load_unaligned(&initstate.masses[j]);

            auto diffx = rposx_j - rposx_i;
            auto diffy = rposy_j - rposy_i;
            auto diffz = rposz_j - rposz_i;

            auto dij = diffx * diffx + diffy * diffy + diffz * diffz;

            auto mask = dij < 1.0; // other way to do ?
            auto mask_0_div = dij == 0;
            auto sqrt = xsimd::select(mask_0_div, b_type(1.0f), xsimd::rsqrt(dij)); // faster sqrt / pow / inv ?
            auto dij_else = 10.0f * (sqrt * sqrt * sqrt);

            dij = xsimd::select(mask, b_type(10.0f), dij_else);

            raccx_i += diffx * dij * rmasse_j;
            raccy_i += diffy * dij * rmasse_j;
            raccz_i += diffz * dij * rmasse_j;

            raccx_i.store_unaligned(&accelerationsx[i]);
            raccy_i.store_unaligned(&accelerationsy[i]);
            raccz_i.store_unaligned(&accelerationsz[i]);

        }
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
