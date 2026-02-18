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



    // // OMP  version
    #pragma omp parallel for
    for (int i = 0; i < n_particles; i++)
    {
        for (int j = 0; j < n_particles; j++)
        {
            if(i != j)
            {
                const float diffx = particles.x[j] - particles.x[i];
                const float diffy = particles.y[j] - particles.y[i];
                const float diffz = particles.z[j] - particles.z[i];

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

                accelerationsx[i] += diffx * dij * initstate.masses[j];
                accelerationsy[i] += diffy * dij * initstate.masses[j];
                accelerationsz[i] += diffz * dij * initstate.masses[j];
            }
        }
    }

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

        auto velox_i = rvelox_i + raccx_i * 2.0f;
        auto veloy_i = rveloy_i + raccy_i * 2.0f;
        auto veloz_i = rveloz_i + raccz_i * 2.0f;

        velox_i.store_unaligned(&velocitiesx[i]);
        veloy_i.store_unaligned(&velocitiesy[i]);
        veloz_i.store_unaligned(&velocitiesz[i]);

        auto posx_i = rposx_i + rvelox_i * 0.1f;
        auto posy_i = rposy_i + rveloy_i * 0.1f;
        auto posz_i = rposz_i + rveloz_i * 0.1f;

        posx_i.store_unaligned(&particles.x[i]);
        posy_i.store_unaligned(&particles.y[i]);
        posz_i.store_unaligned(&particles.z[i]);
    }


    // OMP + xsimd version
    // #pragma omp parallel for
    // for (int i = 0; i < n_particles; i += b_type::size)
    // {
    //     // load registers body i
    //     const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
    //     const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
    //     const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
    //     b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
    //     b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
    //     b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

    //     for (int j = 0; j < n_particles; j += b_type::size)
    //     {
    //         if(i != j)
    //         {
    //             // load register body j
    //             const b_type rposx_j = b_type::load_unaligned(&particles.x[i]);
    //             const b_type rposy_j = b_type::load_unaligned(&particles.y[i]);
    //             const b_type rposz_j = b_type::load_unaligned(&particles.z[i]);
    //             b_type rmasse_j = b_type::load_unaligned(&initstate.masses[j]);

    //             auto diffx = rposx_j - rposx_i;
    //             auto diffy = rposy_j - rposy_i;
    //             auto diffz = rposz_j - rposz_i;

    //             auto dij = diffx * diffx + diffy * diffy + diffz * diffz;

    //             if (dij < 1.0)
    //             {
    //                 dij = 10.0;
    //             }
    //             else
    //             {
    //                 dij = std::sqrt(dij);
    //                 dij = 10.0 / (dij * dij * dij);
    //             }

    //             auto mulx_i = diffx * dij * rmasse_j;
    //             auto resx_i = mulx_i + raccx_i;
    //             auto muly_i = diffy * dij * rmasse_j;
    //             auto resy_i = muly_i + raccy_i;
    //             auto mulz_i = diffz * dij * rmasse_j;
    //             auto resz_i = mulz_i + raccz_i;

    //             raccx_i.store_unaligned(resx_i);
    //             raccy_i.store_unaligned(resy_i);
    //             raccz_i.store_unaligned(resz_i);

    //         }   
    //     }

    //     for (int i = 0; i < n_particles; i += b_type::size){

    //         // load velocities i
    //         b_type rvelox_i = b_type::load_unaligned(&velocitiesx[i]);
    //         b_type rveloy_i = b_type::load_unaligned(&velocitiesy[i]);
    //         b_type rveloz_i = b_type::load_unaligned(&velocitiesz[i]);

    //         auto velox_i = rvelox_i + raccx_i * 2.0f;
    //         auto veloy_i = rveloy_i + raccy_i * 2.0f;
    //         auto veloz_i = rveloz_i + raccz_i * 2.0f;

    //         rvelox_i.store_unaligned(velox_i);
    //         rveloy_i.store_unaligned(veloy_i);
    //         rveloz_i.store_unaligned(veloz_i);

    //         auto posx_i = rposx_i + rvelox_i * 0.1f;
    //         auto posy_i = rposy_i + rveloy_i * 0.1f;
    //         auto posz_i = rposz_i + rveloz_i * 0.1f;

    //         rposx_i.store_unaligned(posx_i);
    //         rposy_i.store_unaligned(posy_i);
    //         rposz_i.store_unaligned(posz_i);
    //     }
    // }

}

#endif // GALAX_MODEL_CPU_FAST
