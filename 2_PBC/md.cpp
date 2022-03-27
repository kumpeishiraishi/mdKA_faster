#include <algorithm>
#include <chrono>
#include <iterator>
#include <functional>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <cmath>
#include <vector>
#include <array>
#include <sstream>
#include "cnpy.h"
//---------------------------------------------------------------
constexpr auto deg = 3;
constexpr auto N   = 1000;
constexpr auto rho = 1.204;

constexpr auto dt     = 5e-3;
constexpr auto margin = 0.28;

constexpr auto Ndof  = deg*N;
constexpr auto Ninv  = 1.0/N;
constexpr auto SKIN2 = (margin*0.5) * (margin*0.5);
constexpr auto dt2   = dt*0.5;
constexpr auto dt4   = dt*0.25;

constexpr auto drVLJrc = 3.89994774528e-02;
constexpr auto VLJrc   = -1.6316891136e-02;
constexpr auto N_A     = N*4/5;

double conf[N][deg], velo[N][deg], force[N][deg], NL_config[N][deg];
std::vector<int> point(N), list(N*50);

const std::array<double, 2> L = {std::pow(N/rho, 1.0/deg), 1.0/std::pow(N/rho, 1.0/deg)};
std::array<double, deg> vij;

enum {X, Y, Z};
//---------------------------------------------------------------
void init_lattice() {
    const auto ln   = std::ceil(std::pow(N, 1.0/deg));
    const auto haba = L[0]/ln;
    const auto lnz  = std::ceil(N/(ln*ln));
    const auto zaba = L[0]/lnz;

    int iz,iy,ix;
    for (int i=0; i<N; i++) {
        iz = std::floor(i/(ln*ln));
        iy = std::floor((i - iz*ln*ln)/ln);
        ix = i - iz*ln*ln - iy*ln;

        conf[i][X] = haba*0.5 + haba * ix;
        conf[i][Y] = haba*0.5 + haba * iy;
        conf[i][Z] = zaba*0.5 + zaba * iz;

        for (int d=0; d<deg; d++) {
            conf[i][d] -= L[0] * std::round(conf[i][d] * L[1]);
        }
    }
}
void init_vel_MB(const double T_targ, std::mt19937 &mt) {
    if (std::abs(T_targ) < 1e-16) {
        std::fill(*velo, *velo+Ndof, 0.0);
        return;
    }
    std::normal_distribution<double> dist_trans(0.0, std::sqrt(T_targ));
    for (int i=0; i<N; i++) {
        velo[i][X] = dist_trans(mt);
        velo[i][Y] = dist_trans(mt);
        velo[i][Z] = dist_trans(mt);
    }

    // remove drift
    double vel1 = 0.0, vel2 = 0.0, vel3 = 0.0;
    for (int i=0; i<N; i++) {
        vel1 += velo[i][X];
        vel2 += velo[i][Y];
        vel3 += velo[i][Z];
    }
    vel1 /= N;
    vel2 /= N;
    vel3 /= N;
    for (int i=0; i<N; i++) {
        velo[i][0] -= vel1;
        velo[i][1] -= vel2;
        velo[i][2] -= vel3;
    }
}
//---------------------------------------------------------------
inline double KABLJ_energy(const int ki, const int kj) {
    switch (ki) {
    case 0:
        switch (kj) {
        case 0:
            return 1.0;
        case 1:
            return 1.5;
        }
    case 1:
        switch (kj) {
        case 0:
            return 1.5;
        case 1:
            return 0.5;
        }
    }
    return 0.0;
}
inline double KABLJ_sij(const int ki, const int kj) {
    switch (ki) {
    case 0:
        switch (kj) {
        case 0:
            return 1.0;
        case 1:
            return 0.8;
        }
    case 1:
        switch (kj) {
        case 0:
            return 0.8;
        case 1:
            return 0.88;
        }
    }
    return 0.0;
}
inline double KABLJ_sij2(const int ki, const int kj) {
    switch (ki) {
    case 0:
        switch (kj) {
        case 0:
            return 1.0;
        case 1:
            return 0.64;
        }
    case 1:
        switch (kj) {
        case 0:
            return 0.64;
        case 1:
            return 0.7744;
        }
    }
    return 0.0;
}
//---------------------------------------------------------------
constexpr auto Nend = N-1;
void generate_NL() {
    double rij2, sij2, rlist2, xij, yij, zij;
    int ki, kj;

    auto nlist = -1;
    for (int i=0; i<Nend; i++) {
        point[i] = nlist+1;

        ki = i>=N_A;
        for (int j=i+1; j<N; j++) {
            kj = j>=N_A;

            xij = conf[i][X] - conf[j][X];
            yij = conf[i][Y] - conf[j][Y];
            zij = conf[i][Z] - conf[j][Z];
            xij -= L[0] * floor(xij * L[1] + 0.5);
            yij -= L[0] * floor(yij * L[1] + 0.5);
            zij -= L[0] * floor(zij * L[1] + 0.5);

            rij2   = (xij*xij + yij*yij + zij*zij);
            sij2   = KABLJ_sij2(ki, kj);
            rlist2 = 6.25*sij2 + 5.0*margin * KABLJ_sij(ki, kj) + margin*margin;
            if (rij2 < rlist2) {
                nlist++;
                list[nlist] = j;
            }
        }
    }
    point[Nend] = nlist+1;
}
void calc_force() {
    int j, pend, si, sj;
    double rij2, sij2, temp, xij, yij, zij, rijsij2, rijsij6;
    std::fill(*force, *force+Ndof, 0.0);

    for (int i=0; i<Nend; i++) {
        pend = point[i+1];
        if (pend == point[i]) continue;

        si = i>=N_A;
        for (int p=point[i]; p<pend; p++) {
            j = list[p];
            sj = j>=N_A;

            xij = conf[i][X] - conf[j][X];
            yij = conf[i][Y] - conf[j][Y];
            zij = conf[i][Z] - conf[j][Z];
            xij -= L[0] * floor(xij * L[1] + 0.5);
            yij -= L[0] * floor(yij * L[1] + 0.5);
            zij -= L[0] * floor(zij * L[1] + 0.5);

            rij2 = (xij*xij + yij*yij + zij*zij);
            sij2 = KABLJ_sij2(si, sj);
            if (rij2 < 6.25*sij2) {
                rijsij2 = rij2/sij2;
                rijsij6 = rijsij2 * rijsij2 * rijsij2;

                temp = -24.0 * (2.0 - rijsij6) / (rijsij6 * rijsij6 * rij2) - drVLJrc/std::sqrt(rij2*sij2);
                temp *= KABLJ_energy(si, sj);

                force[i][X] -= temp * xij;
                force[i][Y] -= temp * yij;
                force[i][Z] -= temp * zij;
                force[j][X] += temp * xij;
                force[j][Y] += temp * yij;
                force[j][Z] += temp * zij;
            }
        }
    }
}
//---------------------------------------------------------------
inline void velocity_update() {
    for (int i=0; i<N; i++) {
        velo[i][X] += dt2*force[i][X];
        velo[i][Y] += dt2*force[i][Y];
        velo[i][Z] += dt2*force[i][Z];
    }
}
inline void position_update() {
    for (int i=0; i<N; i++) {
        conf[i][X] += dt*velo[i][X];
        conf[i][Y] += dt*velo[i][Y];
        conf[i][Z] += dt*velo[i][Z];
    }
}
inline void PBC() {
    for (int i=0; i<N; i++) {
        conf[i][X] -= L[0] * floor(conf[i][X] * L[1] + 0.5);
        conf[i][Y] -= L[0] * floor(conf[i][Y] * L[1] + 0.5);
        conf[i][Z] -= L[0] * floor(conf[i][Z] * L[1] + 0.5);
    }  
}
inline void NL_check() {
    double dev_max = 0.0;
    for (int i=0; i<N; ++i) {
        for (int d=0; d<deg; ++d) {
            vij[d]  = conf[i][d] - NL_config[i][d];
            vij[d] -= L[0] * floor(vij[d] * L[1] + 0.5);
        }
        dev_max = std::max(dev_max, std::inner_product(vij.begin(), vij.end(), vij.begin(), 0.0));
    }
    if (dev_max > SKIN2) {// renew neighbor list
        generate_NL();
        std::copy(*conf, *conf+Ndof, *NL_config);
    }
}
//---------------------------------------------------------------
void NVT(const double T_targ, const int steps) {
    calc_force();
    // Nose-Hoover variables
    double temp, uk, vxi1 = 0.0;
    const auto gkBT = Ndof*T_targ;

    auto t = 0;
    while (t < steps) {
        // Nose-Hoover chain (QMASS = 1.0)
        uk    = std::inner_product(*velo, *velo+Ndof, *velo, 0.0);
        vxi1 += dt4 * (uk - gkBT);
        temp  = std::exp(-vxi1 * dt2);
        std::transform(*velo, *velo+Ndof, *velo,
                       std::bind(std::multiplies<double>(), std::placeholders::_1, temp));
        vxi1 += dt4 * (uk*temp*temp - gkBT);

        velocity_update();
        position_update();
        PBC();
        NL_check();
        calc_force();
        velocity_update();

        // Nose-Hoover chain (QMASS = 1.0)
        uk    = std::inner_product(*velo, *velo+Ndof, *velo, 0.0);
        vxi1 += dt4 * (uk - gkBT);
        temp  = std::exp(-vxi1 * dt2);
        std::transform(*velo, *velo+Ndof, *velo,
                       std::bind(std::multiplies<double>(), std::placeholders::_1, temp));
        vxi1 += dt4 * (uk*temp*temp - gkBT);

        t++;
        // remove drift (use vij as drift)
        if ((t % 100) == 0) {
            vij.fill(0.0);
            for (int i=0; i<N; i++) {
                vij[X] += velo[i][X];
                vij[Y] += velo[i][Y];
                vij[Z] += velo[i][Z];
            }
            for (int d=0; d<deg; d++) {
                vij[d] *= Ninv;
            }
            for (int i=0; i<N; i++) {
                velo[i][X] -= vij[X];
                velo[i][Y] -= vij[Y];
                velo[i][Z] -= vij[Z];
            }
        }
    }
}
//---------------------------------------------------------------
int main() {
    // initialize system
    cnpy::NpyArray arr2 = cnpy::npy_load("start.npy");
    std::vector<double> conf2(arr2.data<double>(), arr2.data<double>() + Ndof);
    std::copy(conf2.begin(), conf2.end(), *conf);

    std::mt19937 mt(123);
    init_vel_MB(1.0, mt);

    // initialize neighbor list
    generate_NL();
    std::copy(*conf, *conf+Ndof, *NL_config);

    auto start2 = std::chrono::system_clock::now();
    NVT(1.0, 1e2/dt);
    auto end2 = std::chrono::system_clock::now();
    auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2-start2).count();
    std::cout << elapsed2 << std::endl;

    for (auto&& x : conf) {
        for (int d=0; d<deg; d++) {
            std::cout << x[d] << std::endl;
        }
    }
}
