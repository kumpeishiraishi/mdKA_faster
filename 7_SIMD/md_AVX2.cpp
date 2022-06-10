#include <algorithm>
#include <chrono>
#include <immintrin.h>
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
#include <x86intrin.h>
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

double __attribute__((aligned(32))) conf[N][4], velo[N][4], force[N][4], NL_config[N][4];
double conf_save[N][deg], velo_save[N][deg];
std::vector<int> point(N), list(N*50);
double vxi1 = 0.0;

const double Lbox = std::pow(N/rho, 1.0/deg);
const double Linv = 1.0/Lbox;

const __m256d rcut2 = _mm256_set_pd(6.25, 6.25, 6.25, 6.25);
const __m256d uhalf = _mm256_set_pd(0.5, 0.5, 0.5, 0.5);
const __m256d vzero = _mm256_set_pd(0, 0, 0, 0);
const __m256d vtwo  = _mm256_set_pd(2.0, 2.0, 2.0, 2.0);
const __m256d vTF   = _mm256_set_pd(24.0, 24.0, 24.0, 24.0);
const __m256d vct   = _mm256_set_pd(drVLJrc, drVLJrc, drVLJrc, drVLJrc);
const __m256d vLinv = _mm256_set_pd(Linv, Linv, Linv, Linv);
const __m256d vLbox = _mm256_set_pd(Lbox, Lbox, Lbox, Lbox);

enum {X, Y, Z};
//---------------------------------------------------------------
void copy_to_save() {
    for (int i=0; i<N; i++) {
        conf_save[i][X] = conf[i][X];
        conf_save[i][Y] = conf[i][Y];
        conf_save[i][Z] = conf[i][Z];
    }
    for (int i=0; i<N; i++) {
        velo_save[i][X] = velo[i][X];
        velo_save[i][Y] = velo[i][Y];
        velo_save[i][Z] = velo[i][Z];
    }
}
//---------------------------------------------------------------
void init_lattice() {
    const auto ln   = std::ceil(std::pow(N, 1.0/deg));
    const auto haba = Lbox/ln;
    const auto lnz  = std::ceil(N/(ln*ln));
    const auto zaba = Lbox/lnz;

    int iz, iy, ix;
    for (int i=0; i<N; i++) {
        iz = std::floor(i/(ln*ln));
        iy = std::floor((i - iz*ln*ln)/ln);
        ix = i - iz*ln*ln - iy*ln;

        conf[i][X] = haba*0.5 + haba * ix;
        conf[i][Y] = haba*0.5 + haba * iy;
        conf[i][Z] = zaba*0.5 + zaba * iz;

        for (int d=0; d<deg; d++) {
            conf[i][d] -= Lbox * std::round(conf[i][d] * Linv);
        }
    }
}
inline void remove_drift() {
    double vel1 = 0.0, vel2 = 0.0, vel3 = 0.0;
    for (int i=0; i<N; i++) {
        vel1 += velo[i][X];
        vel2 += velo[i][Y];
        vel3 += velo[i][Z];
    }
    vel1 *= Ninv;
    vel2 *= Ninv;
    vel3 *= Ninv;
    for (int i=0; i<N; i++) {
        velo[i][X] -= vel1;
        velo[i][Y] -= vel2;
        velo[i][Z] -= vel3;
    }
}
void init_vel_MB(const double T_targ, std::mt19937 &mt) {
    std::normal_distribution<double> dist_trans(0.0, std::sqrt(T_targ));
    for (int i=0; i<N; i++) {
        velo[i][X] = dist_trans(mt);
        velo[i][Y] = dist_trans(mt);
        velo[i][Z] = dist_trans(mt);
    }
    remove_drift();
}
//---------------------------------------------------------------
inline double KABLJ_energy(const int &ki, const int &kj) {
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
inline double KABLJ_sij(const int &ki, const int &kj) {
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
inline double KABLJ_sij2(const int &ki, const int &kj) {
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
    auto nlist = -1;
    for (int i=0; i<Nend; i++) {
        point[i] = nlist+1;

        int ki = i>=N_A;
        double xi = conf[i][X];
        double yi = conf[i][Y];
        double zi = conf[i][Z];

        // initial
        int kj = (i+1)>=N_A;

        double xij = xi - conf[i+1][X];
        double yij = yi - conf[i+1][Y];
        double zij = zi - conf[i+1][Z];
        xij -= Lbox * floor(xij * Linv + 0.5);
        yij -= Lbox * floor(yij * Linv + 0.5);
        zij -= Lbox * floor(zij * Linv + 0.5);

        for (int j=i+2; j<N; j++) {
            double dx2 = xi - conf[j][X];
            double dy2 = yi - conf[j][Y];
            double dz2 = zi - conf[j][Z];
            dx2 -= Lbox * floor(dx2 * Linv + 0.5);
            dy2 -= Lbox * floor(dy2 * Linv + 0.5);
            dz2 -= Lbox * floor(dz2 * Linv + 0.5);

            //----------------------------------------

            double rij2   = (xij*xij + yij*yij + zij*zij);
            double sij2   = KABLJ_sij2(ki, kj);
            double rlist2 = 6.25*sij2 + 5.0*margin * KABLJ_sij(ki, kj) + margin*margin;
            if (rij2 < rlist2) {
                nlist++;
                list[nlist] = j-1;
            }

            //----------------------------------------

            kj = j>=N_A;
            xij = dx2;
            yij = dy2;
            zij = dz2;
        }

        // final
        double rij2   = (xij*xij + yij*yij + zij*zij);
        double sij2   = KABLJ_sij2(ki, kj);
        double rlist2 = 6.25*sij2 + 5.0*margin * KABLJ_sij(ki, kj) + margin*margin;
        if (rij2 < rlist2) {
            nlist++;
            list[nlist] = N-1;
        }
    }
    point[Nend] = nlist+1;
    std::copy(*conf, *conf+Ndof+N, *NL_config);
}
void calc_force() {
    std::fill(*force, *force+Ndof+N, 0.0);
    for (int i=0; i<Nend; i++) {
        const int pstart = point[i];
        const int pend = point[i+1];
        if (pstart == pend) continue;
        const int pend_tmp = pstart+(pend-pstart)/4*4;

        const int si = i>=N_A;
        const __m256d vqi = _mm256_load_pd((double *)(conf + i));
        __m256d vfi = _mm256_load_pd((double *)(force + i));

        // initial
        int ja0 = list[pstart];
        int ja1 = list[pstart+1];
        int ja2 = list[pstart+2];
        int ja3 = list[pstart+3];
        int sj0 = ja0>=N_A;
        int sj1 = ja1>=N_A;
        int sj2 = ja2>=N_A;
        int sj3 = ja3>=N_A;

        __m256d vqja0 = _mm256_load_pd((double *)(conf + ja0));
        __m256d vqja1 = _mm256_load_pd((double *)(conf + ja1));
        __m256d vqja2 = _mm256_load_pd((double *)(conf + ja2));
        __m256d vqja3 = _mm256_load_pd((double *)(conf + ja3));
        __m256d vdra0 = vqi - vqja0;
        __m256d vdra1 = vqi - vqja1;
        __m256d vdra2 = vqi - vqja2;
        __m256d vdra3 = vqi - vqja3;
        vdra0 -= vLbox * _mm256_floor_pd(vdra0 * vLinv + uhalf);
        vdra1 -= vLbox * _mm256_floor_pd(vdra1 * vLinv + uhalf);
        vdra2 -= vLbox * _mm256_floor_pd(vdra2 * vLinv + uhalf);
        vdra3 -= vLbox * _mm256_floor_pd(vdra3 * vLinv + uhalf);

        __m256d tmp0 = _mm256_unpacklo_pd(vdra0, vdra1);
        __m256d tmp1 = _mm256_unpackhi_pd(vdra0, vdra1);
        __m256d tmp2 = _mm256_unpacklo_pd(vdra2, vdra3);
        __m256d tmp3 = _mm256_unpackhi_pd(vdra2, vdra3);
        __m256d vdx = _mm256_permute2f128_pd(tmp0, tmp2, 2*16+1*0);
        __m256d vdy = _mm256_permute2f128_pd(tmp1, tmp3, 2*16+1*0);
        __m256d vdz = _mm256_permute2f128_pd(tmp0, tmp2, 3*16+1*1);
        __m256d vr2 = vdx * vdx + vdy * vdy + vdz * vdz;

        double sij2_0 = KABLJ_sij2(si, sj0);
        double sij2_1 = KABLJ_sij2(si, sj1);
        double sij2_2 = KABLJ_sij2(si, sj2);
        double sij2_3 = KABLJ_sij2(si, sj3);
        __m256d vs2 = _mm256_set_pd(sij2_3, sij2_2, sij2_1, sij2_0);

        double eij_0 = KABLJ_energy(si, sj0);
        double eij_1 = KABLJ_energy(si, sj1);
        double eij_2 = KABLJ_energy(si, sj2);
        double eij_3 = KABLJ_energy(si, sj3);
        __m256d eij = _mm256_set_pd(eij_3, eij_2, eij_1, eij_0);

        __m256d vrs2 = vr2/vs2;
        __m256d vrs6 = vrs2 * vrs2 * vrs2;
        __m256d vrs12 = vrs6 * vrs6;
        __m256d vrs1 = _mm256_sqrt_pd(vrs2);
        __m256d df = (-vTF * (vtwo - vrs6) - vct * vrs12 * vrs1) / (vrs12 * vr2);
        df *= eij;
        __m256d mask = rcut2*vs2 - vr2;
        df = _mm256_blendv_pd(df, vzero, mask);
        if (pend-pstart < 4) df = _mm256_setzero_pd();

        for (int p=pstart+4; p<pend_tmp; p+=4) {
            int jb0 = list[p];
            int jb1 = list[p+1];
            int jb2 = list[p+2];
            int jb3 = list[p+3];
            sj0 = jb0>=N_A;
            sj1 = jb1>=N_A;
            sj2 = jb2>=N_A;
            sj3 = jb3>=N_A;
            __m256d vqjb0 = _mm256_load_pd((double *)(conf + jb0));
            __m256d vqjb1 = _mm256_load_pd((double *)(conf + jb1));
            __m256d vqjb2 = _mm256_load_pd((double *)(conf + jb2));
            __m256d vqjb3 = _mm256_load_pd((double *)(conf + jb3));
            __m256d vdrb0 = vqi - vqjb0;
            __m256d vdrb1 = vqi - vqjb1;
            __m256d vdrb2 = vqi - vqjb2;
            __m256d vdrb3 = vqi - vqjb3;
            vdrb0 -= vLbox * _mm256_floor_pd(vdrb0 * vLinv + uhalf);
            vdrb1 -= vLbox * _mm256_floor_pd(vdrb1 * vLinv + uhalf);
            vdrb2 -= vLbox * _mm256_floor_pd(vdrb2 * vLinv + uhalf);
            vdrb3 -= vLbox * _mm256_floor_pd(vdrb3 * vLinv + uhalf);

            tmp0 = _mm256_unpacklo_pd(vdrb0, vdrb1);
            tmp1 = _mm256_unpackhi_pd(vdrb0, vdrb1);
            tmp2 = _mm256_unpacklo_pd(vdrb2, vdrb3);
            tmp3 = _mm256_unpackhi_pd(vdrb2, vdrb3);
            vdx = _mm256_permute2f128_pd(tmp0, tmp2, 2*16+1*0);
            vdy = _mm256_permute2f128_pd(tmp1, tmp3, 2*16+1*0);
            vdz = _mm256_permute2f128_pd(tmp0, tmp2, 3*16+1*1);
            vr2 = vdx * vdx + vdy * vdy + vdz * vdz;

            sij2_0 = KABLJ_sij2(si, sj0);
            sij2_1 = KABLJ_sij2(si, sj1);
            sij2_2 = KABLJ_sij2(si, sj2);
            sij2_3 = KABLJ_sij2(si, sj3);
            vs2 = _mm256_set_pd(sij2_3, sij2_2, sij2_1, sij2_0);

            vrs2 = vr2/vs2;
            vrs6 = vrs2 * vrs2 * vrs2;
            vrs12 = vrs6 * vrs6;
            vrs1 = _mm256_sqrt_pd(vrs2);

            eij_0 = KABLJ_energy(si, sj0);
            eij_1 = KABLJ_energy(si, sj1);
            eij_2 = KABLJ_energy(si, sj2);
            eij_3 = KABLJ_energy(si, sj3);
            eij = _mm256_set_pd(eij_3, eij_2, eij_1, eij_0);

            //----------------------------------------

            __m256d vdf_0 = _mm256_permute4x64_pd(df, 0);
            __m256d vdf_1 = _mm256_permute4x64_pd(df, 85);
            __m256d vdf_2 = _mm256_permute4x64_pd(df, 170);
            __m256d vdf_3 = _mm256_permute4x64_pd(df, 255);
            __m256d vpj_0 = _mm256_load_pd((double *)(force + ja0));
            __m256d vpj_1 = _mm256_load_pd((double *)(force + ja1));
            __m256d vpj_2 = _mm256_load_pd((double *)(force + ja2));
            __m256d vpj_3 = _mm256_load_pd((double *)(force + ja3));

            vfi -= vdf_0 * vdra0;
            vfi -= vdf_1 * vdra1;
            vfi -= vdf_2 * vdra2;
            vfi -= vdf_3 * vdra3;
            vpj_0 += vdf_0 * vdra0;
            vpj_1 += vdf_1 * vdra1;
            vpj_2 += vdf_2 * vdra2;
            vpj_3 += vdf_3 * vdra3;

            _mm256_store_pd((double *)(force + ja0), vpj_0);
            _mm256_store_pd((double *)(force + ja1), vpj_1);
            _mm256_store_pd((double *)(force + ja2), vpj_2);
            _mm256_store_pd((double *)(force + ja3), vpj_3);

            //----------------------------------------

            ja0 = jb0;
            ja1 = jb1;
            ja2 = jb2;
            ja3 = jb3;
            vdra0 = vdrb0;
            vdra1 = vdrb1;
            vdra2 = vdrb2;
            vdra3 = vdrb3;
            df = (-vTF * (vtwo - vrs6) - vct * vrs12 * vrs1) / (vrs12 * vr2);
            df *= eij;
            mask = rcut2*vs2 - vr2;
            df = _mm256_blendv_pd(df, vzero, mask);
        }

        // final
        __m256d vdf_0 = _mm256_permute4x64_pd(df, 0);
        __m256d vdf_1 = _mm256_permute4x64_pd(df, 85);
        __m256d vdf_2 = _mm256_permute4x64_pd(df, 170);
        __m256d vdf_3 = _mm256_permute4x64_pd(df, 255);
        __m256d vpj_0 = _mm256_load_pd((double *)(force + ja0));
        __m256d vpj_1 = _mm256_load_pd((double *)(force + ja1));
        __m256d vpj_2 = _mm256_load_pd((double *)(force + ja2));
        __m256d vpj_3 = _mm256_load_pd((double *)(force + ja3));

        vfi -= vdf_0 * vdra0;
        vfi -= vdf_1 * vdra1;
        vfi -= vdf_2 * vdra2;
        vfi -= vdf_3 * vdra3;
        vpj_0 += vdf_0 * vdra0;
        vpj_1 += vdf_1 * vdra1;
        vpj_2 += vdf_2 * vdra2;
        vpj_3 += vdf_3 * vdra3;

        _mm256_store_pd((double *)(force + ja0), vpj_0);
        _mm256_store_pd((double *)(force + ja1), vpj_1);
        _mm256_store_pd((double *)(force + ja2), vpj_2);
        _mm256_store_pd((double *)(force + ja3), vpj_3);

        //------------------------------

        _mm256_store_pd((double *)(force + i), vfi);
        double fix = force[i][X];
        double fiy = force[i][Y];
        double fiz = force[i][Z];

        // gomi
        for (int p=pend_tmp; p<pend; p++) {
            int j = list[p];
            int sj = j>=N_A;
            double dx = conf[i][X] - conf[j][X];
            double dy = conf[i][Y] - conf[j][Y];
            double dz = conf[i][Z] - conf[j][Z];
            dx -= Lbox * floor(dx * Linv + 0.5);
            dy -= Lbox * floor(dy * Linv + 0.5);
            dz -= Lbox * floor(dz * Linv + 0.5);

            double rij2 = (dx*dx + dy*dy + dz*dz);
            double sij2 = KABLJ_sij2(si, sj);
            if (rij2 < 6.25*sij2) {
                double rijsij2 = rij2/sij2;
                double rijsij6 = rijsij2 * rijsij2 * rijsij2;
                double rijsij12 = rijsij6 * rijsij6;
                double temp = (-24.0 * (2.0 - rijsij6) - drVLJrc * rijsij12 * std::sqrt(rijsij2)) / (rijsij12 * rij2);
                temp *= KABLJ_energy(si, sj);
                fix -= temp * dx;
                fiy -= temp * dy;
                fiz -= temp * dz;
                force[j][X] += temp * dx;
                force[j][Y] += temp * dy;
                force[j][Z] += temp * dz;
            }
        }
        vqja0 = _mm256_set_pd(0.0, fiz, fiy, fix);
        _mm256_store_pd((double *)(force + i), vqja0);
    }
}
//---------------------------------------------------------------
double calc_potential() {
    double ans = 0.0;
    for (int i=0; i<Nend; i++) {
        int si = i>=N_A;
        int pend = point[i+1];
        double xi = conf[i][X];
        double yi = conf[i][Y];
        double zi = conf[i][Z];
        for (int p=point[i]; p<pend; p++) {
            int j = list[p];
            int sj = j>=N_A;

            double dx = xi - conf[j][X];
            double dy = yi - conf[j][Y];
            double dz = zi - conf[j][Z];
            dx -= Lbox * floor(dx * Linv + 0.5);
            dy -= Lbox * floor(dy * Linv + 0.5);
            dz -= Lbox * floor(dz * Linv + 0.5);

            double rij2 = (dx*dx + dy*dy + dz*dz);
            double sij2 = KABLJ_sij2(si, sj);
            if (rij2 < 6.25*sij2) {
                double rijsij2 = rij2/sij2;
                double rijsij6 = rijsij2 * rijsij2 * rijsij2;
                double temp = 4.0 * (1.0 - rijsij6)/(rijsij6*rijsij6) - VLJrc - drVLJrc*(std::sqrt(rijsij2) - 2.5);
                ans += KABLJ_energy(si, sj)*temp;
            }
        }
    }
    return ans;
}
double calc_potential_N2() {
    double ans = 0.0;
    for (int i=0; i<N; i++) {
        int si = i>=N_A;
        double xi = conf[i][X];
        double yi = conf[i][Y];
        double zi = conf[i][Z];
        for (int j=i+1; j<N; j++) {
            int sj = j>=N_A;

            double dx = xi - conf[j][X];
            double dy = yi - conf[j][Y];
            double dz = zi - conf[j][Z];
            dx -= Lbox * floor(dx * Linv + 0.5);
            dy -= Lbox * floor(dy * Linv + 0.5);
            dz -= Lbox * floor(dz * Linv + 0.5);

            double rij2 = (dx*dx + dy*dy + dz*dz);
            double sij2 = KABLJ_sij2(si, sj);
            if (rij2 < 6.25*sij2) {
                double rijsij2 = rij2/sij2;
                double rijsij6 = rijsij2 * rijsij2 * rijsij2;
                double temp = 4.0 * (1.0 - rijsij6)/(rijsij6*rijsij6) - VLJrc - drVLJrc*(std::sqrt(rijsij2) - 2.5);
                ans += KABLJ_energy(si, sj)*temp;
            }
        }
    }
    return ans;
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
        conf[i][X] -= Lbox * floor(conf[i][X] * Linv + 0.5);
        conf[i][Y] -= Lbox * floor(conf[i][Y] * Linv + 0.5);
        conf[i][Z] -= Lbox * floor(conf[i][Z] * Linv + 0.5);
    }
}
inline void NL_check() {
    double dev_max = 0.0;
    for (int i=0; i<N; i++) {
        double xij = conf[i][X] - NL_config[i][X];
        double yij = conf[i][Y] - NL_config[i][Y];
        double zij = conf[i][Z] - NL_config[i][Z];
        xij -= Lbox * floor(xij * Linv + 0.5);
        yij -= Lbox * floor(yij * Linv + 0.5);
        zij -= Lbox * floor(zij * Linv + 0.5);
        dev_max = std::max(dev_max, (xij*xij + yij*yij + zij*zij));
    }
    if (dev_max > SKIN2) {// renew neighbor list
        generate_NL();
    }
}
//---------------------------------------------------------------
void NVE(const int steps, const std::string& name) {
    calc_force();
    const auto logbin = std::pow(10.0, 1.0/9);
    std::stringstream ss;
    ss << 0;
    copy_to_save();
    cnpy::npz_save(name, "position_"+ss.str(), *conf_save, {N, deg}, "a");
    cnpy::npz_save(name, "velocity_"+ss.str(), *velo_save, {N, deg}, "a");

    int t = 0, counter = 5;
    auto checker = 1e-3 * std::pow(logbin, counter);
    while (t < steps) {
        velocity_update();
        position_update();
        PBC();
        NL_check();
        calc_force();
        velocity_update();

        t++;
        if (dt*t > checker) {
            checker *= logbin;
            ss.str(""); ss.clear(std::stringstream::goodbit);
            ss << t;
            copy_to_save();
            cnpy::npz_save(name, "position_"+ss.str(), *conf_save, {N, deg}, "a");
            cnpy::npz_save(name, "velocity_"+ss.str(), *velo_save, {N, deg}, "a");
        }
    }
}
//---------------------------------------------------------------
void NVT(const double T_targ, const int steps) {
    calc_force();
    // Nose-Hoover variables
    const auto gkBT = Ndof*T_targ;

    auto t = 0;
    while (t < steps) {
        // Nose-Hoover chain (QMASS = 1.0)
        double uk = std::inner_product(*velo, *velo+Ndof+N, *velo, 0.0);
        vxi1 += dt4 * (uk - gkBT);
        double temp = std::exp(-vxi1 * dt2);
        std::transform(*velo, *velo+Ndof+N, *velo,
                       std::bind(std::multiplies<double>(), std::placeholders::_1, temp));
        vxi1 += dt4 * (uk*temp*temp - gkBT);

        velocity_update();
        position_update();
        PBC();
        NL_check();
        calc_force();
        velocity_update();

        // Nose-Hoover chain (QMASS = 1.0)
        uk = std::inner_product(*velo, *velo+Ndof+N, *velo, 0.0);
        vxi1 += dt4 * (uk - gkBT);
        temp = std::exp(-vxi1 * dt2);
        std::transform(*velo, *velo+Ndof+N, *velo,
                       std::bind(std::multiplies<double>(), std::placeholders::_1, temp));
        vxi1 += dt4 * (uk*temp*temp - gkBT);

        t++;
        if (!(t & 127)) {
            remove_drift();
        }
    }
}
//---------------------------------------------------------------
void FIRE2(const double fmax_tol) {
    calc_force();

    ////// FIRE variables //////
    constexpr auto alpha_fire_0 = 0.25;
    constexpr auto finc         = 1.1;
    constexpr auto fdec         = 0.5;
    constexpr auto falpha       = 0.99;
    constexpr auto Npneg_max    = 2000;
    constexpr auto Ndelay       = 20;
    constexpr auto initialdelay = true;

    auto Nppos = 0;
    auto Npneg = 0;
    auto dt    = 4e-3;
    auto alpha_fire = alpha_fire_0;
    const auto dtmax = dt*10;
    const auto dtmin = dt*0.02;

    const auto fmax_tol2 = fmax_tol*fmax_tol;
    //////////////////////////////

    std::fill(*velo, *velo+Ndof+N, 0.0);
    auto t = 0;
    while (1) {
        /////// FIRE ///////
        double P = std::inner_product(*velo, *velo+Ndof+N, *force, 0.0);
        if (P > 0) {
            Nppos += 1;
            Npneg  = 0;
            if (Nppos > Ndelay) {
                dt          = std::min(dt*finc, dtmax);
                alpha_fire *= falpha;
            }
        } else if (P <= 0) {
            Nppos = 0;
            Npneg += 1;

            if (Npneg > Npneg_max) {
                break;
            }

            if (!(initialdelay and t<Ndelay)) {
                if (dt*fdec >= dtmin) {
                    dt *= fdec;
                }
                alpha_fire = alpha_fire_0;
            }
            for (int i=0; i<N; i++) {
                conf[i][X] -= 0.5*dt*velo[i][X];
                conf[i][Y] -= 0.5*dt*velo[i][Y];
                conf[i][Z] -= 0.5*dt*velo[i][Z];
            }
            std::fill(*velo, *velo+Ndof+N, 0.0);
        }
        ////// FIRE end //////

        ////// MD //////
        // velocity update
        for (int i=0; i<N; i++) {
            velo[i][X] += dt*0.5*force[i][X];
            velo[i][Y] += dt*0.5*force[i][Y];
            velo[i][Z] += dt*0.5*force[i][Z];
        }

        // mixing
        double Fnorm = std::inner_product(*velo, *velo+Ndof+N, *velo, 0.0) / std::inner_product(*force, *force+Ndof+N, *force, 0.0);
        Fnorm = std::sqrt(Fnorm);
        for (int i=0; i<N; i++) {
            velo[i][X] = (1.0 - alpha_fire) * velo[i][X] + alpha_fire*Fnorm * force[i][X];
            velo[i][Y] = (1.0 - alpha_fire) * velo[i][Y] + alpha_fire*Fnorm * force[i][Y];
            velo[i][Z] = (1.0 - alpha_fire) * velo[i][Z] + alpha_fire*Fnorm * force[i][Z];
        }

        // position update
        for (int i=0; i<N; i++) {
            conf[i][X] += dt*velo[i][X];
            conf[i][Y] += dt*velo[i][Y];
            conf[i][Z] += dt*velo[i][Z];
        }

        PBC();
        NL_check();
        calc_force();

        // velocity update
        for (int i=0; i<N; i++) {
            velo[i][X] += dt*0.5*force[i][X];
            velo[i][Y] += dt*0.5*force[i][Y];
            velo[i][Z] += dt*0.5*force[i][Z];
        }
        ////// MD end //////

        ////// converge //////
        double fmax = 0.0;
        for (int i=0; i<N; i++) {
            double temp = std::inner_product(*force+deg*i, *force+deg*i+deg, *force+deg*i, 0.0);
            if (fmax < temp) fmax = temp;
        }
        if (fmax < fmax_tol2) {
            return;
        }
        ////// converge end //////
        t++;
    }
}
//---------------------------------------------------------------
int main() {
    // initialize system
    std::fill(*velo, *velo+Ndof+N, 0.0);
    std::fill(*force, *force+Ndof+N, 0.0);
    init_lattice();
    std::mt19937 mt(123);
    init_vel_MB(1.0, mt);
    generate_NL();

    // execute MD
    NVT(1.0, 2e2/dt);
    NVT(0.47, 1e4/dt);
    NVT(0.43, 1e6/dt);

    // save file
    copy_to_save();
    cnpy::npy_save("config.npy", *conf_save, {N, deg}, "a");
}
