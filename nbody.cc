/* 
Ian A. Cosden (icosden@princeton.edu)
Bei Wang (beiwang@princeton.edu)

Modified with Permission from Colfax International for the sole 
use of the attendees of the PICSciE mini-course "Introduction to 
Performance Tuning and Optimization" 
*/

/* Copyright (c) 2013-2015, Colfax International. All Right Reserved.
This file, labs/4/4.01-overview-nbody/nbody.cc,
is a part of Supplementary Code for Practical Exercises for the handbook
"Parallel Programming and Optimization with Intel Xeon Phi Coprocessors",
2nd Edition -- 2015, Colfax International,
ISBN 9780988523401 (paper), 9780988523425 (PDF), 9780988523432 (Kindle).
Redistribution or commercial usage without written permission
from Colfax International is prohibited.
Contact information can be found at http://colfax-intl.com/    */

#include <cmath>
#include <cstdio>
#include <omp.h>
#include <stdlib.h>

#ifdef SoA
struct ParticleArrays { 
  float *x, *y, *z;
  float *vx, *vy, *vz; 
};
#else
struct Particle { 
  float x, y, z;
  float vx, vy, vz; 
};
#endif

#ifdef SoA
void MoveParticles(const int nParticles, ParticleArrays &particle, const float dt) {
#else
void MoveParticles(const int nParticles, Particle* const particle, const float dt) {
#endif
  // Loop over particles that experience force
#ifdef SoA
  float *xp = particle.x;
  float *yp = particle.y;
  float *zp = particle.z;
#endif
  for (int i = 0; i < nParticles; i++) { 

    // Components of the gravity force on particle i
    float Fx = 0, Fy = 0, Fz = 0; 
#ifdef SoA
    float xi = xp[i];
    float yi = yp[i];
    float zi = zp[i];
#else
    float xi = particle[i].x;
    float yi = particle[i].y;
    float zi = particle[i].z;
#endif
      
    // Loop over particles that exert force: vectorization expected here
#ifdef OMP_SIMD
#ifdef Aligned
#pragma omp simd aligned(xp, yp, zp: 64) reduction(+:Fx,Fy,Fz) 
#else
#pragma omp simd reduction(+:Fx,Fy,Fz)
#endif
#endif
    for (int j = 0; j < nParticles; j++) { 
      
      // Avoid singularity and interaction with self
      const float softening = 1e-20;

      // Newton's law of universal gravity
#ifdef SoA
      const float dx = xp[j] - xi;
      const float dy = yp[j] - yi;
      const float dz = zp[j] - zi;
#else
      const float dx = particle[j].x - xi;
      const float dy = particle[j].y - yi;
      const float dz = particle[j].z - zi;
#endif
      const float drSquared  = dx*dx + dy*dy + dz*dz + softening;
#ifdef No_FP_Conv
      const float drPower32  = powf(drSquared, 3.0/2.0);
#else
      const float drPower32  = pow(drSquared, 3.0/2.0);
#endif
      const float drPower32Inv = 1.0 / drPower32;
      // Calculate the net force
      Fx += dx * drPower32Inv;  
      Fy += dy * drPower32Inv;  
      Fz += dz * drPower32Inv;

    }

    // Accelerate particles in response to the gravitational force
#ifdef SoA
    particle.vx[i] += dt*Fx; 
    particle.vy[i] += dt*Fy; 
    particle.vz[i] += dt*Fz;
#else
    particle[i].vx += dt*Fx; 
    particle[i].vy += dt*Fy; 
    particle[i].vz += dt*Fz;
#endif
  }

  // Move particles according to their velocities
  // O(N) work, so using a serial loop

  for (int i = 0 ; i < nParticles; i++) { 
#ifdef SoA
    particle.x[i]  += particle.vx[i]*dt;
    particle.y[i]  += particle.vy[i]*dt;
    particle.z[i]  += particle.vz[i]*dt;
#else
    particle[i].x  += particle[i].vx*dt;
    particle[i].y  += particle[i].vy*dt;
    particle[i].z  += particle[i].vz*dt;
#endif
  }
}

int main(const int argc, const char** argv) {

  // Problem size and other parameters
  const int nParticles = (argc > 1 ? atoi(argv[1]) : 16384);
  const int nSteps = 10;  // Duration of test
  const float dt = 0.01f; // Particle propagation time step

#ifdef SoA
  // Particle data stored now stored as Structure of Arrays (SoA)
  // This is makes vectorization more efficient by improving strided access
  ParticleArrays particle;
#ifdef Aligned 
  particle.x  = (float*)_mm_malloc(nParticles*sizeof(float),64);
  particle.y  = (float*)_mm_malloc(nParticles*sizeof(float),64);
  particle.z  = (float*)_mm_malloc(nParticles*sizeof(float),64);
  particle.vx = (float*)_mm_malloc(nParticles*sizeof(float),64);
  particle.vy = (float*)_mm_malloc(nParticles*sizeof(float),64);
  particle.vz = (float*)_mm_malloc(nParticles*sizeof(float),64);
#else
  particle.x  = new float[nParticles];
  particle.y  = new float[nParticles];
  particle.z  = new float[nParticles];
  particle.vx = new float[nParticles];
  particle.vy = new float[nParticles];
  particle.vz = new float[nParticles];
#endif

  // Initialize random number generator and particles
  for (int i=0; i<nParticles; i++) {
    particle.x[i]  = rand();
    particle.y[i]  = rand();
    particle.z[i]  = rand();
    particle.vx[i] = rand();
    particle.vy[i] = rand();
    particle.vz[i] = rand();   
  }
#else
  // Particle data stored as an Array of Structures (AoS)
  // this is good object-oriented programming style,
  // but inefficient for the purposes of vectorization
  Particle* particle = new Particle[nParticles];

  // Initialize random number generator and particles
  for (int i=0; i<nParticles; i++) {
    particle[i].x = rand();
    particle[i].y = rand();
    particle[i].z = rand();
    particle[i].vx = rand();
    particle[i].vy = rand();
    particle[i].vz = rand();   
  }
#endif

  // Perform benchmark
  printf("\nPropagating %d particles using 1 thread on CPU...\n\n", 
	 nParticles);
#ifdef SoA
  printf("\nParticle data layout: Structure of Arrays (SoA)\n");
#else
  printf("\nParticle data layout: Array of Structures (AoS)\n");
#endif

  double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration is warm-up on Xeon Phi coprocessor
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  for (int step = 1; step <= nSteps; step++) {

    const double tStart = omp_get_wtime(); // Start timing
    MoveParticles(nParticles, particle, dt);
    const double tEnd = omp_get_wtime(); // End timing

    const float HztoInts   = float(nParticles)*float(nParticles-1) ;
    const float HztoGFLOPs = 20.0*1e-9*float(nParticles)*float(nParticles-1);

    if (step > skipSteps) { // Collect statistics
      rate  += HztoGFLOPs/(tEnd - tStart); 
      dRate += HztoGFLOPs*HztoGFLOPs/((tEnd - tStart)*(tEnd-tStart)); 
    }

    printf("%5d %10.3e %10.3e %8.1f %s\n", 
	   step, (tEnd-tStart), HztoInts/(tEnd-tStart), HztoGFLOPs/(tEnd-tStart), (step<=skipSteps?"*":""));
    fflush(stdout);
  }
  rate/=(double)(nSteps-skipSteps); 
  dRate=sqrt(dRate/(double)(nSteps-skipSteps)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1f +- %.1f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");
#ifdef SoA
#ifdef Aligned
  _mm_free(particle.x);
  _mm_free(particle.y);
  _mm_free(particle.z);
  _mm_free(particle.vx);
  _mm_free(particle.vy);
  _mm_free(particle.vz);
#else
  delete particle.x;
  delete particle.y;
  delete particle.z;
  delete particle.vx;
  delete particle.vy;
  delete particle.vz;
#endif
#else
  delete particle;
#endif
}


