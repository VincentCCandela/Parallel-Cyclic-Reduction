#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

// #include "cuda_helper.h" // get mac os to ignore things

// changeable params
#define NITER 1000
// gridpoints in s
#ifndef NX
  #define NX 128
#endif
// gridpoints in v
#ifndef NY
  #define NY 128
#endif
#define DT 0.001f
#define RHO -0.5f
#define SIGMA 0.3f
#define R 0.05f
#define LAMBDA 0.0f // keep at zero because integral term has not been implemented
#define XI 0.0f
#define NU 0.04f
#define KAPPA 2.0f

#define K 45.0f

#define SMAX 100.0f
#define VMAX 1.0f
#define SMIN 30.0f
#define VMIN 0.0f

#define PRINT_TIME 1
#define RUN_CPU 0

#if PRINT_TIME
double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}
#endif

double wakeup_delay()
{
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i=1; i<j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}



void rhs_combined_serial(
  int i, int j,
  const float * __restrict__ u_old,
  float * __restrict__ rhs,
  float dt,float ds,float dv,
  float rho,float sigma,float r,float lambda,float xi,float nu,float kappa,
  float k_strike, float tau,
  float s_min,float v_min
){
  // int i = threadIdx.x;
  // int j = blockIdx.x;
  int id = i + j * NX;

  float s = s_min + i * ds;
  float v = v_min + j * dv;
  if(i == 0){
    rhs[id] = 0.0f;
    return;
  }
  else if(i == NX - 1){
    float s_max = s_min + (NX - 1) * ds;
    rhs[id] = fmaxf(s_max - k_strike *expf(-r * tau), 0.0f);
    return;
  }
  else if(j == 0){
    rhs[i] = u_old[i + 1 * NX];
    return;
  }
  else if(j == NY - 1){
    rhs[i + NX * (NY - 1)] = u_old[i + NX * (NY - 2)];
    return;
  }
  float u_ij  = u_old[id];

  float up    = u_old[i + NX * (j+1)];
  float down  = u_old[i + NX * (j-1)];
  float left  = u_old[i-1 + NX * j];
  float right = u_old[i+1 + NX * j];
  float pp    = u_old[i+1 + NX * (j+1)];
  float pm    = u_old[i+1 + NX * (j-1)];
  float mp    = u_old[i-1 + NX * (j+1)];
  float mm    = u_old[i-1 + NX * (j-1)];

  float u_s = (right - left) / (2.0f * ds);
  float u_ss = (right - 2.0f*u_ij + left) / (ds * ds);
  float u_v = (up - down) / (2.0f * dv);
  float u_vv = (up - 2.0f*u_ij + down) / (dv * dv);
  float u_vs = (pp + mm - mp - pm) / (4.0f * ds * dv);

  rhs[id] = u_ij + 0.5f * dt * (
    0.5f * v * s * s * u_ss +
    (r - lambda * xi) * s * u_s + // remove drift correction for now (we don't have the PIDE term)
    0.5f * sigma * sigma * v * u_vv +
    kappa * (nu - v) * u_v +
    rho * sigma * v * s * u_vs -
    (r + lambda) * u_ij
  );
}

void setup_tri_solve_s_serial(
  int i, int j,
  float *a, float *c, float *d,
  const float * __restrict__ rhs, 
  // float * __restrict__ u_new,
  float dt, 
  float ds, 
  float dv,
  // float sigma, 
  float r, 
  float lambda,
  float xi,
  float s_min,
  float v_min
) {
  // int i = threadIdx.x;
  // int j = blockIdx.x;
  // __shared__ float a[NX];
  // __shared__ float c[NX];
  // __shared__ float d[NX];

  float s = s_min + i * ds;
  float v = v_min + j * dv;

  // tridiagonal row
  float alpha = 0.5f * v * s * s / (ds * ds);     // u_ss coef
  float beta  = (r - lambda * xi) * s / (2.0f * ds); // u_s coef // remove drift correction for now

  float aa = -0.5f * dt * (alpha - beta);
  float bb =  1.0f + dt * alpha + 0.25f * dt * (r + lambda);
  float cc = -0.5f * dt * (alpha + beta);
  float dd = rhs[i + NX * j];

  if(i == 0 || i == NX - 1){
    aa = 0.0f;
    bb = 1.0f;
    cc = 0.0f;
  }
  else if(j == 0){
    aa = 0.0f;
    bb = 1.0f;
    cc = 0.0f;
  }
  else if(j == NY - 1){
    aa = 0.0f;
    bb = 1.0f;
    cc = 0.0f;
  }

  // normalize by bb
  float bb_i = 1.0f / bb;
  aa *= bb_i;
  cc *= bb_i;
  dd *= bb_i;

  a[i] = aa;
  c[i] = cc;
  d[i] = dd;
}


void setup_tri_solve_v_serial(
  int i, int j,
  float *a, float *c, float *d,
  const float * __restrict__ rhs, 
  // float * __restrict__ u_new,
  float dt, 
  float ds, 
  float dv,
  float sigma, 
  float r, 
  float lambda,
  // float xi,
  float kappa,
  float nu,
  float s_min,
  float v_min
) {
  // int j = threadIdx.x;
  // int i = blockIdx.x;
  /*
  */
  // __shared__ float a[NY];
  // __shared__ float c[NY];
  // __shared__ float d[NY];
  // float s = s_min + i * ds;
  float v = v_min + j * dv;

  // tridiagonal row
  float alpha = 0.5f * sigma * sigma * v / (dv * dv);// u_vv coef
  float beta  = kappa * (nu - v) / (2.0f * dv); // u_v coef

  float aa = -0.5f * dt * (alpha - beta);
  float bb =  1.0f + dt * alpha + 0.25f*dt*(r + lambda);
  float cc = -0.5f * dt * (alpha + beta);
  float dd = rhs[i + NX * j];

  if(j == 1){
    bb += aa;
    aa = 0.0f;
  }
  else if(j == NY - 2){
    bb += cc; 
    cc = 0.0f;
  }

  if(j == 0 || j == NY - 1){
    aa = 0.0f;
    bb = 1.0f;
    cc = 0.0f;
    dd = 0.0f;
  }

  // normalize by bb
  float bb_i = 1.0f / bb;
  aa *= bb_i;
  cc *= bb_i;
  dd *= bb_i;

  a[j] = aa;
  c[j] = cc;
  d[j] = dd;
}

void step_tri_solve_serial(
  int idx, int step, int grid_param,
  const float * __restrict__ a_old, 
  const float * __restrict__ c_old, 
  const float * __restrict__ d_old,
  float * __restrict__ a, 
  float * __restrict__ c, 
  float * __restrict__ d
){
  float aa = a_old[idx];
  float cc = c_old[idx];
  float dd = d_old[idx];
  float a_left=0;
  float c_left=0;
  float d_left=0;
  float a_right=0;
  float c_right=0;
  float d_right=0;

  if(idx - step >= 0){
    a_left = a_old[idx-step];
    c_left = c_old[idx-step];
    d_left  = d_old[idx-step];
  }
  if(idx + step < grid_param){
    a_right = a_old[idx+step];
    c_right = c_old[idx+step];
    d_right = d_old[idx+step];
  }

  float bb_new = 1.0f - aa * c_left - cc * a_right;
  float inverse = 1.0f / bb_new;
  a[idx] = (-aa * a_left)  * inverse;
  c[idx] = (-cc * c_right) * inverse;
  d[idx] = (dd - aa * d_left - cc * d_right) * inverse;
}

void transfer_d_v_serial(int i, int j,
  const float * __restrict__ d, float * __restrict__ u_new
){
  if(j == 0){
    u_new [i] = d[1];
  }
  else if(j == NY - 1){
    u_new[i + NX * (NY - 1)] = d[NY - 2];
  }
  else{
    u_new[i + NX * j] = d[j];
  }
}

void transfer_d_s_serial(int i, int j,
  const float * __restrict__ d, float * __restrict__ u_new
){
  u_new[i + NX * j] = d[i];
}

__global__ void rhs_combined(
  const float * __restrict__ u_old,
  float * __restrict__ rhs,
  float dt,float ds,float dv,
  float rho,float sigma,float r,float lambda,float xi,float nu,float kappa,
  float k_strike, float tau,
  float s_min,float v_min
){
  int i = threadIdx.x;
  int j = blockIdx.x;
  int id = i + NX * j;

  float s = s_min + i * ds;
  float v = v_min + j * dv;
  if(i == 0){
    rhs[id] = 0.0f;
    return;
  }
  else if(i == NX - 1){
    float s_max = s_min + (NX - 1) * ds;
    rhs[id] = fmaxf(s_max - k_strike *expf(-r * tau), 0.0f);
    return;
  }
  else if(j == 0){
    rhs[i] = u_old[i + NX];
    return;
  }
  else if(j == NY - 1){
    rhs[i + NX * (NY - 1)] = u_old[i + NX * (NY - 2)];
    return;
  }
  float u_ij  = u_old[id];

  float up    = u_old[i + NX * (j+1)];
  float down  = u_old[i + NX * (j-1)];
  float left  = u_old[i-1 + NX * j];
  float right = u_old[i+1 + NX * j];
  float pp    = u_old[i+1 + NX * (j+1)];
  float pm    = u_old[i+1 + NX * (j-1)];
  float mp    = u_old[i-1 + NX * (j+1)];
  float mm    = u_old[i-1 + NX * (j-1)];

  float u_s = (right - left) / (2.0f * ds);
  float u_ss = (right - 2.0f*u_ij + left) / (ds * ds);
  float u_v = (up - down) / (2.0f * dv);
  float u_vv = (up - 2.0f*u_ij + down) / (dv * dv);
  float u_vs = (pp + mm - mp - pm) / (4.0f * ds * dv);

  rhs[id] = u_ij + 0.5f * dt * (
    0.5f * v * s * s * u_ss +
    (r - lambda * xi) * s * u_s + // remove drift correction for now (we don't have the PIDE term)
    0.5f * sigma * sigma * v * u_vv +
    kappa * (nu - v) * u_v +
    rho * sigma * v * s * u_vs -
    (r + lambda) * u_ij
  );
}

__global__ void tri_solve_v(
  const float * __restrict__ rhs, 
  float * __restrict__ u_new,
  float dt, 
  float ds, 
  float dv,
  float sigma, 
  float r, 
  float lambda,
  // float xi,
  float kappa,
  float nu,
  float s_min,
  float v_min
) {
  int j = threadIdx.x;
  int i = blockIdx.x;

  if(i == 0 || i == NX - 1){
    u_new[i + NX * j] = rhs[i + NX * j];
    return;
  }

  __shared__ float a[NY], a1[NY];
  __shared__ float c[NY], c1[NY];
  __shared__ float d[NY], d1[NY];
  float *a_in = a, *a_out = a1;
  float *c_in = c, *c_out = c1;
  float *d_in = d, *d_out = d1;

  // float s = s_min + i * ds;
  float v = v_min + j * dv;

  // tridiagonal row
  float alpha = 0.5f * sigma * sigma * v / (dv * dv);// u_vv coef
  float beta  = kappa * (nu - v) / (2.0f * dv); // u_v coef

  float aa = -0.5f * dt * (alpha - beta);
  float bb =  1.0f + dt * alpha + 0.25f*dt*(r + lambda);
  float cc = -0.5f * dt * (alpha + beta);
  float dd = rhs[i + NX * j];

  // u[0] = u[1]
  if(j == 1){
    bb += aa;
    aa = 0.0f;
  }
  // u[NY - 2] = u[NY - 1]
  else if(j == NY - 2){
    bb += cc; 
    cc = 0.0f;
  }
  // assign zero to boundary (but we ignore it so it is still neumann)
  if(j == 0 || j == NY - 1){
    aa = 0.0f;
    bb = 1.0f;
    cc = 0.0f;
    dd = 0.0f;
  }

  // normalize by bb
  float bb_i = 1.0f / bb;
  aa *= bb_i;
  cc *= bb_i;
  dd *= bb_i;

  a[j] = aa;
  c[j] = cc;
  d[j] = dd;

  __syncthreads();

  // pcr
  // log(nx) iterations
  for(int step = 1; step < NY; step *= 2){
    float a_left = 0.0f;
    float c_left = 0.0f;
    float d_left = 0.0f;
    float a_right = 0.0f;
    float c_right = 0.0f;
    float d_right = 0.0f;

    if(j - step >= 0){
      a_left = a_in[j - step];
      c_left = c_in[j - step];
      d_left = d_in[j - step];
    }
    if(j + step < NY){
      a_right = a_in[j + step];
      c_right = c_in[j + step];
      d_right = d_in[j + step];
    }
    float bb_new = 1.0f - aa * c_left - cc * a_right;
    float aa_new = -aa * a_left;
    float cc_new = -cc * c_right;
    float dd_new = dd - aa * d_left - cc * d_right;
    // diagonal normalization
    float inverse = 1.0f / bb_new;
    
    aa = aa_new * inverse;
    cc = cc_new * inverse;
    dd = dd_new * inverse;

    a_out[j] = aa;
    c_out[j] = cc;
    d_out[j] = dd;
    __syncthreads();

    float *temp;
    temp = a_out; a_out = a_in; a_in = temp;
    temp = c_out; c_out = c_in; c_in = temp;
    temp = d_out; d_out = d_in; d_in = temp;
  }

  // neumann zero boundary in v
  if(j == 0){
    u_new[i] = d_in[1];
  }
  else if(j == NY - 1){
    u_new[i + NX * (NY - 1)] = d_in[NY - 2];
  }
  else{
    u_new[i + NX * j] = dd;
  }
}


__global__ void tri_solve_s(
  // int nx, 
  // int ny, 
  const float * __restrict__ rhs, 
  float * __restrict__ u_new,
  float dt, 
  float ds, 
  float dv,
  // float sigma, 
  float r, 
  float lambda,
  float xi,
  float s_min,
  float v_min
) {
  int i = threadIdx.x;
  int j = blockIdx.x;
  __shared__ float a[NX], a1[NX];
  __shared__ float c[NX], c1[NX];
  __shared__ float d[NX], d1[NX];
  float *a_in = a, *a_out = a1;
  float *c_in = c, *c_out = c1;
  float *d_in = d, *d_out = d1;

  float s = s_min + i * ds;
  float v = v_min + j * dv;

  // tridiagonal row
  float alpha = 0.5f * v * s * s / (ds * ds);     // u_ss coef
  float beta  = (r - lambda * xi) * s / (2.0f * ds); // u_s coef

  float aa = -0.5f * dt * (alpha - beta);
  float bb =  1.0f + dt * alpha + 0.25f * dt * (r + lambda);
  float cc = -0.5f * dt * (alpha + beta);
  float dd = rhs[i + NX * j];

  //pass through dirichlet
  if(i == 0 || i == NX - 1){
    aa = 0.0f;
    bb = 1.0f;
    cc = 0.0f;
  }
  else if(j == 0){
    aa = 0.0f;
    bb = 1.0f;
    cc = 0.0f;
  }
  else if(j == NY - 1){
    aa = 0.0f;
    bb = 1.0f;
    cc = 0.0f;
  }

  // normalize by bb
  float bb_i = 1.0f / bb;
  aa *= bb_i;
  cc *= bb_i;
  dd *= bb_i;

  a[i] = aa;
  c[i] = cc;
  d[i] = dd;
  __syncthreads();
  // pcr
  // log(nx) iterations
  for(int step = 1; step < NX; step *= 2){
    float a_left = 0.0f;
    float c_left = 0.0f;
    float d_left = 0.0f;
    float a_right = 0.0f;
    float c_right = 0.0f;
    float d_right = 0.0f;

    if(i - step >= 0){
      a_left = a_in[i-step];
      c_left = c_in[i - step];
      d_left = d_in[i - step];
    }
    if(i + step < NX){
      a_right = a_in[i + step];
      c_right = c_in[i + step];
      d_right = d_in[i + step];
    }
    float bb_new = 1.0f - aa * c_left - cc * a_right;
    float aa_new = -aa * a_left;
    float cc_new = -cc * c_right;
    float dd_new = dd - aa * d_left - cc * d_right;
    float inverse = 1.0f / bb_new;
    
    aa = aa_new * inverse;
    cc = cc_new * inverse;
    dd = dd_new * inverse;

    a_out[i] = aa;
    c_out[i] = cc;
    d_out[i] = dd;
    __syncthreads();
    float *temp;
    temp = a_in; a_in = a_out; a_out = temp;
    temp = c_in; c_in = c_out; c_out = temp;
    temp = d_in; d_in = d_out; d_out = temp;
  }  

  u_new[i + NX * j] = dd;
}

int main(int argc, const char **argv){
#if PRINT_TIME
  printf("%d,%d,",NX,NY);
#endif

  /* 
  DEFINE IN HEADER TO MAKE IT ADJUSTABLE
  int NX = 128;
  int NY = 128;
  float dt = 0.001f;
  float rho = -0.5f;
  float sigma = 0.3f;
  float r = 0.05f;
  float lambda = 0.1f;
  float xi = 0.0f;
  float nu = 0.04f;
  float kappa = 2.0f;

  */
  // int nx = NX;
  // int ny = NY;

  wakeup_delay();
  const float ds = (SMAX - SMIN) / (NX - 1);
  const float dv = (VMAX - VMIN) / (NY - 1);

  size_t bytes = sizeof(float) * NX * NY;
  
  float *h_u = (float*)malloc(bytes);
  float *h_u_serial = (float*)malloc(bytes);

  for(int j = 0; j < NY; j++){
    for(int i = 0; i < NX; i++){
      float s = SMIN + i *ds;
      h_u[i + NX * j] = fmaxf(s - K, 0.0f);
    }
  }
  memcpy(h_u_serial, h_u, bytes);

  float *d_u_new;
  float *d_u_old;
  float *d_u_half;
  float *d_rhs;

  cudaMalloc(&d_u_new, bytes);
  cudaMalloc(&d_u_old, bytes);
  cudaMalloc(&d_u_half, bytes);
  cudaMalloc(&d_rhs, bytes);
  cudaMemcpy(d_u_old, h_u, bytes, cudaMemcpyHostToDevice);
  
  // findCudaDevice(argc, argv); we are using only one device assigned

  // dim3 grid(NY), block(NX);

#if PRINT_TIME
  cudaEvent_t start, stop;
  float elapsed_gpu;
  // Create the cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record event on the default stream
  cudaEventRecord(start, 0);
#endif

  for(int step = 0; step < NITER; step++) {
    // combined kernel
    float tau = (step + 1) * DT;
    rhs_combined<<<NY,NX>>>(d_u_old, d_rhs, DT, ds, dv,
    RHO, SIGMA, R, LAMBDA, XI, NU, KAPPA, K, tau ,SMIN, VMIN);

    // implicit
    tri_solve_s<<<NY,NX>>>(d_rhs, d_u_half,
      DT, ds, dv, R, LAMBDA, XI,
      SMIN, VMIN
    );

    // TODO: transpose d_u_half here 
    tri_solve_v<<<NX,NY>>>(d_u_half, d_u_new,
      DT, ds, dv, 
      SIGMA, R, LAMBDA, KAPPA, NU,
      SMIN, VMIN
    );

    float *temp = d_u_old;
    d_u_old = d_u_new;
    d_u_new = temp;
  }

#if PRINT_TIME
  // Stop and destroy the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_gpu, start, stop);
  // printf("\nGPU time: %f (msec)\n", elapsed_gpu);
  printf("%f, ", elapsed_gpu);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
#endif

#if !RUN_CPU
  printf("\n");
#endif


#if RUN_CPU

  cudaMemcpy(h_u, d_u_old, sizeof(float)*NX*NY,
                              cudaMemcpyDeviceToHost);

  float *h_u_rhs_serial = (float *)malloc(sizeof(float) * NX * NY); 
  float *h_u_half = (float *)malloc(sizeof(float)*NX*NY);
  float *h_u_new_serial = (float *)malloc(sizeof(float)*NX*NY);
#if PRINT_TIME
  struct timespec time_start, time_stop;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
#endif

  for(int iteration = 0; iteration < NITER; iteration++){
    float tau = (iteration + 1) * DT;
    for(int j = 0; j < NY; j++){
      for(int i = 0; i < NX; i++){
        rhs_combined_serial(
          i,j,
          h_u_serial,h_u_rhs_serial,
          DT,ds,dv,
          RHO,SIGMA,R,LAMBDA,XI,NU,KAPPA,K,tau,
          SMIN,VMIN
        );
      }
    }
    for(int j = 0; j < NY; j++){ // solve s
      float *a = (float *) malloc(sizeof(float)*NX);
      float *c = (float *) malloc(sizeof(float)*NX);
      float *d = (float *) malloc(sizeof(float)*NX);
      float *a_old = (float *) malloc(sizeof(float)*NX);
      float *c_old = (float *) malloc(sizeof(float)*NX);
      float *d_old = (float *) malloc(sizeof(float)*NX);
      for(int i = 0; i < NX; i++){
        setup_tri_solve_s_serial(
          i,j,
          a,c,d,
          h_u_rhs_serial,
          DT,ds,dv,
          R,LAMBDA,XI,
          SMIN,VMIN
        );
      }
      for(int step = 1; step < NX; step *= 2){
        memcpy(a_old, a, sizeof(float)*NX);
        memcpy(c_old, c, sizeof(float)*NX);
        memcpy(d_old, d, sizeof(float)*NX);
        for(int i = 0; i < NX; i++){
          step_tri_solve_serial(
            i,step,NX,
            a_old, c_old, d_old,
            a, c, d
          );
        }
      }
      for(int i = 0; i < NX; i++){
        transfer_d_s_serial(i,j,d,h_u_half);
      }
      free(a);free(c);free(d);
      free(a_old);free(c_old);free(d_old);

    }
    for(int i = 0; i < NX; i++){ // solve v
      // s direction dirichlet pass through
      if(i == 0 || i == NX - 1){
        for(int j = 0; j < NY; j++){
          h_u_new_serial[i + NX * j] = h_u_half[i + NX * j];
        }
        continue;
      }

      float *a = (float *) malloc(sizeof(float)*NY);
      float *c = (float *) malloc(sizeof(float)*NY);
      float *d = (float *) malloc(sizeof(float)*NY);
      float *a_old = (float *) malloc(sizeof(float)*NY);
      float *c_old = (float *) malloc(sizeof(float)*NY);
      float *d_old = (float *) malloc(sizeof(float)*NY);
      for(int j = 0; j < NY; j++){
        setup_tri_solve_v_serial(
          i,j,
          a,c,d,
          h_u_half,
          DT,ds,dv,
          SIGMA,R,LAMBDA,KAPPA,NU,
          SMIN,VMIN
        );
      }
      for(int step = 1; step < NY; step *= 2){
        memcpy(a_old, a, sizeof(float)*NY);
        memcpy(c_old, c, sizeof(float)*NY);
        memcpy(d_old, d, sizeof(float)*NY);
        for(int j = 0; j < NY; j++){
          step_tri_solve_serial(
            j,step,NY,
            a_old, c_old, d_old,
            a, c, d
          );
        }
      }
      for(int j = 0; j < NY; j++){
        transfer_d_v_serial(i,j,d,h_u_new_serial);
      }
      free(a);free(c);free(d);
      free(a_old);free(c_old);free(d_old);
    }
    float *temp_serial = h_u_serial;
    h_u_serial      = h_u_new_serial;
    h_u_new_serial  = temp_serial;
  }

// #if PRINT_TIME
//   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
//   // measurement = interval(time_start, time_stop);
//   printf("CPU completed in %f milliseconds.\n", interval(time_start, time_stop)*1000);
// #endif

#if PRINT_TIME
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
  // measurement = interval(time_start, time_stop);
  printf("%f\n", interval(time_start, time_stop)*1000);
#endif

#endif



  // float total_error = 0.0;
  // float sum_sq_error  = 0.0;
  // for (int idx = 0; idx < NX * NY; idx++) {
  //   float error = fabs((float)h_u[idx] - (float)h_u_serial[idx]);
  //   total_error += error;
  //   sum_sq_error += error * error;
  // }
  // float rms_error = sqrt(sum_sq_error / (NX * NY));
  // printf("total error: %f\n", total_error);
  // printf("rms: %f\n", rms_error);


  cudaFree(d_u_new);
  cudaFree(d_u_old);
  cudaFree(d_u_half);
  cudaFree(d_rhs);

  free(h_u);
  free(h_u_serial);

#if RUN_CPU
  free(h_u_rhs_serial);free(h_u_half);free(h_u_new_serial);
#endif
  // printf("Finished Successfully!\n");

  return 0;
}

