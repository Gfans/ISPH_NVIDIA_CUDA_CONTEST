/*
  FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2012. Rama Hoetzlein, http://fluids3.com

  Fluids-ZLib license (* see part 1 below)
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. Acknowledgement of the
	 original author is required if you publish this in a paper, or use it
	 in a product. (See fluids3.com for details)
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/********************************************************************************************/
/*   PCISPH is integrated by Xiao Nie for NVIDIA’s 2013 CUDA Campus Programming Contest    */
/*                     https://github.com/Gfans/ISPH_NVIDIA_CUDA_CONTEST                    */
/*   For the PCISPH, please refer to the paper "Predictive-Corrective Incompressible SPH"   */
/********************************************************************************************/

#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>

#include "GL/glut.h"

#include <cuda_gl_interop.h>		
#include "cutil.h"				
#include "cutil_math.h"				

#include "fluid_system_host.cuh"		
#include "fluid_system_kern.cuh"
#include "cudaHeaders.cuh"
#include "common_defs.h"

#define BLOCK_SIZE 256

ParticleParams	fcudaParams;
bufList			fbuf;

float**			g_scanBlockSums;
int**			g_scanBlockSumsInt;
unsigned int	g_numEltsAllocated = 0;
unsigned int	g_numLevelsAllocated = 0;

char* d_idata = NULL;
char* d_odata = NULL;

template<class T>
struct SharedMemory
{
	__device__ inline operator       T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

template<>
struct SharedMemory<float>
{
	__device__ inline operator       float *()
	{
		extern __shared__ float __smem_d[];
		return (float *)__smem_d;
	}

	__device__ inline operator const float *() const
	{
		extern __shared__ float __smem_d[];
		return (float *)__smem_d;
	}
};

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
	ReduceMax(T *g_idata, T *g_odata, unsigned int n)
{
	T *sdata = SharedMemory<T>();

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	T myMax = 0;

	while (i < n)
	{
		myMax = fmaxf(myMax, g_idata[i]);

		if (nIsPow2 || i + blockSize < n)
			myMax = fmaxf(myMax, g_idata[i+blockSize]);

		i += gridSize;
	}

	sdata[tid] = myMax;
	__syncthreads();

	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 256]);
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 128]);
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			sdata[tid] = myMax = fmaxf(myMax, sdata[tid +  64]);
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		volatile T *smem = sdata;

		if (blockSize >=  64)
		{
			smem[tid] = myMax = fmaxf(myMax, smem[tid + 32]);
		}

		if (blockSize >=  32)
		{
			smem[tid] = myMax = fmaxf(myMax, smem[tid + 16]);
		}

		if (blockSize >=  16)
		{
			smem[tid] = myMax = fmaxf(myMax, smem[tid +  8]);
		}

		if (blockSize >=   8)
		{
			smem[tid] = myMax = fmaxf(myMax, smem[tid +  4]);
		}

		if (blockSize >=   4)
		{
			smem[tid] = myMax = fmaxf(myMax, smem[tid +  2]);
		}

		if (blockSize >=   2)
		{
			smem[tid] = myMax = fmaxf(myMax, smem[tid +  1]);
		}
	}

	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

int iDivUp (int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void computeCUDAGridBlockSize (int numParticles, int blockSize, int &numBlocks, int &numThreads)
{
    numThreads = min( blockSize, numParticles );
    numBlocks = iDivUp ( numParticles, numThreads );
}

inline bool isPowerOfTwo(int n) { return ((n&(n-1))==0) ; }

inline int floorPow2(int n) {
	#ifdef WIN32
		return 1 << (int)logb((float)n);
	#else
		int exp;
		frexp((float)n, &exp);
		return 1 << (exp - 1);
	#endif
}

void preallocBlockSums(unsigned int maxNumElements)
{
    assert(g_numEltsAllocated == 0); 

    g_numEltsAllocated = maxNumElements;
    unsigned int blockSize = BLOCK_SIZE;
    unsigned int numElts = maxNumElements;
    int level = 0;

    do {       
        unsigned int numBlocks =   max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) level++;
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums = (float**) malloc(level * sizeof(float*));
    g_numLevelsAllocated = level;
    
    numElts = maxNumElements;
    level = 0;
    
    do {       
        unsigned int numBlocks = max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) CUDA_SAFE_CALL ( cudaMalloc((void**) &g_scanBlockSums[level++], numBlocks * sizeof(float)) );
        numElts = numBlocks;
    } while (numElts > 1);
}

void deallocBlockSums()
{
    for (unsigned int i = 0; i < g_numLevelsAllocated; i++) cudaFree(g_scanBlockSums[i]);
    
    free( (void**)g_scanBlockSums );

    g_scanBlockSums = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}

void prescanArrayRecursive (float *outArray, const float *inArray, int numElements, int level)
{
    unsigned int blockSize = BLOCK_SIZE;
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    unsigned int numEltsLastBlock = numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock) {
        np2LastBlock = 1;
        if(!isPowerOfTwo(numEltsLastBlock)) numThreadsLastBlock = floorPow2(numEltsLastBlock);            
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = sizeof(float) * (2 * numThreadsLastBlock + extraSpace);
    }

    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(float) * (numEltsPerBlock + extraSpace);

	#ifdef DEBUG
		if (numBlocks > 1) assert(g_numEltsAllocated >= numElements);
	#endif

    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
    dim3  threads(numThreads, 1, 1);

    if (numBlocks > 1) {
        prescan<true, false><<< grid, threads, sharedMemSize >>> (outArray, inArray,  g_scanBlockSums[level], numThreads * 2, 0, 0);
        if (np2LastBlock) {
            prescan<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>> (outArray, inArray, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }

        prescanArrayRecursive (g_scanBlockSums[level], g_scanBlockSums[level], numBlocks, level+1);

        uniformAdd<<< grid, threads >>> (outArray, g_scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);
        if (np2LastBlock) {
            uniformAdd<<< 1, numThreadsLastBlock >>>(outArray, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }
    } else if (isPowerOfTwo(numElements)) {
        prescan<false, false><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numThreads * 2, 0, 0);
    } else {
        prescan<false, true><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numElements, 0, 0);
    }
}

void cudaInit(int argc, char **argv)
{   
    CUT_DEVICE_INIT(argc, argv);
 
	cudaDeviceProp p;
	cudaGetDeviceProperties ( &p, 0);
	
	// 输出CUDA设备信息
	printf ( "-- CUDA Device Info --\n" );
	printf ( "Name:       %s\n", p.name );
	printf ( "Capability: %d.%d\n", p.major, p.minor );
	printf ( "Global Mem: %d MB\n", p.totalGlobalMem/1000000 );
	printf ( "Shared/Blk: %d\n", p.sharedMemPerBlock );
	printf ( "Regs/Blk:   %d\n", p.regsPerBlock );
	printf ( "Warp Size:  %d\n", p.warpSize );
	printf ( "Mem Pitch:  %d\n", p.memPitch );
	printf ( "Thrds/Blk:  %d\n", p.maxThreadsPerBlock );
	printf ( "Const Mem:  %d\n", p.totalConstMem );
	printf ( "Clock Rate: %d\n", p.clockRate );	
};

void cudaExit (int argc, char **argv)
{
	CUT_EXIT(argc, argv); 
}

void ParticleClearCUDA ()
{
	CUDA_SAFE_CALL ( cudaFree ( fbuf.pos ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.predicted_pos ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.vel ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.vel_eval ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.correction_pressure_force ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.force ) );		
	CUDA_SAFE_CALL ( cudaFree ( fbuf.press ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.correction_pressure ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.fluid_density ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.predicted_density ) );		
	CUDA_SAFE_CALL ( cudaFree ( fbuf.densityError ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.max_predicted_density_array ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.particle_grid_cell_index ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.grid_particle_offset ) );	
	CUDA_SAFE_CALL ( cudaFree ( fbuf.clr ) );	

	CUDA_SAFE_CALL ( cudaFree ( fbuf.sort_buf ) );	

	CUDA_SAFE_CALL ( cudaFree ( fbuf.particle_index_grid ) );
	CUDA_SAFE_CALL ( cudaFree ( fbuf.grid_particles_num ) );
	CUDA_SAFE_CALL ( cudaFree ( fbuf.grid_off ) );
	CUDA_SAFE_CALL ( cudaFree ( fbuf.grid_active ) );
}

void ParticleSetupCUDA ( int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk , float grid_cell_size, float param_kernel_self)
{	
	fcudaParams.param_num = num;	
	fcudaParams.param_grid_res = res;
	fcudaParams.param_grid_size = size;
	fcudaParams.param_grid_delta = delta;
	fcudaParams.param_grid_min = gmin;
	fcudaParams.param_grid_max = gmax;
	fcudaParams.param_grid_total = total;
	fcudaParams.param_grid_search = gsrch;
	fcudaParams.param_grid_adj_cnt = gsrch*gsrch*gsrch;
	fcudaParams.param_grid_scan_max = res;
	fcudaParams.param_grid_cell_size = grid_cell_size;
	fcudaParams.param_kernel_self = param_kernel_self;
	fcudaParams.param_chk = chk;
	fcudaParams.param_max_loops = MAX_PCISPH_LOOPS;
	fcudaParams.param_min_loops = 3; 

	int cell = 0;
	for (int y = -1; y < 2; y++ ) 
		for (int z = -1; z < 2; z++ ) 
			for (int x = -1; x < 2; x++ ) 
				fcudaParams.param_grid_neighbor_cell_index_offset[cell++] = y * fcudaParams.param_grid_res.x * fcudaParams.param_grid_res.z + z * fcudaParams.param_grid_res.x + x ;			

    computeCUDAGridBlockSize ( fcudaParams.param_num, 256, fcudaParams.param_num_blocks, fcudaParams.param_num_threads);			
    computeCUDAGridBlockSize ( fcudaParams.param_grid_total, 256, fcudaParams.param_grid_blocks, fcudaParams.param_grid_threads);	
    
    fcudaParams.param_size_Points = (fcudaParams.param_num_blocks  * fcudaParams.param_num_threads);     	
	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.pos,										fcudaParams.param_size_Points*sizeof(float)*3 ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.predicted_pos,								fcudaParams.param_size_Points*sizeof(float)*3 ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.vel,										fcudaParams.param_size_Points*sizeof(float)*3 ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.vel_eval,									fcudaParams.param_size_Points*sizeof(float)*3 ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.correction_pressure_force,					fcudaParams.param_size_Points*sizeof(float)*3 ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.force,										fcudaParams.param_size_Points*sizeof(float)*3 ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.press,										fcudaParams.param_size_Points*sizeof(float) ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.correction_pressure,						fcudaParams.param_size_Points*sizeof(float) ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.fluid_density,								fcudaParams.param_size_Points*sizeof(float) ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.predicted_density,							fcudaParams.param_size_Points*sizeof(float) ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.densityError,								fcudaParams.param_size_Points*sizeof(float) ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.max_predicted_density_array,				fcudaParams.param_num_blocks*sizeof (float) ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.particle_grid_cell_index,					fcudaParams.param_size_Points*sizeof(uint) ) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.grid_particle_offset,						fcudaParams.param_size_Points*sizeof(uint)) );	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.clr,										fcudaParams.param_size_Points*sizeof(uint) ) );	

	int temp_size = 6 * (sizeof(float)*3) + 6 * sizeof(float) + 3 * sizeof(uint); 
	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.sort_buf,	fcudaParams.param_size_Points * temp_size ) );	

	fcudaParams.param_size_grid = (fcudaParams.param_grid_blocks * fcudaParams.param_grid_threads);  
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.particle_index_grid,					fcudaParams.param_size_Points*sizeof(uint) ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.grid_particles_num,					fcudaParams.param_size_grid*sizeof(int) ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.grid_off,								fcudaParams.param_size_grid*sizeof(int) ) );
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &fbuf.grid_active,							fcudaParams.param_size_grid*sizeof(int) ) );
		
	CUDA_SAFE_CALL ( cudaMemcpyToSymbol ( "simData", &fcudaParams, sizeof(ParticleParams) ) );
	cudaDeviceSynchronize ();

	deallocBlockSumsInt ();
	preallocBlockSumsInt ( fcudaParams.param_grid_total );
}

void FluidParamCUDA ( float simScale, float smoothRadius, float particleRadius, float mass, float restDen, float3 boundMin, 
					float3 boundMax, float boundStiff, float internalStiff, float visc, float damp, float forceMin, 
					float forceMax, float forceFreq, float groundSlope, float gravX, float gravY, float gravZ, float accelLimit, float velLimit, float density_error_factor)
{
	fcudaParams.param_bound_min = boundMin;
	fcudaParams.param_bound_max = boundMax;
	fcudaParams.param_gravity = make_float3( gravX, gravY, gravZ );
	fcudaParams.param_mass = mass;
	fcudaParams.param_rest_dens = restDen;
	fcudaParams.param_ext_stiff = boundStiff;
	fcudaParams.param_gas_constant = internalStiff;
	fcudaParams.param_radius = particleRadius;
	fcudaParams.param_smooth_radius = smoothRadius;
	fcudaParams.param_particle_spacing = smoothRadius * 0.5 - 0.000001f;
	fcudaParams.param_sim_scale = simScale;
	fcudaParams.param_visc = visc;
	fcudaParams.param_force_min = forceMin;
	fcudaParams.param_force_max = forceMax;
	fcudaParams.param_force_freq = forceFreq;
	fcudaParams.param_ground_slope = groundSlope;
	fcudaParams.param_damp = damp;
	fcudaParams.param_acc_limit = accelLimit;
	fcudaParams.param_acc_limit_square = accelLimit * accelLimit;
	fcudaParams.param_vel_limit = velLimit;
	fcudaParams.param_vel_limit_square = velLimit * velLimit;

	fcudaParams.param_poly6_kern = 315.0f / (64.0f * MY_PI * pow( smoothRadius, 9.0f) );
	fcudaParams.param_spiky_kern = -45.0f / (MY_PI * pow( smoothRadius, 6.0f) );
	fcudaParams.param_lapkern = 45.0f / (MY_PI * pow( smoothRadius, 6.0f) );	

	fcudaParams.param_max_density_error_allowed = 1.0;		
	fcudaParams.param_density_error_factor = density_error_factor;

	fcudaParams.param_add_boundary_force = true;
	fcudaParams.param_force_distance = fcudaParams.param_smooth_radius * 0.25f;
	fcudaParams.param_boundary_force_factor = 256;
	fcudaParams.param_max_boundary_force = fcudaParams.param_boundary_force_factor * fcudaParams.param_particle_spacing * fcudaParams.param_particle_spacing;

	CUDA_SAFE_CALL( cudaMemcpyToSymbol ( "simData", &fcudaParams, sizeof(ParticleParams) ) );
	cudaDeviceSynchronize ();
}

void CopyToCUDA ( float* pos, float* predicted_pos, float* vel, float* veleval, float* correction_pressure_force, float* force, float* pressure, float* correction_pressure, 
	float* density, float* predicted_density, float* densityError, float* max_predicted_density_array, uint* cluster, uint* next_particle_index_in_the_same_cell, char* clr )
{
	int numPoints = fcudaParams.param_num;
	int numBlocks = fcudaParams.param_num_blocks;
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.pos,										pos,										numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.predicted_pos,							predicted_pos,								numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.vel,										vel,										numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.vel_eval,									veleval,									numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.correction_pressure_force,				correction_pressure_force,					numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.force,									force,										numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.press,									pressure,									numPoints*sizeof(float),   cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.correction_pressure,						correction_pressure,						numPoints*sizeof(float),   cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.fluid_density,							density,									numPoints*sizeof(float),   cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.predicted_density,						predicted_density,							numPoints*sizeof(float),   cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.densityError,								densityError,								numPoints*sizeof(float),   cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.max_predicted_density_array,				max_predicted_density_array,				numBlocks*sizeof(float),   cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.grid_particle_offset,						next_particle_index_in_the_same_cell,		numPoints*sizeof(float),   cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy ( fbuf.clr,										clr,										numPoints*sizeof(uint),    cudaMemcpyHostToDevice ) );

	cudaDeviceSynchronize ();	
}

void CopyFromCUDA ( float* pos, float* predicted_pos, float* vel, float* vel_eval, float* correction_pressure_force, float* force, float* pressure, float* correction_pressure, 
	float* density, float* predicted_density, float* densityError, float* max_predicted_density_array, uint* cluster, uint* next_particle_index_in_the_same_cell, char* clr )
{
	int numPoints = fcudaParams.param_num;
	int numBlocks = fcudaParams.param_num_blocks;

	if ( pos != 0x0 ) 
		CUDA_SAFE_CALL( cudaMemcpy ( pos,											fbuf.pos,											numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );

	if ( predicted_pos != 0x0 ) 
		CUDA_SAFE_CALL( cudaMemcpy ( predicted_pos,									fbuf.predicted_pos,									numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );

	if (vel != 0x0)
		CUDA_SAFE_CALL( cudaMemcpy ( vel,											fbuf.vel,											numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
	
	if (vel_eval != 0x0)
		CUDA_SAFE_CALL( cudaMemcpy ( vel_eval,										fbuf.vel_eval,										numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );

	if (correction_pressure_force != 0x0)
		CUDA_SAFE_CALL( cudaMemcpy ( correction_pressure_force,						fbuf.correction_pressure_force,						numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
	
	if (force != 0x0)
		CUDA_SAFE_CALL( cudaMemcpy ( force,											fbuf.force,											numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
	
	if (pressure != 0x0)
		CUDA_SAFE_CALL( cudaMemcpy ( pressure,										fbuf.press,											numPoints*sizeof(float),  cudaMemcpyDeviceToHost ) );

	if (correction_pressure != 0x0)
		CUDA_SAFE_CALL( cudaMemcpy ( correction_pressure,							fbuf.correction_pressure,							numPoints*sizeof(float),  cudaMemcpyDeviceToHost ) );
	
	if (density != 0x0)	
		CUDA_SAFE_CALL( cudaMemcpy ( density,										fbuf.fluid_density,									numPoints*sizeof(float),  cudaMemcpyDeviceToHost ) );

	if (predicted_density != 0x0)
		CUDA_SAFE_CALL( cudaMemcpy ( predicted_density,								fbuf.predicted_density,								numPoints*sizeof(float),  cudaMemcpyDeviceToHost ) );

	if ( densityError != 0x0 ) 
		CUDA_SAFE_CALL( cudaMemcpy ( densityError,									fbuf.densityError,									numPoints*sizeof(float),  cudaMemcpyDeviceToHost ) );

	if ( max_predicted_density_array != 0x0 ) 
		CUDA_SAFE_CALL( cudaMemcpy ( max_predicted_density_array,					fbuf.max_predicted_density_array,					numBlocks*sizeof(float),  cudaMemcpyDeviceToHost ) );

	if ( next_particle_index_in_the_same_cell != 0x0 ) 
		CUDA_SAFE_CALL( cudaMemcpy ( next_particle_index_in_the_same_cell,			fbuf.grid_particle_offset,							numPoints*sizeof(uint),  cudaMemcpyDeviceToHost ) );
	
	if ( clr != 0x0 ) 
		CUDA_SAFE_CALL( cudaMemcpy ( clr,											fbuf.clr,											numPoints*sizeof(uint),  cudaMemcpyDeviceToHost ) );

	cudaDeviceSynchronize ();	
}

void InsertParticlesCUDA ( uint* grid_cell_id, uint* cluster_cell, int* particles_num )
{
	cudaMemset ( fbuf.grid_particles_num, 0,	fcudaParams.param_grid_total * sizeof(int));

	insertParticles<<< fcudaParams.param_num_blocks, fcudaParams.param_num_threads>>> ( fbuf, fcudaParams.param_num );
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: InsertParticlesCUDA: %s\n", cudaGetErrorString(error) );
	}  
	cudaDeviceSynchronize ();
	// 用于验证数据的正确性时将数据传回CPU
	if (grid_cell_id != 0x0) {
		CUDA_SAFE_CALL( cudaMemcpy ( grid_cell_id,	fbuf.particle_grid_cell_index,		fcudaParams.param_num*sizeof(uint),			cudaMemcpyDeviceToHost ) );		
		CUDA_SAFE_CALL( cudaMemcpy ( particles_num,	fbuf.grid_particles_num,			fcudaParams.param_grid_total*sizeof(int),	cudaMemcpyDeviceToHost ) );
	}
}

void PrefixSumCellsCUDA ( int* goff )
{
    prescanArrayRecursiveInt ( fbuf.grid_off, fbuf.grid_particles_num, fcudaParams.param_grid_total, 0);
	cudaDeviceSynchronize ();

	// 用于验证数据的正确性时将数据传回CPU
	if ( goff != 0x0 ) {
		CUDA_SAFE_CALL( cudaMemcpy ( goff,	fbuf.grid_off, fcudaParams.param_grid_total * sizeof(int),  cudaMemcpyDeviceToHost ) );
	}
}

void CountingSortIndexCUDA ( uint* ggrid )
{	
	cudaMemset ( fbuf.particle_index_grid,	GRID_UCHAR,	fcudaParams.param_num * sizeof(int) );

	countingSortIndex <<< fcudaParams.param_num_blocks, fcudaParams.param_num_threads>>> ( fbuf, fcudaParams.param_num );		
	cudaDeviceSynchronize ();

	// 用于验证数据的正确性时将数据传回CPU
	if ( ggrid != 0x0 ) {
		CUDA_SAFE_CALL( cudaMemcpy ( ggrid,	fbuf.particle_index_grid, fcudaParams.param_num * sizeof(uint), cudaMemcpyDeviceToHost ) );
	}
}

void CountingSortFullCUDA ( uint* ggrid )
{
	// 将粒子数据拷贝到临时缓冲区
	int n = fcudaParams.param_num;
	cudaMemcpy ( fbuf.sort_buf + n*BUF_POS,							fbuf.pos,						n*sizeof(float)*3,	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.sort_buf + n*BUF_VEL,							fbuf.vel,						n*sizeof(float)*3,	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.sort_buf + n*BUF_VELEVAL,						fbuf.vel_eval,					n*sizeof(float)*3,	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.sort_buf + n*BUF_FORCE,						fbuf.force,						n*sizeof(float)*3,	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.sort_buf + n*BUF_PRESS,						fbuf.press,						n*sizeof(float),	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.sort_buf + n*BUF_DENS,						fbuf.fluid_density,				n*sizeof(float),	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.sort_buf + n*BUF_PARTICLE_GRID_CELL_INDEX,	fbuf.particle_grid_cell_index,	n*sizeof(uint),		cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.sort_buf + n*BUF_GRID_PARTICLE_OFFSET,		fbuf.grid_particle_offset,		n*sizeof(uint),		cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.sort_buf + n*BUF_CLR,							fbuf.clr,						n*sizeof(uint),		cudaMemcpyDeviceToDevice );

	cudaMemset ( fbuf.particle_index_grid,	GRID_UCHAR,	fcudaParams.param_num * sizeof(int) );

	countingSortFull <<< fcudaParams.param_num_blocks, fcudaParams.param_num_threads>>> ( fbuf, fcudaParams.param_num );		
	cudaDeviceSynchronize ();
}

void ComputeDensityPressureCUDA ()
{
	computeDensityPressureSPH<<< fcudaParams.param_num_blocks, fcudaParams.param_num_threads>>> ( fbuf, fcudaParams.param_num );	
    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: ComputePressureCUDA: %s\n", cudaGetErrorString(error) );
	}    
	cudaDeviceSynchronize ();
}

void ComputeForceCUDA ()
{
	ComputeForceCUDASPH<<< fcudaParams.param_num_blocks, fcudaParams.param_num_threads>>> ( fbuf, fcudaParams.param_num );
    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: ComputeForceCUDA: %s\n", cudaGetErrorString(error) );
	}    
	cudaDeviceSynchronize ();
}

void ComputeOtherForceCUDA ()
{
	ComputeOtherForceCUDAPCISPH<<< fcudaParams.param_num_blocks, fcudaParams.param_num_threads>>> ( fbuf, fcudaParams.param_num );
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: ComputeOtherForceCUDA: %s\n", cudaGetErrorString(error) );
	}    
	cudaDeviceSynchronize ();
}

void PredictPositionAndVelocityCUDA(float time_step)
{
	PredictPositionAndVelocityCUDAPCISPH<<< fcudaParams.param_num_blocks, fcudaParams.param_num_threads>>> (fbuf, fcudaParams.param_num, time_step);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: PredictPositionAndVelocity: %s\n", cudaGetErrorString(error) );
	}    
	cudaDeviceSynchronize ();
}

void ComputePredictedDensityAndPressureCUDA()
{
	ComputePredictedDensityAndPressureCUDAPCISPH<<< fcudaParams.param_num_blocks, fcudaParams.param_num_threads>>> ( fbuf, fcudaParams.param_num);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: ComputePredictedDensityAndPressure: %s\n", cudaGetErrorString(error) );
	}    
	cudaDeviceSynchronize ();
}

void GetMaxPredictedDensityCUDA(float& max_predicted_density)
{
	GetMaxPredictedDensityArrayCUDAPCISPH<float>(fcudaParams.param_num, fcudaParams.param_num_threads, fcudaParams.param_num_blocks, fbuf.predicted_density, fbuf.max_predicted_density_array);

	float* max_predicted_density_value;
	cudaMalloc((void**)&max_predicted_density_value, sizeof(float));
	GetMaxValue<<<1,1>>>(fbuf.max_predicted_density_array, fcudaParams.param_num, max_predicted_density_value);
	cudaMemcpy(&max_predicted_density, max_predicted_density_value, sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C"
	bool isPow2(unsigned int x)
{
	return ((x&(x-1))==0);
}

template <class T>
void GetMaxPredictedDensityArrayCUDAPCISPH(int size, int threads, int blocks, T *d_idata, T *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	if (isPow2(size))
	{
		switch (threads)
		{
		case 512:
			ReduceMax<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 256:
			ReduceMax<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 128:
			ReduceMax<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 64:
			ReduceMax<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 32:
			ReduceMax<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 16:
			ReduceMax<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  8:
			ReduceMax<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  4:
			ReduceMax<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  2:
			ReduceMax<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  1:
			ReduceMax<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		}
	}
	else
	{
		switch (threads)
		{
		case 512:
			ReduceMax<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 256:
			ReduceMax<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 128:
			ReduceMax<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 64:
			ReduceMax<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 32:
			ReduceMax<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 16:
			ReduceMax<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  8:
			ReduceMax<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  4:
			ReduceMax<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  2:
			ReduceMax<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  1:
			ReduceMax<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		}
	}

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: ReduceMax: %s\n", cudaGetErrorString(error) );
	}    
	cudaDeviceSynchronize ();
}

void ComputeCorrectivePressureForceCUDA()
{
	ComputeCorrectivePressureForceCUDAPCISPH<<< fcudaParams.param_num_blocks, fcudaParams.param_num_threads>>> ( fbuf, fcudaParams.param_num );
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: ComputeCorrectivePressureForce: %s\n", cudaGetErrorString(error) );
	}    
	cudaDeviceSynchronize ();
}

void PredictionCorrectionStepCUDA(float time_step)
{
	bool densityErrorTooLarge = true; 
	int iteration = 0;
	while( (iteration < fcudaParams.param_min_loops) || ((densityErrorTooLarge) && (iteration < fcudaParams.param_max_loops)) )	
	{
		PredictPositionAndVelocityCUDA(time_step);

		float max_predicted_density = 0.0;

		ComputePredictedDensityAndPressureCUDA();

		GetMaxPredictedDensityCUDA(max_predicted_density);

		float densityErrorInPercent = max(0.1f * max_predicted_density - 100.0f, 0.0f);

		if(densityErrorInPercent < fcudaParams.param_max_density_error_allowed) 
			densityErrorTooLarge = false;

		ComputeCorrectivePressureForceCUDA();

		iteration++;
	}
}

void CountActiveCUDA ()
{
	int threads = 1;
	int blocks = 1;
	
	assert ( fbuf.grid_active != 0x0 );
	
	cudaMemcpy ( &gridActive, &fcudaParams.param_grid_active, sizeof(int), cudaMemcpyHostToDevice );

	countActiveCells<<< blocks, threads >>> ( fbuf, fcudaParams.param_grid_total );
	cudaDeviceSynchronize ();

	cudaMemcpy ( &fcudaParams.param_grid_active, &gridActive, sizeof(int), cudaMemcpyDeviceToHost );
	
	printf ( "Active cells: %d\n", fcudaParams.param_grid_active );
}

void AdvanceCUDA ( float time_step, float sim_scale )
{
	advanceParticlesCUDASimpleCollision<<< fcudaParams.param_num_blocks, fcudaParams.param_num_threads>>> ( time_step, sim_scale, fbuf, fcudaParams.param_num );
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: AdvanceCUDA: %s\n", cudaGetErrorString(error) );
	}    
    cudaDeviceSynchronize ();
}

void AdvanceCUDAPCISPH(float time_step, float sim_scale )
{
	advanceParticlesPCISPHSimpleCollision<<< fcudaParams.param_num_blocks, fcudaParams.param_num_threads>>> ( time_step, sim_scale, fbuf, fcudaParams.param_num );
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: AdvanceCUDAPCISPH: %s\n", cudaGetErrorString(error) );
	}    
	cudaDeviceSynchronize ();
}

void prefixSumToGPU ( char* inArray, int num, int siz )
{
    CUDA_SAFE_CALL ( cudaMalloc( (void**) &d_idata, num*siz ));
    CUDA_SAFE_CALL ( cudaMalloc( (void**) &d_odata, num*siz ));
    CUDA_SAFE_CALL ( cudaMemcpy( d_idata, inArray, num*siz, cudaMemcpyHostToDevice) );
}

void prefixSumFromGPU ( char* outArray, int num, int siz )
{		
	CUDA_SAFE_CALL ( cudaMemcpy( outArray, d_odata, num*siz, cudaMemcpyDeviceToHost));

	if(d_idata != 0x0) CUDA_SAFE_CALL ( cudaFree( (void**) &d_idata ));
    if(d_odata != 0x0) CUDA_SAFE_CALL ( cudaFree( (void**) &d_odata ));
	d_idata = NULL;
	d_odata = NULL;
}

void prefixSum ( int num )
{
	prescanArray ( (float*) d_odata, (float*) d_idata, num );
}

void prefixSumInt ( int num )
{	
	prescanArrayInt ( (int*) d_odata, (int*) d_idata, num );
}

void preallocBlockSumsInt (unsigned int maxNumElements)
{
#ifdef ENABLE_DEBUG
	 assert(g_numEltsAllocated == 0);
#endif

    g_numEltsAllocated = maxNumElements;
    unsigned int blockSize = BLOCK_SIZE;
    unsigned int numElts = maxNumElements;
    int level = 0;

    do {       
        unsigned int numBlocks =   max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) 
			level++;
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSumsInt = (int**) malloc(level * sizeof(int*));
    g_numLevelsAllocated = level;
    
    numElts = maxNumElements;
    level = 0;
    
    do {       
        unsigned int numBlocks = max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) CUDA_SAFE_CALL ( cudaMalloc((void**) &g_scanBlockSumsInt[level++], numBlocks * sizeof(int)) );
        numElts = numBlocks;
    } while (numElts > 1);
}

void deallocBlockSumsInt()
{
    for (unsigned int i = 0; i < g_numLevelsAllocated; i++) 
		cudaFree(g_scanBlockSumsInt[i]);    
    free( (void**)g_scanBlockSumsInt );

    g_scanBlockSumsInt = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}

void prescanArray ( float *d_odata, float *d_idata, int num )
{	
	preallocBlockSums( num );
    prescanArrayRecursive ( d_odata, d_idata, num, 0);
	deallocBlockSums();
}

void prescanArrayInt ( int *d_odata, int *d_idata, int num )
{	
	preallocBlockSumsInt ( num );
    prescanArrayRecursiveInt ( d_odata, d_idata, num, 0);
	deallocBlockSumsInt ();
}

void prescanArrayRecursiveInt (int *outArray, const int *inArray, int numElements, int level)
{
    unsigned int blockSize = BLOCK_SIZE; 
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    unsigned int numEltsLastBlock = numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock) {
        np2LastBlock = 1;
        if(!isPowerOfTwo(numEltsLastBlock)) numThreadsLastBlock = floorPow2(numEltsLastBlock);            
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = sizeof(float) * (2 * numThreadsLastBlock + extraSpace);
    }

    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(float) * (numEltsPerBlock + extraSpace);

	#ifdef DEBUG
		if (numBlocks > 1) assert(g_numEltsAllocated >= numElements);
	#endif

    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
    dim3  threads(numThreads, 1, 1);

    if (numBlocks > 1) {
        prescanInt <true, false><<< grid, threads, sharedMemSize >>> (outArray, inArray,  g_scanBlockSumsInt[level], numThreads * 2, 0, 0);
        if (np2LastBlock) {
            prescanInt <true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>> (outArray, inArray, g_scanBlockSumsInt[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }

        prescanArrayRecursiveInt (g_scanBlockSumsInt[level], g_scanBlockSumsInt[level], numBlocks, level+1);

        uniformAddInt <<< grid, threads >>> (outArray, g_scanBlockSumsInt[level], numElements - numEltsLastBlock, 0, 0);
        if (np2LastBlock) {
            uniformAddInt <<< 1, numThreadsLastBlock >>>(outArray, g_scanBlockSumsInt[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }
    } else if (isPowerOfTwo(numElements)) {
        prescanInt <false, false><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numThreads * 2, 0, 0);
    } else {
        prescanInt <false, true><<< grid, threads, sharedMemSize >>> (outArray, inArray, 0, numElements, 0, 0);
    }
}