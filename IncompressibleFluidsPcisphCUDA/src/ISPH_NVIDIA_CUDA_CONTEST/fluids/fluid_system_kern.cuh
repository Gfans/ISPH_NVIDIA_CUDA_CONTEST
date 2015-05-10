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

#ifndef DEF_KERN_CUDA
#define DEF_KERN_CUDA

#include <stdio.h>
#include <math.h>

typedef unsigned int		uint;
typedef unsigned short int	ushort;

const int max_num_adj_grid_cells_gpu = 27; 

struct bufList {
	float3*			pos;
	float3*			predicted_pos;					
	float3*			vel;
	float3*			vel_eval;						
	float3*			correction_pressure_force;		
	float3*			force;
	float*			press;
	float*			correction_pressure;			
	float*			fluid_density;
	float*			predicted_density;				
	float*			densityError;					
	float*			max_predicted_density_array;	
	uint*			particle_grid_cell_index;		
	uint*			grid_particle_offset;			
	uint*			clr;							

	char*			sort_buf;						

	uint*			particle_index_grid;			
	int*			grid_particles_num;				
	int*			grid_off;						
	int*			grid_active;
};

#define BUF_POS								0
#define BUF_PREDICTED_POS					(sizeof(float3))
#define BUF_VEL								(BUF_PREDICTED_POS					+ sizeof(float3))
#define BUF_VELEVAL							(BUF_VEL							+ sizeof(float3))
#define BUF_CORRECTION_PRESSURE_FORCE 		(BUF_VELEVAL						+ sizeof(float3))
#define BUF_FORCE							(BUF_CORRECTION_PRESSURE_FORCE		+ sizeof(float3))
#define BUF_PRESS							(BUF_FORCE							+ sizeof(float3))
#define BUF_CORRECTION_PRESS				(BUF_PRESS							+ sizeof(float))
#define BUF_DENS							(BUF_CORRECTION_PRESS				+ sizeof(float))
#define BUF_PREDICTED_DEN					(BUF_DENS							+ sizeof(float))
#define BUF_DENSITY_ERROR					(BUF_PREDICTED_DEN					+ sizeof(float))
#define BUF_MAX_PREDICTED_DENS				(BUF_DENSITY_ERROR					+ sizeof(float))
#define BUF_PARTICLE_GRID_CELL_INDEX		(BUF_MAX_PREDICTED_DENS				+ sizeof(float))
#define BUF_GRID_PARTICLE_OFFSET			(BUF_PARTICLE_GRID_CELL_INDEX		+ sizeof(uint))
#define BUF_CLR								(BUF_GRID_PARTICLE_OFFSET			+ sizeof(uint))

// 粒子参数
struct ParticleParams {
	float3			param_bound_min;
	float3			param_bound_max;
	float3			param_gravity;
	float3			param_grid_size;
	float3			param_grid_delta;
	float3			param_grid_min;
	float3			param_grid_max;
	int3			param_grid_res;
	int3			param_grid_scan_max;
	float			param_particle_spacing;
	float			param_mass;
	float			param_rest_dens;
	float			param_ext_stiff;
	float			param_gas_constant;	
	float			param_radius;
	float			param_smooth_radius;
	float			param_sim_scale;
	float			param_visc;
	float			param_force_min;
	float			param_force_max;
	float			param_force_freq;
	float			param_ground_slope;
	float			param_damp;
	float			param_acc_limit;
	float			param_acc_limit_square;
	float			param_vel_limit;
	float			param_vel_limit_square;
	float			param_poly6_kern;
	float			param_spiky_kern;
	float			param_lapkern;
	float			param_max_density_error_allowed;
	float			param_density_error_factor;
	float			param_grid_cell_size;
	float			param_kernel_self;
	int				param_num_threads;
	int				param_num_blocks;
	int				param_grid_threads;
	int				param_grid_blocks;	
	int				param_size_Points; 
	int				param_size_hash;
	int				param_size_grid;
	int				param_stride;
	int				param_num;
	int				param_grid_search;
	int				param_grid_total;
	int				param_grid_adj_cnt;
	int				param_grid_active;
	int				param_chk;
	int				param_grid_neighbor_cell_index_offset[max_num_adj_grid_cells_gpu];
	int				param_max_loops;
	int				param_min_loops;
	bool			param_add_boundary_force;
	float			param_force_distance;
	float			param_max_boundary_force;
	float			param_inv_force_distance;
	float			param_boundary_force_factor;
};

// 前序求和定义 -  在Fermi & Kepler架构上为 32 banks， 第一代CUDA GPU为16 banks
#define NUM_BANKS		32
#define LOG_NUM_BANKS	 4

#ifndef CUDA_KERNEL
extern "C" {
	ParticleParams	simData;
	uint		gridActive;
}		
__global__ void insertParticles ( bufList buf, int pnum );
__global__ void countingSortIndex ( bufList buf, int pnum );		
__global__ void countingSortFull ( bufList buf, int pnum );		
__global__ void computeDensityPressureSPH ( bufList buf, int pnum );		
__global__ void ComputeForceCUDASPH( bufList buf, int pnum );
__global__ void ComputeOtherForceCUDAPCISPH( bufList buf, int pnum );
__global__ void PredictPositionAndVelocityCUDAPCISPH(bufList buf, int pnum, float time_step);
__global__ void ComputePredictedDensityAndPressureCUDAPCISPH(bufList buf, int pnum);
__global__ void ComputeCorrectivePressureForceCUDAPCISPH(bufList buf, int pnum);
__global__ void advanceParticlesCUDA ( float time_step, float sim_scale, bufList buf, int numPnts );
__global__ void advanceParticlesCUDASimpleCollision ( float time_step, float sim_scale, bufList buf, int numPnts );
__global__ void advanceParticlesPCISPH( float time, float dt, float sim_scale, bufList buf, int numPnts );
__global__ void advanceParticlesPCISPHSimpleCollision( float time_step, float sim_scale, bufList buf, int numPnts );
__global__ void GetMaxValue(float* idata, int numPnts, float* max_predicted_density);

__global__ void countActiveCells ( bufList buf, int pnum );	

#include "prefix_sum.cu"

template <bool storeSum, bool isNP2> __global__ void prescan(float *g_odata, const float *g_idata, float *g_blockSums, int n, int blockIndex, int baseIndex) {
	int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
	extern __shared__ float s_data[];
	loadSharedChunkFromMem<isNP2>(s_data, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
	prescanBlock<storeSum>(s_data, blockIndex, g_blockSums); 
	storeSharedChunkToMem<isNP2>(g_odata, s_data, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
}
template <bool storeSum, bool isNP2> __global__ void prescanInt (int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex) {
	int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
	extern __shared__ int s_dataInt [];

	loadSharedChunkFromMemInt <isNP2>(s_dataInt, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
	prescanBlockInt<storeSum>(s_dataInt, blockIndex, g_blockSums); 
	storeSharedChunkToMemInt <isNP2>(g_odata, s_dataInt, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
}
__global__ void uniformAddInt (int*  g_data, int *uniforms, int n, int blockOffset, int baseIndex);	
__global__ void uniformAdd    (float*g_data, float *uniforms, int n, int blockOffset, int baseIndex);	
#endif


#define EPSILON				0.00001f
#define GRID_UCHAR			0xFF
#define GRID_UNDEF			0xFFFFFFFF


#endif
