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

#define CUDA_KERNEL
#include "fluid_system_kern.cuh"
#include "cudaHeaders.cuh"
#include "common_defs.h"

#include "cutil_math.h"
#include "sm_35_atomic_functions.h"
#include "radixsort.cu"						

__constant__ ParticleParams		simData;
__device__ uint					gridActive;
#define LUT_SIZE_CUDA 100000

__device__ int GetGridCell ( float3 pos , float3 gridMin, int3 gridRes, float cellSize, int3& grid_cell)
{
	float px = pos.x - gridMin.x;
	float py = pos.y - gridMin.y;
	float pz = pos.z - gridMin.z;

	if(px < 0.0)
		px = 0.0;
	if(py < 0.0)
		py = 0.0;
	if(pz < 0.0)
		pz = 0.0;

	grid_cell.x = (int)(px / cellSize);
	grid_cell.y = (int)(py / cellSize);
	grid_cell.z = (int)(pz / cellSize);

	if(grid_cell.x > gridRes.x - 1)
		grid_cell.x = gridRes.x - 1;
	if(grid_cell.y > gridRes.y - 1)
		grid_cell.y = gridRes.y - 1;
	if(grid_cell.z > gridRes.z - 1)
		grid_cell.z = gridRes.z - 1;

	return (int)(grid_cell.y * gridRes.x  * gridRes.z + grid_cell.z * gridRes.x + grid_cell.x);	//cell index 的计算应该保持一致性
}

__device__ void collisionHandlingSimScaleCUDA(float3* pos, float3* vel)
{
	const float  sim_scale		= simData.param_sim_scale;
	const float3 vec_bound_min  = simData.param_bound_min * sim_scale;
	const float3 vec_bound_max  = simData.param_bound_max * sim_scale;

	float damping = 0.1;

	float reflect = 1.1;

	// 边界碰撞处理
	if (pos->x < vec_bound_min.x) 
	{
		pos->x = vec_bound_min.x;
		float3 axis = make_float3(-1, 0, 0);
		*vel -= axis * dot(axis,*vel) * reflect;
		vel->x *=  damping;
	}

	if (pos->x > vec_bound_max.x) 
	{
		pos->x = vec_bound_max.x;
		float3 axis = make_float3(1, 0, 0);
		*vel -= axis * dot(axis,*vel) * reflect;
		vel->x *=  damping;
	}

	if (pos->y < vec_bound_min.y)
	{
		pos->y = vec_bound_min.y;
		float3 axis = make_float3(0, -1, 0);
		*vel -= axis * dot(axis,*vel) * reflect;
		vel->y *=  damping;
	}

	if (pos->y > vec_bound_max.y) 
	{
		pos->y = vec_bound_max.y;		
		float3 axis = make_float3(0, 1, 0);
		*vel -= axis * dot(axis,*vel) * reflect;
		vel->y *=  damping;
	}

	if (pos->z < vec_bound_min.z) 
	{
		pos->z = vec_bound_min.z;
		float3 axis = make_float3(0, 0, -1);
		*vel -= axis * dot(axis,*vel) * reflect;
		vel->z *=  damping;
	}

	if (pos->z > vec_bound_max.z) 
	{
		pos->z = vec_bound_max.z;
		float3 axis = make_float3(0, 0, 1);
		*vel -= axis * dot(axis,*vel) * reflect;
		vel->z *=  damping;
	}
}

__device__ inline void boxBoundaryForce(const float3& position, float3& force )
{
	const float  sim_scale		= simData.param_sim_scale;
	const float3 vec_bound_min  = simData.param_bound_min * sim_scale;
	const float3 vec_bound_max  = simData.param_bound_max * sim_scale; 

	if( position.x < vec_bound_min.x + simData.param_force_distance )
	{
		force += (make_float3(1.0,0.0,0.0) * ((vec_bound_min.x + simData.param_force_distance - position.x) * simData.param_inv_force_distance * 2.0 * simData.param_max_boundary_force)); 
	}
	if( position.x > vec_bound_max.x - simData.param_force_distance )
	{
		force += (make_float3(-1.0,0.0,0.0) * ((position.x + simData.param_force_distance - vec_bound_max.x) * simData.param_inv_force_distance * 2.0 * simData.param_max_boundary_force));
	}

	if( position.y < vec_bound_min.y + simData.param_force_distance )
	{
		force += (make_float3(0.0,1.0,0.0) * ((vec_bound_min.y + simData.param_force_distance - position.y) * simData.param_inv_force_distance * 2.0 * simData.param_max_boundary_force));
	}
	if( position.y > vec_bound_max.y - simData.param_force_distance )
	{
		force += (make_float3(0.0,-1.0,0.0) * ((position.y + simData.param_force_distance - vec_bound_max.y) * simData.param_inv_force_distance * 2.0 * simData.param_max_boundary_force));
	}

	if( position.z < vec_bound_min.z + simData.param_force_distance )
	{
		force += (make_float3(0.0,0.0,1.0) * ((vec_bound_min.z + simData.param_force_distance - position.z) * simData.param_inv_force_distance * 2.0 * simData.param_max_boundary_force));
	}

	if( position.z > vec_bound_max.z - simData.param_force_distance )
	{
		force += (make_float3(0.0,0.0,-1.0) * ((position.z + simData.param_force_distance - vec_bound_max.z) * simData.param_inv_force_distance * 2.0 * simData.param_max_boundary_force));
	}
}

__device__ int g_mutex = 0;

// GPU 同步函数
__device__ void __gpu_sync(int goalVal)
{

	__threadfence ();

	// 仅将线程0用于同步
	if (threadIdx.x == 0) 
		atomicAdd(&g_mutex, 1);

	while(g_mutex < goalVal) {			
	}

	if ( blockIdx.x == 0 && threadIdx.x == 0 ) g_mutex = 0;
	
	__syncthreads();
}



__global__ void insertParticles ( bufList buf, int pnum )
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;					
	if ( index >= pnum ) 
		return;

	const	float3			gridMin			= simData.param_grid_min;
	const	float3			gridDelta		= simData.param_grid_delta;
	const	int3			gridRes			= simData.param_grid_res;
	const	int3			gridScan		= simData.param_grid_scan_max;
	const	float			sim_scale		= simData.param_sim_scale;
	const	float			poffset			= simData.param_smooth_radius / sim_scale;
	const	float			cellSize		= simData.param_grid_cell_size / sim_scale;
	
	int3   gridCellID;
	int    gridCellIndex    = GetGridCell(buf.pos[index], gridMin, gridRes, cellSize, gridCellID);
	
	if ( gridCellID.x >= 0 && gridCellID.x < gridScan.x && gridCellID.y >= 0 && gridCellID.y < gridScan.y && gridCellID.z >= 0 && gridCellID.z < gridScan.z ) {
		buf.particle_grid_cell_index[index] = gridCellIndex;												
		buf.grid_particle_offset[index] = atomicAdd ( &buf.grid_particles_num[ gridCellIndex ], 1 );		

	} else {
		buf.particle_grid_cell_index[index] = GRID_UNDEF;
	}
}

__global__ void countingSortIndex ( bufList buf, int pnum )
{
	uint tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;				
	if ( tid >= pnum ) 
		return;

	uint icell_idx = buf.particle_grid_cell_index[tid];
	uint particle_offset =  buf.grid_particle_offset[tid];
	int sort_ndx = buf.grid_off[ icell_idx ] + particle_offset;		
	if ( icell_idx != GRID_UNDEF ) {
		buf.particle_index_grid[ sort_ndx ] = tid;					
	}
}

// 计数排序
__global__ void countingSortFull ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;												
	if ( i >= pnum ) return;

	uint cell_index = *(uint*) (buf.sort_buf + pnum*BUF_PARTICLE_GRID_CELL_INDEX + i*sizeof(uint) );
	uint grid_particle_offset =  *(uint*) (buf.sort_buf + pnum*BUF_GRID_PARTICLE_OFFSET + i*sizeof(uint) );
	int sort_index = buf.grid_off[ cell_index ] + grid_particle_offset;									
	if ( cell_index != GRID_UNDEF ) {
		buf.particle_index_grid[sort_index]				= sort_index;														
		char* bpos										= buf.sort_buf + i*sizeof(float3);
		buf.pos[sort_index]								= *(float3*) (bpos);
		buf.predicted_pos[sort_index]					= *(float3*) (bpos + pnum*BUF_PREDICTED_POS );
		buf.vel[sort_index]								= *(float3*) (bpos + pnum*BUF_VEL );
		buf.vel_eval[sort_index]						= *(float3*) (bpos + pnum*BUF_VELEVAL );
		buf.correction_pressure_force[sort_index]		= *(float3*) (bpos + pnum*BUF_CORRECTION_PRESSURE_FORCE );
		buf.force[sort_index]							= *(float3*) (bpos + pnum*BUF_FORCE );
		buf.press[sort_index]							= *(float*) (buf.sort_buf + pnum*BUF_PRESS + i*sizeof(float) );
		buf.correction_pressure[sort_index]				= *(float*) (buf.sort_buf + pnum*BUF_PRESS + i*sizeof(float) );
		buf.fluid_density[sort_index]					= *(float*) (buf.sort_buf + pnum*BUF_DENS + i*sizeof(float) );
		buf.predicted_density[sort_index]				= *(float*) (buf.sort_buf + pnum*BUF_PREDICTED_DEN + i*sizeof(float) );
		buf.densityError[sort_index]					= *(float*) (buf.sort_buf + pnum*BUF_DENSITY_ERROR + i*sizeof(float) );
		buf.max_predicted_density_array[sort_index]		= *(float*) (buf.sort_buf + pnum*BUF_MAX_PREDICTED_DENS + i*sizeof(float) );
		buf.clr[sort_index]								= *(uint*) (buf.sort_buf + pnum*BUF_CLR+ i*sizeof(uint) );			
		buf.particle_grid_cell_index[sort_index]		= cell_index;
		buf.grid_particle_offset[sort_index]			= grid_particle_offset;		
	}
}

__global__ void countActiveCells ( bufList buf, int pnum )
{
	
	if ( threadIdx.x == 0 ) {				
		gridActive = -1;

		int last_ndx = buf.grid_off [ simData.param_grid_total-1 ] + buf.grid_particles_num[ simData.param_grid_total-1 ] - 1;
		int last_p = buf.particle_index_grid[ last_ndx ];
		int last_cell = buf.particle_grid_cell_index[ last_p ];
		int first_p = buf.particle_index_grid[ 0 ];
		int first_cell = buf.particle_grid_cell_index[ first_p ] ;

		int id, cell, cnt = 0, curr = 0;
		cell = first_cell;
		while ( cell < last_cell ) {			
			buf.grid_active[ cnt ] = cell;		
			cnt++;
			curr += buf.grid_particles_num[cell];						
			cell = buf.particle_grid_cell_index [ curr ];			
		}
		gridActive = cnt;
	}
	__syncthreads();
}

__device__ float kernelM4CUDA(float dist, float sr)
{
	float s = dist / sr;
	float result;
	float factor = 2.546479089470325472f / (sr * sr * sr);
	if(dist < 0.0f || dist >= sr)
		return 0.0f;
	else
	{
		if(s < 0.5f)
		{
			result = 1.0f - 6.0 * s * s + 6.0f * s * s * s;
		}
		else
		{
			float tmp = 1.0f - s;
			result = 2.0 * tmp * tmp * tmp;
		}
	}
	return factor * result;
}

__device__ float kernelM4LutCUDA(float dist, float sr)
{
	int index = dist / sr * LUT_SIZE_CUDA;

	if(index >= LUT_SIZE_CUDA ) return 0.0f;
	else return kernelM4CUDA(index*sr/LUT_SIZE_CUDA, sr);
}

__global__ void computeDensityPressureSPH ( bufList buf, int pnum )
{
	uint tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;								
	if ( tid >= pnum ) 
		return;

	uint i_cell_index = buf.particle_grid_cell_index[ tid ];
	if ( i_cell_index == GRID_UNDEF ) 
		return;									

	const float3 ipos								= buf.pos[ tid ];
	const float  sim_scale_square					= simData.param_sim_scale * simData.param_sim_scale;
	const float  smooth_radius						= simData.param_smooth_radius;
	const float  smooth_radius_square				= smooth_radius * smooth_radius;	
	const float  mass								= simData.param_mass;

	float sum  = 0.0;
	for (int cell=0; cell < simData.param_grid_adj_cnt; cell++) {
		int neighbor_cell_index = i_cell_index + simData.param_grid_neighbor_cell_index_offset[cell];
	
		if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > simData.param_grid_total - 1))
		{
			continue;
		}	

	
		if ( buf.grid_particles_num[neighbor_cell_index] == 0 ) 
		{
			continue;
		}
		int cell_start = buf.grid_off[ neighbor_cell_index ];
		int cell_end = cell_start + buf.grid_particles_num[ neighbor_cell_index ];
		for ( int idx = cell_start; idx < cell_end; idx++ ) {
			int j = buf.particle_index_grid[idx];
			if ( tid==j )		
			{
				continue;
			}
			float3 vector_i_minus_j = ipos - buf.pos[j];	
			const float dx = vector_i_minus_j.x;
			const float dy = vector_i_minus_j.y;
			const float dz = vector_i_minus_j.z;
			const float dist_square_sim_scale = sim_scale_square*(dx*dx + dy*dy + dz*dz);
			if ( dist_square_sim_scale <= smooth_radius_square) {
				const float dist = sqrt(dist_square_sim_scale);
				float kernelValue = kernelM4LutCUDA(dist, smooth_radius);
				sum += kernelValue * mass;
			} 
		}
	}
	
	sum += simData.param_kernel_self * mass;	
	buf.fluid_density[ tid ] = sum;	

	buf.press[ tid ] = max(0.0f, ( sum - simData.param_rest_dens ) * simData.param_gas_constant);	
}

__device__ float3 ComputeBoundaryForce(int i, float sim_scale, bufList buf, int numPnts)
{
	if ( buf.particle_grid_cell_index[i] == GRID_UNDEF ) {
		buf.pos[i] = make_float3(-1000,-1000,-1000);
		buf.vel[i] = make_float3(0,0,0);
		return;
	}

	register float3 norm;
	register float  adj;
	register float3 pos = buf.pos[i];
	register float3 veval = buf.vel_eval[i];

	register float3 force = make_float3(0.0, 0.0, 0.0);

	register float diff = simData.param_radius - (pos.y - (simData.param_bound_min.y + (pos.x-simData.param_bound_min.x)*simData.param_ground_slope )) * sim_scale;
	if ( diff > EPSILON ) {
		norm = make_float3( -simData.param_ground_slope, 1.0 - simData.param_ground_slope, 0);
		adj = simData.param_ext_stiff * diff - simData.param_damp * dot(norm, veval );
		norm *= adj; force += norm;
	}

	diff = simData.param_radius - ( simData.param_bound_max.y - pos.y )*sim_scale;
	if ( diff > EPSILON ) {
		norm = make_float3(0, -1, 0);
		adj = simData.param_ext_stiff * diff - simData.param_damp * dot(norm, veval );
		norm *= adj; force += norm;
	}

	diff = simData.param_radius - (pos.x - (simData.param_bound_min.x + (sin(simData.param_force_freq)+1)*0.5 * simData.param_force_min))*sim_scale;
	if ( diff > EPSILON ) {
		norm = make_float3( 1, 0, 0);
		adj = (simData.param_force_min+1) * simData.param_ext_stiff * diff - simData.param_damp * dot(norm, veval );
		norm *= adj; force += norm;
	}
	diff = simData.param_radius - ( (simData.param_bound_max.x - (sin(simData.param_force_freq)+1)*0.5*simData.param_force_max) - pos.x)*sim_scale;
	if ( diff > EPSILON ) {
		norm = make_float3(-1, 0, 0);
		adj = (simData.param_force_max+1) * simData.param_ext_stiff * diff - simData.param_damp * dot(norm, veval );
		norm *= adj; force += norm;
	}

	diff = simData.param_radius - (pos.z - simData.param_bound_min.z ) * sim_scale;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, 1 );
		adj = simData.param_ext_stiff * diff - simData.param_damp * dot(norm, veval );
		norm *= adj; force += norm;
	}
	diff = simData.param_radius - ( simData.param_bound_max.z - pos.z )*sim_scale;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, -1 );
		adj = simData.param_ext_stiff * diff - simData.param_damp * dot(norm, veval );
		norm *= adj; force += norm;
	}

	return force;
}

__device__ float kernelPressureGradCUDA(float dist, float sr)
{
	if(dist == 0)
		return 0.0f;
	if(dist > sr)
		return 0.0f;

	float kernelPressureConst = -45.f/((float(MY_PI)*sr*sr*sr*sr*sr*sr));
	return kernelPressureConst / dist * (sr-dist)*(sr-dist);
}

__device__ float kernelPressureGradLut(float dist, float sr)
{
	int index = dist / sr * LUT_SIZE_CUDA;
	if(index >= LUT_SIZE_CUDA) return 0.0f;
	else return kernelPressureGradCUDA(index*sr/LUT_SIZE_CUDA, sr);
}

__device__ float kernelViscosityLaplacian(float dist, float sr)
{
	if(dist > sr)
		return 0.0f;
	float kernelViscosityConst = 45.f/((float(MY_PI)*sr*sr*sr*sr*sr*sr));
	return kernelViscosityConst * (sr - dist);
}

__global__ void ComputeForceCUDASPH( bufList buf, int pnum)
{			
	uint tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		
	if ( tid >= pnum ) 
		return;

	uint i_cell_index = buf.particle_grid_cell_index[ tid ];
	if ( i_cell_index == GRID_UNDEF )
		return;										

	const float3 ipos					= buf.pos[ tid ];
	const float3 iveleval				= buf.vel_eval[ tid ];
	const float  ipress					= buf.press[ tid ];
	const float  idensity				= buf.fluid_density[ tid ];
	
	const float  mass					= simData.param_mass;
	const float	 sim_scale				= simData.param_sim_scale;
	const float  sim_scale_square		= sim_scale * sim_scale;	
	const float  smooth_radius			= simData.param_smooth_radius;
	const float  smooth_radius_square	= smooth_radius * smooth_radius;	
	const float  vterm					= simData.param_lapkern * simData.param_visc;
	const float  pVol					= mass / idensity;

	float3 force						= make_float3(0,0,0);	

	for (int cell=0; cell < simData.param_grid_adj_cnt; cell++) 
	{
		int neighbor_cell_index = i_cell_index + simData.param_grid_neighbor_cell_index_offset[cell];
		
		if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > simData.param_grid_total - 1))
		{
			continue;
		}	

		if ( buf.grid_particles_num[neighbor_cell_index] == 0 )		
		{
			continue;
		}
		int cell_start = buf.grid_off[ neighbor_cell_index ];
		int cell_end = cell_start + buf.grid_particles_num[ neighbor_cell_index ];
		for ( int cndx = cell_start; cndx < cell_end; cndx++ ) 
		{										
			int j = buf.particle_index_grid[ cndx ];		
			if ( tid==j )		
			{
				continue;
			}
			float3 vector_i_minus_j = ( ipos - buf.pos[ j ] );	
			const float dx = vector_i_minus_j.x;
			const float dy = vector_i_minus_j.y;
			const float dz = vector_i_minus_j.z;
			const float dist_square_sim_scale = sim_scale_square*(dx*dx + dy*dy + dz*dz);
		
			if ( dist_square_sim_scale < smooth_radius_square && dist_square_sim_scale > 0) {	
				float jdist = sqrt(dist_square_sim_scale);
				float kernelGradientValue = kernelPressureGradLut(jdist, smooth_radius);
				float3 kernelGradient = vector_i_minus_j * sim_scale * kernelGradientValue;
				
				float grad = 0.5f * (ipress + buf.press[j])/(idensity * buf.fluid_density[j]);
				force -= kernelGradient * grad * mass * mass;

				float kernelVisc = kernelViscosityLaplacian(jdist, smooth_radius);
				float3 v_ij = iveleval - buf.vel_eval[ j ];
				float nVol = mass / buf.fluid_density[j];
				force -= v_ij * pVol * nVol * simData.param_visc * kernelVisc;
			}	
		}
	}

	if (simData.param_add_boundary_force)
	{
		boxBoundaryForce(ipos, force);
	}

	buf.force[ tid ] = force + simData.param_gravity * mass;
}

__global__ void ComputeOtherForceCUDAPCISPH( bufList buf, int pnum)
{		
	uint tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;				
	if ( tid >= pnum ) 
		return;

	uint i_cell_index = buf.particle_grid_cell_index[ tid ];
	if ( i_cell_index == GRID_UNDEF )	
		return;		

	const float3   ipos					= buf.pos[ tid ];
	const float3   iveleval				= buf.vel_eval[ tid ];
	const float	   sim_scale			= simData.param_sim_scale;
	const float    sim_scale_square		= sim_scale * sim_scale;	
	const float    smooth_radius		= simData.param_smooth_radius;
	const float    smooth_radius_square	= smooth_radius * smooth_radius;	
	const float	   mass					= simData.param_mass;
	const float    vterm				= simData.param_lapkern * simData.param_visc;
	const float    restVolume			= mass / simData.param_rest_dens;

	float3 force = make_float3(0,0,0);	

	for (int cell=0; cell < simData.param_grid_adj_cnt; cell++) {
		int neighbor_cell_index = i_cell_index + simData.param_grid_neighbor_cell_index_offset[cell];
	
		if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > simData.param_grid_total - 1))
		{
			continue;
		}	

		if ( buf.grid_particles_num[neighbor_cell_index] == 0 )			
		{
			continue;
		}

		int cell_start = buf.grid_off[ neighbor_cell_index ];
		int cell_end = cell_start + buf.grid_particles_num[ neighbor_cell_index ];
		for ( int cndx = cell_start; cndx < cell_end; cndx++ ) {	
			int j = buf.particle_index_grid[ cndx ];		
			if ( tid==j )		
			{
				continue;
			}			
			float3 vector_i_minus_j = ( ipos - buf.pos[ j ] );		
			const float dx = vector_i_minus_j.x;
			const float dy = vector_i_minus_j.y;
			const float dz = vector_i_minus_j.z;
			const float dist_square_sim_scale = sim_scale_square*(dx*dx + dy*dy + dz*dz);
			if ( dist_square_sim_scale < smooth_radius_square && dist_square_sim_scale > 0) {	
				float jdist = sqrt(dist_square_sim_scale);
				float kernelVisc = kernelViscosityLaplacian(jdist, smooth_radius);
				float3 v_ij = iveleval - buf.vel_eval[ j ];
				force -= v_ij * restVolume * restVolume * simData.param_visc * kernelVisc;
			}	
		}
	}

	force = force + simData.param_gravity * mass;

	if (simData.param_add_boundary_force)
	{
		boxBoundaryForce(ipos, force);
	}

	buf.force[ tid ] = force;

	buf.correction_pressure[tid] = 0.0f;
	buf.correction_pressure_force[tid] = make_float3(0, 0, 0);
}

__global__ void PredictPositionAndVelocityCUDAPCISPH(bufList buf, int pnum, float time_step)
{
	uint tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;				
	if ( tid >= pnum ) 
		return;

	const float	   sim_scale			= simData.param_sim_scale;

	float3 acceleration = (buf.force[tid] + buf.correction_pressure_force[tid]) * (1.0f / simData.param_mass);
	float3 predictedVelocity = buf.vel_eval[tid] + acceleration * time_step;

	float3 pos = buf.pos[tid] * sim_scale + predictedVelocity * time_step;		

	collisionHandlingSimScaleCUDA(&pos, &predictedVelocity);

	buf.predicted_pos[tid] = pos;

}

__global__ void ComputePredictedDensityAndPressureCUDAPCISPH(bufList buf, int pnum)
{
	uint tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	
	if ( tid >= pnum ) 
		return;

	uint i_cell_index = buf.particle_grid_cell_index[ tid ];
	if ( i_cell_index == GRID_UNDEF )
		return;	

	const float3 ipredicted_pos						= buf.predicted_pos[tid];
	const float  smooth_radius						= simData.param_smooth_radius;
	const float  smooth_radius_square				= smooth_radius * smooth_radius;	
	const float  sim_scale_square					= simData.param_sim_scale * simData.param_sim_scale;
	const float  mass								= simData.param_mass;

	float predictedSPHDensity  = 0.0;
	for (int cell=0; cell < simData.param_grid_adj_cnt; cell++) 
	{
		int neighbor_cell_index = i_cell_index + simData.param_grid_neighbor_cell_index_offset[cell];
		
		if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > simData.param_grid_total - 1))
		{
			continue;
		}	

		if ( buf.grid_particles_num[neighbor_cell_index] == 0 )		
		{
			continue;
		}

		int cell_start = buf.grid_off[ neighbor_cell_index ];
		int cell_end = cell_start + buf.grid_particles_num[ neighbor_cell_index ];
		for ( int cndx = cell_start; cndx < cell_end; cndx++ )
		{										
			int j = buf.particle_index_grid[ cndx ];		
			if ( tid==j )	
			{
				continue;
			}
			float3 vector_i_minus_j = ipredicted_pos - buf.predicted_pos[j];		
			const float dx = vector_i_minus_j.x;
			const float dy = vector_i_minus_j.y;
			const float dz = vector_i_minus_j.z;
			const float dist_square_sim_scale = dx*dx + dy*dy + dz*dz;
			if ( dist_square_sim_scale <= smooth_radius_square && dist_square_sim_scale > 0) 
			{
				const float dist = sqrt(dist_square_sim_scale);
				float kernelValue = kernelM4LutCUDA(dist, smooth_radius);
				predictedSPHDensity += kernelValue * mass;
			}
		}
	}

	predictedSPHDensity += simData.param_kernel_self * mass;	

	buf.densityError[tid] = max(predictedSPHDensity - simData.param_rest_dens, 0.0f);

	buf.correction_pressure[tid] += max( buf.densityError[tid] * simData.param_density_error_factor, 0.0f);
	
	buf.predicted_density[tid] = predictedSPHDensity;
}

__global__ void ComputeCorrectivePressureForceCUDAPCISPH(bufList buf, int pnum)
{
	uint tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		
	if ( tid >= pnum) 
		return;

	uint i_cell_index = buf.particle_grid_cell_index[ tid ];
	if ( i_cell_index == GRID_UNDEF )
		return;	

	const float3 ipos					= buf.pos[ tid ];
	const float3 iveleval				= buf.vel_eval[ tid ];
	const float  ipress					= buf.correction_pressure[ tid ];

	const float  mass					= simData.param_mass;
	const float	 sim_scale				= simData.param_sim_scale;
	const float  sim_scale_square		= sim_scale * sim_scale;	
	const float  smooth_radius			= simData.param_smooth_radius;
	const float  smooth_radius_square	= smooth_radius * smooth_radius;	
	const float  rest_volume			= mass / simData.param_rest_dens;

	float3 force						= make_float3(0,0,0);

	for (int cell=0; cell < simData.param_grid_adj_cnt; cell++) 
	{
		int neighbor_cell_index = i_cell_index + simData.param_grid_neighbor_cell_index_offset[cell];
	
		if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > simData.param_grid_total - 1))
		{
			continue;
		}	

		if ( buf.grid_particles_num[neighbor_cell_index] == 0 )			
		{
			continue;
		}
		int cell_start = buf.grid_off[ neighbor_cell_index ];
		int cell_end = cell_start + buf.grid_particles_num[ neighbor_cell_index ];
		for ( int cndx = cell_start; cndx < cell_end; cndx++ ) 
		{
			int j = buf.particle_index_grid[ cndx ];		
			if ( tid==j )		
			{
				continue;
			}
			float3 vector_i_minus_j = ( ipos - buf.pos[ j ] );		
			const float dx = vector_i_minus_j.x;
			const float dy = vector_i_minus_j.y;
			const float dz = vector_i_minus_j.z;
			const float dist_square_sim_scale = sim_scale_square*(dx*dx + dy*dy + dz*dz);
	
			if ( dist_square_sim_scale < smooth_radius_square && dist_square_sim_scale > 0) 
			{
				float jdist = sqrt(dist_square_sim_scale);
				float kernelGradientValue = kernelPressureGradLut(jdist, smooth_radius);
				float3 kernelGradient = vector_i_minus_j * sim_scale * kernelGradientValue;			
				float grad = 0.5f * (ipress + buf.correction_pressure[j]) * rest_volume * rest_volume;
				force -= kernelGradient * grad;
			}
		}
	}

	buf.correction_pressure_force[tid] = force;
}

__global__ void advanceParticlesCUDA ( float time_step, float sim_scale, bufList buf, int numPnts)
{		
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		
	if ( i >= numPnts ) 
		return;

	register float3 norm;
	register float adj;
	register float3 pos = buf.pos[i];
	register float3 veval = buf.vel_eval[i];
					
	register float3 accel = buf.force[i] * (1.0f / simData.param_mass);

	register float diff = simData.param_radius - (pos.y - (simData.param_bound_min.y + (pos.x-simData.param_bound_min.x)*simData.param_ground_slope )) * sim_scale;
	if ( diff > EPSILON ) {
		norm = make_float3( -simData.param_ground_slope, 1.0 - simData.param_ground_slope, 0);
		adj = simData.param_ext_stiff * diff - simData.param_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	diff = simData.param_radius - ( simData.param_bound_max.y - pos.y )*sim_scale;
	if ( diff > EPSILON ) {
		norm = make_float3(0, -1, 0);
		adj = simData.param_ext_stiff * diff - simData.param_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	diff = simData.param_radius - (pos.x - (simData.param_bound_min.x + (sin(simData.param_force_freq)+1)*0.5 * simData.param_force_min))*sim_scale;
	if ( diff > EPSILON ) {
		norm = make_float3( 1, 0, 0);
		adj = (simData.param_force_min+1) * simData.param_ext_stiff * diff - simData.param_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	diff = simData.param_radius - ( (simData.param_bound_max.x - (sin(simData.param_force_freq)+1)*0.5*simData.param_force_max) - pos.x)*sim_scale;
	if ( diff > EPSILON ) {
		norm = make_float3(-1, 0, 0);
		adj = (simData.param_force_max+1) * simData.param_ext_stiff * diff - simData.param_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	diff = simData.param_radius - (pos.z - simData.param_bound_min.z ) * sim_scale;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, 1 );
		adj = simData.param_ext_stiff * diff - simData.param_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
	diff = simData.param_radius - ( simData.param_bound_max.z - pos.z )*sim_scale;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, -1 );
		adj = simData.param_ext_stiff * diff - simData.param_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	accel += simData.param_gravity;

	register float speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
	if ( speed > simData.param_acc_limit_square ) {
		accel *= simData.param_acc_limit / sqrt(speed);
	}

	float3 vel = buf.vel[i];
	speed = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
	if ( speed > simData.param_vel_limit_square ) {
		speed = simData.param_vel_limit_square;
		vel *= simData.param_vel_limit / sqrt(speed);
	}

	if ( speed > simData.param_vel_limit_square*0.2) {
		adj = simData.param_vel_limit_square*0.2;
		buf.clr[i] += ((  buf.clr[i] & 0xFF) < 0xFD ) ? +0x00000002 : 0;		
		buf.clr[i] += (( (buf.clr[i]>>8) & 0xFF) < 0xFD ) ? +0x00000200 : 0;	
		buf.clr[i] += (( (buf.clr[i]>>16) & 0xFF) < 0xFD ) ? +0x00020000 : 0;	
	}
	if ( speed < 0.03 ) {		
		int v = int(speed/.01)+1;
		buf.clr[i] += ((  buf.clr[i] & 0xFF) > 0x80 ) ? -0x00000001 * v : 0;	
		buf.clr[i] += (( (buf.clr[i]>>8) & 0xFF) > 0x80 ) ? -0x00000100 * v : 0;
	}

	float3 vnext = accel*time_step + vel;				
	buf.vel_eval[i] = (vel + vnext) * 0.5;					
	buf.vel[i] = vnext;
	buf.pos[i] += vnext * (time_step/sim_scale);		
}

__global__ void advanceParticlesCUDASimpleCollision ( float time_step, float sim_scale, bufList buf, int numPnts )
{		
	uint tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;				
	if ( tid >= numPnts ) 
		return;
	
	float3 acceleration = buf.force[tid] / simData.param_mass;

	float3 veval = buf.vel_eval[tid];
	veval += acceleration * time_step;

	float3 pos = buf.pos[tid] * simData.param_sim_scale;	
	pos += veval * time_step;

	collisionHandlingSimScaleCUDA(&pos, &veval);

	buf.pos[tid] = pos / simData.param_sim_scale;			
	buf.vel_eval[tid] = veval;
}

__global__ void advanceParticlesPCISPH( float time, float dt, float sim_scale, bufList buf, int numPnts )
{		
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;			
	if ( i >= numPnts ) return;

	if ( buf.particle_grid_cell_index[i] == GRID_UNDEF ) {
		buf.pos[i] = make_float3(-1000,-1000,-1000);
		buf.vel[i] = make_float3(0,0,0);
		return;
	}
				
	register float3 force = buf.force[i];

	force += ComputeBoundaryForce(i, sim_scale, buf, numPnts);

	register float speed = force.x*force.x + force.y*force.y + force.z*force.z;
	if ( speed > simData.param_acc_limit_square ) {
		force *= simData.param_acc_limit / sqrt(speed);
	}

	float3 vel = buf.vel[i];
	speed = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
	if ( speed > simData.param_vel_limit_square ) {
		speed = simData.param_vel_limit_square;
		vel *= simData.param_vel_limit / sqrt(speed);
	}

	if ( speed > simData.param_vel_limit_square*0.2) {
		float adj = simData.param_vel_limit_square*0.2;
		buf.clr[i] += ((  buf.clr[i] & 0xFF) < 0xFD ) ? +0x00000002 : 0;		
		buf.clr[i] += (( (buf.clr[i]>>8) & 0xFF) < 0xFD ) ? +0x00000200 : 0;	
		buf.clr[i] += (( (buf.clr[i]>>16) & 0xFF) < 0xFD ) ? +0x00020000 : 0;
	}
	if ( speed < 0.03 ) {		
		int v = int(speed/.01)+1;
		buf.clr[i] += ((  buf.clr[i] & 0xFF) > 0x80 ) ? -0x00000001 * v : 0;		
		buf.clr[i] += (( (buf.clr[i]>>8) & 0xFF) > 0x80 ) ? -0x00000100 * v : 0;
	}

	float3 vnext = force*dt + vel;					
	buf.vel_eval[i] = (vel + vnext) * 0.5;			
	buf.vel[i] = vnext;
	buf.pos[i] += vnext * (dt/sim_scale);		
}

__global__ void advanceParticlesPCISPHSimpleCollision(float time_step, float sim_scale, bufList buf, int numPnts )
{		
	uint tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;			
	if ( tid >= numPnts ) 
		return;

	float3 acceleration = (buf.force[tid] + buf.correction_pressure_force[tid]) / simData.param_mass;
	
	float3 veval = buf.vel_eval[tid];
	veval += acceleration * time_step;

	float3 pos = buf.pos[tid] * simData.param_sim_scale;
	pos += veval * time_step;

	collisionHandlingSimScaleCUDA(&pos, &veval);

	buf.pos[tid] = pos / simData.param_sim_scale;		
	buf.vel_eval[tid] = veval;
}

__global__ void GetMaxValue(float* idata, int numPnts, float* max_predicted_density)
{
	uint tid = threadIdx.x;
	if (tid == 0)
	{
		float maxValue = 0.0f;
		for (int i = 0; i < numPnts; ++i)
		{
			if(idata[i] > maxValue)
				maxValue = idata[tid];		
		}
		*max_predicted_density = maxValue;
	}
}