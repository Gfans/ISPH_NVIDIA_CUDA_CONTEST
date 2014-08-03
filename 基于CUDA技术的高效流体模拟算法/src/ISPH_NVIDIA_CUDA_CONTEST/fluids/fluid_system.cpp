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

#include "gl_helper.h"
#include <assert.h>
#include <stdio.h>
#include <conio.h>
#include <utility> 

#include "camera3d.h"

#include <gl/glut.h>

#include "common_defs.h"
#include "mtime.h"
#include "fluid_system.h"

#include "fluid_system_host.cuh"
#include "cutil.h"
#include "cuda_runtime_api.h"

#define EPSILON			0.00001f			
const uint NUM_PARTICLES = 4096;
const uint GRID_RESOLUTION = 1024;

void ParticleSystem::TransferToCUDA ()
{ 
	CopyToCUDA ( (float*) pos_, (float*) predictedPosition_, (float*) vel_, (float*) vel_eval_, (float*) correction_pressure_force_, (float*) force_, pressure_, correction_pressure_, 
		density_, predicted_density_, densityError_, max_predicted_density_array_, cluster_cell_, next_particle_index_in_the_same_cell_, (char*) clr_ ); 
}
void ParticleSystem::TransferFromCUDA ()	
{
	CopyFromCUDA ( (float*) pos_, (float*) predictedPosition_, (float*) vel_, (float*) vel_eval_, (float*) correction_pressure_force_, (float*) force_, pressure_, correction_pressure_,
		density_, predicted_density_, densityError_, max_predicted_density_array_, cluster_cell_, next_particle_index_in_the_same_cell_, (char*) clr_ ); 
}

ParticleSystem::ParticleSystem ()
{
	frame_ = 0;
	time_step_ = 0.003;
	time_ = 0;

	// 程序运行方法---6种方法可选
	param_ [ PRUN_MODE ] = RUN_CPU_SPH;	//RUN_CPU_SPH; RUN_CUDA_INDEX_SPH; RUN_CUDA_FULL_SPH; RUN_CPU_PCISPH; RUN_CUDA_INDEX_PCISPH; RUN_CUDA_FULL_PCISPH;

	// 是否加载3D模型数据
	toggle_ [ PUSELOADEDSCENE] = false;

	num_points_ = 0;
	max_points_ = 0;
	good_points_ = 0;

	param_[PEXAMPLE] = 0;

	pos_ = 0x0;
	predictedPosition_ = 0x0;
	vel_ = 0x0;
	vel_eval_ = 0x0;
	correction_pressure_force_ = 0x0;
	force_ = 0x0;
	sumGradW_ = 0x0;
	sumGradWDot_ = 0x0;
	pressure_ = 0x0;
	correction_pressure_  = 0x0;
	density_ = 0x0;
	predicted_density_ = 0x0;
	densityError_ = 0x0;
	max_predicted_density_array_ = 0x0;
	particle_grid_cell_index_ = 0x0;
	next_particle_index_in_the_same_cell_ = 0x0;
	index_ = 0x0;
	clr_ = 0x0;

	cluster_cell_ = 0x0;
	age_ = 0x0;
	neighbor_index_ = 0x0;
	neighbor_particle_numbers_ = 0x0;

	grid_head_cell_particle_index_array_ = 0x0;
	grid_particles_number_ = 0x0;
	grid_total_ = 0;
	grid_search_ = 0;
	grid_adj_cnt_ = 0;

	neighbor_particles_num_ = 0;
	neighbor_particles_max_num_ = 0;
	neighbor_table_ = 0x0;
	neighbor_dist_ = 0x0;

	pack_fluid_particle_buf_ = 0x0;
	pack_grid_buf_ = 0x0;

	selected_ = -1;

	if ( !xml.Load ( "scene.xml" ) ) {
		error.PrintF ( "fluid", "ERROR: Problem loading scene.xml. Check formatting.\n" );
		error.Exit ();
	}
}

ParticleSystem::~ParticleSystem()
{
	particles.clear();
}

void ParticleSystem::Setup(bool bStart)
{
	frame_ = 0;
	time_ = 0;							

	SetupDefaultParams ();

	SetupExampleParams ( bStart );

	param_ [ PGRIDSIZEREALSCALE] = param_[PSMOOTHRADIUS] / param_[PGRID_DENSITY];

	SetupKernels ();

	SetupSpacing ();

	computeGasConstAndTimeStep(param_ [ PMAXDENSITYERRORALLOWED]);	

	if( param_[PRUN_MODE] == RUN_CPU_PCISPH || param_[PRUN_MODE] == RUN_CUDA_INDEX_PCISPH || param_[PRUN_MODE] == RUN_CUDA_FULL_PCISPH)
	{
		time_step_ = time_step_pcisph_;
		ClearNeighborTable ();
		AllocateTemporalParticlesMemory(param_[PMAXNUM]);
		Vector3DF init_sample_particle_volume_min = Vector3DF(-20.0, 0.0, -20);
		Vector3DF init_sample_particle_volume_max = Vector3DF(20.0, 40.0, 20);
		SetupInitParticleVolume(init_sample_particle_volume_min, init_sample_particle_volume_max, param_[PSPACINGGRAPHICSWORLD], 0.1);		
		Vector3DF init_grid_volume_min = init_sample_particle_volume_min - (param_[PSMOOTHRADIUS] / param_[PSIMSCALE]);
		Vector3DF init_grid_volume_max = init_sample_particle_volume_max + (param_[PSMOOTHRADIUS] / param_[PSIMSCALE]);
		SetupSampleGridAllocatePCISPH ( init_grid_volume_min, init_grid_volume_max, param_[PSIMSCALE], param_[PGRIDSIZEREALSCALE], 1.0 );	
		ComputeDensityErrorFactor(num_points_);							
		DeallocateTemporalParticleMemory();								
	}

	ClearNeighborTable ();

	// 从文件中加载粒子位置信息,并设置其速度，颜色等属性值
	if(toggle_[PUSELOADEDSCENE] == true)
	{
		// 从文件中读取bunny模型数据
		particles.clear();
		const char* file_name = "models/bunny.txt";
		num_points_ = readInFluidParticleNum(file_name);

		Vector3DF minCorner = vec_[PINITPARTICLEMIN];
		Vector3DF maxCorner = vec_[PINITPARTICLEMAX];
		const float lengthX = maxCorner.x-minCorner.x;
		const float lengthY = maxCorner.y-minCorner.y;
		const float lengthZ = maxCorner.z-minCorner.z;

		const int numParticlesX  = ceil( lengthX / param_[PSPACINGGRAPHICSWORLD] );
		const int numParticlesY  = ceil( lengthY / param_[PSPACINGGRAPHICSWORLD] );
		const int numParticlesZ  = ceil( lengthZ / param_[PSPACINGGRAPHICSWORLD] );
		const int numParticles   = numParticlesX * numParticlesY * numParticlesZ;

		AllocateParticlesMemory(num_points_ + numParticles);

		Vector3DF minVec;		
		Vector3DF maxVec;		
		readInFluidParticles(file_name, num_points_, minVec, maxVec);
		SetupInitParticleVolumeFromFile(minVec, maxVec);
		SetupAdditonalParticleVolume ( vec_[PINITPARTICLEMIN], vec_[PINITPARTICLEMAX], param_[PSPACINGGRAPHICSWORLD], 0.1 , numParticles);

		num_points_ = num_points_ + numParticles;
	}
	else 
	{
		num_points_ = 0;

		AllocateParticlesMemory ( param_[PMAXNUM] );	

		SetupInitParticleVolume ( vec_[PINITPARTICLEMIN], vec_[PINITPARTICLEMAX], param_[PSPACINGGRAPHICSWORLD], 0.1 );		
	}			

	AllocatePackBuf ();

	SetupGridAllocate ( vec_[PGRIDVOLUMEMIN], vec_[PGRIDVOLUMEMAX], param_[PSIMSCALE], param_[PGRIDSIZEREALSCALE], 1.0 );			

#ifdef USE_CUDA

	ParticleClearCUDA();

	int3 grid_res = make_int3(grid_res_.x, grid_res_.y, grid_res_.z);
	float3 grid_size = make_float3(grid_size_.x, grid_size_.y, grid_size_.z);
	float3 grid_delta = make_float3(grid_delta_.x, grid_delta_.y, grid_delta_.z);
	float3 grid_min = make_float3(grid_min_.x, grid_min_.y, grid_min_.z);
	float3 grid_max = make_float3(grid_max_.x, grid_max_.y, grid_max_.z);
	ParticleSetupCUDA ( num_points(), grid_search_, grid_res, grid_size, grid_delta, grid_min, grid_max, grid_total_, (int) vec_[PEMIT_RATE].x , param_[PGRIDSIZEREALSCALE], param_[PKERNELSELF]);

	Vector3DF grav = vec_[PPLANE_GRAV_DIR];
	float3 boundMin = make_float3(vec_[PBOUNDARYMIN].x, vec_[PBOUNDARYMIN].y, vec_[PBOUNDARYMIN].z);
	float3 boundMax = make_float3(vec_[PBOUNDARYMAX].x, vec_[PBOUNDARYMAX].y, vec_[PBOUNDARYMAX].z);
	FluidParamCUDA ( param_[PSIMSCALE], param_[PSMOOTHRADIUS], param_[PCOLLISIONRADIUS], param_[PMASS], param_[PRESTDENSITY], boundMin, boundMax, 
		param_[PBOUNDARYSTIFF], param_[PGASCONSTANT], param_[PVISC], param_[PBOUNDARYDAMP], param_[PFORCE_MIN], param_[PFORCE_MAX], 
		param_[PFORCE_FREQ], param_[PGROUND_SLOPE], grav.x, grav.y, grav.z, param_[PACCEL_LIMIT], param_[PVEL_LIMIT], param_[PDENSITYERRORFACTOR]);

	TransferToCUDA ();		// 数据拷贝CPU -> GPU

#endif 
}

void ParticleSystem::ExitParticleSystem ()
{
	CUDA_SAFE_CALL(cudaFreeHost(pos_)); 
	CUDA_SAFE_CALL(cudaFreeHost( predictedPosition_ ));
	CUDA_SAFE_CALL(cudaFreeHost( vel_ ));
	CUDA_SAFE_CALL(cudaFreeHost( vel_eval_ ));
	CUDA_SAFE_CALL(cudaFreeHost( correction_pressure_force_ ));
	CUDA_SAFE_CALL(cudaFreeHost( force_ ));
	CUDA_SAFE_CALL(cudaFreeHost( sumGradW_));
	CUDA_SAFE_CALL(cudaFreeHost( sumGradWDot_));
	CUDA_SAFE_CALL(cudaFreeHost( pressure_ ));
	CUDA_SAFE_CALL(cudaFreeHost( correction_pressure_));
	CUDA_SAFE_CALL(cudaFreeHost( density_ ));
	CUDA_SAFE_CALL(cudaFreeHost( predicted_density_));
	CUDA_SAFE_CALL(cudaFreeHost( densityError_));
	CUDA_SAFE_CALL(cudaFreeHost( max_predicted_density_array_));
	CUDA_SAFE_CALL(cudaFreeHost( particle_grid_cell_index_ ));
	CUDA_SAFE_CALL(cudaFreeHost( next_particle_index_in_the_same_cell_ ));
	CUDA_SAFE_CALL(cudaFreeHost( index_));
	CUDA_SAFE_CALL(cudaFreeHost( clr_ ));

	CUDA_SAFE_CALL(cudaFreeHost( cluster_cell_ ));
	CUDA_SAFE_CALL(cudaFreeHost( age_ ));
	CUDA_SAFE_CALL(cudaFreeHost( neighbor_index_ ));
	CUDA_SAFE_CALL(cudaFreeHost( neighbor_particle_numbers_ ));

	ParticleClearCUDA();

	cudaExit (0,0);
}

// 分配临时内存，保存粒子信息
void ParticleSystem::AllocateTemporalParticlesMemory( uint num_particles )
{
	unsigned int flag = cudaHostAllocDefault;

	if (pos_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(pos_)); 
		pos_ = 0x0;
	}
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&pos_, num_particles*sizeof(Vector3DF), flag));

	if (next_particle_index_in_the_same_cell_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(next_particle_index_in_the_same_cell_)); 
		next_particle_index_in_the_same_cell_ = 0x0;
	}
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&next_particle_index_in_the_same_cell_, num_particles*sizeof(uint), flag));

	if (index_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(index_)); 
		index_ = 0x0;
	}
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&index_, num_particles*sizeof(uint), flag));

	if (particle_grid_cell_index_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(particle_grid_cell_index_)); 
		particle_grid_cell_index_ = 0x0;
	}
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&particle_grid_cell_index_, num_particles*sizeof(uint), flag));

	if (cluster_cell_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(cluster_cell_)); 
		cluster_cell_ = 0x0;
	}
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&cluster_cell_, num_particles*sizeof(uint), flag));

	if (vel_eval_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(vel_eval_)); 
		vel_eval_ = 0x0;
	}
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&vel_eval_, num_particles*sizeof(Vector3DF), flag));

	if (clr_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(clr_)); 
		clr_ = 0x0;
	}
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&clr_, num_particles*sizeof(DWORD), flag));

	if (sumGradW_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(sumGradW_)); 
		sumGradW_ = 0x0;
	}
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&sumGradW_, num_particles*sizeof(Vector3DF), flag));

	if (sumGradWDot_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(sumGradWDot_)); 
		sumGradWDot_ = 0x0;
	}
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&sumGradWDot_, num_particles*sizeof(float), flag));

	if (neighbor_particle_numbers_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(neighbor_particle_numbers_)); 
		neighbor_particle_numbers_ = 0x0;
	}
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&neighbor_particle_numbers_, num_particles*sizeof(uint), flag));

	if (neighbor_index_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(neighbor_index_)); 
		neighbor_index_ = 0x0;
	}
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&neighbor_index_, num_particles*sizeof(uint), flag));

	max_points_ = num_particles;
}

void ParticleSystem::DeallocateTemporalParticleMemory()
{
	if (pos_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(pos_)); 
		pos_ = 0x0;
	}

	if (next_particle_index_in_the_same_cell_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(next_particle_index_in_the_same_cell_)); 
		next_particle_index_in_the_same_cell_ = 0x0;
	}

	if (index_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(index_)); 
		index_ = 0x0;
	}

	if (particle_grid_cell_index_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(particle_grid_cell_index_)); 
		particle_grid_cell_index_ = 0x0;
	}

	if (cluster_cell_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(cluster_cell_)); 
		cluster_cell_ = 0x0;
	}

	if (sumGradW_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(sumGradW_)); 
		sumGradW_ = 0x0;
	}

	if (sumGradWDot_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(sumGradWDot_)); 
		sumGradWDot_ = 0x0;
	}

	if (neighbor_particle_numbers_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(neighbor_particle_numbers_)); 
		neighbor_particle_numbers_ = 0x0;
	}

	if ( grid_head_cell_particle_index_array_ != 0x0 ) 
	{
		free(grid_head_cell_particle_index_array_); 
		grid_head_cell_particle_index_array_ = 0x0;
	}

	if ( grid_particles_number_ != 0x0 ) 
	{
		free(grid_particles_number_); 
		grid_particles_number_ = 0x0;
	}

	if ( pack_grid_buf_ != 0x0 )
	{
		free(pack_grid_buf_); 
		pack_grid_buf_ = 0x0;
	}

	if (neighbor_index_ != 0x0)
	{
		CUDA_SAFE_CALL(cudaFreeHost(neighbor_index_)); 
		neighbor_index_ = 0x0;
	}

	num_points_ = 0;
	max_points_ = 0;
	grid_total_ = 0;

	grid_res_.Set(0, 0, 0);
	grid_min_.Set(0, 0, 0);
	grid_max_.Set(0, 0, 0);
	grid_size_.Set(0, 0, 0);
	grid_delta_.Set(0, 0, 0);

	grid_search_ = 0;
	grid_adj_cnt_ = 0;
	memset(grid_neighbor_cell_index_offset_, 0, max_num_adj_grid_cells_cpu * sizeof(int));
}

// 分配内存，保存粒子数据
void ParticleSystem::AllocateParticlesMemory ( int cnt )
{
	int nump = 0;		

	Vector3DF* srcPos = pos_;
	unsigned int flag = cudaHostAllocDefault;

	CUDA_SAFE_CALL(cudaHostAlloc((void**)&pos_, cnt*sizeof(Vector3DF), flag));
	if ( srcPos != 0x0 )
	{ 
		memcpy ( pos_, srcPos, nump *sizeof(Vector3DF)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcPos)); 
	}

	Vector3DF* src_predicted_pos = predictedPosition_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&predictedPosition_, cnt*sizeof(Vector3DF), flag));
	if ( src_predicted_pos != 0x0 )	
	{ 
		memcpy ( predictedPosition_, src_predicted_pos, nump *sizeof(Vector3DF)); 
		CUDA_SAFE_CALL(cudaFreeHost(src_predicted_pos)); 
	}

	Vector3DF* srcVel = vel_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&vel_, cnt*sizeof(Vector3DF), flag));
	if ( srcVel != 0x0 )	
	{ 
		memcpy ( vel_, srcVel, nump *sizeof(Vector3DF)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcVel)); 
	}

	Vector3DF* srcVelEval = vel_eval_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&vel_eval_, cnt*sizeof(Vector3DF), flag));
	if ( srcVelEval != 0x0 ) 
	{ 
		memcpy ( vel_eval_, srcVelEval, nump *sizeof(Vector3DF)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcVelEval)); 
	}

	Vector3DF* src_correction_pressure_force = correction_pressure_force_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&correction_pressure_force_, cnt*sizeof(Vector3DF), flag));
	if ( src_correction_pressure_force != 0x0 ) 
	{ 
		memcpy ( correction_pressure_force_, src_correction_pressure_force, nump *sizeof(Vector3DF)); 
		CUDA_SAFE_CALL(cudaFreeHost(src_correction_pressure_force)); 
	}

	Vector3DF* srcForce = force_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&force_, cnt*sizeof(Vector3DF), flag));
	if ( srcForce != 0x0 )	
	{ 
		memcpy ( force_, srcForce, nump *sizeof(Vector3DF));
		CUDA_SAFE_CALL(cudaFreeHost(srcForce)); 
	}

	Vector3DF* src_sumGradW = sumGradW_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&sumGradW_, cnt*sizeof(Vector3DF), flag));
	if ( src_sumGradW != 0x0 ) 
	{ 
		memcpy ( sumGradW_, src_sumGradW, nump *sizeof(Vector3DF)); 
		CUDA_SAFE_CALL(cudaFreeHost(src_sumGradW)); 
	}

	float* src_sumGradWDot = sumGradWDot_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&sumGradWDot_, cnt*sizeof(float), flag));
	if ( src_sumGradWDot != 0x0 ) 
	{ 
		memcpy ( sumGradWDot_, src_sumGradWDot, nump *sizeof(float)); 
		CUDA_SAFE_CALL(cudaFreeHost(src_sumGradWDot)); 
	}

	float* srcPress = pressure_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&pressure_, cnt*sizeof(float), flag));
	if ( srcPress != 0x0 ) 
	{ 
		memcpy ( pressure_, srcPress, nump *sizeof(float));
		CUDA_SAFE_CALL(cudaFreeHost(srcPress)); 
	}	

	float* srcCorrectionPress = correction_pressure_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&correction_pressure_, cnt*sizeof(float), flag));
	if ( srcCorrectionPress != 0x0 ) 
	{ 
		memcpy ( correction_pressure_, srcCorrectionPress, nump *sizeof(float));
		CUDA_SAFE_CALL(cudaFreeHost(srcCorrectionPress)); 
	}	

	float* srcCurrentDensity = density_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&density_, cnt*sizeof(float), flag));
	if ( srcCurrentDensity != 0x0 ) 
	{ 
		memcpy ( density_, srcCurrentDensity, nump *sizeof(float)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcCurrentDensity)); 
	}	

	float* srcPredictedDensity = predicted_density_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&predicted_density_, cnt*sizeof(float), flag));
	if ( srcPredictedDensity != 0x0 ) 
	{ 
		memcpy ( predicted_density_, srcPredictedDensity, nump *sizeof(float)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcPredictedDensity)); 
	}	

	float* srcDensity = densityError_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&densityError_, cnt*sizeof(float), flag));
	if ( srcDensity != 0x0 ) 
	{ 
		memcpy ( densityError_, srcDensity, nump *sizeof(float)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcDensity)); 
	}	

	float* srcMaxPredictedDensityArray = max_predicted_density_array_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&max_predicted_density_array_, cnt*sizeof(float), flag));
	if ( srcMaxPredictedDensityArray != 0x0 ) 
	{ 
		memcpy ( max_predicted_density_array_, srcMaxPredictedDensityArray, nump *sizeof(float)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcMaxPredictedDensityArray)); 
	}

	uint* srcGCell = particle_grid_cell_index_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&particle_grid_cell_index_, cnt*sizeof(uint), flag));
	if ( srcGCell != 0x0 )	
	{ 
		memcpy ( particle_grid_cell_index_, srcGCell, nump *sizeof(uint)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcGCell)); 
	}

	uint* srcNext = next_particle_index_in_the_same_cell_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&next_particle_index_in_the_same_cell_, cnt*sizeof(uint), flag));
	if ( srcNext != 0x0 )	
	{ 
		memcpy ( next_particle_index_in_the_same_cell_, srcNext, nump *sizeof(uint)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcNext)); 
	}

	uint* srcIndex = index_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&index_, cnt*sizeof(uint), flag));
	if ( srcIndex != 0x0 )	
	{ 
		memcpy ( index_, srcIndex, nump *sizeof(uint)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcIndex)); 
	}

	DWORD* srcClr = clr_;	
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&clr_, cnt*sizeof(DWORD), flag));
	if ( srcClr != 0x0 )	
	{ 
		memcpy ( clr_, srcClr, nump *sizeof(DWORD)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcClr)); 
	}

	uint* srcCell = cluster_cell_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&cluster_cell_, cnt*sizeof(uint), flag));
	if ( srcCell != 0x0 )	
	{ 
		memcpy ( cluster_cell_, srcCell, nump *sizeof(uint)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcCell)); 
	}

	unsigned short* srcAge = age_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&age_, cnt*sizeof(unsigned short), flag));
	if ( srcAge != 0x0 )	
	{ 
		memcpy ( age_, srcAge, nump *sizeof(unsigned short)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcAge)); 
	}

	uint* srcNbrNdx = neighbor_index_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&neighbor_index_, cnt*sizeof(uint), flag));
	if ( srcNbrNdx != 0x0 )	
	{ 
		memcpy ( neighbor_index_, srcNbrNdx, nump *sizeof(uint));
		CUDA_SAFE_CALL(cudaFreeHost(srcNbrNdx)); 
	}

	uint* srcNbrCnt = neighbor_particle_numbers_;
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&neighbor_particle_numbers_, cnt*sizeof(uint), flag));
	if ( srcNbrCnt != 0x0 )	
	{ 
		memcpy ( neighbor_particle_numbers_, srcNbrCnt, nump *sizeof(uint)); 
		CUDA_SAFE_CALL(cudaFreeHost(srcNbrCnt)); 
	}	

	param_[PSTAT_PMEM] = sizeof(Particle) * cnt;

	max_points_ = cnt;
}

int ParticleSystem::AddParticleToBuf ()
{
	if ( num_points_ >= max_points_ ) 
		return -1;
	int offsetBufPnt = num_points_;
	(pos_ + offsetBufPnt)->Set ( 0,0,0 );
	(predictedPosition_ + offsetBufPnt)->Set ( 0,0,0 );
	(vel_ + offsetBufPnt)->Set ( 0,0,0 );
	(vel_eval_ + offsetBufPnt)->Set ( 0,0,0 );
	(correction_pressure_force_ + offsetBufPnt)->Set ( 0,0,0 );
	(force_ + offsetBufPnt)->Set ( 0,0,0 );
	(sumGradW_ + offsetBufPnt)->Set ( 0,0,0 );
	*(sumGradWDot_ + offsetBufPnt) = 0;
	*(pressure_ + offsetBufPnt) = 0;
	*(correction_pressure_ + offsetBufPnt) = 0;
	*(density_ + offsetBufPnt) = 0;
	*(predicted_density_ + offsetBufPnt) = 0;
	*(densityError_ + offsetBufPnt) = 0;
	*(particle_grid_cell_index_ + offsetBufPnt) = -1;
	*(next_particle_index_in_the_same_cell_ + offsetBufPnt) = -1;
	*(cluster_cell_ + offsetBufPnt) = -1;

	++num_points_;
	return offsetBufPnt;
}

void ParticleSystem::ComputeDensityErrorFactor(uint num_particles)
{
	CreatePreParticlesISPH(num_particles);

	uint max_num_neighbors = 0;
	uint particle_with_max_num_neighbors = 0;

	ComputeGradWValuesSimple(num_particles, max_num_neighbors, particle_with_max_num_neighbors);

	ComputeFactorSimple(num_particles, max_num_neighbors, particle_with_max_num_neighbors);	
}

void ParticleSystem::CreatePreParticlesISPH(uint num_particles)
{
	InsertParticlesCPU(num_particles);
}

void ParticleSystem::ComputeGradWValues(uint num_particles)
{
	const float sim_scale					= param_[PSIMSCALE];
	const float smooth_radius				= param_[PSMOOTHRADIUS];

	for (int i = 0; i < num_particles; ++i)
	{
		sumGradW_[i].Set(0, 0, 0);
		sumGradWDot_[i] = 0;

		int j_index = neighbor_index_[i];
		for (int nbr=0; nbr < neighbor_particle_numbers_[i]; nbr++ ) 
		{
			int j = neighbor_table_[j_index];				
			Vector3DF pos_i_minus_j = pos_[i] - pos_[j];   
			float jdist = neighbor_dist_[j_index];			

			float kernelGradientValue = spiky_kern_/jdist * (smooth_radius - jdist) * (smooth_radius - jdist);


			Vector3DF gradVec = pos_i_minus_j * kernelGradientValue * sim_scale;	

			sumGradW_[i] += gradVec;
			sumGradWDot_[i] += gradVec.Dot(gradVec);

		}

	}
}

void ParticleSystem::ComputeGradWValuesSimple(uint num_particles, uint& max_num_neighbors, uint& index)
{
	const float sim_scale					= param_[PSIMSCALE];
	const float sim_scale_square			= sim_scale * sim_scale;
	const float smooth_radius				= param_[PSMOOTHRADIUS];
	const float smooth_radius_square		= smooth_radius * smooth_radius;

	for (int i = 0; i < num_particles; ++i)
	{
		sumGradW_[i].Set(0, 0, 0);
		sumGradWDot_[i] = 0;

		int neighbor_nums = 0;

		const uint i_cell_index = particle_grid_cell_index_[i];
		if ( i_cell_index != GRID_UNDEF) 
		{
			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) 
			{
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];

				if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1))
				{
					continue;
				}	

				int j = grid_head_cell_particle_index_array_ [neighbor_cell_index] ;	// get head particle index in the grid cell

				while ( j != GRID_UNDEF ) 
				{
					if ( i==j ) 
					{
						j = next_particle_index_in_the_same_cell_[j] ; 
						continue; 
					}
					Vector3DF pos_i_minus_j = pos_[i];
					pos_i_minus_j -= pos_[j];				
					const float dist_square_sim_scale = sim_scale_square*(pos_i_minus_j.x*pos_i_minus_j.x + pos_i_minus_j.y*pos_i_minus_j.y + pos_i_minus_j.z*pos_i_minus_j.z);
					if ( dist_square_sim_scale <= smooth_radius_square ) 
					{
						const float jdist = sqrt(dist_square_sim_scale);		

						Vector3DF gradVec = pos_i_minus_j * sim_scale * kernelPressureGrad(jdist, smooth_radius);

						sumGradW_[i] += gradVec;
						sumGradWDot_[i] += gradVec.Dot(gradVec);

						neighbor_nums++;
					}
					j = *(next_particle_index_in_the_same_cell_+j);
				}
			}
		}

		if (neighbor_nums > max_num_neighbors)
		{
			max_num_neighbors = neighbor_nums;
			index = i;
		}

	}
}

void ParticleSystem::ComputeFactor(uint num_particles)
{
	int maxNeighs = 0;

	for (int i = 0; i < num_particles; ++i)
	{
		if (neighbor_particle_numbers_[i] > maxNeighs)
			maxNeighs = neighbor_particle_numbers_[i];
	}

	for (int i = 0; i < num_particles; ++i)
	{
		if (neighbor_particle_numbers_[i] == maxNeighs)
		{
			float restVol = param_[PMASS] / param_[PRESTDENSITY];
			float preFactor = restVol * restVol * time_step_pcisph_ * time_step_pcisph_;
			Vector3DF temp_plus =  sumGradW_[i];
			Vector3DF temp_minus = sumGradW_ [i] *  (-1.0f);
			float gradWTerm = temp_minus.Dot(temp_plus) - sumGradWDot_[i]; 
			float divisor = preFactor * gradWTerm;

			if(divisor == 0) 
			{
				printf("precompute densErrFactor: division by 0 /n");
				exit(0);
			}

			param_[PDENSITYERRORFACTOR] = -1.0 / divisor;

			return;
		}
	}
}

void ParticleSystem::ComputeFactorSimple(uint num_particles, uint& max_num_neighbors, uint i)
{
	float restVol = param_[PMASS] / param_[PRESTDENSITY];
	float preFactor = restVol * restVol * time_step_pcisph_ * time_step_pcisph_;
	Vector3DF temp_plus =  sumGradW_[i];
	Vector3DF temp_minus = sumGradW_ [i] *  (-1.0f);
	float gradWTerm = temp_minus.Dot(temp_plus) - sumGradWDot_[i]; 
	float divisor = preFactor * gradWTerm;

	if(divisor == 0) 
	{
		printf("pre-compute densErrFactor: division by 0 /n");
		exit(0);
	}

	param_[PDENSITYERRORFACTOR] = -1.0 / divisor;

	return;
}

void ParticleSystem::collisionHandlingSimScale(Vector3DF& pos, Vector3DF& vel)
{
	const float     sim_scale		= param_[PSIMSCALE];
	const Vector3DF vec_bound_min	= vec_[PBOUNDARYMIN] * sim_scale;
	const Vector3DF vec_bound_max	= vec_[PBOUNDARYMAX] * sim_scale;

	float damping = 0.1;

	float reflect = 1.1;

	// 碰撞处理
	if (pos.x < vec_bound_min.x) 
	{
		pos.x = vec_bound_min.x;
		Vector3DF axis(1, 0, 0);
		vel -= axis * (float)axis.Dot(vel) * reflect;
		vel.x *=  damping;
	}

	if (pos.x > vec_bound_max.x) 
	{
		pos.x = vec_bound_max.x;
		Vector3DF axis(-1, 0, 0);
		vel -= axis * (float)axis.Dot(vel) * reflect;
		vel.x *=  damping;
	}

	if (pos.y < vec_bound_min.y)
	{
		pos.y = vec_bound_min.y;
		Vector3DF axis(0, 1, 0);
		vel -= axis * (float)axis.Dot(vel) * reflect;
		vel.y *=  damping;
	}

	if (pos.y > vec_bound_max.y) 
	{
		pos.y = vec_bound_max.y;		
		Vector3DF axis(0, -1, 0);
		vel -= axis * (float)axis.Dot(vel) * reflect;
		vel.y *=  damping;
	}

	if (pos.z < vec_bound_min.z) 
	{
		pos.z = vec_bound_min.z;
		Vector3DF axis(0, 0, 1);
		vel -= axis * (float)axis.Dot(vel) * reflect;
		vel.z *=  damping;
	}

	if (pos.z > vec_bound_max.z) 
	{
		pos.z = vec_bound_max.z;
		Vector3DF axis(0, 0, -1);
		vel -= axis * (float)axis.Dot(vel) * reflect;
		vel.z *=  damping;
	}
}

inline float frand()
{
	return rand() / (float) RAND_MAX;
}

void ParticleSystem::SetupInitParticleVolume ( const Vector3DF &minCorner, const Vector3DF &maxCorner, const float particleSpacing, const float jitter )
{
	srand(2013);

	float spacingRealWorldSize = param_[PSPACINGREALWORLD];
	float particleVolumeRealWorldSize = spacingRealWorldSize * spacingRealWorldSize *spacingRealWorldSize;
	float mass = param_[PRESTDENSITY] * particleVolumeRealWorldSize;

	const float lengthX = maxCorner.x-minCorner.x;
	const float lengthY = maxCorner.y-minCorner.y;
	const float lengthZ = maxCorner.z-minCorner.z;

	const int numParticlesX  = ceil( lengthX / particleSpacing );
	const int numParticlesY  = ceil( lengthY / particleSpacing );
	const int numParticlesZ  = ceil( lengthZ / particleSpacing );
	const int numParticles   = numParticlesX * numParticlesY * numParticlesZ;

	float tmpX, tmpY, tmpZ;
	if(numParticlesX % 2 == 0)
		tmpX = 0.0;
	else
		tmpX = 0.5;
	if(numParticlesZ % 2 == 0)
		tmpZ = 0.0;
	else
		tmpZ = 0.5;

	int i = 0;
	num_points_ = 0;
	for(int iy = 0; iy < numParticlesY; iy++)
	{
		float y = 0.0 + (iy + 0.5) * particleSpacing;
		for(int ix = 0; ix < numParticlesX; ix++)
		{
			float x = minCorner.x + (ix + tmpX) * particleSpacing;
			for(int iz = 0; iz < numParticlesZ; iz++)
			{	
				float z = minCorner.z + (iz + tmpZ) * particleSpacing;		

				if ( num_points_ < max_points_ ) 
				{
					pos_[i].Set( x + (frand() - 0.5) * jitter, y + (frand() - 0.5) * jitter, z + (frand() - 0.5) * jitter);
					vel_eval_[i].Set(0.0, 0.0, 0.0);
					clr_[i] = COLORA( (x-minCorner.x)/lengthX, (y-minCorner.y)/lengthY, (z-minCorner.z)/lengthZ, 1); 
					num_points_++;
				}
				++i;
			}
		}
	}

}

void ParticleSystem::SetupAdditonalParticleVolume ( const Vector3DF &minCorner, const Vector3DF &maxCorner, const float particleSpacing, const float jitter  ,const int numParticles)
{
	srand(2013);

	float spacingRealWorldSize = param_[PSPACINGREALWORLD];
	float particleVolumeRealWorldSize = spacingRealWorldSize * spacingRealWorldSize *spacingRealWorldSize;
	float mass = param_[PRESTDENSITY] * particleVolumeRealWorldSize;

	const float lengthX = maxCorner.x-minCorner.x;
	const float lengthY = maxCorner.y-minCorner.y;
	const float lengthZ = maxCorner.z-minCorner.z;

	const int numParticlesX  = ceil( lengthX / particleSpacing );
	const int numParticlesY  = ceil( lengthY / particleSpacing );
	const int numParticlesZ  = ceil( lengthZ / particleSpacing );

	float tmpX, tmpY, tmpZ;
	if(numParticlesX % 2 == 0)
		tmpX = 0.0;
	else
		tmpX = 0.5;
	if(numParticlesZ % 2 == 0)
		tmpZ = 0.0;
	else
		tmpZ = 0.5;

	int i = num_points_;
	int num_particles = 0;
	for(int iy = 0; iy < numParticlesY; iy++)
	{
		float y = 0.0 + (iy + 0.5) * particleSpacing;
		for(int ix = 0; ix < numParticlesX; ix++)
		{
			float x = minCorner.x + (ix + tmpX) * particleSpacing;
			for(int iz = 0; iz < numParticlesZ; iz++)
			{	
				float z = minCorner.z + (iz + tmpZ) * particleSpacing;		

				if ( num_particles < numParticles ) 
				{
					pos_[i].Set( x + (frand() - 0.5) * jitter, y + (frand() - 0.5) * jitter, z + (frand() - 0.5) * jitter);
					vel_eval_[i].Set(0.0, 0.0, 0.0);
					clr_[i] = COLORA( (x-minCorner.x)/lengthX, (y-minCorner.y)/lengthY, (z-minCorner.z)/lengthZ, 1); 
					num_particles++;
				}
				++i;
			}
		}
	}

}

void ParticleSystem::SetupInitParticleVolumeLoad(const Vector3DF& minVec, const Vector3DF& maxVec)
{
	const float lengthX = maxVec.x-minVec.x;
	const float lengthY = maxVec.y-minVec.y;
	const float lengthZ = maxVec.z-minVec.z;
	const float inv_sim_scale = 1.0f / param_[PSIMSCALE];

	int numUnloadedParticles = particles.size();
	for (int i = 0; i < numUnloadedParticles; ++i)
	{
		Vector3D position = particles[i].position;
		pos_[i].Set(position.v[0] * inv_sim_scale, position.v[1] * inv_sim_scale, position.v[2] * inv_sim_scale);
		vel_eval_[i].Set(0.0, 0.0, 0.0);
		clr_[i] = COLORA( (position.v[0]-minVec.x)/lengthX, (position.v[1]-minVec.y)/lengthY, (position.v[2]-minVec.z)/lengthZ, 1); 
	}
}

void ParticleSystem::SetupInitParticleVolumeFromFile(const Vector3DF& minVec, const Vector3DF& maxVec)
{	
	const float lengthX = maxVec.x-minVec.x;
	const float lengthY = maxVec.y-minVec.y;
	const float lengthZ = maxVec.z-minVec.z;
	const float inv_sim_scale = 1.0f / param_[PSIMSCALE];

	for (int i = 0; i < num_points_; ++i)
	{
		pos_[i] *= inv_sim_scale;
		vel_eval_[i].Set(0.0, 0.0, 0.0);
		clr_[i] = COLORA( (pos_[i].x-minVec.x)/lengthX, (pos_[i].y-minVec.y)/lengthY, (pos_[i].z-minVec.z)/lengthZ, 1); 
	}
}

void ParticleSystem::SetupSampleParticleVolumePCISPH ( const Vector3DF & minCorner, const Vector3DF & maxCorner, const float particleSpacing, const float jitter)
{
	srand(2014);

	float spacingRealWorldSize = param_[PSPACINGREALWORLD];
	float particleVolumeRealWorldSize = spacingRealWorldSize * spacingRealWorldSize *spacingRealWorldSize;
	float mass = param_[PRESTDENSITY] * particleVolumeRealWorldSize;

	const float lengthX = maxCorner.x-minCorner.x;
	const float lengthY = maxCorner.y-minCorner.y;
	const float lengthZ = maxCorner.z-minCorner.z;

	const int numParticlesX  = ceil( lengthX / particleSpacing );
	const int numParticlesY  = ceil( lengthY / particleSpacing );
	const int numParticlesZ  = ceil( lengthZ / particleSpacing );
	const int numParticles   = numParticlesX * numParticlesY * numParticlesZ;

	float tmpX, tmpY, tmpZ;
	if(numParticlesX % 2 == 0)
		tmpX = 0.0;
	else
		tmpX = 0.5;
	if(numParticlesZ % 2 == 0)
		tmpZ = 0.0;
	else
		tmpZ = 0.5;

	int i = 0;
	for(int iy = 0; iy < numParticlesY; iy++)
	{
		float y = 0.0 + (iy + 0.5) * particleSpacing;
		for(int ix = 0; ix < numParticlesX; ix++)
		{
			float x = minCorner.x + (ix + tmpX) * particleSpacing;
			for(int iz = 0; iz < numParticlesZ; iz++)
			{	
				float z = minCorner.z + (iz + tmpZ) * particleSpacing;		

				if ( num_points_ < max_points_ ) 
				{
					pos_[i].Set( x + (frand() - 0.5) * jitter, y + (frand() - 0.5) * jitter, z + (frand() - 0.5) * jitter);
					vel_eval_[i].Set(0.0, 0.0, 0.0);
					clr_[i] = COLORA( (x-minCorner.x)/lengthX, (y-minCorner.y)/lengthY, (z-minCorner.z)/lengthZ, 1); 
					(sumGradW_ + num_points_)->Set ( 0,0,0 );
					*(sumGradWDot_ + num_points_) = 0;
					num_points_++;
				}
				++i;
			}
		}
	}
}

void ParticleSystem::Record ( int param, std::string name, mint::Time& start )
{
	mint::Time stop;
	stop.SetSystemTime ( ACC_NSEC );
	stop = stop - start;
	param_ [ param ] = stop.GetMSec();
	if ( toggle_[PPROFILE] ) printf ("%s:  %s\n", name.c_str(), stop.GetReadableTime().c_str() );
}

void ParticleSystem::RunCPUSPH ()
{
	mint::Time start;
	start.SetSystemTime ( ACC_NSEC );
	InsertParticlesCPU (num_points());
	Record ( PTIME_INSERT, "Insert CPU SPH", start );			
	start.SetSystemTime ( ACC_NSEC );
	ComputePressureGrid ();
	Record ( PTIME_PRESS, "Press CPU SPH", start );
	start.SetSystemTime ( ACC_NSEC );
	ComputeForceGrid ();							
	Record ( PTIME_FORCE, "Force CPU SPH", start );
	start.SetSystemTime ( ACC_NSEC );
	AdvanceStepSimpleCollision (time_step_);
	Record ( PTIME_ADVANCE, "Advance CPU SPH", start );

}

void ParticleSystem::RunCUDAIndexSPH ()
{
	mint::Time start;
	start.SetSystemTime ( ACC_NSEC );
	InsertParticlesCUDA ( 0x0, 0x0, 0x0 );
	Record ( PTIME_INSERT, "Insert CUDA", start );	
	start.SetSystemTime ( ACC_NSEC );
	PrefixSumCellsCUDA ( 0x0 );
	CountingSortIndexCUDA ( 0x0 );
	Record ( PTIME_SORT, "Index Sort CUDA", start );
	start.SetSystemTime ( ACC_NSEC );
	ComputeDensityPressureCUDA();
	Record ( PTIME_PRESS, "Press CUDA", start );		
	start.SetSystemTime ( ACC_NSEC );
	ComputeForceCUDA ();	
	Record ( PTIME_FORCE, "Force CUDA", start );
	start.SetSystemTime ( ACC_NSEC );
	AdvanceCUDA ( time_step_, param_[PSIMSCALE] );			
	Record ( PTIME_ADVANCE, "Advance CUDA", start );
	TransferFromCUDA ();		
}

void ParticleSystem::RunCUDAFullSPH()
{
	mint::Time start;
	start.SetSystemTime ( ACC_NSEC );
	InsertParticlesCUDA ( 0x0, 0x0, 0x0 );
	Record ( PTIME_INSERT, "Insert CUDA", start );			
	start.SetSystemTime ( ACC_NSEC );
	PrefixSumCellsCUDA ( 0x0 );
	CountingSortFullCUDA ( 0x0 );
	Record ( PTIME_SORT, "Full Sort CUDA", start );
	start.SetSystemTime ( ACC_NSEC );
	ComputeDensityPressureCUDA();
	Record ( PTIME_PRESS, "Press CUDA", start );		
	start.SetSystemTime ( ACC_NSEC );
	ComputeForceCUDA ();	
	Record ( PTIME_FORCE, "Force CUDA", start );
	start.SetSystemTime ( ACC_NSEC );
	AdvanceCUDA ( time_step_, param_[PSIMSCALE] );			
	Record ( PTIME_ADVANCE, "Advance CUDA", start );
	TransferFromCUDA ();	
}

void ParticleSystem::RunCPUPCISPH()
{
	mint::Time start;
	start.SetSystemTime ( ACC_NSEC );
	InsertParticlesCPU (num_points());
	Record ( PTIME_INSERT, "Insert CPU PCISPH", start );			
	start.SetSystemTime ( ACC_NSEC );
	ComputeOtherForceCpu();
	Record ( PTIME_OTHER_FORCE, "Other PCISPH Force CPU", start );
	start.SetSystemTime ( ACC_NSEC );
	PredictionCorrectionStepCpu();
	Record ( PTIME_PCI_STEP, "Prediction Correction Step CPU", start );
	start.SetSystemTime ( ACC_NSEC );
	AdvanceStepSimpleCollision(time_step_pcisph_);
	Record ( PTIME_ADVANCE, "Advance CPU", start );
}

void ParticleSystem::RunCUDAIndexPCISPH()
{
	mint::Time start;
	start.SetSystemTime ( ACC_NSEC );
	InsertParticlesCUDA ( 0x0, 0x0, 0x0 );						
	Record ( PTIME_INSERT, "Insert CUDA PCISPH", start );			
	start.SetSystemTime ( ACC_NSEC );
	PrefixSumCellsCUDA ( 0x0 );									
	CountingSortIndexCUDA ( 0x0 );							
	Record ( PTIME_SORT, "Full Sort CUDA", start );	
	start.SetSystemTime ( ACC_NSEC );
	ComputeOtherForceCUDA();									
	Record ( PTIME_OTHER_FORCE, "Other PCISPH Force CUDA", start );
	start.SetSystemTime ( ACC_NSEC );
	PredictionCorrectionStepCUDA(time_step_);	
	Record ( PTIME_PCI_STEP, "Prediction Correction Step CUDA", start );
	start.SetSystemTime ( ACC_NSEC );
	AdvanceCUDAPCISPH ( time_step_, param_[PSIMSCALE] );			
	Record ( PTIME_ADVANCE, "Advance CUDA", start );
	TransferFromCUDA ();										
}

void ParticleSystem::RunCUDAFullPCISPH()
{
	mint::Time start;
	start.SetSystemTime ( ACC_NSEC );
	InsertParticlesCUDA ( 0x0, 0x0, 0x0 );						
	Record ( PTIME_INSERT, "Insert CUDA PCISPH", start );			
	start.SetSystemTime ( ACC_NSEC );
	PrefixSumCellsCUDA ( 0x0 );									
	CountingSortFullCUDA ( 0x0 );								
	Record ( PTIME_SORT, "Full Sort CUDA", start );	
	start.SetSystemTime ( ACC_NSEC );
	ComputeOtherForceCUDA();									
	Record ( PTIME_OTHER_FORCE, "Other PCISPH Force CUDA", start );
	start.SetSystemTime ( ACC_NSEC );
	PredictionCorrectionStepCUDA(time_step_);	
	Record ( PTIME_PCI_STEP, "Prediction Correction Step CUDA", start );
	start.SetSystemTime ( ACC_NSEC );
	AdvanceCUDAPCISPH ( time_step_, param_[PSIMSCALE] );			
	Record ( PTIME_ADVANCE, "Advance CUDA", start );
	TransferFromCUDA ();										
}

void ParticleSystem::PredictionCorrectionStepCpu()
{
	bool densityErrorTooLarge = true;

	int iteration = 0;
	while( (iteration < param_[PMINLOOPPCISPH]) || ((densityErrorTooLarge) && (iteration < param_[PMAXLOOPPCISPH])) )
	{
		for ( int i=0; i < num_points(); i++ ) 
			predictPositionAndVelocity(i);

		maxPredictedDensity = 0.0;

		for ( int i=0; i < num_points(); i++ ) 
			computePredictedDensityAndPressure(i);

		float densityErrorInPercent = max(0.1f * maxPredictedDensity - 100.0f, 0.0f); 

		if (toggle_ [ PPRINTDEBUGGINGINFO])
			printf("ERROR: %f\% \n", densityErrorInPercent);

		if(densityErrorInPercent < param_[PMAXDENSITYERRORALLOWED]) 
			densityErrorTooLarge = false;

		for ( int i=0; i < num_points(); i++ ) 
			computeCorrectivePressureForce(i);

		iteration++;
	}

	int outCnt;
	if (toggle_ [ PPRINTDEBUGGINGINFO])
		outCnt = 1;
	else 
		outCnt = 50;
}

void ParticleSystem::predictPositionAndVelocity(uint index)
{
	const float sim_scale = param_[PSIMSCALE];

	Vector3DF acceleration = (force_[index] + correction_pressure_force_[index]) * (1.0f / param_[PMASS]);
	Vector3DF predictedVelocity = vel_eval_[index] + acceleration * (float)time_step_pcisph_; 

	predictedPosition_[index] = pos_[index] * sim_scale + predictedVelocity * (float)time_step_pcisph_;		

	collisionHandlingSimScale(predictedPosition_[index], predictedVelocity);
}

void ParticleSystem::computePredictedDensityAndPressure(uint index)
{
	float sum = 0.0;

	float predictedSPHDensity			= param_[PKERNELSELF] * param_[PMASS];

	Vector3DF ipredicted_pos			= predictedPosition_[index];
	const float smooth_radius			= param_[PSMOOTHRADIUS];
	const float smooth_radius_square	= smooth_radius * smooth_radius;
	const float	sim_scale_square		= param_[PSIMSCALE] * param_[PSIMSCALE];
	const float mass					= param_[PMASS];

	const uint i_cell_index = particle_grid_cell_index_[index];
	if ( i_cell_index != GRID_UNDEF ) 
	{
		for (int cell=0; cell < max_num_adj_grid_cells_cpu; cell++) 
		{
			const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];

			if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1))
			{
				continue;
			}

			int j = grid_head_cell_particle_index_array_ [neighbor_cell_index] ;	

			while ( j != GRID_UNDEF ) 
			{
				if ( index==j ) 
				{
					j = next_particle_index_in_the_same_cell_[j] ; 
					continue; 
				}

				Vector3DF dist_vector = predictedPosition_[j];
				dist_vector -= ipredicted_pos;
				const float dist_square_sim_scale = dist_vector.x*dist_vector.x + dist_vector.y*dist_vector.y + dist_vector.z*dist_vector.z;
				if ( dist_square_sim_scale <= smooth_radius_square ) 
				{
					const float dist = sqrt(dist_square_sim_scale);
					float kernelValue = kernelM4Lut(dist, smooth_radius);
					predictedSPHDensity += kernelValue * mass;
				}
				j = *(next_particle_index_in_the_same_cell_+j);

			}
		}
	}

	densityError_[index] = max(predictedSPHDensity - param_[PRESTDENSITY], 0.0f);

	correction_pressure_[index] += max( densityError_[index] * param_[PDENSITYERRORFACTOR], 0.0f);

	maxPredictedDensity = max(maxPredictedDensity, predictedSPHDensity);

	predicted_density_[index] = predictedSPHDensity;
}

void ParticleSystem::computeCorrectivePressureForce(uint index)
{
	correction_pressure_force_[index].Set(0,0,0);

	const float p_i						= correction_pressure_[index];
	const float idensSq					= predicted_density_[index] * predicted_density_[index];
	const float mass					= param_[PMASS];
	const float sim_scale				= param_[PSIMSCALE];
	const float sim_scale_square		= sim_scale * sim_scale;
	const float smooth_radius			= param_[PSMOOTHRADIUS];
	const float smooth_radius_square	= smooth_radius * smooth_radius;
	const float rest_volume				= mass / param_[PRESTDENSITY];

	const uint i_cell_index = particle_grid_cell_index_[index];
	if ( i_cell_index != GRID_UNDEF ) 
	{
		for (int cell=0; cell < max_num_adj_grid_cells_cpu; cell++) 
		{
			const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];

			if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1))
			{
				continue;
			}

			int j = grid_head_cell_particle_index_array_ [neighbor_cell_index] ;	

			while ( j != GRID_UNDEF ) 
			{
				if ( index==j ) 
				{
					j = next_particle_index_in_the_same_cell_[j] ; 
					continue; 
				}

				Vector3DF vector_i_minus_j = pos_[index] - pos_[j];
				const float dx = vector_i_minus_j.x;
				const float dy = vector_i_minus_j.y;
				const float dz = vector_i_minus_j.z;

				const float dist_square_sim_scale = sim_scale_square * (dx*dx + dy*dy + dz*dz);
				if ( dist_square_sim_scale <= smooth_radius_square ) 
				{
					const float jdist = sqrt(dist_square_sim_scale);
					float kernelGradientValue = kernelPressureGradLut(jdist, smooth_radius);
					Vector3DF kernelGradient = vector_i_minus_j * sim_scale * kernelGradientValue;
					const float p_j = correction_pressure_[j];
					float grad = 0.5f * (p_i + p_j) ;  

					correction_pressure_force_[index] -= kernelGradient * grad * rest_volume * rest_volume;	

				}
				j = next_particle_index_in_the_same_cell_[j] ;	
			}
		}
	}
}

void ParticleSystem::Run (int width, int height)
{
	// 计时器清零
	param_[ PTIME_INSERT ] = 0.0;
	param_[ PTIME_SORT ] = 0.0;
	param_[ PTIME_COUNT ] = 0.0;
	param_[ PTIME_PRESS ] = 0.0;
	param_[ PTIME_FORCE ] = 0.0;
	param_[ PTIME_ADVANCE ] = 0.0;
	param_[ PTIME_OTHER_FORCE ] = 0.0;
	param_[ PTIME_PCI_STEP ] = 0.0;

	// 运行模拟程序
	// 使用6种不同方法来模拟
	switch ( (int) param_[PRUN_MODE] ) 
	{			
	case RUN_CPU_SPH:		
		RunCPUSPH();		
		break;
	case RUN_CUDA_INDEX_SPH:	
		RunCUDAIndexSPH();		
		break;
	case RUN_CUDA_FULL_SPH:		
		RunCUDAFullSPH();		
		break;
	case RUN_CPU_PCISPH:
		RunCPUPCISPH();
		break;
	case RUN_CUDA_INDEX_PCISPH:
		RunCUDAIndexPCISPH();
		break;
	case RUN_CUDA_FULL_PCISPH:
		RunCUDAFullPCISPH();
		break;		
	};

	time_ += time_step_;
	frame_++;

}

void ParticleSystem::AllocatePackBuf ()
{
	if ( pack_fluid_particle_buf_ != 0x0 ) 
		free ( pack_fluid_particle_buf_ );	
	pack_fluid_particle_buf_ = (char*) malloc ( sizeof(Particle) * max_points_ );
}

void ParticleSystem::DebugPrintMemory ()
{
	int psize = 4*sizeof(Vector3DF) + sizeof(DWORD) + sizeof(unsigned short) + 2*sizeof(float) + sizeof(int) + sizeof(int)+sizeof(int);
	int gsize = 2*sizeof(int);
	int nsize = sizeof(int) + sizeof(float);

	printf ( "MEMORY:\n");
	printf ( "  Fluid (size):			%d bytes\n",	   sizeof(Particle) );
	printf ( "  Particles:              %d, %f MB (%f)\n", num_points_, (psize*num_points_)/1048576.0, (psize*max_points_)/1048576.0);
	printf ( "  Acceleration Grid:      %d, %f MB\n",	   grid_total_, (gsize*grid_total_)/1048576.0 );
	printf ( "  Acceleration Neighbors: %d, %f MB (%f)\n", neighbor_particles_num_, (nsize*neighbor_particles_num_)/1048576.0, (nsize*neighbor_particles_max_num_)/1048576.0 );

}

void ParticleSystem::DrawDomain(Vector3DF& domain_min, Vector3DF& domain_max)
{	
	glColor3f ( 1.0, 0.0, 0.0 );

	/*
	/8------------ /7
	/|             / |
	/	|			 /  |
	/5-|-----------6	|
	|  |			|
	|	|			|	|
	|	|			|	|
	|	4-----------|---3
	|	/			|  /
	| /			| /
	|1 ----------- 2	

	*/
	/*
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	*/
	// ground (1234)
	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glEnd();

	//ceil(5678)
	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glEnd();

	//left face (14,58,15,48)
	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glEnd();

	//right face (23,67,26,37)
	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glEnd();

	//back face(43,78,37,48)
	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_min.y, domain_min.z);//3
	glVertex3f(domain_max.x, domain_max.y, domain_min.z);//7
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_min.z);//4
	glVertex3f(domain_min.x, domain_max.y, domain_min.z);//8
	glEnd();

	//front face(12,56,15,26)
	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_min.x, domain_min.y, domain_max.z);//1
	glVertex3f(domain_min.x, domain_max.y, domain_max.z);//5
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(domain_max.x, domain_min.y, domain_max.z);//2
	glVertex3f(domain_max.x, domain_max.y, domain_max.z);//6
	glEnd();
}

void ParticleSystem::AdvanceStepCPU (float time_step)
{
	const float acc_limit = param_[PACCEL_LIMIT];		
	const float acc_limit_square = acc_limit*acc_limit;
	const float speed_limit = param_[PVEL_LIMIT];	   
	const float speed_limit_square = speed_limit*speed_limit;

	const float sim_scale = param_[PSIMSCALE];

	Vector3DF norm;
	Vector4DF clr;
	float adj;
	float speed;
	float diff; 

	for ( int i=0; i < num_points(); i++ ) {
		if ( particle_grid_cell_index_[i] == GRID_UNDEF) 
			continue;

		Vector3DF acceleration = force_[i];
		acceleration /= param_[PMASS];

		BoundaryCollisions(&pos_[i], &vel_eval_[i], acceleration);

		acceleration += vec_[PPLANE_GRAV_DIR];

		if ( param_[PPOINT_GRAV_AMT] > 0 ) {
			norm.x = ( pos_[i].x - vec_[PPOINT_GRAV_POS].x );
			norm.y = ( pos_[i].y - vec_[PPOINT_GRAV_POS].y );
			norm.z = ( pos_[i].z - vec_[PPOINT_GRAV_POS].z );
			norm.Normalize ();
			norm *= param_[PPOINT_GRAV_AMT];
			acceleration -= norm;
		}

		speed = acceleration.x*acceleration.x + acceleration.y*acceleration.y + acceleration.z*acceleration.z;
		if ( speed > acc_limit_square ) {
			acceleration *= acc_limit / sqrt(speed);
		}		

		speed = vel_[i].x*vel_[i].x + vel_[i].y*vel_[i].y + vel_[i].z*vel_[i].z;
		if ( speed > speed_limit_square ) {
			speed = speed_limit_square;
			vel_[i] *= speed_limit / sqrt(speed);
		}		

		Vector3DF vnext = vel_[i] + acceleration * (float)time_step;							

		vel_eval_[i] = vel_[i];
		vel_eval_[i] += vnext;
		vel_eval_[i] *= 0.5;							
		vel_[i] = vnext;
		vnext *= time_step/sim_scale;		
		pos_[i] += vnext;						

		if ( speed > speed_limit_square*0.1) {
			adj = speed_limit_square*0.1;
			clr.fromClr ( clr_[i] );
			clr += float(2/255.0);
			clr.Clamp ( 1, 1, 1, 1);
			clr_[i] = clr.toClr();
		}
		if ( speed < 0.01 ) {
			clr.fromClr (clr_[i]);
			clr.x -= float(1/255.0);		if ( clr.x < 0.2 ) clr.x = 0.2;
			clr.y -= float(1/255.0);		if ( clr.y < 0.2 ) clr.y = 0.2;
			clr_[i] = clr.toClr();
		}

		if ( toggle_[PWRAP_X] ) {
			diff = pos_[i].x - (vec_[PBOUNDARYMIN].x + 2);			
			if ( diff <= 0 ) {
				pos_[i].x = (vec_[PBOUNDARYMAX].x - 2) + diff*2;				
				pos_[i].z = 10;
			}
		}	
	}
}

void ParticleSystem::AdvanceStepSimpleCollision(float time_step)
{
	const float acc_limit = param_[PACCEL_LIMIT];		
	const float acc_limit_square = acc_limit*acc_limit;
	const float speed_limit = param_[PVEL_LIMIT];	    
	const float speed_limit_square = speed_limit*speed_limit;

	const float sim_scale = param_[PSIMSCALE];

	Vector3DF norm;
	Vector4DF clr;
	float adj;
	float speed;
	float diff; 

	for ( int i=0; i < num_points(); i++ ) {
		if ( particle_grid_cell_index_[i] == GRID_UNDEF) 
			continue;

		Vector3DF acceleration = force_[i] + correction_pressure_force_[i];
		acceleration /= param_[PMASS];
  
		vel_eval_[i] += acceleration * time_step;
		pos_[i] *= sim_scale;		
		pos_[i] += vel_eval_[i] * time_step;

		collisionHandlingSimScale(pos_[i], vel_eval_[i]);

		pos_[i] /= sim_scale;		
	}
}

void ParticleSystem::BoundaryCollisions(Vector3DF* ipos, Vector3DF* iveleval, Vector3DF& acceleration)
{
	const float radius = param_[PCOLLISIONRADIUS];
	const float sim_scale = param_[PSIMSCALE];
	const Vector3DF bound_min = vec_[PBOUNDARYMIN] * sim_scale;
	const Vector3DF bound_max = vec_[PBOUNDARYMAX] * sim_scale;

	const float stiff = param_[PBOUNDARYSTIFF];
	const float damp = param_[PBOUNDARYDAMP];

	float diff;
	float adj;
	Vector3DF norm;

	diff = radius - ( ipos->y - (bound_min.y+ (ipos->x-bound_min.x)*param_[PGROUND_SLOPE] ) );
	if (diff > EPSILON ) {			
		norm.Set ( -param_[PGROUND_SLOPE], 1.0 - param_[PGROUND_SLOPE], 0 );
		adj = stiff * diff - damp * norm.Dot ( *iveleval );
		acceleration.x += adj * norm.x; 
		acceleration.y += adj * norm.y; 
		acceleration.z += adj * norm.z;
	}		
	diff = radius - ( bound_max.y - ipos->y );
	if (diff > EPSILON) {
		norm.Set ( 0, -1, 0 );
		adj = stiff * diff - damp * norm.Dot ( *iveleval );
		acceleration.x += adj * norm.x; 	
		acceleration.y += adj * norm.y; 
		acceleration.z += adj * norm.z;
	}		

	if ( !toggle_[PWRAP_X] ) {
		diff = radius - ( ipos->x - (bound_min.x + (sin(time_*param_[PFORCE_FREQ])+1)*0.5 * param_[PFORCE_MIN]) );	
		if (diff > EPSILON ) {
			norm.Set ( 1.0, 0, 0 );
			adj = (param_[ PFORCE_MIN ]+1) * stiff * diff - damp * norm.Dot ( *iveleval ) ;
			acceleration.x += adj * norm.x; 
			acceleration.y += adj * norm.y; 
			acceleration.z += adj * norm.z;					
		}

		diff = radius - ( (bound_max.x - (sin(time_*param_[PFORCE_FREQ])+1)*0.5* param_[PFORCE_MAX]) - ipos->x );	
		if (diff > EPSILON) {
			norm.Set ( -1, 0, 0 );
			adj = (param_[ PFORCE_MAX ]+1) * stiff * diff - damp * norm.Dot ( *iveleval );
			acceleration.x += adj * norm.x; 
			acceleration.y += adj * norm.y; 
			acceleration.z += adj * norm.z;
		}
	}

	diff = radius - ( ipos->z - bound_min.z );			
	if (diff > EPSILON) {
		norm.Set ( 0, 0, 1 );
		adj = stiff * diff - damp * norm.Dot ( *iveleval );
		acceleration.x += adj * norm.x; 
		acceleration.y += adj * norm.y; 
		acceleration.z += adj * norm.z;
	}
	diff = radius - ( bound_max.z - ipos->z );
	if (diff > EPSILON) {
		norm.Set ( 0, 0, -1 );
		adj = stiff * diff - damp * norm.Dot ( *iveleval );
		acceleration.x += adj * norm.x; 
		acceleration.y += adj * norm.y; 
		acceleration.z += adj * norm.z;
	}

	if ( toggle_[PWALL_BARRIER] ) {
		diff = 2 * radius - ( ipos->x - 0 );					
		if (diff < 2*radius && diff > EPSILON && fabs(ipos->y) < 3 && ipos->z < 10) {
			norm.Set ( 1.0, 0, 0 );
			adj = 2*stiff * diff - damp * norm.Dot ( *iveleval ) ;	
			acceleration.x += adj * norm.x; 
			acceleration.y += adj * norm.y; 
			acceleration.z += adj * norm.z;					
		}
	}

	if ( toggle_[PLEVY_BARRIER] ) {
		diff = 2 * radius - ( ipos->x - 0 );					
		if (diff < 2*radius && diff > EPSILON && fabs(ipos->y) > 5 && ipos->z < 10) {
			norm.Set ( 1.0, 0, 0 );
			adj = 2*stiff * diff - damp * norm.Dot ( *iveleval ) ;	
			acceleration.x += adj * norm.x; 
			acceleration.y += adj * norm.y; 
			acceleration.z += adj * norm.z;					
		}
	}

	if ( toggle_[PDRAIN_BARRIER] ) {
		diff = 2 * radius - ( ipos->z - bound_min.z-15 );
		if (diff < 2*radius && diff > EPSILON && (fabs(ipos->x)>3 || fabs(ipos->y)>3) ) {
			norm.Set ( 0, 0, 1);
			adj = stiff * diff - damp * norm.Dot ( *iveleval );
			acceleration.x += adj * norm.x; 
			acceleration.y += adj * norm.y; 
			acceleration.z += adj * norm.z;
		}
	}
}

void ParticleSystem::ClearNeighborTable ()
{
	if ( neighbor_table_ != 0x0 )	
		free (neighbor_table_);
	if ( neighbor_dist_ != 0x0)		
		free (neighbor_dist_ );
	neighbor_table_ = 0x0;
	neighbor_dist_ = 0x0;
	neighbor_particles_num_ = 0;
	neighbor_particles_max_num_ = 0;
}

void ParticleSystem::ResetNeighbors ()
{
	neighbor_particles_num_ = 0;
}

int ParticleSystem::AddNeighbor ()
{
	if ( neighbor_particles_num_ >= neighbor_particles_max_num_ ) {
		neighbor_particles_max_num_ = 2*neighbor_particles_max_num_ + 1;		
		int* saveTable = neighbor_table_;
		neighbor_table_ = (int*) malloc ( neighbor_particles_max_num_ * sizeof(int) );
		if ( saveTable != 0x0 ) {
			memcpy ( neighbor_table_, saveTable, neighbor_particles_num_*sizeof(int) );
			free ( saveTable );
		}
		float* saveDist = neighbor_dist_;
		neighbor_dist_ = (float*) malloc ( neighbor_particles_max_num_ * sizeof(float) );
		if ( saveDist != 0x0 ) {
			memcpy ( neighbor_dist_, saveDist, neighbor_particles_num_*sizeof(int) );
			free ( saveDist );
		}
	};
	neighbor_particles_num_++;
	return neighbor_particles_num_-1;
}

void ParticleSystem::ClearNeighbors ( int i )
{
	neighbor_particle_numbers_[i] = 0;
}

int ParticleSystem::AddNeighbor( int i, int j, float d )
{
	int k = AddNeighbor();
	neighbor_table_[k] = j;
	neighbor_dist_[k] = d;
	if (*(neighbor_particle_numbers_+i) == 0 ) 
		*(neighbor_index_+i) = k;
	(*(neighbor_particle_numbers_+i))++;
	return k;
}

void ParticleSystem::SetupGridAllocate ( const Vector3DF &grid_volume_min, const Vector3DF &grid_volume_max, const float sim_scale, const float cell_size, const float border )
{
	float world_cellsize = cell_size / sim_scale;

	grid_min_ = grid_volume_min;
	grid_max_ = grid_volume_max;

	grid_size_ = grid_volume_max;
	grid_size_ -= grid_volume_min;
	grid_res_.x = (int)ceil ( grid_size_.x / world_cellsize );	
	grid_res_.y = (int)ceil ( grid_size_.y / world_cellsize );
	grid_res_.z = (int)ceil ( grid_size_.z / world_cellsize );

	if (grid_res_.x == 0)
		grid_res_.x = 1;
	if (grid_res_.y == 0)
		grid_res_.y = 1;
	if (grid_res_.z == 0)
		grid_res_.z = 1;	

	grid_total_ = (int)(grid_res_.x * grid_res_.y * grid_res_.z);
	if(grid_total_ > 10000000)
	{
		printf("too many cells in initGrid(). aborting...\n");
		exit(0);
	}

	grid_size_.x = grid_res_.x * world_cellsize;			
	grid_size_.y = grid_res_.y * world_cellsize;
	grid_size_.z = grid_res_.z * world_cellsize;
	grid_delta_ = grid_res_;								
	grid_delta_ /= grid_size_;

	if ( grid_head_cell_particle_index_array_ != 0x0 ) 
		free (grid_head_cell_particle_index_array_);  
	if ( grid_particles_number_ != 0x0 ) 
		free (grid_particles_number_); 
	grid_head_cell_particle_index_array_ = (uint*) malloc ( sizeof(uint*) * grid_total_ );
	grid_particles_number_ = (uint*) malloc ( sizeof(uint*) * grid_total_ );
	memset ( grid_head_cell_particle_index_array_, GRID_UNDEF, grid_total_*sizeof(uint) );
	memset ( grid_particles_number_, GRID_UNDEF, grid_total_*sizeof(uint) );

	param_[PSTAT_GMEM] = 12 * grid_total_;		

	grid_search_ =  3;
	int cell = 0;
	for (int y = -1; y < 2; y++ ) 
		for (int z = -1; z < 2; z++ ) 
			for (int x = -1; x < 2; x++ ) 
				grid_neighbor_cell_index_offset_[cell++] = y * grid_res_.x *grid_res_.z + z * grid_res_.x + x ;		

	grid_adj_cnt_ = grid_search_ * grid_search_ * grid_search_;

	if ( pack_grid_buf_ != 0x0 ) 
		free ( pack_grid_buf_ );
	pack_grid_buf_ = (int*) malloc ( sizeof(int) * grid_total_ );


}

void ParticleSystem::SetupSampleGridAllocatePCISPH ( const Vector3DF &domain_volume_min, const Vector3DF &domain_volume_max, float sim_scale, float cell_size, float border )
{
	float world_cellsize = cell_size / sim_scale;

	grid_min_ = domain_volume_min;
	grid_max_ = domain_volume_max;

	grid_size_ = grid_max_;
	grid_size_ -= grid_min_;
	grid_res_.x = ceil ( grid_size_.x / world_cellsize );		
	grid_res_.y = ceil ( grid_size_.y / world_cellsize );
	grid_res_.z = ceil ( grid_size_.z / world_cellsize );
	grid_size_.x = grid_res_.x * world_cellsize;				
	grid_size_.y = grid_res_.y * world_cellsize;
	grid_size_.z = grid_res_.z * world_cellsize;
	grid_delta_ = grid_res_;									
	grid_delta_ /= grid_size_;

	grid_total_ = (int)(grid_res_.x * grid_res_.y * grid_res_.z);

	if ( grid_head_cell_particle_index_array_ != 0x0 ) 
		free (grid_head_cell_particle_index_array_);  
	grid_head_cell_particle_index_array_ = (uint*) malloc ( sizeof(uint*) * grid_total_ );
	memset ( grid_head_cell_particle_index_array_, GRID_UNDEF, grid_total_*sizeof(uint) );

	if ( grid_particles_number_ != 0x0 ) 
		free (grid_particles_number_); 
	grid_particles_number_ = (uint*) malloc ( sizeof(uint*) * grid_total_ );
	memset ( grid_particles_number_, GRID_UNDEF, grid_total_*sizeof(uint) );

	grid_search_ =  3;
	int cell = 0;
	for (int y = -1; y < 2; y++ ) 
		for (int z = -1; z < 2; z++ ) 
			for (int x = -1; x < 2; x++ ) 
				grid_neighbor_cell_index_offset_[cell++] = y * grid_res_.x *grid_res_.z + z * grid_res_.x + x ;			

	grid_adj_cnt_ = grid_search_ * grid_search_ * grid_search_;
}

int ParticleSystem::GetGridCell ( int p, Vector3DI& gc )
{
	return GetGridCell ( *(pos_+p), gc );
}

int ParticleSystem::GetGridCell ( Vector3DF& pos, Vector3DI& gc )
{
	float px = pos.x - vec_[PGRIDVOLUMEMIN].x;
	float py = pos.y - vec_[PGRIDVOLUMEMIN].y;
	float pz = pos.z - vec_[PGRIDVOLUMEMIN].z;

	if(px < 0.0)
		px = 0.0;
	if(py < 0.0)
		py = 0.0;
	if(pz < 0.0)
		pz = 0.0;

	const float cellSize = param_[PGRIDSIZEREALSCALE] / param_[PSIMSCALE];
	gc.x = (int)(px / cellSize);
	gc.y = (int)(py / cellSize);
	gc.z = (int)(pz / cellSize);

	if(gc.x > grid_res_.x - 1)
		gc.x = grid_res_.x - 1;
	if(gc.y > grid_res_.y - 1)
		gc.y = grid_res_.y - 1;
	if(gc.z > grid_res_.z - 1)
		gc.z = grid_res_.z - 1;

	return (int)(gc.y * grid_res_.x  * grid_res_.z + gc.z * grid_res_.x + gc.x);	
}

Vector3DI ParticleSystem::GetCell ( int c )
{
	Vector3DI gc;
	int xz = grid_res_.x*grid_res_.z;
	gc.y = c / xz;				c -= gc.y*xz;
	gc.z = c / grid_res_.x;		c -= gc.z*grid_res_.x;
	gc.x = c;
	return gc;
}

void ParticleSystem::InsertParticlesCPU (uint num_particle)
{	
	memset ( next_particle_index_in_the_same_cell_,		GRID_UNDEF,		num_particle*sizeof(uint) );
	memset ( particle_grid_cell_index_,					GRID_UNDEF,		num_particle*sizeof(uint) );
	memset ( cluster_cell_,								GRID_UNDEF,		num_particle*sizeof(uint) );

	memset ( grid_head_cell_particle_index_array_,		GRID_UNDEF,		grid_total_*sizeof(uint) );
	memset ( grid_particles_number_,					0,				grid_total_*sizeof(uint) );

	const int xns = grid_res_.x;
	const int yns = grid_res_.y;
	const int zns = grid_res_.z;

	param_[ PSTAT_OCCUPANCY ] = 0.0;
	param_ [ PSTAT_GRIDCOUNT ] = 0.0;

	for ( int idx = 0; idx < num_particle; idx++ ) 
	{
		Vector3DI gridCell;
		const int gridCellIndex = GetGridCell ( pos_[idx], gridCell );		

		if ( gridCell.x >= 0 && gridCell.x < xns && gridCell.y >= 0 && gridCell.y < yns && gridCell.z >= 0 && gridCell.z < zns ) 
		{
			particle_grid_cell_index_[idx] = gridCellIndex;
			next_particle_index_in_the_same_cell_[idx] = grid_head_cell_particle_index_array_[gridCellIndex];				
			if ( next_particle_index_in_the_same_cell_[idx] == GRID_UNDEF ) 
				param_[ PSTAT_OCCUPANCY ] += 1.0;
			grid_head_cell_particle_index_array_[gridCellIndex] = idx;	
			grid_particles_number_[gridCellIndex]++;
			param_ [ PSTAT_GRIDCOUNT ] += 1.0;
		} 
		else 
		{
#ifdef DEBUG
			Vector3DF vel = *(vel_ + idx);
			Vector3DF ve  = *(vel_eval_ + idx);
			float pr = *(pressure_ + idx);
			float dn = *(density_ + idx);
			printf ( "WARNING: Out of Bounds: %d, P<%f %f %f>, V<%f %f %f>, prs:%f, dns:%f\n", idx, pos_[idx].x, pos_[idx].y, pos_[idx].z, vel.x, vel.y, vel.z, pr, dn );
			pos_[idx].x = -1; pos_[idx].y = -1; pos_[idx].z = -1;
#endif
		}
	}
}

void ParticleSystem::readInFluidParticles(const char* filename, int& num_points, Vector3DF& minVec, Vector3DF& maxVec)
{
	float px, py, pz;
	float min_x, min_y, min_z;
	float max_x, max_y, max_z;
	try 
	{
		infileParticles.open(filename);
	}
	catch(...)
	{
		cout << "Error opening file" << endl;
	}

	int cnt;
	if(infileParticles)
	{
		infileParticles >> cnt;

		infileParticles >> min_x >> min_y >> min_z;
		infileParticles >> max_x >> max_y >> max_z;

		minVec.Set(min_x, min_y, min_z);
		maxVec.Set(max_x, max_y, max_z);
	}
	else
	{
		cerr << "wrong filename!" << endl; 
		exit(-1);
	}

	int readCnt = 0;
	while(infileParticles && readCnt < cnt)
	{
		infileParticles >> px >> py >> pz;							 	
		pos_[readCnt].Set(px, py, pz);	
		readCnt++;
	}

	infileParticles.close();
	num_points = cnt;
}

int ParticleSystem::readInFluidParticleNum(const char* filename)
{
	try 
	{
		infileParticles.open(filename);
	}
	catch(...)
	{
		cout << "Error opening file" << endl;
	}

	int cnt;
	if(infileParticles)
	{
		infileParticles >> cnt;
	}
	else
	{
		cerr << "wrong filename!" << endl; 
		exit(-1);
	}

	infileParticles.close();

	return cnt;
}

void ParticleSystem::SaveResults ()
{
	if ( save_neighbor_index_ != 0x0 ) free ( save_neighbor_index_ );
	if ( save_neighbor_cnt_ != 0x0 ) free ( save_neighbor_cnt_ );
	if ( save_neighbors_ != 0x0 )	free ( save_neighbors_ );

	save_neighbor_index_ = (uint*) malloc ( sizeof(uint) * num_points() );
	save_neighbor_cnt_ = (uint*) malloc ( sizeof(uint) * num_points() );
	save_neighbors_ = (uint*) malloc ( sizeof(uint) * neighbor_particles_num_ );
	memcpy ( save_neighbor_index_, neighbor_index_, sizeof(uint) * num_points() );
	memcpy ( save_neighbor_cnt_, neighbor_particle_numbers_, sizeof(uint) * num_points() );
	memcpy ( save_neighbors_, neighbor_table_, sizeof(uint) * neighbor_particles_num_ );
}

void ParticleSystem::FindNeighborsGrid (uint num_particle)
{
	float		sim_scale_square		= param_[PSIMSCALE] * param_[PSIMSCALE];
	const float smooth_radius			= param_[PSMOOTHRADIUS];
	const float smooth_radius_square	= smooth_radius * smooth_radius;
	ResetNeighbors ();

	for (int i=0; i < num_particle; i++ ) {
		ClearNeighbors ( i );
		const uint i_cell_index = particle_grid_cell_index_[i];
		if ( i_cell_index != GRID_UNDEF ) {
			for (int cell=0; cell < max_num_adj_grid_cells_cpu; cell++) {
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
				if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1))
				{
					continue;
				}		

				int j = grid_head_cell_particle_index_array_ [neighbor_cell_index] ;	
				while ( j != GRID_UNDEF ) {
					if ( i==j ) 
					{
						j = next_particle_index_in_the_same_cell_[j] ; 
						continue; 
					}

					Vector3DF pos_j_minus_i = pos_[j] - pos_[i];
					const float dist_square_sim_scale = sim_scale_square*(pos_j_minus_i.x*pos_j_minus_i.x + pos_j_minus_i.y*pos_j_minus_i.y + pos_j_minus_i.z*pos_j_minus_i.z);
					if ( dist_square_sim_scale <= smooth_radius_square ) {
						AddNeighbor( i, j, sqrt(dist_square_sim_scale) );
					}
					j = next_particle_index_in_the_same_cell_[j] ;	
				}
			}
		}
	}
}

void ParticleSystem::ComputePressureGrid ()
{
	const float	sim_scale_square		 = param_[PSIMSCALE] * param_[PSIMSCALE];
	const float smooth_radius			 = param_[PSMOOTHRADIUS];
	const float smooth_radius_square	 = smooth_radius * smooth_radius;
	const float mass					 = param_[PMASS];

	const float own_density_contribution = param_[PKERNELSELF] * mass;

	float minDens = 10e10;
	float maxDens = 0.0;
	for ( int i = 0; i < num_points(); i++ ) 
	{
		density_[i] = own_density_contribution;

		int neighbor_nums = 0;
		int search_nums = 0;
		float sum = 0.0;

		const uint i_cell_index = particle_grid_cell_index_[i];
		if ( i_cell_index != GRID_UNDEF) 
		{
			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) 
			{
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];

				if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1))
				{
					continue;
				}	

				int j = grid_head_cell_particle_index_array_ [neighbor_cell_index] ;	

				while ( j != GRID_UNDEF ) 
				{
					if ( i==j ) 
					{
						j = next_particle_index_in_the_same_cell_[j] ; 
						continue; 
					}
					Vector3DF dst_graphics_scale = pos_[j];
					dst_graphics_scale -= pos_[i];
					const float dist_square_sim_scale = sim_scale_square*(dst_graphics_scale.x*dst_graphics_scale.x + dst_graphics_scale.y*dst_graphics_scale.y + dst_graphics_scale.z*dst_graphics_scale.z);
					if ( dist_square_sim_scale <= smooth_radius_square ) 
					{
						const float dist = sqrt(dist_square_sim_scale);
						float kernelValue = kernelM4Lut(dist, smooth_radius);
						density_[i] += kernelValue * mass;

						neighbor_nums++;
					}
					search_nums++;
					j = next_particle_index_in_the_same_cell_[j] ;	
				}
			}
		}

		if (density_[i] < minDens)
			minDens = density_[i];
		if (density_[i] > maxDens)
			maxDens = density_[i];

		pressure_[i] = max(0.0f, ( density_[i] - param_[PRESTDENSITY] ) * param_[PGASCONSTANT]);		

		param_ [ PSTAT_NEIGHCNT ] = float(neighbor_nums);
		param_ [ PSTAT_SEARCHCNT ] = float(search_nums);
		if ( param_[PSTAT_NEIGHCNT] > param_ [ PSTAT_NEIGHCNTMAX ] ) 
			param_ [ PSTAT_NEIGHCNTMAX ] = param_[PSTAT_NEIGHCNT];
		if ( param_[PSTAT_SEARCHCNT] > param_ [ PSTAT_SEARCHCNTMAX ] ) 
			param_ [ PSTAT_SEARCHCNTMAX ] = param_[PSTAT_SEARCHCNT];
	}
}

void ParticleSystem::ComputeForceGrid ()
{
	const float mass					= param_[PMASS];
	const float sim_scale				= param_[PSIMSCALE];
	const float sim_scale_square		= sim_scale * sim_scale;	
	const float smooth_radius			= param_[PSMOOTHRADIUS];
	const float smooth_radius_square	= smooth_radius * smooth_radius;
	const float visc					= param_[PVISC];
	Vector3DF   vec_gravity				= vec_[PPLANE_GRAV_DIR];
	const float vterm					= lap_kern_ * visc;

	for ( int i=0; i < num_points(); i++ ) 
	{
		force_[i].Set ( 0, 0, 0 );				
		Vector3DF force(0, 0, 0);
		const uint i_cell_index = particle_grid_cell_index_[i];
		Vector3DF ipos = pos_[i];
		Vector3DF iveleval = vel_eval_[i];
		float	  ipress   = pressure_[i];
		float	  idensity = density_[i];
		if ( i_cell_index != GRID_UNDEF ) 
		{
			for (int cell=0; cell < max_num_adj_grid_cells_cpu; cell++) 
			{
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];

				if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1))
				{
					continue;
				}	

				int j = grid_head_cell_particle_index_array_ [neighbor_cell_index] ;	
				while ( j != GRID_UNDEF ) {
					if ( i==j ) 
					{ 
						j = next_particle_index_in_the_same_cell_[j]; 
						continue; 
					}

					Vector3DF vector_i_minus_j = ipos - pos_[j];
					const float dx = vector_i_minus_j.x;
					const float dy = vector_i_minus_j.y;
					const float dz = vector_i_minus_j.z;

					const float dist_square_sim_scale = sim_scale_square*(dx*dx + dy*dy + dz*dz);
					if ( dist_square_sim_scale <= smooth_radius_square && dist_square_sim_scale > 0) {
						const float jdist = sqrt(dist_square_sim_scale);
						const float jpress = pressure_[j];				
						const float h_minus_r = smooth_radius-jdist;
						const float pterm = -0.5f * h_minus_r * spiky_kern_ * ( ipress + jpress ) / jdist;
						const float dterm = h_minus_r /(idensity * density_[j]);

						Vector3DF vel_j_minus_i = vel_eval_[j];		
						vel_j_minus_i -= iveleval;

						force += vector_i_minus_j * sim_scale * pterm * dterm;		

						force += vel_j_minus_i * vterm * dterm;
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}

		force *= mass * mass;
		force += vec_gravity * mass;

		if (addBoundaryForce)
		{
			force += boxBoundaryForce(i);
		}	

		force_[i] = force;

	}
}

void ParticleSystem::computeGasConstAndTimeStep(float densityVariation)
{
	float maxParticleSpeed = 4.0f;
	float courantFactor = 0.4f;

	if(densityVariation >= 1.0)
	{
		time_step_pcisph_ = 0.001; 

		param_[PGASCONSTANT] = 70000;
		float speedOfSound = sqrt(param_[PGASCONSTANT]);
		float relevantSpeed = max(speedOfSound, maxParticleSpeed);	
		time_step_wcsph_ = courantFactor * param_[PSMOOTHRADIUS] / relevantSpeed;

		param_[PGASCONSTANT] = 1000.0f;					
		speedOfSound = sqrt(param_[PGASCONSTANT]);
		relevantSpeed = max(speedOfSound, maxParticleSpeed);	
		time_step_sph_ = courantFactor * param_[PSMOOTHRADIUS] / relevantSpeed;
	}
	else
	{
		time_step_pcisph_ = 0.0005;

		param_[PGASCONSTANT] = 6000000; 
		float speedOfSound = sqrt(param_[PGASCONSTANT]);
		float relevantSpeed = max(speedOfSound, maxParticleSpeed);	
		time_step_wcsph_ = courantFactor * param_[PSMOOTHRADIUS] / relevantSpeed;

		param_[PGASCONSTANT] = 1000.0f;				
		speedOfSound = sqrt(param_[PGASCONSTANT]);
		relevantSpeed = max(speedOfSound, maxParticleSpeed);	
		time_step_sph_ = courantFactor * param_[PSMOOTHRADIUS] / relevantSpeed;
	}

	time_step_ = time_step_sph_;

}

void ParticleSystem::ComputeOtherForceCpu()
{
	const float	mass						= param_[PMASS];
	const float sim_scale					= param_[PSIMSCALE];
	const float sim_scale_square			= sim_scale * sim_scale;
	const float smooth_radius			 = param_[PSMOOTHRADIUS];
	const float smooth_radius_square	 = smooth_radius * smooth_radius;
	const float visc						= param_[PVISC];
	Vector3DF   vec_gravity					= vec_[PPLANE_GRAV_DIR];

	const float	mass_square					= mass * mass;
	const float restVolume					= mass / param_[PRESTDENSITY];

	for ( int i=0; i < num_points(); i++ ) 
	{
		force_[i].Set ( 0, 0, 0 );				
		const uint i_cell_index = particle_grid_cell_index_[i];

		if ( particle_grid_cell_index_[i] != GRID_UNDEF ) {
			for (int cell=0; cell < grid_adj_cnt_; cell++) {
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];

				if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1))
				{
					continue;
				}	

				int j = grid_head_cell_particle_index_array_ [neighbor_cell_index] ;	
				while ( j != GRID_UNDEF ) {
					if ( i==j ) 
					{ 
						j = next_particle_index_in_the_same_cell_[j];
						continue; 
					}		

					Vector3DF vector_i_minus_j = pos_[i] - pos_[j];
					const float dx = vector_i_minus_j.x;
					const float dy = vector_i_minus_j.y;
					const float dz = vector_i_minus_j.z;

					const float dist_square_sim_scale = sim_scale_square*(dx*dx + dy*dy + dz*dz);
					if ( dist_square_sim_scale <= smooth_radius_square ) 
					{
						const float dist = sqrt(dist_square_sim_scale);
						Vector3DF vel_eval_j_i = vel_eval_[j] - vel_eval_[i];		
						force_[i] += vel_eval_j_i * visc * restVolume * restVolume * (smooth_radius-dist) * lap_kern_;
					}
					j = next_particle_index_in_the_same_cell_[j];	
				}
			}
		}

		force_[i] += vec_gravity * mass;

		if (addBoundaryForce)
		{
			force_[i] += boxBoundaryForce(i);
		}

		correction_pressure_[i] = 0.0f;
		correction_pressure_force_[i].Set(0,0,0);
	}
}

void ParticleSystem::ComputeForceGridNC ()
{
	const float d					= param_[PSIMSCALE];
	const float sim_scale_square	= d * d;
	const float smooth_radius		= param_[PSMOOTHRADIUS];
	const float visc				= param_[PVISC];

	Vector3DF*	ipos = pos_;
	Vector3DF*	iveleval = vel_eval_;
	Vector3DF*	iforce = force_;
	float*		ipress = pressure_;
	float*		idensity = density_;
	uint*		i_neighbor_index =	neighbor_index_;
	uint*		i_neighbor_particle_nums = neighbor_particle_numbers_;

	for (int i=0; i < num_points(); i++ ) {
		iforce->Set ( 0, 0, 0 );	
		int j_index = *i_neighbor_index;

		for (int nbr=0; nbr < *i_neighbor_particle_nums; nbr++ ) {
			int j = *(neighbor_table_+j_index);
			Vector3DF jpos = *(pos_ + j);
			float jpress = *(pressure_ + j);
			float jdensity = *(density_ + j);
			Vector3DF jveleval = *(vel_eval_ + j);
			float jdist = *(neighbor_dist_ + j_index);			
			float dx = ( ipos->x - jpos.x);		
			float dy = ( ipos->y - jpos.y);
			float dz = ( ipos->z - jpos.z);
			const float c = ( smooth_radius - jdist );
			const float pterm = d * -0.5f * c * spiky_kern_ * ( *ipress + jpress ) / jdist;
			const float dterm = c / ((*idensity) * jdensity);
			const float vterm = lap_kern_ * visc;
			iforce->x += ( pterm * dx + vterm * ( jveleval.x - iveleval->x) ) * dterm;
			iforce->y += ( pterm * dy + vterm * ( jveleval.y - iveleval->y) ) * dterm;
			iforce->z += ( pterm * dz + vterm * ( jveleval.z - iveleval->z) ) * dterm;
			j_index++;
		}				
		ipos++;
		iveleval++;
		iforce++;
		ipress++;
		idensity++;
		i_neighbor_index++;
	}
}


void ParticleSystem::SetupRender ()
{
	glEnable ( GL_TEXTURE_2D );

	glGenTextures ( 1, (GLuint*) texture_ );
	glBindTexture ( GL_TEXTURE_2D, texture_[0] );
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);	
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );	
	glPixelStorei( GL_UNPACK_ALIGNMENT, 4);	
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F, 8, 8, 0, GL_RGB, GL_FLOAT, 0);

	glGenBuffers ( 3, (GLuint*) vbo_ );

	const int udiv = 6;
	const int vdiv = 6;
	const float du = 180.0 / udiv;
	const float dv = 360.0 / vdiv;
	const float r = 1.0;

	Vector3DF* buf = (Vector3DF*) malloc ( sizeof(Vector3DF) * (udiv+2)*(vdiv+2)*2 );
	Vector3DF* dat = buf;

	sphere_points_ = 0;
	for ( float tilt=-90; tilt <= 90.0; tilt += du) {
		for ( float ang=0; ang <= 360; ang += dv) {
			float x = sin ( ang*DEGtoRAD) * cos ( tilt*DEGtoRAD );
			float y = cos ( ang*DEGtoRAD) * cos ( tilt*DEGtoRAD );
			float z = sin ( tilt*DEGtoRAD ) ;
			float x1 = sin ( ang*DEGtoRAD) * cos ( (tilt+du)*DEGtoRAD ) ;
			float y1 = cos ( ang*DEGtoRAD) * cos ( (tilt+du)*DEGtoRAD ) ;
			float z1 = sin ( (tilt+du)*DEGtoRAD );

			dat->x = x*r;
			dat->y = y*r;
			dat->z = z*r;
			dat++;
			dat->x = x1*r;
			dat->y = y1*r;
			dat->z = z1*r;
			dat++;
			sphere_points_ += 2;
		}
	}
	glBindBuffer ( GL_ARRAY_BUFFER, vbo_[2] );
	glBufferData ( GL_ARRAY_BUFFER, sphere_points_*sizeof(Vector3DF), buf, GL_STATIC_DRAW);
	glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );

	free ( buf );

	image_.read ( "ball32.bmp", "ball32a.bmp" );
}


void ParticleSystem::DrawCell ( int gx, int gy, int gz )
{
	Vector3DF gd (1, 1, 1);
	Vector3DF gc;
	gd /= grid_delta_;		
	gc.Set ( gx, gy, gz );
	gc /= grid_delta_;
	gc += grid_min_;
	glBegin ( GL_LINES );
	glVertex3f ( gc.x, gc.y, gc.z ); glVertex3f ( gc.x+gd.x, gc.y, gc.z );
	glVertex3f ( gc.x, gc.y+gd.y, gc.z ); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z );
	glVertex3f ( gc.x, gc.y, gc.z+gd.z ); glVertex3f ( gc.x+gd.x, gc.y, gc.z+gd.z );
	glVertex3f ( gc.x, gc.y+gd.y, gc.z+gd.z ); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z+gd.z );

	glVertex3f ( gc.x, gc.y, gc.z ); glVertex3f ( gc.x, gc.y+gd.y, gc.z );
	glVertex3f ( gc.x+gd.x, gc.y, gc.z ); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z );
	glVertex3f ( gc.x, gc.y, gc.z+gd.z ); glVertex3f ( gc.x, gc.y+gd.y, gc.z+gd.z );
	glVertex3f ( gc.x+gd.x, gc.y, gc.z+gd.z ); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z+gd.z );

	glVertex3f ( gc.x, gc.y, gc.z ); glVertex3f ( gc.x, gc.y, gc.z+gd.x );
	glVertex3f ( gc.x, gc.y+gd.y, gc.z ); glVertex3f ( gc.x, gc.y+gd.y, gc.z+gd.z );
	glVertex3f ( gc.x+gd.x, gc.y, gc.z ); glVertex3f ( gc.x+gd.x, gc.y, gc.z+gd.z );
	glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z+gd.z );
	glEnd ();
}

void ParticleSystem::DrawGrid ()
{
	Vector3DF gd (1, 1, 1);
	Vector3DF gc;
	gd /= grid_delta_;		

	glBegin ( GL_LINES );	
	for (int z=0; z <= grid_res_.z; z++ ) {
		for (int y=0; y <= grid_res_.y; y++ ) {
			gc.Set ( 1, y, z);	
			gc /= grid_delta_;	
			gc += grid_min_;
			glVertex3f ( grid_min_.x, gc.y, gc.z );	
			glVertex3f ( grid_max_.x, gc.y, gc.z );
		}
	}
	for (int z=0; z <= grid_res_.z; z++ ) {
		for (int x=0; x <= grid_res_.x; x++ ) {
			gc.Set ( x, 1, z);	
			gc /= grid_delta_;	
			gc += grid_min_;
			glVertex3f ( gc.x, grid_min_.y, gc.z );	
			glVertex3f ( gc.x, grid_max_.y, gc.z );
		}
	}
	for (int y=0; y <= grid_res_.y; y++ ) {
		for (int x=0; x <= grid_res_.x; x++ ) {
			gc.Set ( x, y, 1);	
			gc /= grid_delta_;	
			gc += grid_min_;
			glVertex3f ( gc.x, gc.y, grid_min_.z );	
			glVertex3f ( gc.x, gc.y, grid_max_.z );
		}
	}
	glEnd ();
}

void ParticleSystem::DrawParticle ( int p, int r1, int r2, Vector3DF clr )
{
	Vector3DF* ppos = pos_ + p;
	DWORD* pclr = clr_ + p;

	glDisable ( GL_DEPTH_TEST );

	glPointSize ( r2 );	
	glBegin ( GL_POINTS );
	glColor3f ( clr.x, clr.y, clr.z ); glVertex3f ( ppos->x, ppos->y, ppos->z );
	glEnd ();

	glEnable ( GL_DEPTH_TEST );
}

void ParticleSystem::DrawNeighbors ( int p )
{
	if ( p == -1 ) return;

	Vector3DF* ppos = pos_ + p;
	Vector3DF jpos;
	CLRVAL jclr;
	int j;

	glBegin ( GL_LINES );
	int cnt = *(neighbor_particle_numbers_ + p);
	int ndx = *(neighbor_index_ + p);
	for ( int n=0; n < cnt; n++ ) {
		j = neighbor_table_[ ndx ];
		jpos = *(pos_ + j);
		jclr = *(clr_ + j);
		glColor4f ( (RED(jclr)+1.0)*0.5, (GRN(jclr)+1.0)*0.5, (BLUE(jclr)+1.0)*0.5, ALPH(jclr) );
		glVertex3f ( ppos->x, ppos->y, ppos->z );

		jpos -= *ppos; jpos *= 0.9;		
		glVertex3f ( ppos->x + jpos.x, ppos->y + jpos.y, ppos->z + jpos.z );
		ndx++;
	}
	glEnd ();
}

void ParticleSystem::DrawCircle ( Vector3DF pos, float r, Vector3DF clr, Camera3D& cam )
{
	glPushMatrix ();

	glTranslatef ( pos.x, pos.y, pos.z );
	glMultMatrixf ( cam.getInvView().GetDataF() );
	glColor3f ( clr.x, clr.y, clr.z );
	glBegin ( GL_LINE_LOOP );
	float x, y;
	for (float a=0; a < 360; a += 10.0 ) {
		x = cos ( a*DEGtoRAD )*r;
		y = sin ( a*DEGtoRAD )*r;
		glVertex3f ( x, y, 0 );
	}
	glEnd ();

	glPopMatrix ();
}

void ParticleSystem::DrawText ()
{
	char msg[100];
	for (int n = 0; n < num_points(); n++) {

		sprintf ( msg, "%d", n );
		glColor4f ( (RED(clr_[n])+1.0)*0.5, (GRN(clr_[n])+1.0)*0.5, (BLUE(clr_[n])+1.0)*0.5, ALPH(clr_[n]) );
		drawText3D ( pos_[n].x, pos_[n].y, pos_[n].z, msg );
	}
}

void ParticleSystem::Draw ( Camera3D& cam, float rad )
{
	float* pdens;		

	glDisable ( GL_LIGHTING );

	if (toggle_[PDRAWGRIDCELLS])
	{
		glColor4f ( 0.0, 0.0, 1.0, 0.1);
		DrawGrid ();
	}

	if (toggle_[PDRAWDOMAIN])
	{
		DrawDomain(vec_[PBOUNDARYMIN], vec_[PBOUNDARYMAX]);
	}

	if (toggle_[PDRAWGRIDBOUND])
	{
		DrawDomain(vec_[PGRIDVOLUMEMIN], vec_[PGRIDVOLUMEMAX]);
	}

	if ( param_[PDRAWTEXT] == 1.0 ) {
		DrawText ();
	};

	switch ( (int) param_[PDRAWMODE] ) {
	case 0: 
		{
			glPointSize ( 6 );
			glEnable ( GL_POINT_SIZE );		
			glEnable( GL_BLEND ); 
			glBindBuffer ( GL_ARRAY_BUFFER, vbo_[0] );
			glBufferData ( GL_ARRAY_BUFFER, num_points()*sizeof(Vector3DF), pos_, GL_DYNAMIC_DRAW);		
			glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );				
			glBindBuffer ( GL_ARRAY_BUFFER, vbo_[1] );
			glBufferData ( GL_ARRAY_BUFFER, num_points()*sizeof(uint), clr_, GL_DYNAMIC_DRAW);
			glColorPointer ( 4, GL_UNSIGNED_BYTE, 0, 0x0 ); 
			glEnableClientState ( GL_VERTEX_ARRAY );
			glEnableClientState ( GL_COLOR_ARRAY );          
			glNormal3f ( 0, 0.001, 1 );
			glColor3f ( 1, 1, 1 );
			glDrawArrays ( GL_POINTS, 0, num_points() );
			glDisableClientState ( GL_VERTEX_ARRAY );
			glDisableClientState ( GL_COLOR_ARRAY );
		} 
		break;

	case 1: 
		{
			glEnable ( GL_LIGHTING );		
			glEnable(GL_BLEND); 
			glEnable(GL_ALPHA_TEST); 
			glAlphaFunc( GL_GREATER, 0.5 ); 
			glEnable ( GL_COLOR_MATERIAL );
			glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE ); //根据glColor所设置的值来指定材料参数

			// 点精灵大小
			glPointSize ( 32 );	
			glEnable ( GL_POINT_SIZE );		
			glEnable(GL_POINT_SPRITE); 		
			float quadratic[] =  { 0.0f, 0.3f, 0.00f };
			glEnable (  GL_POINT_DISTANCE_ATTENUATION  );
			glPointParameterfv(  GL_POINT_DISTANCE_ATTENUATION, quadratic );
			float maxSize = 64.0f;
			glGetFloatv( GL_POINT_SIZE_MAX, &maxSize );
			glPointSize( maxSize );
			glPointParameterf( GL_POINT_SIZE_MAX, maxSize );
			glPointParameterf( GL_POINT_SIZE_MIN, 1.0f );

			// 纹理&混合模式
			glEnable ( GL_TEXTURE_2D );
			glBindTexture ( GL_TEXTURE_2D, image_.getID() );
			glTexEnvi (GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
			glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_BLEND );
			glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) ;

			glBindBuffer ( GL_ARRAY_BUFFER, vbo_[0] );
			glBufferData ( GL_ARRAY_BUFFER, num_points()*sizeof(Vector3DF), pos_, GL_DYNAMIC_DRAW);		
			glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );				
			glBindBuffer ( GL_ARRAY_BUFFER, vbo_[1] );
			glBufferData ( GL_ARRAY_BUFFER, num_points()*sizeof(uint), clr_, GL_DYNAMIC_DRAW);
			glColorPointer ( 4, GL_UNSIGNED_BYTE, 0, 0x0 ); 
			glEnableClientState ( GL_VERTEX_ARRAY );
			glEnableClientState ( GL_COLOR_ARRAY );

			glNormal3f ( 0, 1, 0.001  );
			glColor3f ( 1, 1, 1 );
			glDrawArrays ( GL_POINTS, 0, num_points() );

			glDisableClientState ( GL_VERTEX_ARRAY );
			glDisableClientState ( GL_COLOR_ARRAY );
			glDisable (GL_POINT_SPRITE); 
			glDisable ( GL_ALPHA_TEST );
			glDisable ( GL_TEXTURE_2D );
		}
		break;

	case 2: 
		{
			glEnable ( GL_LIGHTING );
			pdens = density_;

			for (int n = 0; n < num_points(); n++) {
				glPushMatrix ();
				glTranslatef ( pos_[n].x, pos_[n].y, pos_[n].z );		
				glScalef ( rad, rad, rad );			
				glColor4f ( RED(clr_[n]), GRN(clr_[n]), BLUE(clr_[n]), ALPH(clr_[n]) );
				drawSphere ();
				glPopMatrix ();		
			}
		}
		break;
	};
}

std::string ParticleSystem::GetModeStr ()
{
	char buf[100];

	switch ( (int) param_[PRUN_MODE] ) {
	case RUN_CPU_SPH:		
		sprintf ( buf, "SIMULATE CPU SPH");		
		break;
	case RUN_CUDA_INDEX_SPH:	
		sprintf ( buf, "SIMULATE CUDA Index Sort SPH" ); 
		break;
	case RUN_CUDA_FULL_SPH:	
		sprintf ( buf, "SIMULATE CUDA Full Sort SPH" ); 
		break;
	case RUN_CPU_PCISPH:		
		sprintf ( buf, "SIMULATE CPU PCISPH");		
		break;
	case RUN_CUDA_INDEX_PCISPH: 
		sprintf ( buf, "SIMULATE CUDA Index Sort PCISPH" );	
		break;
	case RUN_CUDA_FULL_PCISPH: 
		sprintf ( buf, "SIMULATE CUDA Full Sort PCISPH" );	
		break;
	};
	return buf;
};

int ParticleSystem::SelectParticle ( int x, int y, int wx, int wy, Camera3D& cam )
{
	Vector4DF pnt;
	Vector3DF* ppos = pos_;

	for (int n = 0; n < num_points(); n++ ) {
		pnt = cam.project ( *ppos );
		pnt.x = (pnt.x+1.0)*0.5 * wx;
		pnt.y = (pnt.y+1.0)*0.5 * wy;

		if ( x > pnt.x-8 && x < pnt.x+8 && y > pnt.y-8 && y < pnt.y+8 ) {
			selected_ = n;
			return n;
		}
		ppos++;
	}
	selected_ = -1;
	return -1;
}

void ParticleSystem::DrawParticleInfo ( int p )
{
	char disp[256];

	glColor4f ( 1.0, 1.0, 1.0, 1.0 );
	sprintf ( disp, "Particle: %d", p );		drawText2D ( 10, 20, disp ); 

	Vector3DI gc;
	int gs = GetGridCell ( p, gc );
	sprintf ( disp, "Grid Cell:    <%d, %d, %d> id: %d", gc.x, gc.y, gc.z, gs );		drawText2D ( 10, 40, disp ); 

	int cc = *(cluster_cell_ + p);
	gc = GetCell ( cc );
	sprintf ( disp, "Cluster Cell: <%d, %d, %d> id: %d", gc.x, gc.y, gc.z, cc );		drawText2D ( 10, 50, disp ); 

	sprintf ( disp, "Neighbors:    " );

	int cnt = *(neighbor_particle_numbers_ + p);
	int ndx = *(neighbor_index_ + p);
	for ( int n=0; n < cnt; n++ ) {
		sprintf ( disp, "%s%d, ", disp, neighbor_table_[ ndx ] );
		ndx++;
	}
	drawText2D ( 10, 70, disp );

	if ( cc != -1 ) {
		sprintf ( disp, "Cluster Group: ");		drawText2D ( 10, 90, disp);
		int cadj;
		int stotal = 0;
		for (int n=0; n < max_num_adj_grid_cells_cpu; n++ ) {		
			cadj = cc + grid_neighbor_cell_index_offset_[n];

			if (cadj == GRID_UNDEF || (cadj < 0 || cadj > grid_total_ - 1))
			{
				continue;
			}
			gc = GetCell ( cadj );
			sprintf ( disp, "<%d, %d, %d> id: %d, cnt: %d ", gc.x, gc.y, gc.z, cc+grid_neighbor_cell_index_offset_[n], grid_particles_number_[ cadj ] );	
			drawText2D ( 20, 100+n*10, disp );
			stotal += grid_particles_number_[cadj];
		}

		sprintf ( disp, "Search Overhead: %f (%d of %d), %.2f%% occupancy", float(stotal)/ cnt, cnt, stotal, float(cnt)*100.0/stotal );
		drawText2D ( 10, 380, disp );
	}	
}

void ParticleSystem::SetupKernels ()
{
	if (param_[PSPACINGREALWORLD] == 0)
	{
		param_[PSPACINGREALWORLD] = pow ( (float)param_[PMASS] / (float)param_[PRESTDENSITY], 1/3.0f );
	}
	param_[PKERNELSELF] = kernelM4(0.0f, param_[PSMOOTHRADIUS]);				
	poly6_kern_ = 315.0f / (64.0f * MY_PI * pow( param_[PSMOOTHRADIUS], 9) );	
	spiky_kern_ = -45.0f / (MY_PI * pow( param_[PSMOOTHRADIUS], 6) );			
	lap_kern_ = 45.0f / (MY_PI * pow( param_[PSMOOTHRADIUS], 6) );				

	float sr = param_[PSMOOTHRADIUS];
	for(int i=0; i<lutSize; i++)
	{
		float dist = sr * i / lutSize;
		lutKernelM4[i] = kernelM4(dist, sr);
		lutKernelPressureGrad[i] = kernelPressureGrad(dist, sr);
	}	

}

void ParticleSystem::SetupDefaultParams ()
{
	param_ [ PMAXNUM ]					= 1048576;			
	param_ [ PSIMSCALE ]				= 0.005;			
	param_ [ PGRID_DENSITY ]			= 1.0;
	param_ [ PVISC ]					= 1.002; //4.0;			
	param_ [ PRESTDENSITY ]				= 1000.0;			
	param_ [ PMASS ]					= 0.001953125;				
	param_ [ PCOLLISIONRADIUS ]			= 0.00775438;		
	param_ [ PSPACINGREALWORLD ]		= 0.0125;			
	param_ [ PSMOOTHRADIUS ]			= 0.025025;			
	param_ [ PGRIDSIZEREALSCALE]		= param_[PSMOOTHRADIUS] / param_[PGRID_DENSITY];
	param_ [ PGASCONSTANT ]				= 1000.0;
	param_ [ PBOUNDARYSTIFF ]			= 2.5;
	param_ [ PBOUNDARYDAMP ]			= 1.0;
	param_ [ PACCEL_LIMIT ]				= 150.0;			
	param_ [ PVEL_LIMIT ]				= 3.0;				
	param_ [ PSPACINGGRAPHICSWORLD ]	= 0.0;				
	param_ [ PGROUND_SLOPE ]			= 0.0;
	param_ [ PFORCE_MIN ]				= 0.0;
	param_ [ PFORCE_MAX ]				= 0.0;	
	param_ [ PDRAWMODE]					= 2;			
	param_ [ PDRAWTEXT]					= 0;				
	param_ [ PPOINT_GRAV_AMT]			= 0.0;
	param_ [ PSTAT_NEIGHCNTMAX]			= 0 ;
	param_ [ PSTAT_SEARCHCNTMAX]		= 0 ;
	param_ [ PFORCE_FREQ ]				= 8.0;	
	param_ [ PMINLOOPPCISPH]			= 3;				
	param_ [ PMAXLOOPPCISPH]			= MAX_PCISPH_LOOPS;
	param_ [ PMAXDENSITYERRORALLOWED]   = 5.0;				

	vec_ [ PEMIT_POS ].Set ( 0, 0, 0 );
	vec_ [ PEMIT_ANG ].Set ( 0, 90, 1.0 );
	vec_ [ PEMIT_RATE ].Set ( 0, 0, 0 );
	vec_ [ PPOINT_GRAV_POS ].Set ( 0, 50, 0 );
	vec_ [ PPLANE_GRAV_DIR ].Set ( 0, -9.8, 0 );

	toggle_ [ PPAUSE ]					= false;								
	toggle_ [ PWRAP_X ]					= false;
	toggle_ [ PWALL_BARRIER ]			= false;
	toggle_ [ PLEVY_BARRIER ]			= false;
	toggle_ [ PDRAIN_BARRIER ]			= false;
	toggle_ [ PPROFILE ]				= false;
	toggle_ [ PCAPTURE ]				= false;
	toggle_ [ PPRINTDEBUGGINGINFO]		= false;
	toggle_ [ PDRAWDOMAIN ]				= false;
	toggle_ [ PDRAWGRIDBOUND]			= false;
	toggle_ [ PDRAWGRIDCELLS]			= false;

}

void ParticleSystem::ParseXML ( std::string name, int id, bool bStart )
{
	xml.setBase ( name, id );

	xml.assignValueF ( &time_step_, "DT" );
	xml.assignValueStr ( scene_name_, "Name" );
	if (bStart)	xml.assignValueF ( &param_[PMAXNUM],"Num" );
	xml.assignValueF ( &param_[PGRID_DENSITY],		"GridDensity" );
	xml.assignValueF ( &param_[PSIMSCALE],			"SimScale" );
	xml.assignValueF ( &param_[PVISC],				"Viscosity" );
	xml.assignValueF ( &param_[PRESTDENSITY],		"RestDensity" );
	xml.assignValueF ( &param_[PSPACINGGRAPHICSWORLD],	"SpacingGraphicsWorld" );
	xml.assignValueF ( &param_[PMASS],				"Mass" );
	xml.assignValueF ( &param_[PCOLLISIONRADIUS],	"Radius" );
	xml.assignValueF ( &param_[PSPACINGREALWORLD],	"SearchDist" );
	xml.assignValueF ( &param_[PGASCONSTANT],		"IntStiff" );
	xml.assignValueF ( &param_[PBOUNDARYSTIFF],		"BoundStiff" );
	xml.assignValueF ( &param_[PBOUNDARYDAMP],		"BoundDamp" );
	xml.assignValueF ( &param_[PACCEL_LIMIT],		"AccelLimit" );
	xml.assignValueF ( &param_[PVEL_LIMIT],			"VelLimit" );
	xml.assignValueF ( &param_[PPOINT_GRAV_AMT],	"PointGravAmt" );	
	xml.assignValueF ( &param_[PGROUND_SLOPE],		"GroundSlope" );
	xml.assignValueF ( &param_[PFORCE_MIN],			"WaveForceMin" );
	xml.assignValueF ( &param_[PFORCE_MAX],			"WaveForceMax" );
	xml.assignValueF ( &param_[PFORCE_FREQ],		"WaveForceFreq" );
	xml.assignValueF ( &param_[PDRAWMODE],			"DrawMode" );
	xml.assignValueF ( &param_[PDRAWTEXT],			"drawText2D" );

	xml.assignValueV3 ( &vec_[PBOUNDARYMIN],		"BoundaryMin" );
	xml.assignValueV3 ( &vec_[PBOUNDARYMAX],		"BoundaryMax" );
	xml.assignValueV3 ( &vec_[PINITPARTICLEMIN],	"InitParticleVolumeMin" );
	xml.assignValueV3 ( &vec_[PINITPARTICLEMAX],	"InitParticleVolumMax" );
	xml.assignValueV3 ( &vec_[PPOINT_GRAV_POS],		"PointGravPos" );
	xml.assignValueV3 ( &vec_[PPLANE_GRAV_DIR],		"PlaneGravDir" );

}

void ParticleSystem::SetupExampleParams ( bool bStart )
{
	switch ( (int) param_[PEXAMPLE] ) {
	case 0:	
		if (toggle_[PUSELOADEDSCENE] == true)
		{
			vec_ [ PBOUNDARYMIN ].Set ( -80, 0, -80 );
			vec_ [ PBOUNDARYMAX ].Set (  80, 160, 80 );
			vec_ [ PINITPARTICLEMIN ].Set ( -80, 0, -80 );
			vec_ [ PINITPARTICLEMAX ].Set ( 80, 10, 80 );
		} 
		else
		{
			vec_ [ PBOUNDARYMIN ].Set ( -50, 0, -50 );
			vec_ [ PBOUNDARYMAX ].Set (  50, 100, 50 );
			vec_ [ PINITPARTICLEMIN ].Set ( 6, 0, -50 );
			vec_ [ PINITPARTICLEMAX ].Set ( 50, 38, 50 );
		}
		param_ [ PFORCE_MIN ] = 0.0;	
		param_ [ PGROUND_SLOPE ] = 0.0;
		break;
	case 1:	
		if (toggle_[PUSELOADEDSCENE] == true)
		{
			vec_ [ PBOUNDARYMIN ].Set ( -80, 0, -80 );
			vec_ [ PBOUNDARYMAX ].Set (  80, 160, 80 );
			vec_ [ PINITPARTICLEMIN ].Set ( -80, 0, -80 );
			vec_ [ PINITPARTICLEMAX ].Set ( 80, 20, 80 );
		} 
		else
		{
			vec_ [ PBOUNDARYMIN ].Set ( -80, 0, -80 );
			vec_ [ PBOUNDARYMAX ].Set (  80, 160, 80 );
			vec_ [ PINITPARTICLEMIN ].Set ( -80, 0, -80 );
			vec_ [ PINITPARTICLEMAX ].Set ( 80, 20, 80 );
		}
		param_ [ PFORCE_MIN ] = 20.0;	
		param_ [ PGROUND_SLOPE ] = 0.10;
		break;
	case 2:	
		if (toggle_[PUSELOADEDSCENE] == true)
		{
			vec_ [ PBOUNDARYMIN ].Set ( -80, 0, -80 );
			vec_ [ PBOUNDARYMAX ].Set (  80, 160, 80 );
			vec_ [ PINITPARTICLEMIN ].Set ( -80, 0, -80 );
			vec_ [ PINITPARTICLEMAX ].Set ( 80, 30, 80 );
		} 
		else
		{
			vec_ [ PBOUNDARYMIN ].Set ( -80, 0, -80 );
			vec_ [ PBOUNDARYMAX ].Set (  80, 160, 80 );
			vec_ [ PINITPARTICLEMIN ].Set ( -80, 0, -80 );
			vec_ [ PINITPARTICLEMAX ].Set ( 80, 60, 80 );
		}
		param_ [ PFORCE_MIN ] = 20.0;
		param_ [ PFORCE_MAX ] = 20.0;
		vec_ [ PPLANE_GRAV_DIR ].Set ( 0.0, -9.8, 0 );	
		break;
	}

	// 从xml文件中加载场景
	ParseXML ( "Scene", (int) param_[PEXAMPLE], bStart );
}

void ParticleSystem::SetupSpacing ()
{
	if ( param_[PSPACINGGRAPHICSWORLD] == 0 ) {

		if (param_ [PSPACINGREALWORLD] == 0)
		{
			param_ [PSPACINGREALWORLD] = pow ( (float)param_[PMASS] / param_[PRESTDENSITY], 1/3.0f );	
		}

		param_ [PSPACINGGRAPHICSWORLD] = param_ [ PSPACINGREALWORLD ] / param_[ PSIMSCALE ];		
	} else {
		param_ [PSPACINGREALWORLD] = param_[PSPACINGGRAPHICSWORLD] * param_[PSIMSCALE];
		param_ [PRESTDENSITY] = param_[PMASS] / pow ( (float)param_[PSPACINGREALWORLD], 3.0f );
	}

	vec_[PGRIDVOLUMEMIN] = vec_[PBOUNDARYMIN] - (param_[PSMOOTHRADIUS] / param_[PSIMSCALE]);	
	vec_[PGRIDVOLUMEMAX] = vec_[PBOUNDARYMAX] + (param_[PSMOOTHRADIUS] / param_[PSIMSCALE]);
}

void ParticleSystem::SetupBoundaryParams()
{
	addBoundaryForce = true;
	boundaryForceFactor = 256;
	float spacing = param_ [PSPACINGREALWORLD];
	maxBoundaryForce = boundaryForceFactor * spacing * spacing; 
	forceDistance = 0.25 * spacing;
}

float ParticleSystem::kernelM4(float dist, float sr)
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

float ParticleSystem::kernelM4Lut(float dist, float sr)
{
	int index = dist / sr * lutSize;

	if(index >= lutSize) return 0.0f;
	else return lutKernelM4[index];
}

float ParticleSystem::kernelPressureGrad(float dist, float sr)
{
	if(dist == 0)
		return 0.0f;
	if(dist > sr)
		return 0.0f;

	float kernelPressureConst = -45.f/((float(MY_PI)*sr*sr*sr*sr*sr*sr));
	return kernelPressureConst / dist * (sr-dist)*(sr-dist);
}

float ParticleSystem::kernelPressureGradLut(float dist, float sr)
{
	int index = dist / sr * lutSize;
	if(index >= lutSize) return 0.0f;
	else return lutKernelPressureGrad[index];
}

Vector3DF ParticleSystem::boxBoundaryForce(const uint i)
{
	static const float invForceDist = 1.0 / forceDistance;
	static const float forceStrength = maxBoundaryForce;
	static const Vector3DF bound_min = vec_[PBOUNDARYMIN];
	static const Vector3DF bound_max = vec_[PBOUNDARYMAX];
	float distToWall, factor;
	Vector3DF force(0.0, 0.0, 0.0);

	if (pos_[i].y < bound_min.y+forceDistance) {
		distToWall = bound_min.y+forceDistance - pos_[i].y;
		factor = distToWall * invForceDist;
		force += Vector3DF(0, 1, 0) * factor * 2.0f * forceStrength;
	} else if (pos_[i].y > bound_max.y-forceDistance) {
		distToWall = pos_[i].y - (bound_max.y-forceDistance);
		factor = distToWall * invForceDist;
		force += Vector3DF(0, -1, 0) * factor * forceStrength;
	}

	if (pos_[i].x < bound_min.x+forceDistance) {
		distToWall = bound_min.x+forceDistance - pos_[i].x;
		factor = distToWall * invForceDist;
		force += Vector3DF(1, 0, 0) * factor * forceStrength;
	}
	if (pos_[i].x > bound_max.x-forceDistance) {
		distToWall = pos_[i].x - (bound_max.x-forceDistance);
		factor = distToWall * invForceDist;
		force += Vector3DF(-1, 0, 0) * 1 * factor * forceStrength;  
	}		

	if (pos_[i].z < bound_min.z+forceDistance) {
		distToWall = bound_min.z+forceDistance - pos_[i].z;
		factor = distToWall * invForceDist;
		force += Vector3DF(0, 0, 1) * factor * forceStrength;
	} 
	if (pos_[i].z > bound_max.z-forceDistance) {
		distToWall = pos_[i].z - (bound_max.z-forceDistance);
		factor = distToWall * invForceDist;
		force += Vector3DF(0, 0, -1) * factor * forceStrength;
	}

	return force;
}
