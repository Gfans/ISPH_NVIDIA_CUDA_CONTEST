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
/*   PCISPH is integrated by Xiao Nie for NVIDIA°Øs 2013 CUDA Campus Programming Contest    */
/*                     https://github.com/Gfans/ISPH_NVIDIA_CUDA_CONTEST                    */
/*   For the PCISPH, please refer to the paper "Predictive-Corrective Incompressible SPH"   */
/********************************************************************************************/

#ifndef DEF_CAMERA_3D
#define	DEF_CAMERA_3D

#include "matrix.h"
#include "vector.h"
#include "pivotx.h"	

#define DEG_TO_RAD			(MY_PI/180.0)

class  Camera3D : public PivotX {
public:
	enum eProjection {
		Perspective = 0,
		Parallel = 1
	};
	Camera3D ();

	void draw_gl();

	// …Ë÷√
	void setAspect ( float asp )					{ m_aspect = asp;			updateMatrices(); }
	void setPos ( float x, float y, float z )		{ m_fromPos.Set(x,y,z);		updateMatrices(); }
	void setToPos ( float x, float y, float z )		{ m_toPos.Set(x,y,z);		updateMatrices(); }
	void setFov (float fov)							{ m_fov = fov;				updateMatrices(); }
	void setNearFar (float n, float f )				{ m_near = n; m_far = f;	updateMatrices(); }
	void setTile ( float x1, float y1, float x2, float y2 )		{ m_tile.Set ( x1, y1, x2, y2 );		updateMatrices(); }
	void setProjection (eProjection proj_type);
	void setModelMatrix ();
	void setModelMatrix ( Matrix4F& model );

	// “∆∂Ø
	void setOrbit  ( float ax, float ay, float az, Vector3DF tp, float dist, float dolly );
	void setOrbit  ( Vector3DF angs, Vector3DF tp, float dist, float dolly );
	void setAngles ( float ax, float ay, float az );
	void moveOrbit ( float ax, float ay, float az, float dist );		
	void moveToPos ( float tx, float ty, float tz );		
	void moveRelative ( float dx, float dy, float dz );

	// frustum ≤‚ ‘
	bool pointInFrustum ( float x, float y, float z );
	bool boxInFrustum ( Vector3DF bmin, Vector3DF bmax);
	float calculateLOD ( Vector3DF pnt, float minlod, float maxlod, float maxdist );

	// ∏®÷˙∫Ø ˝
	void updateMatrices ();					
	void updateFrustum ();						
	Vector3DF inverseRay ( float x, float y, float z );
	Vector4DF project ( Vector3DF& p );
	Vector4DF project ( Vector3DF& p, Matrix4F& vm );		

	void getVectors ( Vector3DF& dir, Vector3DF& up, Vector3DF& side )	{ dir = m_dirVec; up = m_upVec; side = m_sideVec; }
	void getBounds ( float dst, Vector3DF& min, Vector3DF& max );
	float getNear ()				{ return m_near; }
	float getFar ()					{ return m_far; }
	float getFov ()					{ return m_fov; }
	float getDolly()				{ return m_dolly; }	
	float getOrbitDist()			{ return m_orbitDist; }
	Vector3DF& getUpDir ()			{ return m_upDir; }
	Vector4DF& getTile ()			{ return m_tile; }
	Matrix4F& getViewMatrix ()		{ return m_viewMatrix; }
	Matrix4F& getInvView ()			{ return m_invRotMatrix; }
	Matrix4F& getProjMatrix ()		{ return m_tileProjMatrix; }	
	Matrix4F& getFullProjMatrix ()	{ return m_projMatrix; }
	Matrix4F& getModelMatrix()		{ return m_modelMatrix; }
	Matrix4F& getMVMatrix()			{ return m_mvMatrix; }
	float getAspect ()				{ return m_aspect; }

public:
	eProjection		m_projType;								

	// …„œÒª˙≤Œ ˝								
	float			m_dolly;									
	float			m_orbitDist;
	float			m_fov, m_aspect;							
	float			m_near, m_far;							
	Vector3DF		m_dirVec, m_sideVec, m_upVec;				
	Vector3DF		m_upDir;
	Vector4DF		m_tile;

	// ±‰ªªæÿ’Û
	Matrix4F		m_rotateMatrix;							
	Matrix4F		m_viewMatrix;							
	Matrix4F		m_projMatrix;							
	Matrix4F		m_invRotMatrix;							
	Matrix4F		m_invProjMatrix;
	Matrix4F		m_tileProjMatrix;						
	Matrix4F		m_modelMatrix;
	Matrix4F		m_mvMatrix;
	float			m_frustum[6][4];							

	bool			m_ops[8];
	int				m_wire;

};

#endif


