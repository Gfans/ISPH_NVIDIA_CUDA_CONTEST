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
/*   PCISPH is integrated by Xiao Nie for NVIDIA¡¯s 2013 CUDA Campus Programming Contest    */
/*                     https://github.com/Gfans/ISPH_NVIDIA_CUDA_CONTEST                    */
/*   For the PCISPH, please refer to the paper "Predictive-Corrective Incompressible SPH"   */
/********************************************************************************************/

#include "gl_helper.h"

#include "camera3d.h"

Camera3D::Camera3D ()
{	
	
	m_projType = Perspective;
	m_wire = 0;

	m_upDir.Set ( 0.0, 1.0, 0 );			
	m_aspect = 800/600.0;
	m_dolly = 5.0;
	m_fov = 40.0;	
	m_near = 0.1;
	m_far = 6000.0;
	m_tile.Set ( 0, 0, 1, 1 );

	for (int n=0; n < 8; n++ ) m_ops[n] = false;	
	m_ops[0] = false;

	setOrbit ( 0, 45, 0, Vector3DF(0,0,0), 120.0, 5.0 );
	updateMatrices ();

}

void Camera3D::draw_gl ()
{
	Vector3DF pnt; 
	int va, vb;
	
	if ( !m_ops[0] ) return;

	if ( m_ops[5] ) {
		glPushMatrix ();
		glEnable ( GL_LIGHTING );
		glColor3f ( 1, 1, 1 );	
		Vector3DF bmin, bmax, vmin, vmax;
		int lod;
		for (float y=0; y < 100; y += 10.0 ) {
		for (float z=-100; z < 100; z += 10.0 ) {
			for (float x=-100; x < 100; x += 10.0 ) {
				bmin.Set ( x, y, z );
				bmax.Set ( x+8, y+8, z+8 );
				if ( boxInFrustum ( bmin, bmax ) ) {				
					lod = (int) calculateLOD ( bmin, 1, 5, 300.0 );
				}
			}
		}
		}
		glPopMatrix ();
	}

	glDisable ( GL_LIGHTING );	
	glLoadMatrixf ( getViewMatrix().GetDataF() );

	if ( m_ops[2] ) {
		glBegin ( GL_POINTS );
		glColor3f ( 1, 1, 0 );
		Vector3DF norm;
		Vector3DF side, up;
		for (int n=0; n < 6; n++ ) {
			norm.Set ( m_frustum[n][0], m_frustum[n][1], m_frustum[n][2] );
			glColor3f ( n/6.0, 1.0- (n/6.0), 0.5 );
			side = Vector3DF(0,1,0); side.Cross ( norm ); side.Normalize ();	
			up = side; up.Cross ( norm ); up.Normalize();
			norm *= m_frustum[n][3];
			for (float y=-50; y < 50; y += 1.0 ) {
				for (float x=-50; x < 50; x += 1.0 ) {
					if ( x*x+y*y < 1000 ) {
                        pnt = side;
                        Vector3DF tv = up;

                        tv *= y;
                        pnt *= x;
                        pnt += tv;
                        pnt -= norm;

						glVertex3f ( pnt.x, pnt.y, pnt.z );
					}
				}
			}
		}
		glEnd (); 
	}

	if ( m_ops[4] ) {
		glColor3f ( 1, 1, 1 );
		glBegin ( GL_POINTS );
		for (float z=-100; z < 100; z += 4.0 ) {
			for (float y=0; y < 100; y += 4.0 ) {
				for (float x=-100; x < 100; x += 4.0 ) {
					if ( pointInFrustum ( x, y, z) ) {
						glVertex3f ( x, y, z );
					}
				}
			}
		}
		glEnd ();
	}
	
	if ( m_ops[3] ) {
		glBegin ( GL_LINES );
		glColor3f ( 0, 1, 0);
		for (float x = 0; x <= 1.0; x+= 0.5 ) {
			for (float y = 0; y <= 1.0; y+= 0.5 ) {
				pnt = inverseRay ( x, y, m_far );
				pnt += m_fromPos;
				glVertex3f ( m_fromPos.x, m_fromPos.y, m_fromPos.z );		
				glVertex3f ( pnt.x, pnt.y, pnt.z );
			}
		}
		glEnd ();
	}

	Vector3DF pnts[8];
	Vector3DI edge[12];
	pnts[0].Set (  0,  0,  0 );	pnts[1].Set ( 10,  0,  0 ); pnts[2].Set ( 10,  0, 10 ); pnts[3].Set (  0,  0, 10 );		
	pnts[4].Set (  0, 10,  0 );	pnts[5].Set ( 10, 10,  0 ); pnts[6].Set ( 10, 10, 10 ); pnts[7].Set (  0, 10, 10 );		
	edge[0].Set ( 0, 1, 0 ); edge[1].Set ( 1, 2, 0 ); edge[2].Set ( 2, 3, 0 ); edge[3].Set ( 3, 0, 0 );					
	edge[4].Set ( 4, 5, 0 ); edge[5].Set ( 5, 6, 0 ); edge[6].Set ( 6, 7, 0 ); edge[7].Set ( 7, 4, 0 );					
	edge[8].Set ( 0, 4, 0 ); edge[9].Set ( 1, 5, 0 ); edge[10].Set ( 2, 6, 0 ); edge[11].Set ( 3, 7, 0 );				
	
	if ( m_ops[6] ) {
		glBegin ( GL_LINES );
		glColor3f ( 1, 1, 1);
		for (int e = 0; e < 12; e++ ) {
			va = edge[e].x;
			vb = edge[e].y;
			glVertex3f ( pnts[va].x, pnts[va].y, pnts[va].z );
			glVertex3f ( pnts[vb].x, pnts[vb].y, pnts[vb].z );
		}
		glEnd ();	
	}

	glPushMatrix ();
	glLoadMatrixf ( getViewMatrix().GetDataF() );
	glTranslatef ( m_fromPos.x, m_fromPos.y, m_fromPos.z );
	glMultMatrixf ( m_invRotMatrix.GetDataF() );				

	if ( m_ops[6] ) {
		glBegin ( GL_LINES );
		glColor3f ( 1, 0, 0);
		Vector4DF proja, projb;
		for (int e = 0; e < 12; e++ ) {
			va = edge[e].x;
			vb = edge[e].y;
			proja = project ( pnts[va] );
			projb = project ( pnts[vb] );
			if ( proja.w > 0 && projb.w > 0 && proja.w < 1 && projb.w < 1) {	
				glVertex3f ( proja.x, proja.y, proja.z );
				glVertex3f ( projb.x, projb.y, projb.z );
			}
		}
		glEnd ();
	}

	glBegin ( GL_LINES );
	float to_d = (m_fromPos - m_toPos).Length();
	glColor3f ( .8,.8,.8); glVertex3f ( 0, 0, 0 );	glVertex3f ( 0, 0, -to_d );
	glColor3f ( 1,0,0); glVertex3f ( 0, 0, 0 );		glVertex3f ( 10, 0, 0 );
	glColor3f ( 0,1,0); glVertex3f ( 0, 0, 0 );		glVertex3f ( 0, 10, 0 );
	glColor3f ( 0,0,1); glVertex3f ( 0, 0, 0 );		glVertex3f ( 0, 0, 10 );
	glEnd ();

	if ( m_ops[1] ) {

		float sy = tan ( m_fov * DEGtoRAD / 2.0);
		float sx = sy * m_aspect;
		glColor3f ( 0.8, 0.8, 0.8 );
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( -m_near*sx,  m_near*sy, -m_near );
		glVertex3f (  m_near*sx,  m_near*sy, -m_near );
		glVertex3f (  m_near*sx, -m_near*sy, -m_near );
		glVertex3f ( -m_near*sx, -m_near*sy, -m_near );
		glEnd ();

		glBegin ( GL_LINE_LOOP );
		glVertex3f ( -m_far*sx,  m_far*sy, -m_far );
		glVertex3f (  m_far*sx,  m_far*sy, -m_far );
		glVertex3f (  m_far*sx, -m_far*sy, -m_far );
		glVertex3f ( -m_far*sx, -m_far*sy, -m_far );
		glEnd ();

		float l, r, t, b;
		l = -sx + 2.0*sx*m_tile.x;					
		r = -sx + 2.0*sx*m_tile.z;
		t =  sy - 2.0*sy*m_tile.y;
		b =  sy - 2.0*sy*m_tile.w;
		glColor3f ( 0.8, 0.8, 0.0 );
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( l * m_near, t * m_near, -m_near );
		glVertex3f ( r * m_near, t * m_near, -m_near );
		glVertex3f ( r * m_near, b * m_near, -m_near );
		glVertex3f ( l * m_near, b * m_near, -m_near );		
		glEnd ();

		glBegin ( GL_LINE_LOOP );
		glVertex3f ( l * m_far, t * m_far, -m_far );
		glVertex3f ( r * m_far, t * m_far, -m_far );
		glVertex3f ( r * m_far, b * m_far, -m_far );
		glVertex3f ( l * m_far, b * m_far, -m_far );		
		glEnd ();
	}

	glPopMatrix ();
}

bool Camera3D::pointInFrustum ( float x, float y, float z )
{
	int p;
	for ( p = 0; p < 6; p++ )
		if( m_frustum[p][0] * x + m_frustum[p][1] * y + m_frustum[p][2] * z + m_frustum[p][3] <= 0 )
			return false;
	return true;
}

bool Camera3D::boxInFrustum ( Vector3DF bmin, Vector3DF bmax)
{
	Vector3DF vmin, vmax;
	int p;
	bool ret = true;	
	for ( p = 0; p < 6; p++ ) {
		vmin.x = ( m_frustum[p][0] > 0 ) ? bmin.x : bmax.x;		
		vmax.x = ( m_frustum[p][0] > 0 ) ? bmax.x : bmin.x;		
		vmin.y = ( m_frustum[p][1] > 0 ) ? bmin.y : bmax.y;
		vmax.y = ( m_frustum[p][1] > 0 ) ? bmax.y : bmin.y;
		vmin.z = ( m_frustum[p][2] > 0 ) ? bmin.z : bmax.z;
		vmax.z = ( m_frustum[p][2] > 0 ) ? bmax.z : bmin.z;
		if ( m_frustum[p][0]*vmax.x + m_frustum[p][1]*vmax.y + m_frustum[p][2]*vmax.z + m_frustum[p][3] <= 0 ) return false;		
		else if ( m_frustum[p][0]*vmin.x + m_frustum[p][1]*vmin.y + m_frustum[p][2]*vmin.z + m_frustum[p][3] <= 0 ) ret = true;		
	}
	return ret;			
}

void Camera3D::setOrbit  ( Vector3DF angs, Vector3DF tp, float dist, float dolly )
{
	setOrbit ( angs.x, angs.y, angs.z, tp, dist, dolly );
}

void Camera3D::setOrbit ( float ax, float ay, float az, Vector3DF tp, float dist, float dolly )
{
	m_angEuler.Set ( ax, ay, az );
	m_orbitDist = dist;
	m_dolly = dolly;
	float dx, dy, dz;
	dx = cos ( m_angEuler.y * DEGtoRAD ) * sin ( m_angEuler.x * DEGtoRAD ) ;
	dy = sin ( m_angEuler.y * DEGtoRAD );
	dz = cos ( m_angEuler.y * DEGtoRAD ) * cos ( m_angEuler.x * DEGtoRAD );
	m_fromPos.x = tp.x + dx * m_orbitDist;
	m_fromPos.y = tp.y + dy * m_orbitDist;
	m_fromPos.z = tp.z + dz * m_orbitDist;
	m_toPos.x = m_fromPos.x - dx * m_dolly;
	m_toPos.y = m_fromPos.y - dy * m_dolly;
	m_toPos.z = m_fromPos.z - dz * m_dolly;
	updateMatrices ();
}

void Camera3D::moveOrbit ( float ax, float ay, float az, float dd )
{
	m_angEuler += Vector3DF(ax,ay,az);
	m_orbitDist += dd;
	
	float dx, dy, dz;
	dx = cos ( m_angEuler.y * DEGtoRAD ) * sin ( m_angEuler.x * DEGtoRAD ) ;
	dy = sin ( m_angEuler.y * DEGtoRAD );
	dz = cos ( m_angEuler.y * DEGtoRAD ) * cos ( m_angEuler.x * DEGtoRAD );
	m_fromPos.x = m_toPos.x + dx * m_orbitDist;
	m_fromPos.y = m_toPos.y + dy * m_orbitDist;
	m_fromPos.z = m_toPos.z + dz * m_orbitDist;
	updateMatrices ();
}

void Camera3D::moveToPos ( float tx, float ty, float tz )
{
	m_toPos += Vector3DF(tx,ty,tz);

	float dx, dy, dz;
	dx = cos ( m_angEuler.y * DEGtoRAD ) * sin ( m_angEuler.x * DEGtoRAD ) ;
	dy = sin ( m_angEuler.y * DEGtoRAD );
	dz = cos ( m_angEuler.y * DEGtoRAD ) * cos ( m_angEuler.x * DEGtoRAD );
	m_fromPos.x = m_toPos.x + dx * m_orbitDist;
	m_fromPos.y = m_toPos.y + dy * m_orbitDist;
	m_fromPos.z = m_toPos.z + dz * m_orbitDist;
	updateMatrices ();
}

void Camera3D::setAngles ( float ax, float ay, float az )
{
	m_angEuler = Vector3DF(ax,ay,az);
	m_toPos.x = m_fromPos.x - cos ( m_angEuler.y * DEGtoRAD ) * sin ( m_angEuler.x * DEGtoRAD ) * m_dolly;
	m_toPos.y = m_fromPos.y - sin ( m_angEuler.y * DEGtoRAD ) * m_dolly;
	m_toPos.z = m_fromPos.z - cos ( m_angEuler.y * DEGtoRAD ) * cos ( m_angEuler.x * DEGtoRAD ) * m_dolly;
	updateMatrices ();
}


void Camera3D::moveRelative ( float dx, float dy, float dz )
{
	Vector3DF vec ( dx, dy, dz );
	vec *= m_invRotMatrix;
	m_toPos += vec;
	m_fromPos += vec;
	updateMatrices ();
}

void Camera3D::setProjection (eProjection proj_type)
{
	m_projType = proj_type;
}

void Camera3D::updateMatrices ()
{
	Matrix4F basis;
	Vector3DF temp;	
	
	
	m_dirVec = m_toPos;										
	m_dirVec -= m_fromPos;				
	m_dirVec.Normalize ();
	m_sideVec = m_dirVec;
	m_sideVec.Cross ( m_upDir );
	m_sideVec.Normalize ();
	m_upVec = m_sideVec;
	m_upVec.Cross ( m_dirVec );
	m_upVec.Normalize();
	m_dirVec *= -1;
	
	m_rotateMatrix.Basis (m_sideVec, m_upVec, m_dirVec );
	m_viewMatrix = m_rotateMatrix;

	m_viewMatrix.PreTranslate ( Vector3DF(-m_fromPos.x, -m_fromPos.y, -m_fromPos.z ) );

	float sx = tan ( m_fov * 0.5 * DEGtoRAD ) * m_near;
	float sy = sx / m_aspect;
	m_projMatrix = 0.0;
	m_projMatrix(0,0) = 2.0*m_near / sx;			
	m_projMatrix(1,1) = 2.0*m_near / sy;
	m_projMatrix(2,2) = -(m_far + m_near)/(m_far - m_near);			
	m_projMatrix(2,3) = -(2.0*m_far * m_near)/(m_far - m_near);		
	m_projMatrix(3,2) = -1.0;

	float l, r, t, b;
	l = -sx + 2.0*sx*m_tile.x;						
	r = -sx + 2.0*sx*m_tile.z;
	t =  sy - 2.0*sy*m_tile.y;
	b =  sy - 2.0*sy*m_tile.w;
	m_tileProjMatrix = 0.0;
	m_tileProjMatrix(0,0) = 2.0*m_near / (r - l);
	m_tileProjMatrix(1,1) = 2.0*m_near / (t - b);
	m_tileProjMatrix(0,2) = (r + l) / (r - l);		
	m_tileProjMatrix(1,2) = (t + b) / (t - b);		
	m_tileProjMatrix(2,2) = m_projMatrix(2,2);		
	m_tileProjMatrix(2,3) = m_projMatrix(2,3);		
	m_tileProjMatrix(3,2) = -1.0;

    Vector3DF tvz(0, 0, 0);
	
	m_invRotMatrix.InverseView ( m_rotateMatrix.GetDataF(), tvz );		
	m_invProjMatrix.InverseProj ( m_tileProjMatrix.GetDataF() );							

	updateFrustum ();
}

void Camera3D::updateFrustum ()
{
	Matrix4F mv;
	mv = m_tileProjMatrix;					
	mv *= m_viewMatrix;
	float* mvm = mv.GetDataF();
	float t;

   m_frustum[0][0] = mvm[ 3] - mvm[ 0];
   m_frustum[0][1] = mvm[ 7] - mvm[ 4];
   m_frustum[0][2] = mvm[11] - mvm[ 8];
   m_frustum[0][3] = mvm[15] - mvm[12];
   t = sqrt( m_frustum[0][0] * m_frustum[0][0] + m_frustum[0][1] * m_frustum[0][1] + m_frustum[0][2] * m_frustum[0][2] );
   m_frustum[0][0] /= t; m_frustum[0][1] /= t; m_frustum[0][2] /= t; m_frustum[0][3] /= t;

   m_frustum[1][0] = mvm[ 3] + mvm[ 0];
   m_frustum[1][1] = mvm[ 7] + mvm[ 4];
   m_frustum[1][2] = mvm[11] + mvm[ 8];
   m_frustum[1][3] = mvm[15] + mvm[12];
   t = sqrt( m_frustum[1][0] * m_frustum[1][0] + m_frustum[1][1] * m_frustum[1][1] + m_frustum[1][2]    * m_frustum[1][2] );
   m_frustum[1][0] /= t; m_frustum[1][1] /= t; m_frustum[1][2] /= t; m_frustum[1][3] /= t;

   m_frustum[2][0] = mvm[ 3] + mvm[ 1];
   m_frustum[2][1] = mvm[ 7] + mvm[ 5];
   m_frustum[2][2] = mvm[11] + mvm[ 9];
   m_frustum[2][3] = mvm[15] + mvm[13];
   t = sqrt( m_frustum[2][0] * m_frustum[2][0] + m_frustum[2][1] * m_frustum[2][1] + m_frustum[2][2]    * m_frustum[2][2] );
   m_frustum[2][0] /= t; m_frustum[2][1] /= t; m_frustum[2][2] /= t; m_frustum[2][3] /= t;

   m_frustum[3][0] = mvm[ 3] - mvm[ 1];
   m_frustum[3][1] = mvm[ 7] - mvm[ 5];
   m_frustum[3][2] = mvm[11] - mvm[ 9];
   m_frustum[3][3] = mvm[15] - mvm[13];
   t = sqrt( m_frustum[3][0] * m_frustum[3][0] + m_frustum[3][1] * m_frustum[3][1] + m_frustum[3][2]    * m_frustum[3][2] );
   m_frustum[3][0] /= t; m_frustum[3][1] /= t; m_frustum[3][2] /= t; m_frustum[3][3] /= t;

   m_frustum[4][0] = mvm[ 3] - mvm[ 2];
   m_frustum[4][1] = mvm[ 7] - mvm[ 6];
   m_frustum[4][2] = mvm[11] - mvm[10];
   m_frustum[4][3] = mvm[15] - mvm[14];
   t = sqrt( m_frustum[4][0] * m_frustum[4][0] + m_frustum[4][1] * m_frustum[4][1] + m_frustum[4][2]    * m_frustum[4][2] );
   m_frustum[4][0] /= t; m_frustum[4][1] /= t; m_frustum[4][2] /= t; m_frustum[4][3] /= t;

   m_frustum[5][0] = mvm[ 3] + mvm[ 2];
   m_frustum[5][1] = mvm[ 7] + mvm[ 6];
   m_frustum[5][2] = mvm[11] + mvm[10];
   m_frustum[5][3] = mvm[15] + mvm[14];
   t = sqrt( m_frustum[5][0] * m_frustum[5][0] + m_frustum[5][1] * m_frustum[5][1] + m_frustum[5][2]    * m_frustum[5][2] );
   m_frustum[5][0] /= t; m_frustum[5][1] /= t; m_frustum[5][2] /= t; m_frustum[5][3] /= t;
}

float Camera3D::calculateLOD ( Vector3DF pnt, float minlod, float maxlod, float maxdist )
{
	Vector3DF vec = pnt;
	vec -= m_fromPos;
	float lod = minlod + (vec.Length() * (maxlod-minlod) / maxdist );	
	lod = (lod < minlod) ? minlod : lod;
	lod = (lod > maxlod) ? maxlod : lod;
	return lod;
}

void Camera3D::setModelMatrix ()
{
	glGetFloatv ( GL_MODELVIEW_MATRIX, m_modelMatrix.GetDataF() );
}

void Camera3D::setModelMatrix ( Matrix4F& model )
{
	m_modelMatrix = model;
	m_mvMatrix = model;
	m_mvMatrix *= m_viewMatrix;
	glLoadMatrixf ( m_mvMatrix.GetDataF() );
}

Vector3DF Camera3D::inverseRay (float x, float y, float z)
{	
	float sx = tan ( m_fov * 0.5 * DEGtoRAD);
	float sy = sx / m_aspect;
	float tu, tv;
	tu = m_tile.x + x * (m_tile.z-m_tile.x);
	tv = m_tile.y + y * (m_tile.w-m_tile.y);
	Vector4DF pnt ( (tu*2.0-1.0) * z*sx, (1.0-tv*2.0) * z*sy, -z, 1 );
	pnt *= m_invRotMatrix;
	return pnt;
}

Vector4DF Camera3D::project ( Vector3DF& p, Matrix4F& vm )
{
	Vector4DF q = p;								
	
	q *= vm;										
	
	q *= m_projMatrix;								

	q /= q.w;										
	
	q.x *= 0.5;
	q.y *= -0.5;
	q.z = q.z*0.5 + 0.5;							
		
	return q;
}

Vector4DF Camera3D::project ( Vector3DF& p )
{
	Vector4DF q = p;								
	q *= m_viewMatrix;								

	q *= m_projMatrix;								
	
	q /= q.w;										

	q.x *= 0.5;
	q.y *= -0.5;
	q.z = q.z*0.5 + 0.5;							
		
	return q;
}


