#ifndef DEF_PIVOT_H
#define DEF_PIVOT_H

#include <string>

#include "vector.h"
#include "matrix.h"

class PivotX {
public:
	PivotX()	{ m_fromPos.Set(0,0,0); m_toPos.Set(0,0,0); m_angEuler.Set(0,0,0); m_scale.Set(1,1,1); m_trans.Identity(); }
	PivotX( Vector3DF& f, Vector3DF& t, Vector3DF& s, Vector3DF& a) { m_fromPos=f; m_toPos=t; m_scale=s; m_angEuler=a; }

	// base class should have a virtual destructor (2013-06-25)
	virtual ~PivotX(){};

	void setPivot ( float x, float y, float z, float rx, float ry, float rz );
	void setPivot ( Vector3DF& pos, Vector3DF& ang ) { m_fromPos = pos; m_angEuler = ang; }
	void setPivot ( PivotX  piv )	{ m_fromPos = piv.m_fromPos; m_toPos = piv.m_toPos; m_angEuler = piv.m_angEuler; updateTform(); }		
	void setPivot ( PivotX& piv )	{ m_fromPos = piv.m_fromPos; m_toPos = piv.m_toPos; m_angEuler = piv.m_angEuler; updateTform(); }

	void setIdentity ()		{ m_fromPos.Set(0,0,0); m_toPos.Set(0,0,0); m_angEuler.Set(0,0,0); m_scale.Set(1,1,1); m_trans.Identity(); }

	void setAng ( float rx, float ry, float rz )	{ m_angEuler.Set(rx,ry,rz);	updateTform(); }
	void setAng ( Vector3DF& a )					{ m_angEuler = a;			updateTform(); }

	void setPos ( float x, float y, float z )		{ m_fromPos.Set(x,y,z);		updateTform(); }
	void setPos ( Vector3DF& p )					{ m_fromPos = p;				updateTform(); }

	void setToPos ( float x, float y, float z )		{ m_toPos.Set(x,y,z);		updateTform(); }

	void updateTform ();
	void setTform ( Matrix4F& t )		{ m_trans = t; }
	inline Matrix4F& getTform ()		{ return m_trans; }
	inline float* getTformData ()		{ return m_trans.GetDataF(); }

	// Pivot		
	PivotX getPivot ()	{ return PivotX(m_fromPos, m_toPos, m_scale, m_angEuler); }
	Vector3DF& getPos ()			{ return m_fromPos; }
	Vector3DF& getToPos ()			{ return m_toPos; }
	Vector3DF& getAng ()			{ return m_angEuler; }
	Vector3DF getDir ()			{ 
		return m_toPos - m_fromPos; 
	}

protected:

	Vector3DF	m_fromPos;
	Vector3DF	m_toPos;
	Vector3DF	m_scale;
	Vector3DF	m_angEuler;
	Matrix4F	m_trans;

};

#endif



