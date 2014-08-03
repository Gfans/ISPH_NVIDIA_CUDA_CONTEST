
#include "pivotx.h"

void PivotX::setPivot ( float x, float y, float z, float rx, float ry, float rz )
{
	m_fromPos.Set ( x,y,z);
	m_angEuler.Set ( rx,ry,rz );
}

void PivotX::updateTform ()
{
	m_trans.RotateZYXT ( m_angEuler, m_fromPos );
}