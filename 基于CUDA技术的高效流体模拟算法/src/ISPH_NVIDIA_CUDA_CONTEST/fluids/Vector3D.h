#ifndef VECTOR3D__H
#define VECTOR3D__H VECTOR3D__H

#include <cmath>
#include "Matrix3x3.h"

using namespace std; 
#define EPSILON 1e-9
#define M_PI 3.14159

//--------------------------------------------------------------------
class Vector3D
//--------------------------------------------------------------------
{
	public:
		float v[3];
	
	public:
		
		inline Vector3D();
		inline Vector3D(float v0, float v1, float v2);
		inline ~Vector3D();
		
		inline void set (float v0, float v1, float v2);
		inline void setZero();
		inline void makeNegative();
		inline bool isValid();
		inline float getSquaredLength() const;
		inline float getLength() const;
		
		inline float getSquaredLengthXZ() const;
		inline float getLengthXZ() const;

		inline float normalize ();
		static inline Vector3D crossProduct(const Vector3D &a, const Vector3D &b);
		static inline float dotProduct(const Vector3D &a, const Vector3D &b);
		static inline Matrix3x3 vectorProduct(const Vector3D &a, const Vector3D &b);
		static inline Vector3D vectorMatrixProduct(const Matrix3x3 &a, const Vector3D &b);
		static inline float getDistanceSq(float x1, float y1, float z1, float x2, float y2, float z2);
		static inline float getDistanceSq(const Vector3D &a, const Vector3D &b);
		static inline float getDistanceSqAndVector(float x1, float y1, float z1, float x2, float y2, float z2, float *dx, float *dy, float *dz);
		
		static inline void solveCubic(float b, float c, float d, float *x1, float *x2, float *x3);
		static inline void largestEigenvalue(const Matrix3x3 &m, float &eigenValue, Vector3D &eigenVec);
		
		inline Vector3D& operator= (const Vector3D& V);
		inline Vector3D& operator+= (const Vector3D V);
		inline Vector3D& operator+= (float sum);
		inline Vector3D operator+ (const Vector3D V) const;
		inline Vector3D& operator-= (const Vector3D V);
		inline Vector3D& operator-= (float sub);
		inline Vector3D operator- (const Vector3D V) const;
		inline Vector3D operator- (float sub);
		inline Vector3D operator- ();
		inline Vector3D& operator*= (const Vector3D V);
		inline Vector3D& operator*= (float m);
		inline Vector3D operator* (const Vector3D V) const;
		inline Vector3D operator* (float m) const;
		inline Vector3D& operator/= (const Vector3D V);
		inline Vector3D& operator/= (float d);
		inline Vector3D operator/ (float d) const;
		inline bool operator == (const Vector3D &a) const;
		inline bool operator != (const Vector3D &a) const;
		inline float& operator[] (int index);			

};

//--------------------------------------------------------------------
class ThetaComparator
{
	public:
	static void setBaseVectors(Vector3D bV1, Vector3D bV2)
	{
		baseVec1 = bV1;
		baseVec2 = bV2;
	}
	bool operator()(Vector3D &vecA, Vector3D &vecB)
	{
		float thetaA = atan2(Vector3D::dotProduct(baseVec2, vecA), Vector3D::dotProduct(baseVec1, vecA));
		float thetaB = atan2(Vector3D::dotProduct(baseVec2, vecB), Vector3D::dotProduct(baseVec1, vecB));
		return thetaA < thetaB;
	}
	
	private:
	static Vector3D baseVec1, baseVec2;
};
//--------------------------------------------------------------------

//--------------------------------------------------------------------
Vector3D::Vector3D()
//--------------------------------------------------------------------
{
	setZero();
}

//--------------------------------------------------------------------
Vector3D::Vector3D(float v0, float v1, float v2)
//--------------------------------------------------------------------
{
	v[0] = v0;
	v[1] = v1;
	v[2] = v2;
}

//--------------------------------------------------------------------
Vector3D::~Vector3D()
//--------------------------------------------------------------------
{
}


//--------------------------------------------------------------------
void Vector3D::set(float v0, float v1, float v2)
//--------------------------------------------------------------------
{
	v[0] = v0;
	v[1] = v1;
	v[2] = v2;	
}


//--------------------------------------------------------------------
void Vector3D::setZero()
//--------------------------------------------------------------------
{
	v[0] = 0;
	v[1] = 0;
	v[2] = 0;	
}

//--------------------------------------------------------------------
void Vector3D::makeNegative()
//--------------------------------------------------------------------
{
	v[0] = -v[0];
	v[1] = -v[1];
	v[2] = -v[2];	
}

//--------------------------------------------------------------------
// check if particle has valid position (==not nan) to detect instabilities
//--------------------------------------------------------------------
bool Vector3D::isValid()
{
	return(v[0] == v[0] && v[1] == v[1] && v[2] == v[2]);
}


//--------------------------------------------------------------------
float Vector3D::getSquaredLength() const
//--------------------------------------------------------------------
{
	return(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

//--------------------------------------------------------------------
float Vector3D::getLength() const
//--------------------------------------------------------------------
{
	return (float)sqrt(getSquaredLength());
}

//--------------------------------------------------------------------
float Vector3D::getSquaredLengthXZ() const
//--------------------------------------------------------------------
{
	return(v[0]*v[0] + v[2]*v[2]);
}

//--------------------------------------------------------------------
float Vector3D::getLengthXZ() const
//--------------------------------------------------------------------
{
	return (float)sqrt(getSquaredLengthXZ());
}

//--------------------------------------------------------------------
float Vector3D::normalize()
//--------------------------------------------------------------------
// normalize and return length
{
	float length = getLength();
	if (length == 0.0f)
		return 0;

	float rezLength = 1.0f / length;
	v[0] *= rezLength;
	v[1] *= rezLength;
	v[2] *= rezLength;
	return length;
}

//--------------------------------------------------------------------
Vector3D Vector3D::crossProduct(const Vector3D &a, const Vector3D &b)
//--------------------------------------------------------------------
{
	Vector3D result;

	result.v[0] = a.v[1] * b.v[2] - a.v[2] * b.v[1];
	result.v[1] = a.v[2] * b.v[0] - a.v[0] * b.v[2];
	result.v[2] = a.v[0] * b.v[1] - a.v[1] * b.v[0];

	return(result);
}

//--------------------------------------------------------------------
float Vector3D::dotProduct(const Vector3D &a, const Vector3D& b)
//--------------------------------------------------------------------
{
	return(a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2]);
}

//--------------------------------------------------------------------
Matrix3x3 Vector3D::vectorProduct(const Vector3D &a, const Vector3D &b)
//--------------------------------------------------------------------
{
	Matrix3x3 result;

	result.elements[0][0] = a.v[0] * b.v[0];
	result.elements[0][1] = a.v[0] * b.v[1];
	result.elements[0][2] = a.v[0] * b.v[2];
	
	result.elements[1][0] = a.v[1] * b.v[0];
	result.elements[1][1] = a.v[1] * b.v[1];
	result.elements[1][2] = a.v[1] * b.v[2];
	
	result.elements[2][0] = a.v[2] * b.v[0];
	result.elements[2][1] = a.v[2] * b.v[1];
	result.elements[2][2] = a.v[2] * b.v[2];

	return(result);
}

//--------------------------------------------------------------------
Vector3D Vector3D::vectorMatrixProduct(const Matrix3x3 &a, const Vector3D &b)
//--------------------------------------------------------------------
{
	Vector3D result;
	result.setZero();
	for(int i=0;i<3;++i)
	{
		for(int j=0;j<3;++j)
		{
			result[i] += a.elements[i][j] * b.v[j];
		}
	}
	return result;
}


// -----------------------------------------------------------------------------------------------
float Vector3D::getDistanceSq(float x1, float y1, float z1,
										float x2, float y2, float z2)
// -----------------------------------------------------------------------------------------------
{
	// calculate distance in each coordinate direction
	float distX = x2 - x1;
	float distY = y2 - y1;
	float distZ = z2 - z1;
	
/*
	//if(fp->isPeriodic())
		float maxX = boxLength * 0.5;
		float maxZ = boxWidth  * 0.5;

		// if distance in x direction is larger than half of the box length
		// use the complementary distance
		if(distX > maxX)
			distX -= boxLength;
		else
			if(distX < -maxX)
				distX += boxLength;
				 
		if(distZ > maxZ)
			distZ -= boxWidth;
		else
			if(distZ < -maxZ)
				distZ += boxWidth;	
	}
	*/
	
	// return squared eucledian distance
	return distX * distX + distY * distY + distZ * distZ;
}

// -----------------------------------------------------------------------------------------------
float Vector3D::getDistanceSq(const Vector3D &a, const Vector3D &b)
// -----------------------------------------------------------------------------------------------
{
	float deltaX = a.v[0] - b.v[0];
	float deltaY = a.v[1] - b.v[1];
	float deltaZ = a.v[2] - b.v[2];
	return deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ;	
}

		

// -----------------------------------------------------------------------------------------------
float Vector3D::getDistanceSqAndVector(float x1, float y1, float z1,
										float x2, float y2, float z2,
										float *dx, float *dy, float *dz)
// -----------------------------------------------------------------------------------------------
{
	// calculate distance in each coordinate direction
	float distX = x2 - x1;
	float distY = y2 - y1;
	float distZ = z2 - z1;
		
	// store difference in each coordinate 
	*dx = distX;
	*dy = distY;
	*dz = distZ;	
	
	// return squared eucledian distance
	return distX * distX + distY * distY + distZ * distZ;
}


//--------------------------------------------------------------------
Vector3D& Vector3D::operator= (const Vector3D& V)
//--------------------------------------------------------------------
{
	v[0] = V.v[0];
	v[1] = V.v[1];
	v[2] = V.v[2];
	return (*this);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator+= (const Vector3D V)
//--------------------------------------------------------------------
{
	v[0] += V.v[0];
	v[1] += V.v[1];
	v[2] += V.v[2];
	return (*this);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator+= (float sum)
//--------------------------------------------------------------------
{
	v[0] += sum;
	v[1] += sum;
	v[2] += sum;
	return (*this);
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator+ (const Vector3D V) const
//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = v[0] + V.v[0];
	res.v[1] = v[1] + V.v[1];
	res.v[2] = v[2] + V.v[2];
	return (res); 
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator-= (const Vector3D V)
//--------------------------------------------------------------------
{
	v[0] -= V.v[0];
	v[1] -= V.v[1];
	v[2] -= V.v[2];
	return (*this);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator-= (float sub)
//--------------------------------------------------------------------
{
	v[0] -= sub;
	v[1] -= sub;
	v[2] -= sub;
	return (*this);
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator- (const Vector3D V) const
//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = v[0] - V.v[0];
	res.v[1] = v[1] - V.v[1];
	res.v[2] = v[2] - V.v[2];
	return (res); 
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator- (float sub)
	//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = v[0] - sub;
	res.v[1] = v[1] - sub;
	res.v[2] = v[2] - sub;
	return (res);
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator- ()
	//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = -v[0];
	res.v[1] = -v[1];
	res.v[2] = -v[2];
	return (res);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator*= (const Vector3D V)
//--------------------------------------------------------------------
{
	v[0] *= V.v[0];
	v[1] *= V.v[1];
	v[2] *= V.v[2];
	return (*this);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator*= (float m)
//--------------------------------------------------------------------
{
	v[0] *= m;
	v[1] *= m;
	v[2] *= m;
	return (*this);
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator* (const Vector3D V) const
//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = v[0] * V.v[0];
	res.v[1] = v[1] * V.v[1];
	res.v[2] = v[2] * V.v[2];
	return (res); 
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator* (float m) const
//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = v[0] * m;
	res.v[1] = v[1] * m;
	res.v[2] = v[2] * m;
	return (res); 
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator/= (const Vector3D V)
//--------------------------------------------------------------------
{
	v[0] /= V.v[0];
	v[1] /= V.v[1];
	v[2] /= V.v[2];
	return (*this);
}

//--------------------------------------------------------------------
Vector3D& Vector3D::operator/= (float d)
//--------------------------------------------------------------------
{
	v[0] /= d;
	v[1] /= d;
	v[2] /= d;
	return (*this);
}

//--------------------------------------------------------------------
Vector3D Vector3D::operator/ (float d) const
//--------------------------------------------------------------------
{
	Vector3D res;
	res.v[0] = v[0] / d;
	res.v[1] = v[1] / d;
	res.v[2] = v[2] / d;
	return (res); 
}

//--------------------------------------------------------------------
bool Vector3D::operator == (const Vector3D &a) const
//--------------------------------------------------------------------
{
	return(v[0] == a.v[0] && v[1] == a.v[1] && v[2] == a.v[2]);
}

//--------------------------------------------------------------------
bool Vector3D::operator != (const Vector3D &a) const
//--------------------------------------------------------------------
{
	return(v[0] != a.v[0] || v[1] != a.v[1] || v[2] != a.v[2]);
}

//--------------------------------------------------------------------
float& Vector3D::operator[] (int index)
//--------------------------------------------------------------------
{
	return (v[index]);
}

//---------------------------------------------------
void Vector3D::solveCubic(float b, float c, float d, float *x1, float *x2, float *x3)
//---------------------------------------------------
{
/* solves x^3 + b*x^2 + c*x + d = 0, returns real part of solutions */
  float e,f,g,u,s, sq, cosf;
  float r,rx,phi, x,y;

  u = -b/3;
  e = 3*u*u + 2*u*b + c;
  f = u*u*u + u*u*b + u*c + d;
  s = -e/3;
  g = -e*e*e/27;

  sq = f*f - 4*g;
  if (sq >= 0) {
    sq = sqrt(sq);
    r = (-f + sq)/2; phi = 0.0;
    if (r < 0) { r = -r; phi = M_PI; }
    if (r > EPSILON) r = exp(1.0/3*log(r));
  }
  else {
    sq = sqrt(-sq);
    x = -f/2; y = sq/2;
    r = sqrt(x*x + y*y);
    if (r < EPSILON) { r = 0.0; phi = 0.0; }
    else {
      cosf = x/r;
      if (cosf > 1.0) cosf = 1.0;
      if (cosf < -1.0) cosf = -1.0;
      phi = acos(cosf)/3; r = exp(1.0/3*log(r));
    }
  }
  if (r > EPSILON) rx = r + s/r; else rx = 0.0;
  *x1 = rx*cos(phi) + u;
  *x2 = rx*cos(phi + 2.0*M_PI/3.0) + u;
  *x3 = rx*cos(phi + 4.0*M_PI/3.0) + u;
/* the y's are needed if the solutions are complex
   not needed here because the Eigenvalues of a symmetric matix
   are always real
  if (r > EPSILON) ry = r - s/r; else ry = 0.0;
  *y1 = ry*sin(phi);
  *y2 = ry*sin(phi + 2.0*M_PI/3.0);
  *y3 = ry*sin(phi + 4.0*M_PI/3.0);
*/
}
//---------------------------------------------------
void Vector3D::largestEigenvalue(const Matrix3x3 &m, float &eigenValue, Vector3D &eigenVec)
//---------------------------------------------------
{
/* The proc returns lambda, the largest eigenvalue and
   a corresponing eigenvector */
  float inv1,inv2,inv3;
  float a[3][3];
  float x[3];
  float l1,l2,l3,l;
  int i,j, i0,i1, j0,j1;
  int mi0 = 0; int mi1 = 0; int mj0 = 0; int mj1 = 0; int mj2 = 0;
  float det,d0,d1, max, s;

  float a00 = m.elements[0][0];
  float a01 = m.elements[0][1];
  float a02 = m.elements[0][2];
  float a10 = m.elements[1][0];
  float a11 = m.elements[1][1];
  float a12 = m.elements[1][2];
  float a20 = m.elements[2][0];
  float a21 = m.elements[2][1];
  float a22 = m.elements[2][2];
  
  s = a00+a01+a02 + a10+a11+a12 + a20+a21+a22;
  if (fabs(s) < 1.0) s = 1.0;
  else s = 1.0/s;
  if (s < 0.0) s = -s;

  a[0][0] = s*a00; a[0][1] = s*a01; a[0][2] = s*a02;
  a[1][0] = s*a10; a[1][1] = s*a11; a[1][2] = s*a12;
  a[2][0] = s*a20; a[2][1] = s*a21; a[2][2] = s*a22;

  inv1 = a[0][0] + a[1][1] + a[2][2];
  inv2 = a[0][0]*a[1][1]-a[0][1]*a[1][0] +
         a[0][0]*a[2][2]-a[0][2]*a[2][0] +
         a[1][1]*a[2][2]-a[1][2]*a[2][1];
  inv3 = a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1]
        -a[0][0]*a[1][2]*a[2][1] - a[0][1]*a[1][0]*a[2][2] - a[0][2]*a[1][1]*a[2][0];

  solveCubic(-inv1,inv2,-inv3, &l1,&l2,&l3);

//  if (l1 > l2) l = l1; 
//	else l = l2;  /* tension only */
//  if (l3 > l) l = l3;
  
  // largest Eigenvalue
  if (l1 < l2) l = l1; 
	else l = l2; 
  if (l3 < l) l = l3;


  a[0][0] -= l; a[1][1] -= l; a[2][2] -= l;
  eigenValue = l/s;

  max = 0.0;
  i0 = 1; i1 = 2;
  for (i = 0 ; i < 3; i++) {
    if (i == 1) i0--; if (i == 2) i1--;
    j0 = 1; j1 = 2;
    for (j = 0; j < 3; j++) {
      if (j == 1) j0--; if (j == 2) j1--;
      det = fabs(a[i0][j0]*a[i1][j1] - a[i0][j1]*a[i1][j0]);
      if (det > max) {
        max = det;
        mi0 = i0; mi1 = i1;
        mj0 = j0; mj1 = j1; mj2 = 3-j0-j1;
      }
    }
  }
  if (max > EPSILON) {	/* single eigenvalue */
    x[mj2] = -1.0;
    det = a[mi0][mj0]*a[mi1][mj1] - a[mi0][mj1]*a[mi1][mj0];
    d0  = a[mi0][mj2]*a[mi1][mj1] - a[mi0][mj1]*a[mi1][mj2];
    d1  = a[mi0][mj0]*a[mi1][mj2] - a[mi0][mj2]*a[mi1][mj0];
    x[mj0] = d0/det;
    x[mj1] = d1/det;
  }
  else {
    max = 0.0;
    for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
        if (fabs(a[i][j]) > max) {
          mi0 = i; mj0 = j; max = fabs(a[i][j]);
        }
      }
    }
    if (max > EPSILON) { /* double eigenvalue */
      mj1 = mj0+1; if (mj1 > 2) mj1 = 0; mj2 = 3-mj0-mj1;
      x[mj1] = -1.0;
      x[mj0] = a[mi0][mj1]/a[mi0][mj0];
      x[mj2] = 0.0;
    }
    else {  /* triple eigenvalue */
      x[0] = 1.0; x[1] = 0.0; x[2] = 0.0;
    }
  }
  eigenVec.set(x[0], x[1], x[2]);
  eigenVec.normalize();
}



#endif

