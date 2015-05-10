#ifndef MATRIX_3_X_3_H
#define MATRIX_3_X_3_H MATRIX_3_X_3_H

/**
 * Represents a matrix of size 3x3.
 */
class Matrix3x3
{
	public:
		/**
		 * Constructor. The elements are not set.
		 */
		Matrix3x3();
		Matrix3x3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33);
		/**
		 * Multiplies 'matrix' from the right.
		 */
		Matrix3x3 operator*(const Matrix3x3& matrix);
		/**
		 * Multiplies all elements with factor.
		 */
		Matrix3x3 operator*(float factor);
		/**
		 * Matrix addition.
		 */
		Matrix3x3 operator+(const Matrix3x3& summand);
		/**
		 * Sets all elements to zero.
		 */
		void setZero();
		/**
		 * is Zero
		 */
		bool isZero();
		/**
		 * Returns the tranpose of this matrix.
		 */
		Matrix3x3 getTransposedMatrix();
		/**
		 * Returns an unit matrix.
		 */
		static Matrix3x3 unitMatrix();
		/**
		 * Calculates the inverse of matrix and stores it in result. If the
		 * matrix could not be inverted, result is not changed and false will be
		 * returned.
		 */
		static bool calculateInverse(const Matrix3x3& matrix, Matrix3x3& result);
		/**
		 * Contains the elements of this matrix.
		 */
		float elements[3][3];
	
};

#endif
