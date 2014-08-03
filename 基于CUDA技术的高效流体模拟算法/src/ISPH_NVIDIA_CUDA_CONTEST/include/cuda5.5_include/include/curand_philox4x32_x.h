 /* Copyright 2010-2013 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */
/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CURAND_PHILOX4X32_X__H_
#define CURAND_PHILOX4X32_X__H_

#define ROUND_OPTIMIZED 

typedef unsigned int uint32;
typedef unsigned int uint32x2[2];
typedef unsigned int uint32x4[4];
typedef unsigned long long uint64;

#define PHILOX_W32_0   ((uint32)0x9E3779B9)
#define PHILOX_W32_1   ((uint32)0xBB67AE85)
#define PHILOX_M4x32_0 ((uint32)0xD2511F53)
#define PHILOX_M4x32_1 ((uint32)0xCD9E8D57)

struct curandStatePhilox4_32_10 {
	uint32x2 key;
	uint32x4 ctr;
	int STATE;
	uint32x4 output;
	int boxmuller_flag;
	int boxmuller_flag_double;
	float boxmuller_extra;
	double boxmuller_extra_double;
};

typedef struct curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;


QUALIFIERS void Philox_State_Incr_carefully(curandStatePhilox4_32_10_t* s, uint64 n=1)
{
	uint32 vtn;
	vtn = n;
	s->ctr[0] += n;
	const unsigned rshift = 8* sizeof(uint32);
	for(size_t i = 1; i < 4; ++i) {
		if(rshift) {
			n >>= rshift/2;
			n >>= rshift/2;
		}else {
			n=0;
		}
		if( s->ctr[i-1] < vtn )
			++n;
		if( n==0 ) break;
		vtn = n;
		s->ctr[i] += n;
	}
}

QUALIFIERS void Philox_State_Incr_carefully_hi(curandStatePhilox4_32_10_t* s, uint64 n=1)
{
	uint32 vtn;
	vtn = n;
	s->ctr[2] += n;
	const unsigned rshift = 8* sizeof(uint32);
	for(size_t i = 3; i < 4; ++i) {
		if(rshift) {
			n >>= rshift/2;
			n >>= rshift/2;
		}else {
			n=0;
		}
		if( s->ctr[i-1] < vtn )
			++n;
		if( n==0 ) break;
		vtn = n;
		s->ctr[i] += n;
	}
}


QUALIFIERS void Philox_State_Incr(curandStatePhilox4_32_10_t* s, uint64 n)
{
    int check = n >> 4*sizeof(uint32);
    check = n >> 4*sizeof(uint32);
    if(check)
        Philox_State_Incr_carefully(s,n);
	s->ctr[0] += n;
	if(n <= s->ctr[0])
		return;
	++s->ctr[1];
	if(!!s->ctr[1]) return;
	++s->ctr[2];
	if(!!s->ctr[2]) return;
	++s->ctr[3];
	for(int i=0; i<4; i++)
		++s->ctr[i];
}

QUALIFIERS void Philox_State_Incr_hi(curandStatePhilox4_32_10_t* s, uint64 n)
{
    int check = n >> 4*sizeof(uint32);
    check = n >> 4*sizeof(uint32);
    if(check)
        Philox_State_Incr_carefully_hi(s,n);
	s->ctr[2] += n;
	if(n <= s->ctr[2])
		return;
	++s->ctr[3];
	for(int i=0; i<4; i++)
		++s->ctr[i];
}



QUALIFIERS void Philox_State_Incr(curandStatePhilox4_32_10_t* s)
{
	++s->ctr[0];
	if(!!s->ctr[0])
		return;
	++s->ctr[1];
	if(!!s->ctr[1]) return;
	++s->ctr[2];
	if(!!s->ctr[2]) return;
	++s->ctr[3];
	for(int i=0; i<4; i++)
		++s->ctr[i];
}


QUALIFIERS uint32 mulhilo32(uint32 a, uint32 b, uint32* hip)
{
#ifndef __CUDA_ARCH__
    // host code
	uint64 product = ((uint64)a) * ((uint64)b);
	*hip = product >> 32;
	return (uint32)product;
#else
    // device code
    *hip = __umulhi(a,b);
    return a*b;
#endif
}

QUALIFIERS uint32* _philox4x32round(uint32* ctr, uint32* key)
{
	uint32 hi0;
	uint32 hi1;
	uint32 lo0 = mulhilo32(PHILOX_M4x32_0, ctr[0], &hi0);
	uint32 lo1 = mulhilo32(PHILOX_M4x32_1, ctr[2], &hi1);

	ctr[0] = hi1^ctr[1]^key[0];
	ctr[1] = lo1;
	ctr[2] = hi0^ctr[3]^key[1];
	ctr[3] = lo0;

	return ctr;
}

QUALIFIERS uint32* _philox4x32bumpkey( uint32* key)
{
	key[0] += PHILOX_W32_0;
	key[1] += PHILOX_W32_1;
	return key;
}


QUALIFIERS void curand_Philox4x32_10(uint32x4 ctr, uint32x2 key, uint32* out)
{
	//curandStatePhilox4_32_10 localState = *state;
	//uint32x4 _ctr  = {state->ctr[0],state->ctr[1],state->ctr[2],state->ctr[3]};
	//uint32* ctr = _ctr;
	//uint32* ctr = localState.ctr;
	//uint32x2 _key  = {state->key[0], state->key[1]};
	//uint32* key = _key;
	//uint32* key = localState.key;

	//uint32* out  = state->output;
	ctr = _philox4x32round(ctr, key);                                 // 1 
	key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key);  // 2
    key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key);  // 3
	key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key);  // 4
    key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key);  // 5
	key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key);  // 6
    key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key);  // 7
	key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key);  // 8
    key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key);  // 9
	key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key);  // 10
	for(int i=0; i<4; i++)	out[i] = ctr[i];
	return;
}


#endif
