#ifndef CUDA_VECTOR_UINT2x4_H
#define CUDA_VECTOR_UINT2x4_H

///////////////////////////////////////////////////////////////////////////////////
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif

#include "cuda_helper.h"
#include "cuda_vectors.h"

static __inline__ __device__ uint2x4 make_uint2x4(uint2 s0, uint2 s1, uint2 s2, uint2 s3)
{
	uint2x4 t;
	t.x = s0; t.y = s1; t.z = s2; t.w = s3;
	return t;
}

#endif