#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <cstdio>
#include <cerrno>

// #define NMEMCPY 1 // Must be power of 2, acceptable = {1,2,4,8}
// #define NMEMCPYLOG2 0
// #define MEMCPYLARGEMASK 0xffffffff
// #define MEMCPYSMALLMASK 0x00000000

// #define NMEMCPY 2 // Must be power of 2, acceptable = {1,2,4,8}
// #define NMEMCPYLOG2 1
// #define MEMCPYLARGEMASK 0xfffffffe
// #define MEMCPYSMALLMASK 0x00000001

#define NMEMCPY 4 // Must be power of 2, acceptable = {1,2,4,8}
#define NMEMCPYLOG2 2
#define MEMCPYLARGEMASK 0xfffffffc
#define MEMCPYSMALLMASK 0x00000003

// #define NMEMCPY 8 // Must be power of 2, acceptable = {1,2,4,8}
// #define NMEMCPYLOG2 3
// #define MEMCPYLARGEMASK 0xfffffff8
// #define MEMCPYSMALLMASK 0x00000007

__host__ __device__
constexpr uint32_t log2(uint32_t n) {
    return ( (n<2) ? ((n!=0) ? 1 : 0) : 1+log2(n/2));
}

template<typename T>
__host__ __device__
constexpr T ceil(T a, T b) {
    return (1 + ((a - 1)/b) );
}


__host__ __device__
constexpr uint32_t bitsNeeded(uint32_t n) {
  return n <= 1 ? 0 : 1 + bitsNeeded((n + 1) / 2);
}

template<typename T>
__host__ __device__
constexpr T pow(T base, T exponent) {
    return exponent == 0 ? 1 : base * pow(base, exponent - 1);
}


__host__ __device__
constexpr uint32_t wrap(uint32_t value, uint32_t limit) {
    return (((value) < (limit)) ? (value) : ((value) - (limit)));
}

#define WRAP(v, l) wrap(v, l)
/* __host__ __device__ */
/* int constexpr mod (int a, int b) */
/* { */
/*     if(b < 0) //you can check for b == 0 separately and do what you want */
/* 	return -mod(-a, -b);    */
/*     int ret = a % b; */
/*     if(ret < 0) */
/* 	ret+=b; */
/*     return ret; */
/* } */

#define cuda_err_chk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess) 
    {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(1);
    }
}

#define PRINT_ERROR							\
    do {								\
	fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n",	\
		__LINE__, __FILE__, errno, strerror(errno)); exit(1);	\
    } while(0)

#define UNUSED(expr) do { (void)(expr); } while (0)

#endif
