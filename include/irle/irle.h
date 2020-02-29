#include <common.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <inttypes.h>

#define _MIN(a,b) \
({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })
    
#define COMPRESSION_TYPE int64_t

constexpr   uint64_t CHUNK_SIZE_() { return 1 * 1024; }
constexpr   uint16_t BLK_SIZE_() { return 1024; }
constexpr   uint16_t MAX_LITERAL_SIZE_() { return 128; }
constexpr   uint8_t  MINIMUM_REPEAT_() { return 3; }
constexpr   uint8_t  MAXIMUM_REPEAT_() { return 127 + MINIMUM_REPEAT_(); }
constexpr   uint64_t OUTPUT_CHUNK_SIZE_() { return CHUNK_SIZE_() + CHUNK_SIZE_()/ sizeof(COMPRESSION_TYPE) + 1; }
// constexpr   uint64_t OUTPUT_CHUNK_SIZE_() { return CHUNK_SIZE_() + (CHUNK_SIZE_() - 1) / MAX_LITERAL_SIZE_() + 1; }

#define CHUNK_SIZE                CHUNK_SIZE_()
#define BLK_SIZE                  BLK_SIZE_()			  
#define MAX_LITERAL_SIZE          MAX_LITERAL_SIZE_()
#define MINIMUM_REPEAT            MINIMUM_REPEAT_()
#define MAXIMUM_REPEAT            MAXIMUM_REPEAT_()
#define OUTPUT_CHUNK_SIZE         OUTPUT_CHUNK_SIZE_() //maximum output chunk size

const int64_t MAX_DELTA = 127;
const int64_t MIN_DELTA = -128;
const int64_t BASE_128_MASK = 0x7f;

#define SIZEOF_BYTE 8

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c\n"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0') 

namespace irle {
    __host__ __device__ int8_t read_byte(uint8_t* &buf) {
        return *(buf++);
    }

    __host__ __device__ uint64_t read_long(uint8_t* &in) {
        uint64_t result = 0;
        int64_t offset = 0;
        int8_t ch = read_byte(in);
        if (ch >= 0) {
            result = static_cast<uint64_t>(ch);
        } else {
            result = static_cast<uint64_t>(ch) & BASE_128_MASK;
            while ((ch = read_byte(in)) < 0) {
            offset += 7;
            result |= (static_cast<uint64_t>(ch) & BASE_128_MASK) << offset;
            }
            result |= static_cast<uint64_t>(ch) << (offset + 7);
        }
        return result;
    }

    template <class T>
    __host__ __device__ void decode(uint8_t* in, const uint64_t* ptr, const uint64_t tid, const uint64_t in_n_bytes, T* out) {
        uint64_t position = 0;
        bool repeating = false;
        in += ptr[tid];
        out += tid * CHUNK_SIZE / sizeof(T);

        int8_t delta;
        uint64_t remainingValues = 0;
        int64_t value;

        const uint64_t cur_chunk_size = _MIN(CHUNK_SIZE, in_n_bytes - tid * CHUNK_SIZE);

        while (position < cur_chunk_size) {
            // If we are out of values, read more.
            if (remainingValues == 0) {
                int8_t ch = read_byte(in);
                if (ch < 0) {
                    remainingValues = static_cast<uint64_t>(-ch);
                    repeating = false;
                } else {
                    remainingValues = static_cast<uint64_t>(ch) + MINIMUM_REPEAT;
                    repeating = true;
                    delta = read_byte(in);
                    value = static_cast<int64_t>(read_long(in));
                }
            }
            uint64_t count = _MIN(CHUNK_SIZE - position, remainingValues);
            uint64_t consumed = 0;
            if (repeating) {
                for (uint64_t i = 0; i < count; ++i) {
                    if (tid == 1) break;
                    out[position + i] = value + static_cast<int64_t>(i) * delta;
                }
                consumed = count;
            value += static_cast<int64_t>(consumed) * delta;
            } else {
                for (uint64_t i = 0; i < count; ++i) {
                    if (tid == 1) break;

                    out[position + i] = static_cast<int64_t>(read_long(in));
                }
                consumed = count;
            }
            remainingValues -= consumed;
            position += count;
        }
    }

    template <class T>
    __global__ void kernel_decompress(uint8_t* in, const uint64_t* ptr, const uint64_t n_chunks, const uint64_t in_n_bytes, T* out) {
        uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n_chunks) {
            decode(in, ptr, tid, in_n_bytes, out);
        }
    }

    template <class T>
    __host__ __device__ void encode(const uint64_t tid, const T* in, const uint64_t in_n_bytes, uint8_t* out, uint64_t* ptr) {
        /**
         * (c) Copyright [2014-2015] Hewlett-Packard Development Company, L.P
        */
        #define write_byte(b) { \
            cur_out[pos++] = b; \
        }
        #define write_ulong(val) { \
            while (1) { \
                if ((val & ~0x7f) == 0) { \
                    write_byte(static_cast<uint8_t>(val)); \
                    break; \
                } else { \
                    write_byte(static_cast<uint8_t>(0x80 | (val & 0x7f))); \
                    val = (static_cast<uint64_t>(val) >> 7); \
                } \
            } \
        } 
        #define write_out \
            if (num_literals != 0) { \
                if (repeat) { \
                    write_byte(static_cast<uint8_t>(static_cast<uint64_t>(num_literals) - MINIMUM_REPEAT)); \
                    write_byte(static_cast<uint8_t>(delta)); \
                    write_ulong(literals[0]); \
                } else { \
                    write_byte(static_cast<uint8_t>(-num_literals)); \
                    for(size_t i=0; i < num_literals; ++i) \
                        write_ulong(literals[i]); \
                } \
                repeat = 0; \
                num_literals = 0; \
                tail_run = 0; \
            }  

        const uint64_t offset_in = tid * CHUNK_SIZE;
        const uint64_t in_n_digits = _MIN(in_n_bytes - offset_in, CHUNK_SIZE) / sizeof(T);

        const T* cur_in = in + offset_in / sizeof(T);
        // const T* cur_in = (const T*) ((void*)in + offset_in);
        uint8_t* cur_out = out + tid * OUTPUT_CHUNK_SIZE;
        uint64_t pos = 0;
        uint64_t num_literals = 0, tail_run = 0;
        T literals[MAX_LITERAL_SIZE];
        bool repeat = false; 
        int64_t delta = 0;

        for (uint64_t i=0; i<in_n_digits; ++i) {
            T value = cur_in[i];
            if (num_literals == 0) {
                literals[num_literals++] = value;
                tail_run = 1;
            } else if (repeat) {
                if (value == literals[0] + delta * static_cast<int64_t>(num_literals)) {
                num_literals += 1;
                if (num_literals == MAXIMUM_REPEAT) {
                    write_out;
                }
                } else {
                write_out;
                literals[num_literals++] = value;
                tail_run = 1;
                }
            } else {
                if (tail_run == 1) {
                delta = value - literals[num_literals - 1];
                if (delta < MIN_DELTA || delta > MAX_DELTA) {
                    tail_run = 1;
                } else {
                    tail_run = 2;
                }
                } else if (value == literals[num_literals - 1] + delta) {
                tail_run += 1;
                } else {
                delta = value - literals[num_literals - 1];
                if (delta < MIN_DELTA || delta > MAX_DELTA) {
                    tail_run = 1;
                } else {
                    tail_run = 2;
                }
                }
                if (tail_run == MINIMUM_REPEAT) {
                if (num_literals + 1 == MINIMUM_REPEAT) {
                    repeat = true;
                    num_literals += 1;
                } else {
                    num_literals -= static_cast<int>(MINIMUM_REPEAT - 1);
                    int64_t base = literals[num_literals];
                    write_out;
                    literals[0] = base;
                    repeat = true;
                    num_literals = MINIMUM_REPEAT;
                }
                } else {
                literals[num_literals++] = value;
                if (num_literals == MAX_LITERAL_SIZE) {
                    write_out;
                }
                }
            }
        }
        write_out;
        #undef write_out
        #undef write_byte
        #undef write_ulong

        ptr[tid + 1] = pos;
    }

    __global__ void kernel_shift(const uint8_t* in, const uint64_t* ptr, const uint32_t n_chunks, uint8_t* out) {
        uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n_chunks) {
            const uint8_t* cur_in = in + tid * OUTPUT_CHUNK_SIZE;
            uint8_t* cur_out = out + ptr[tid];
            memcpy(cur_out, cur_in, sizeof(uint8_t) * (ptr[tid + 1] - ptr[tid]));
        }
    }
    
    template<class T>
    __global__ void kernel_compress(T* in, const uint64_t in_n_bytes, const uint64_t n_chunks, uint8_t* out, uint64_t* ptr) {
        uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n_chunks) {
            encode(tid, in, in_n_bytes, out, ptr);
        }
    }

    template<class T>
    __host__ void decompress_gpu(uint8_t* in, T*& out, const uint64_t in_n_bytes, uint64_t *out_n_bytes) {
        uint8_t* d_in;
        T* d_out;

        uint64_t* d_ptr;

        uint32_t n_ptr = *((uint32_t*)in);
        uint32_t n_chunks = n_ptr - 1;
        uint64_t grid_size = ceil<uint64_t>(n_chunks, BLK_SIZE);

        uint64_t header_byte = n_ptr * sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t);
        uint64_t data_byte = in_n_bytes - header_byte;

        uint64_t exp_out_n_bytes = *(uint64_t*)(in + header_byte - sizeof(uint64_t));

        cuda_err_chk(cudaMalloc((void**)&d_in, data_byte));
        cuda_err_chk(cudaMalloc((void**)&d_ptr, sizeof(uint64_t) * n_ptr));
        cuda_err_chk(cudaMemcpy(d_in, in + header_byte, data_byte, cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(d_ptr, in + sizeof(uint32_t), sizeof(uint64_t) * n_ptr, cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMalloc((void**)&d_out, n_chunks * CHUNK_SIZE));

        kernel_decompress<<<grid_size, BLK_SIZE>>>(d_in, d_ptr, n_chunks, exp_out_n_bytes, d_out);

	    cuda_err_chk(cudaDeviceSynchronize());

        out = new T[exp_out_n_bytes / sizeof(T)];
        cudaMemcpy(out, d_out, exp_out_n_bytes, cudaMemcpyDeviceToHost);

        *out_n_bytes = exp_out_n_bytes;
    }

    template<class T>
    __host__ void compress_gpu(T* in, uint8_t*& out, const uint64_t in_n_bytes, uint64_t *out_n_bytes) {
        T *d_in;
        uint64_t *d_ptr;
        uint8_t *d_inter, *d_out;

        uint32_t n_chunks = (in_n_bytes - 1) / CHUNK_SIZE + 1;
        uint64_t grid_size = ceil<uint64_t>(n_chunks, BLK_SIZE);
        uint64_t data_n_bytes, exp_out_n_bytes;

        cuda_err_chk(cudaMalloc((void**)&d_in, in_n_bytes));
        cuda_err_chk(cudaMalloc((void**)&d_inter, n_chunks * OUTPUT_CHUNK_SIZE));
        cuda_err_chk(cudaMalloc((void**)&d_ptr, (n_chunks + 1) * sizeof(uint64_t)));
        cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));

        kernel_compress<T><<<grid_size, BLK_SIZE>>>(d_in, in_n_bytes, n_chunks, d_inter, d_ptr);
        cuda_err_chk(cudaDeviceSynchronize());
        
        thrust::inclusive_scan(thrust::device, d_ptr, d_ptr + n_chunks + 1, d_ptr);
        cuda_err_chk(cudaMemcpy(&data_n_bytes, d_ptr + n_chunks, sizeof(uint64_t), cudaMemcpyDeviceToHost));

        cuda_err_chk(cudaMalloc((void**)&d_out, data_n_bytes * sizeof(uint8_t)));
        
        kernel_shift<<<grid_size, BLK_SIZE>>>(d_inter, d_ptr, n_chunks, d_out);
        
        exp_out_n_bytes = sizeof(uint32_t) + sizeof(uint64_t) * (n_chunks + 1) + data_n_bytes + sizeof(uint64_t);
        out = new uint8_t[exp_out_n_bytes];
        uint64_t ptr_len = sizeof(uint64_t) * (n_chunks + 1);
        *(uint32_t*)out = n_chunks + 1;

        cuda_err_chk(cudaMemcpy(out + sizeof(uint32_t), d_ptr, sizeof(uint64_t) * (n_chunks + 1), cudaMemcpyDeviceToHost));

        *(uint64_t*)(out + sizeof(uint32_t) + ptr_len) = in_n_bytes;
        cuda_err_chk(cudaMemcpy(out + sizeof(uint32_t) + ptr_len + sizeof(uint64_t), d_out, data_n_bytes, cudaMemcpyDeviceToHost));

        cuda_err_chk(cudaFree(d_in));
        cuda_err_chk(cudaFree(d_inter));
        cuda_err_chk(cudaFree(d_out));
        cuda_err_chk(cudaFree(d_ptr));

        *out_n_bytes = exp_out_n_bytes;
    }
}
