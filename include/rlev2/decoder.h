#ifndef _RLEV2_DECODER_H_
#define _RLEV2_DECODER_H_

#include <common.h>

#include "utils.h"

namespace rlev2 {

    __host__ __device__ inline void write_val(const int64_t& val, int64_t* &out) {
        *(out ++) = val;
    }

    __host__ __device__ inline uint8_t read_byte(uint8_t* &in) {
        return *(in ++);
    }

    __host__ __device__ int64_t read_long(uint8_t* &in, uint8_t num_bytes) {
        int64_t ret = 0;
        while (num_bytes-- > 0) {
            ret |= (read_byte(in) << (num_bytes * 8));
        }
        return ret;
    }

    __host__ __device__ uint32_t decode_short_repeat(uint8_t* &in, uint8_t first, int64_t* &out) {
        const uint8_t num_bytes = ((first >> 3) & 0x07) + 1;
        const uint8_t count = (first & 0x07) + MINIMUM_REPEAT;

        const auto val = read_long(in, num_bytes);

        for (uint8_t i=0; i<count; ++i) {
            write_val(val, out);
        }
        
        return 1 + num_bytes;
    }

    __host__ __device__ uint32_t readLongs(uint8_t*& in, int64_t*& data, uint64_t len, uint64_t fb) {
        uint32_t ret = 0;

        uint64_t bitsLeft = 0;
        uint32_t curByte = 0;
        // TODO: unroll to improve performance
        for(uint64_t i = 0; i < len; i++) {
            uint64_t result = 0;
            uint64_t bitsLeftToRead = fb;
            while (bitsLeftToRead > bitsLeft) {
                result <<= bitsLeft;
                result |= curByte & ((1 << bitsLeft) - 1);
                bitsLeftToRead -= bitsLeft;
                curByte = read_byte(in);
                ++ ret;
                bitsLeft = 8;
            }

            // handle the left over bits
            if (bitsLeftToRead > 0) {
                result <<= bitsLeftToRead;
                bitsLeft -= static_cast<uint32_t>(bitsLeftToRead);
                result |= (curByte >> bitsLeft) & ((1 << bitsLeftToRead) - 1);
            }
            write_val(static_cast<int64_t>(result), data);
        }

        return ret;
    }

    __host__ __device__ uint64_t readVulong(uint8_t*& in) {
        uint64_t ret = 0, b;
        uint64_t offset = 0;
        do {
            b = read_byte(in);
            ret |= (0x7f & b) << offset;
            offset += 7;
        } while (b >= 0x80);
        return ret;
    }

    __host__ __device__ inline int64_t unZigZag(uint64_t value) {
        return value >> 1 ^ -(value & 1);
    }
    __host__ __device__ inline int64_t readVslong(uint8_t*& in) {
        return unZigZag(readVulong(in));
    }


    __host__ __device__ uint32_t decode_delta(uint8_t* &in, uint8_t first, int64_t* &out) {
        const uint8_t encoded_fbw = (first >> 1) & 0x1f;
        const uint8_t fbw = get_decoded_bit_width(encoded_fbw);
        const uint16_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte(in)) + 1;
        
        auto base_val = static_cast<int64_t>(readVulong(in));
        auto base_delta = static_cast<int64_t>(readVslong(in));

        write_val(base_val, out);
        base_val += base_delta;
        write_val(base_val, out);

        auto curr = out;

        bool increasing = (base_delta > 0);
        if (fbw != 0) {
            uint32_t consumed = readLongs(in, out, len - 2, fbw);
            if (increasing) {
                for (uint16_t i=0; i<len; ++i) {
                    base_val = curr[i] += base_val;
                }
            } else {
                for (uint16_t i=0; i<len; ++i) {
                    base_val = curr[i] = base_val - curr[i];
                }
            }
            return consumed + 4; //2 for header, 2 for base val and base delta
        } else {
            for (uint16_t i=2; i<len; ++i) {
                base_val += base_delta;
                write_val(base_val, out);
            }
            return 4;
        }
    }

    __host__ __device__ uint32_t decode_direct(uint8_t* &in, uint8_t first, int64_t* &out) {
        const uint8_t encoded_fbw = (first >> 1) & 0x1f;
        const uint8_t fbw = get_decoded_bit_width(encoded_fbw);
        const uint16_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte(in)) + 1;

        return 2 + readLongs(in, out, len, fbw);
    }

    __host__ __device__ uint32_t decode_patched_base(uint8_t* &in, uint8_t first, int64_t* &out) {
        const uint8_t encoded_fbw = (first >> 1) & 0x1f;
        const uint8_t fbw = get_decoded_bit_width(encoded_fbw);
        const uint16_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte(in)) + 1;

        const auto third = read_byte(in);
        const auto forth = read_byte(in);

        const uint8_t bw = ((third >> 5) & 0x07) + 1;
        const uint8_t pw = get_decoded_bit_width(third & 0x1f);
        const uint8_t pgw = ((forth >> 5) & 0x07) + 1;
        const uint8_t pll = forth & 0x1f;

        const auto base  = read_long(in, bw);

        uint32_t consumed = 4 + 2; // 4 bytes header + 2 byte long

        auto curr = out;
        consumed += readLongs(in, out, len, fbw);

        uint32_t cfb = get_closest_bit(pw + pgw);

        // TODO Patched List should be no more than 10 percent of MAX_LIT_SIZE
        int64_t patch_list[MAX_LITERAL_SIZE], *p_list = patch_list;
        consumed += readLongs(in, p_list, pll, cfb);

        const uint16_t patch_mask = (static_cast<uint16_t>(1) << pw) - 1; // get rid of patch gap
        int gap_idx = 0;
        for (uint8_t p=0; p<pll; ++p) {
            gap_idx += patch_list[p] >> pw;
            curr[gap_idx] |= static_cast<int64_t>(patch_list[p] & patch_mask) << fbw;
        }

        for (uint16_t i=0; i<len; ++i) {
            curr[i] += base;
        }
        return consumed;
    }

    __host__ __device__ void block_decode(const uint64_t tid, uint8_t* in, const uint64_t* offsets, int64_t* out) {
        uint32_t consumed = 0;

        auto curr_out = out + tid * CHUNK_SIZE;
        uint8_t* curr_in = in + offsets[tid];
        const uint32_t my_chunk_size = static_cast<uint32_t>(offsets[tid + 1] - offsets[tid]);

        // printf("chunk size: %u\n", my_chunk_size);
        while (consumed < my_chunk_size) {
            const auto first = read_byte(curr_in);

            const auto encoding = static_cast<uint8_t>(first & 0xC0); // 0bxx000000
            switch(encoding) {
            case HEADER_SHORT_REPEAT:
                consumed += decode_short_repeat(curr_in, first, curr_out);
                break;
            case HEADER_DIRECT:
                consumed += decode_direct(curr_in, first, curr_out);
                break;
            case HEADER_PACTED_BASE:
                consumed += decode_patched_base(curr_in, first, curr_out);
                break;
            case HEADER_DELTA:  
                consumed += decode_delta(curr_in, first, curr_out);
                break;
            }

            // printf("consumed: %u\n", consumed);

        }
    }

    __global__ void kernel_decompress(uint8_t* in, const uint64_t *ptr, const uint32_t n_chunks, int64_t* out) {
        uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n_chunks) block_decode(tid, in, ptr, out);
    }

    __host__ void decompress_gpu(const uint8_t* in, const uint64_t in_n_bytes, int64_t*& out, uint64_t& out_n_bytes) {
        uint8_t *d_in;
        int64_t *d_out;
        uint64_t *d_ptr;
        
        uint32_t n_ptr = *((uint32_t*)in);
        const uint64_t header_bytes = sizeof(uint32_t) + sizeof(uint64_t) * n_ptr + sizeof(uint64_t);
        const uint64_t data_bytes = in_n_bytes - header_bytes;
        const uint64_t raw_data_bytes = *((uint64_t*)(in + header_bytes - sizeof(uint64_t)));

        cuda_err_chk(cudaMalloc((void**)&d_in, data_bytes));
        cuda_err_chk(cudaMalloc((void**)&d_ptr, sizeof(uint64_t) * n_ptr));
        cuda_err_chk(cudaMalloc((void**)&d_out, raw_data_bytes));

        cuda_err_chk(cudaMemcpy(d_in, in + header_bytes, data_bytes, cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(d_ptr, in + sizeof(uint32_t), sizeof(uint64_t) * n_ptr, cudaMemcpyHostToDevice));

        uint64_t grid_size = ceil<uint64_t>(n_ptr - 1, BLK_SIZE);
        kernel_decompress<<<grid_size, BLK_SIZE>>>(d_in, d_ptr, n_ptr - 1, d_out);

	    cuda_err_chk(cudaDeviceSynchronize());

        out = new int64_t[raw_data_bytes / sizeof(int64_t)];
	    cuda_err_chk(cudaMemcpy(out, d_out, raw_data_bytes, cudaMemcpyDeviceToHost));

        cuda_err_chk(cudaFree(d_in));
        cuda_err_chk(cudaFree(d_out));
        cuda_err_chk(cudaFree(d_ptr));

        out_n_bytes = raw_data_bytes;
    }

}

#endif