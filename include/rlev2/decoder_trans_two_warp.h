#ifndef _RLEV2_DECODER_DOUBLE_WARP_H_
#define _RLEV2_DECODER_DOUBLE_WARP_H_

#include <cuda/atomic>
#include "utils.h"

namespace rlev2 {
	template <int READ_UNIT>
	__global__ void decompress_func_read_sync(const uint8_t* __restrict__ in, const uint64_t n_chunks, const blk_off_t* __restrict__ blk_off, const col_len_t* __restrict__ col_len, INPUT_T* __restrict__ out) {
		__shared__ uint8_t in_[WARP_SIZE][DECODE_BUFFER_COUNT];
		__shared__ cuda::atomic<uint8_t, cuda::thread_scope_block> in_cnt_[WARP_SIZE];

		__shared__ INPUT_T out_buffer[WARP_SIZE][WRITE_VEC_SIZE];

		int tid = threadIdx.x;
		int cid = blockIdx.x;
		int which = threadIdx.y;

		uint32_t used_bytes = 0;
		uint32_t mychunk_size = col_len[cid * BLK_SIZE + tid];

		if (which == 0) {
			in_cnt_[tid] = 0;
		}

		__syncthreads();

		if (which == 0) { // reading warp
			uint64_t in_start_idx = blk_off[cid];
			uint32_t* in_4B = (uint32_t *)(&(in[in_start_idx]));
			uint32_t in_4B_off = 0;	

			uint8_t in_tail_ = 0;

// if (tid == ERR_THREAD) printf("thrad %d with chunksize %u\n", tid, mychunk_size);
			const uint32_t t_read_mask = (0xffffffff >> (32 - tid));
			while (true) {
				bool read = used_bytes < mychunk_size;
				unsigned read_sync = __ballot_sync(0xffffffff, read);

				if (read) {
					while (in_cnt_[tid].load(cuda::memory_order_acquire) + 4 > DECODE_BUFFER_COUNT) {
						__nanosleep(1000);
					}

					__syncwarp(read_sync);

					*(uint32_t*)(&in_[tid][in_tail_]) = in_4B[in_4B_off + __popc(read_sync & t_read_mask)];  
					in_cnt_[tid].fetch_add(4, cuda::memory_order_release);

					in_tail_ = (in_tail_ + 4) % DECODE_BUFFER_COUNT;
					in_4B_off += __popc(read_sync);
					used_bytes += 4;
				} else {
					break;
				}
// printf("thread %d waiting with mask %u with READ count %u\n", tid, read_sync, used_bytes);
			}
		} else if (which == 1) { // compute warp
			INPUT_T* out_8B = out + (cid * CHUNK_SIZE / sizeof(INPUT_T) + tid * READ_UNIT); 
			uint32_t out_ptr = 0;

			uint8_t in_head_ = 0;
			uint8_t *in_ptr_ = &(in_[tid][0]);
			uint8_t out_buffer_ptr = 0;
			uint8_t out_counter = 0;

			auto deque_int = [&]() {
				return;
				//TODO: 
				*reinterpret_cast<longlong4*>(out_8B + out_counter) = *reinterpret_cast<longlong4*>(out_buffer[tid]);
				
				out_counter += WRITE_VEC_SIZE;
				if (out_counter == READ_UNIT) {
					out_counter = 0;
					out_8B += BLK_SIZE * READ_UNIT;
				}    
				out_buffer_ptr = 0;
			};

			auto write_int = [&](INPUT_T i) {
 #ifdef DEBUG
 if (cid == ERR_CHUNK && tid == ERR_THREAD) printf("thread %d write int %u at idx %d\n", tid, i, (out_8B + out_buffer_ptr - out));
 #endif 

				*(out_8B + out_buffer_ptr) = i; 
				out_buffer_ptr ++;
				if (out_buffer_ptr == READ_UNIT) {
					out_buffer_ptr = 0;
					out_8B += BLK_SIZE * READ_UNIT;
				}
				return;

				if (READ_UNIT >= 4) {
					out_ptr ++;
					if (out_buffer_ptr == WRITE_VEC_SIZE) {
						deque_int();
					}
					out_buffer[tid][out_buffer_ptr++] = i;
				} else {
					*(out_8B + out_buffer_ptr) = i; 
					out_buffer_ptr ++;
					if (out_buffer_ptr == READ_UNIT) {
						out_buffer_ptr = 0;
						out_8B += BLK_SIZE * READ_UNIT;
					}
				}
			};
			
			auto read_byte = [&]() {
				while (in_cnt_[tid].load(cuda::memory_order_acquire) == 0) {
					__nanosleep(1000);
#ifdef DEBUG
// printf("chunk %d thread %d loop with %u < %u\n", cid, tid, used_bytes, mychunk_size);
#endif
					// if (out_buffer_ptr == WRITE_VEC_SIZE) {
					// 	deque_int();
					// }
					
				}

				// auto curr32 = in_[tid][in_head_ / 4].load(cuda::memory_order_relaxed);
				// auto ret = ((uint8_t*)&curr32)[in_head_%4];

				auto ret = in_ptr_[in_head_];
#ifdef DEBUG
// if (cid == ERR_CHUNK && tid == ERR_THREAD) printf("thread %d read byte %x with %u < %u\n", tid, ret, used_bytes, mychunk_size);
#endif
				in_head_ = (in_head_ + 1) % DECODE_BUFFER_COUNT;
				in_cnt_[tid].fetch_sub(1, cuda::memory_order_release);
				used_bytes += 1;
				return ret;
			};

			auto read_uvarint = [&]() {
				UINPUT_T out_int = 0;
				int offset = 0;
				uint8_t b = 0;
				do {
					b = read_byte();
					out_int |= (VARINT_MASK & b) << offset;
					offset += 7;
				} while (b >= 0x80);
				return out_int;
			};

			auto read_svarint = [&]() {
				auto ret = static_cast<INPUT_T>(read_uvarint());
				return ret >> 1 ^ -(ret & 1);
			};
			
			while (used_bytes < mychunk_size) {
				/*
				bool compute = used_bytes < mychunk_size;
				unsigned compute_sync = __ballot_sync(0xffffffff, compute);
				
				if (!compute) break;
				*/
				auto first = read_byte();
// #ifdef DEBUG
// if (cid == ERR_CHUNK && tid == ERR_THREAD) {
// 	switch(first & HEADER_MASK) {
// 		case HEADER_SHORT_REPEAT: {
// 			printf("===== case short repeat\n");
// 		} break;
// 		case HEADER_DIRECT: {
// 			printf("===== case direct\n");
// 		} break;
// 		case HEADER_DELTA: {
// 			printf("===== case DELTA\n");
// 		} break;
// 		case HEADER_PACTED_BASE: {
// 			printf("===== case patched base\n");
// 		} break;
// 		default: {
// 			printf("+++++ case should not exeist\n");
// 		} break;
// 	}
// }
// #endif
				switch(first & HEADER_MASK) {
				case HEADER_SHORT_REPEAT: {
					auto nbytes = ((first >> 3) & 0x07) + 1;
					auto count = (first & 0x07) + MINIMUM_REPEAT;
					INPUT_T tmp_int = 0;
					while (nbytes-- > 0) {
						tmp_int |= ((INPUT_T)read_byte() << (nbytes * 8));
					}
					while (count-- > 0) {
						write_int(tmp_int);
					}
				} break;
				case HEADER_DIRECT: {
					uint8_t encoded_fbw = (first >> 1) & 0x1f;
					uint8_t fbw = get_decoded_bit_width(encoded_fbw);
					uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;
					uint8_t bits_left = 0 /* bits left over from unused bits of last byte */, curr_byte = 0;
					while (len-- > 0) {
						UINPUT_T result = 0;
						uint8_t bits_to_read = fbw;
						while (bits_to_read > bits_left) {
							result <<= bits_left;
							result |= curr_byte & ((1 << bits_left) - 1);
							bits_to_read -= bits_left;
							curr_byte = read_byte();
							bits_left = 8;
						}

						if (bits_to_read > 0) {
							result <<= bits_to_read;
							bits_left -= bits_to_read;
							result |= (curr_byte >> bits_left) & ((1 << bits_to_read) - 1);
						}

						write_int(static_cast<INPUT_T>(result));
					}
				} break;
				case HEADER_DELTA: {
					uint8_t encoded_fbw = (first >> 1) & 0x1f;
					uint8_t fbw = get_decoded_bit_width(encoded_fbw);
					uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;

					INPUT_T base_val = static_cast<INPUT_T>(read_uvarint());
        			INPUT_T base_delta = static_cast<INPUT_T>(read_svarint());
// #ifdef DEBUG
// if (cid == ERR_CHUNK && tid == ERR_THREAD) printf("tid %d read base delta: %ld\n", tid, base_delta);
// #endif
					write_int(base_val);
					base_val += base_delta;
					write_int(base_val);

					len -= 2;
					int multiplier = (base_delta > 0 ? 1 : -1);
					if (encoded_fbw != 0) {
						uint8_t bits_left = 0, curr_byte = 0;
						while (len-- > 0) {
							UINPUT_T result = 0;
							uint8_t bits_to_read = fbw;
							while (bits_to_read > bits_left) {
								result <<= bits_left;
								result |= curr_byte & ((1 << bits_left) - 1);
								bits_to_read -= bits_left;
								curr_byte = read_byte();
								bits_left = 8;
							}

							if (bits_to_read > 0) {
								result <<= bits_to_read;
								bits_left -= bits_to_read;
								result |= (curr_byte >> bits_left) & ((1 << bits_to_read) - 1);
							}

							INPUT_T dlt = static_cast<INPUT_T>(result) * multiplier;
							base_val += dlt; 
							write_int(base_val);
						}
					} else {
						while (len-- > 0) {
							base_val += base_delta;
							write_int(base_val);
						}
					}
				} break;	
				case HEADER_PACTED_BASE: {
					uint8_t encoded_fbw = (first >> 1) & 0x1f;
					uint8_t fbw = get_decoded_bit_width(encoded_fbw);
					uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;

					uint8_t third = read_byte();
        			uint8_t forth = read_byte();

					uint8_t bw = ((third >> 5) & 0x07) + 1;
					uint8_t pw = get_decoded_bit_width(third & 0x1f);
					uint8_t pgw = ((forth >> 5) & 0x07) + 1;
					uint8_t pll = forth & 0x1f;

					uint16_t patch_mask = (static_cast<uint16_t>(1) << pw) - 1;

					uint32_t base_out_ptr = out_ptr;

					INPUT_T base_val = 0 ;
					while (bw-- > 0) {
						base_val |= ((INPUT_T)read_byte() << (bw * 8));
					}
					uint8_t bits_left = 0 /* bits left over from unused bits of last byte */, curr_byte = 0;
					while (len-- > 0) {
						UINPUT_T result = 0;
						uint8_t bits_to_read = fbw;
						while (bits_to_read > bits_left) {
							result <<= bits_left;
							result |= curr_byte & ((1 << bits_left) - 1);
							bits_to_read -= bits_left;
							curr_byte = read_byte();
							bits_left = 8;
						}

						if (bits_to_read > 0) {
							result <<= bits_to_read;
							bits_left -= bits_to_read;
							result |= (curr_byte >> bits_left) & ((1 << bits_to_read) - 1);
						}
						
						write_int(static_cast<INPUT_T>(result) + base_val);
					}

					bits_left = 0, curr_byte = 0;
					uint8_t cfb = get_closest_bit(pw + pgw);
					int patch_gap = 0;
					while (pll-- > 0) {
						UINPUT_T result = 0;
						uint8_t bits_to_read = cfb;
						while (bits_to_read > bits_left) {
							result <<= bits_left;
							result |= curr_byte & ((1 << bits_left) - 1);
							bits_to_read -= bits_left;
							curr_byte = read_byte();
							bits_left = 8;
						}
						
						if (bits_to_read > 0) {
							result <<= bits_to_read;
							bits_left -= bits_to_read;
							result |= (curr_byte >> bits_left) & ((1 << bits_to_read) - 1);
						}

						patch_gap += result >> pw;
						uint32_t direct_out_ptr = base_out_ptr + patch_gap;

						if (out_ptr - direct_out_ptr >= WRITE_VEC_SIZE || out_buffer_ptr == 0) {
							out[(direct_out_ptr / READ_UNIT) * BLK_SIZE * READ_UNIT + (direct_out_ptr % READ_UNIT) + tid * READ_UNIT] |= static_cast<INPUT_T>(result & patch_mask) << fbw;
						} else {
							out_buffer[tid][direct_out_ptr % WRITE_VEC_SIZE] |= static_cast<INPUT_T>(result & patch_mask) << fbw;
						}
						
					}
				} break;
				}
			}
			if (out_buffer_ptr > 0) {
				deque_int();
			}
		}
    }
};

#endif