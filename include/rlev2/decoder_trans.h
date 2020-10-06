#ifndef _RLEV2_DECODER_TRANPOSE_H_
#define _RLEV2_DECODER_TRANPOSE_H_
#include "cuda_profiler_api.h"
#include "utils.h"

#include <cuda/atomic>

namespace rlev2 {
	template <int READ_UNIT>
	__global__ void decompress_func_template(const uint8_t* __restrict__ in, const uint64_t n_chunks, const blk_off_t* __restrict__ blk_off, const col_len_t* __restrict__ col_len, int64_t* __restrict__ out) {
		auto tid = threadIdx.x;
		auto cid = blockIdx.x;
		
		uint64_t mychunk_size = col_len[cid * BLK_SIZE + tid];
		uint64_t in_start_idx = blk_off[cid];

		uint64_t out_start_idx = cid * CHUNK_SIZE / sizeof(int64_t);

		uint32_t* in_4B = (uint32_t *)(&(in[in_start_idx]));
		uint32_t in_4B_off = 0;

		int64_t* out_8B = out + (out_start_idx + tid * READ_UNIT); 
		// printf("out off %lu for thread %d\n", out_start_idx + tid * 1, tid);

		// uint8_t input_buffer[DECODE_BUFFER_COUNT];

		// __shared__ uint8_t shm_buffer[SHM_BUFFER_COUNT];
		// uint8_t *input_buffer = &shm_buffer[DECODE_BUFFER_COUNT * tid];

		uint8_t shm_buffer[DECODE_BUFFER_COUNT];
		uint8_t *input_buffer = &shm_buffer[0];

		uint8_t curr_schm = 0;

		uint8_t input_buffer_head = 0;
		uint8_t input_buffer_tail = 0;
		uint8_t input_buffer_count = 0;

		uint8_t curr_fbw = 0, curr_fbw_left = 0;

		uint8_t first_byte;

		// for complicated encoding schemes, input_buffer may not have enough data
		// needs a structure to hold the info of last encoding scheme
		int curr_len = 0; 

		int64_t curr_64; // current outint, might not be complete
		uint16_t bits_left = 0;
		uint8_t bits_left_over; // leftover from last byte since bit packing

		bool dal_read_base = false;
		int64_t base_val, base_delta, *base_out; // for delta encoding

		int bw, pw, pgw, pll, patch_gap, curr_pwb_left; // for patched base encoding

		uint64_t used_bytes = 0;

        int curr_write_offset = 0;

		auto read_byte = [&]() {
			auto ret = input_buffer[input_buffer_head];
			input_buffer_count -= 1;
			used_bytes += 1;
			input_buffer_head = (input_buffer_head + 1) % DECODE_BUFFER_COUNT;
			return ret;
		};

		auto write_int = [&](int64_t i) {
			*(out_8B + curr_write_offset) = i; 
            curr_write_offset ++;
            if (curr_write_offset == READ_UNIT) {
                curr_write_offset = 0;
                out_8B += BLK_SIZE * READ_UNIT;
            }

			curr_len --;
			curr_64 = 0;
			curr_fbw_left = curr_fbw;
		};

		auto read_uvarint = [&]() {
			uint64_t out_int = 0;
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
			auto ret = static_cast<int64_t>(read_uvarint());
			return ret >> 1 ^ -(ret & 1);
		};

		const uint32_t t_read_mask = (0xffffffff >> (32 - tid));

        while (used_bytes < mychunk_size) {
			
            auto mask = __activemask();
	        bool read;
			#pragma unroll
			for (int read_iter=0; read_iter<2; ++read_iter) {
				
				read = used_bytes + input_buffer_count < mychunk_size;
				uint32_t read_sync = __ballot_sync(mask, read);
				if (read) {
					*(uint32_t *)(&(input_buffer[input_buffer_tail])) = in_4B[in_4B_off + __popc(read_sync & t_read_mask)];  
					input_buffer_tail = (input_buffer_tail + 4) % DECODE_BUFFER_COUNT;
					input_buffer_count += 4;
					in_4B_off += __popc(read_sync); 
				} 
				__syncwarp(mask);
			}


			if (curr_schm == 0) {
				first_byte = read_byte();
				curr_schm = first_byte & HEADER_MASK;
				curr_fbw = get_decoded_bit_width((first_byte >> 1) & 0x1f);
				curr_fbw_left = curr_fbw;

				if (curr_schm != HEADER_SHORT_REPEAT) {
					curr_len = (((first_byte & 0x01) << 8) | read_byte()) + 1;
					bits_left = 0; bits_left_over = 0; curr_64 = 0;

					dal_read_base = false;
					
					if (curr_schm == HEADER_PACTED_BASE) {
						auto third = read_byte();
						auto forth = read_byte();

						bw = ((third >> 5) & 0x07) + 1;
						pw = get_decoded_bit_width(third & 0x1f);
						pgw = ((forth >> 5) & 0x07) + 1;
						pll = forth & 0x1f;
						patch_gap = 0;

						curr_pwb_left = get_closest_bit(pw + pgw);
					}
				}
			}

			switch(curr_schm) {
			case HEADER_SHORT_REPEAT: {
				auto num_bytes = ((first_byte >> 3) & 0x07) + 1;
				if (num_bytes <= input_buffer_count) { 
					int64_t tmp_int = 0;
					while (num_bytes-- > 0) {
						tmp_int |= ((int64_t)read_byte() << (num_bytes * 8));
					}
					auto cnt = (first_byte & 0x07) + MINIMUM_REPEAT;
					while (cnt-- > 0) {
						// *(out_8B + curr_write_offset) = tmp_int; 
                        // curr_write_offset ++;
                        // if (curr_write_offset == READ_UNIT) {
                        //     curr_write_offset = 0;
						//     out_8B += BLK_SIZE * READ_UNIT;
                        // }
						write_int(tmp_int);
					}
					curr_schm = 0;
				} 
			}	break;
			case HEADER_DIRECT: {
				while (curr_len > 0) {
					while (curr_fbw_left > bits_left) {
						if (input_buffer_count == 0) goto main_loop;
						curr_64 <<= bits_left;
						curr_64 |= bits_left_over & ((1 << bits_left) - 1);
						curr_fbw_left -= bits_left;
						bits_left_over = read_byte();
						bits_left = 8;
					}

					if (curr_fbw_left <= bits_left) {
						if (curr_fbw_left > 0) {
							curr_64 <<= curr_fbw_left;
							bits_left -= curr_fbw_left;
							curr_64 |= (bits_left_over >> bits_left) & ((1 << curr_fbw_left) - 1);
						}

						write_int(curr_64);
					} 
				}
				
				curr_schm = 0;
			}	break;
			case HEADER_DELTA: {
				if (!dal_read_base) {
					if (read && input_buffer_count < 64 + curr_fbw) break;
					dal_read_base = true;

					base_val = read_uvarint();
					base_delta = read_svarint();
					write_int(base_val);
					base_val += base_delta;
					write_int(base_val);
				}

				if (((first_byte >> 1) & 0x1f) != 0) {
					// var delta
					while (curr_len > 0) {
						while (curr_fbw_left > bits_left) {
							if (input_buffer_count == 0) goto main_loop;
							curr_64 <<= bits_left;
							curr_64 |= bits_left_over & ((1 << bits_left) - 1);
							curr_fbw_left -= bits_left;
							bits_left_over = read_byte();
							bits_left = 8;
						}

						if (curr_fbw_left <= bits_left) {
							if (curr_fbw_left > 0) {
								curr_64 <<= curr_fbw_left;
								bits_left -= curr_fbw_left;
								curr_64 |= (bits_left_over >> bits_left) & ((1 << curr_fbw_left) - 1);
							}
							base_val += curr_64; //TODO: THIS IS NOT ALWAYS +.
							write_int(base_val);
						} 
					}
				} else {
					// fixed delta encoding
					while (curr_len > 0) {
						base_val += base_delta;
						write_int(base_val);
					}
				}
				curr_schm = 0;
			}	break;
			case HEADER_PACTED_BASE: {
				if (!dal_read_base) {
                    if (input_buffer_count < bw) break;
					dal_read_base = true;

					base_val = 0;
					auto fbw = bw;
					while (fbw-- > 0) {
						base_val |= (read_byte() << (fbw * 8));
					}
					base_out = out_8B;
				} 

                while (curr_len > 0) {
					while (curr_fbw_left > bits_left) {
						if (input_buffer_count == 0) goto main_loop;
						curr_64 <<= bits_left;
						curr_64 |= bits_left_over & ((1 << bits_left) - 1);
						curr_fbw_left -= bits_left;
						bits_left_over = read_byte();
						bits_left = 8;
					}

					if (curr_fbw_left <= bits_left) {
						if (curr_fbw_left > 0) {
							curr_64 <<= curr_fbw_left;
							bits_left -= curr_fbw_left;
							curr_64 |= (bits_left_over >> bits_left) & ((1 << curr_fbw_left) - 1);
						}
						curr_64 += base_val;
						write_int(curr_64);
					} 
				}

				auto patch_mask = (static_cast<uint16_t>(1) << pw) - 1;
				while (pll > 0) {
					while (curr_pwb_left > bits_left) {
						if (input_buffer_count == 0) goto main_loop;
						curr_64 <<= bits_left;
						curr_64 |= bits_left_over & ((1 << bits_left) - 1);
						curr_pwb_left -= bits_left;
						bits_left_over = read_byte();
						bits_left = 8;
					}

					if (curr_pwb_left <= bits_left) {
						if (curr_pwb_left > 0) {
							curr_64 <<= curr_pwb_left;
							bits_left -= curr_pwb_left;
							curr_64 |= (bits_left_over >> bits_left) & ((1 << curr_pwb_left) - 1);
						}

						patch_gap += curr_64 >> pw;
						base_out[(patch_gap / READ_UNIT) * BLK_SIZE + (patch_gap % READ_UNIT)] |= static_cast<int64_t>(curr_64 & patch_mask) << curr_fbw;

						pll --;
						curr_64 = 0;
						curr_pwb_left = get_closest_bit(pw + pgw);
					} 
				}
				curr_schm = 0; 
			}	break;
			default: {
				curr_schm = 0;
			} break;
			}
			main_loop:
			__syncwarp(mask);
        }
		

    }

	template <int READ_UNIT>
	__global__ void decompress_func_read_sync(const uint8_t* __restrict__ in, const uint64_t n_chunks, const blk_off_t* __restrict__ blk_off, const col_len_t* __restrict__ col_len, int64_t* __restrict__ out) {
		__shared__ uint8_t in_[BLK_SIZE][DECODE_BUFFER_COUNT];
		__shared__ cuda::atomic<uint8_t, cuda::thread_scope_block> in_cnt_[BLK_SIZE];

		__shared__ int64_t out_buffer[BLK_SIZE][WRITE_VEC_SIZE];

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
			int64_t* out_8B = out + (cid * CHUNK_SIZE / sizeof(int64_t) + tid * READ_UNIT); 
			uint32_t out_ptr = 0;

			uint8_t in_head_ = 0;
			uint8_t *in_ptr_ = &(in_[tid][0]);
			uint8_t out_buffer_ptr = 0;
			uint8_t out_counter = 0;

			auto deque_int = [&]() {
				*reinterpret_cast<longlong4*>(out_8B + out_counter) = *reinterpret_cast<longlong4*>(out_buffer[tid]);
				
				out_counter += WRITE_VEC_SIZE;
				if (out_counter == READ_UNIT) {
					out_counter = 0;
					out_8B += BLK_SIZE * READ_UNIT;
				}    
				out_buffer_ptr = 0;
			};

			auto write_int = [&](int64_t i) {
				out_ptr ++;
				// *(out_8B + out_buffer_ptr) = i; 
				// out_buffer_ptr ++;
				// if (out_buffer_ptr == READ_UNIT) {
				// 	out_buffer_ptr = 0;
				// 	out_8B += BLK_SIZE * READ_UNIT;
				// }
				
				if (out_buffer_ptr == WRITE_VEC_SIZE) {
					deque_int();
				}

				out_buffer[tid][out_buffer_ptr++] = i;
				// out_buffer_ptr = (out_buffer_ptr + 1) % WRITE_VEC_SIZE;
				// if (out_buffer_ptr == 0) {
				// 	*reinterpret_cast<longlong4*>(out_8B) = *reinterpret_cast<longlong4*>(out_buffer[tid]);
				// 	out_8B += BLK_SIZE * READ_UNIT;
				// }
				
			};
			
			auto read_byte = [&]() {
				while (in_cnt_[tid].load(cuda::memory_order_acquire) == 0) {
					// __nanosleep(1000);
					
					if (out_buffer_ptr == WRITE_VEC_SIZE) {
						deque_int();
					}
					
				}

				// auto curr32 = in_[tid][in_head_ / 4].load(cuda::memory_order_relaxed);
				// auto ret = ((uint8_t*)&curr32)[in_head_%4];

				auto ret = in_ptr_[in_head_];
// #ifdef DEBUG
// if (tid == ERR_THREAD) printf("read[%u]: %x\n", curr_head, ((uint8_t*)&curr32)[curr_head%4]);
// #endif
				in_head_ = (in_head_ + 1) % DECODE_BUFFER_COUNT;
				in_cnt_[tid].fetch_sub(1, cuda::memory_order_release);
				used_bytes += 1;
				return ret;
			};

			auto read_uvarint = [&]() {
				uint64_t out_int = 0;
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
				auto ret = static_cast<int64_t>(read_uvarint());
				return ret >> 1 ^ -(ret & 1);
			};
			
			while (used_bytes < mychunk_size) {
				/*
				bool compute = used_bytes < mychunk_size;
				unsigned compute_sync = __ballot_sync(0xffffffff, compute);
				
				if (!compute) break;
				*/
				auto first = read_byte();
				switch(first & HEADER_MASK) {
				case HEADER_SHORT_REPEAT: {
					auto nbytes = ((first >> 3) & 0x07) + 1;
					auto count = (first & 0x07) + MINIMUM_REPEAT;
					int64_t tmp_int = 0;
					while (nbytes-- > 0) {
						tmp_int |= ((int64_t)read_byte() << (nbytes * 8));
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
						uint64_t result = 0;
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

						write_int(static_cast<int64_t>(result));
					}
				} break;
				case HEADER_DELTA: {
					uint8_t encoded_fbw = (first >> 1) & 0x1f;
					uint8_t fbw = get_decoded_bit_width(encoded_fbw);
					uint8_t len = ((static_cast<uint16_t>(first & 0x01) << 8) | read_byte()) + 1;

					int64_t base_val = static_cast<int64_t>(read_uvarint());
        			int64_t base_delta = static_cast<int64_t>(read_svarint());

					write_int(base_val);
					base_val += base_delta;
					write_int(base_val);

					len -= 2;
					int multiplier = (base_delta > 0 ? 1 : -1);
					if (encoded_fbw != 0) {
						uint8_t bits_left = 0, curr_byte = 0;
						while (len-- > 0) {
							uint64_t result = 0;
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

							int64_t dlt = static_cast<int64_t>(result) * multiplier;
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

					int64_t base_val = 0 ;
					while (bw-- > 0) {
						base_val |= ((int64_t)read_byte() << (bw * 8));
					}
					uint8_t bits_left = 0 /* bits left over from unused bits of last byte */, curr_byte = 0;
					while (len-- > 0) {
						uint64_t result = 0;
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
						
						write_int(static_cast<int64_t>(result) + base_val);
					}

					bits_left = 0, curr_byte = 0;
					uint8_t cfb = get_closest_bit(pw + pgw);
					int patch_gap = 0;
					while (pll-- > 0) {
						uint64_t result = 0;
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

						// It is not possible to have PATCHED BASE ENCODING WITH size < 4(WRITE_VEC_SIZE)
						// if (virtual_offset < WRITE_VEC_SIZE) {
						// 	// still in local buffer
						// 	out_buffer[tid][virtual_offset % WRITE_VEC_SIZE] |= static_cast<int64_t>(result & patch_mask) << fbw;
						// } else {

						if (out_ptr - direct_out_ptr >= WRITE_VEC_SIZE || out_buffer_ptr == 0) {
							out[(direct_out_ptr / READ_UNIT) * BLK_SIZE * READ_UNIT + (direct_out_ptr % READ_UNIT) + tid * READ_UNIT] |= static_cast<int64_t>(result & patch_mask) << fbw;
						} else {
							out_buffer[tid][direct_out_ptr % WRITE_VEC_SIZE] |= static_cast<int64_t>(result & patch_mask) << fbw;
						}
						
						// }
						// auto index = (patch_gap / READ_UNIT) * BLK_SIZE + (patch_gap % READ_UNIT);
						// Needs to determine whether the index is still in local bufer or global mem
					}
				} break;
				}
			}
			if (out_buffer_ptr > 0) {
				deque_int();
			}
		}
    }

	template<int READ_UNIT>
	__host__
	void decompress_gpu(const uint8_t *in, const uint64_t in_n_bytes, const uint64_t n_chunks,
			blk_off_t *blk_off, col_len_t *col_len,
			int64_t *&out, uint64_t &out_n_bytes) {
		printf("Calling decompress kernel.\n");

		initialize_bit_maps();
		uint8_t *d_in;
		int64_t *d_out;
		blk_off_t *d_blk_off;
		col_len_t *d_col_len;

		auto exp_out_n_bytes = blk_off[n_chunks];
		out_n_bytes = exp_out_n_bytes;


		cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
		cuda_err_chk(cudaMalloc(&d_out, exp_out_n_bytes));
		cuda_err_chk(cudaMalloc(&d_blk_off, sizeof(blk_off_t) * n_chunks));
		cuda_err_chk(cudaMalloc(&d_col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE));
			
		cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
		cuda_err_chk(cudaMemcpy(d_blk_off, blk_off, sizeof(blk_off_t) * n_chunks, cudaMemcpyHostToDevice));
		cuda_err_chk(cudaMemcpy(d_col_len, col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE, cudaMemcpyHostToDevice));


		std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();
		// decompress_func_write_sync<<<n_chunks, dim3(BLK_SIZE, 2, 1)>>>(d_in, n_chunks, d_blk_off, d_col_len, d_out);
		decompress_func_read_sync<READ_UNIT><<<n_chunks, dim3(BLK_SIZE, 2, 1)>>>(d_in, n_chunks, d_blk_off, d_col_len, d_out);
		cuda_err_chk(cudaDeviceSynchronize());
		std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);		

		std::cout << "kernel time: " << total.count() << " secs\n";
	
		out = new int64_t[exp_out_n_bytes / sizeof(int64_t)];
		cuda_err_chk(cudaMemcpy(out, d_out, exp_out_n_bytes, cudaMemcpyDeviceToHost));
		
		cuda_err_chk(cudaFree(d_in));
		cuda_err_chk(cudaFree(d_out));
		cuda_err_chk(cudaFree(d_blk_off));
		cuda_err_chk(cudaFree(d_col_len));
	}

}
#endif
