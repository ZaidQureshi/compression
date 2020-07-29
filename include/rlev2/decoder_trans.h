#ifndef _RLEV2_DECODER_TRANPOSE_H_
#define _RLEV2_DECODER_TRANPOSE_H_

#include "utils.h"
#include "decoder_non_template.h"
#include <stdio.h>

#define DEBUG

// constexpr int DECODE_BUFFER_COUNT = 256;
// constexpr int SHM_BUFFER_COUNT = DECODE_BUFFER_COUNT * BLK_SIZE;

namespace rlev2 {
    template<int read_unit>
	__global__ void decompress_func(const uint8_t* __restrict__ in, const uint64_t n_chunks, const blk_off_t* __restrict__ blk_off, const col_len_t* __restrict__ col_len, int64_t* __restrict__ out) {
		auto tid = threadIdx.x;
		auto cid = blockIdx.x;
		
		uint64_t mychunk_size = col_len[cid * BLK_SIZE + tid];
		uint64_t in_start_idx = blk_off[cid];

		uint64_t out_start_idx = cid * CHUNK_SIZE / sizeof(int64_t);

		uint32_t* in_4B = (uint32_t *)(&(in[in_start_idx]));
		uint32_t in_4B_off = 0;

		int64_t* out_8B = out + (out_start_idx + tid * read_unit); 
		// printf("out off %lu for thread %d\n", out_start_idx + tid * 1, tid);

		// uint8_t input_buffer[DECODE_BUFFER_COUNT];

		__shared__ uint8_t shm_buffer[SHM_BUFFER_COUNT];
		uint8_t *input_buffer = &shm_buffer[DECODE_BUFFER_COUNT * tid];



		uint8_t input_buffer_head = 0;
		uint8_t input_buffer_tail = 0;
		int input_buffer_count = 0;

		uint8_t curr_fbw = 0, curr_fbw_left = 0;

		bool read_first = false, read_second;
		uint8_t first_byte;

		// for complicated encoding schemes, input_buffer may not have enough data
		// needs a structure to hold the info of last encoding scheme
		int curr_len = 0; 

		int64_t curr_64; // current outint, might not be complete
		uint16_t bits_left = 0;
		uint8_t bits_left_over; // leftover from last byte since bit packing

		bool dal_read_base = false;
		int64_t base_val, base_delta, *base_out; // for delta encoding
        int base_write_offset;

		int bw, pw, pgw, pll, patch_gap, curr_pwb_left; // for patched base encoding

        int curr_write_offset = 0;
		uint64_t used_bytes = 0;

		auto read_byte = [&]() {
// #ifdef DEBUG
// if (cid == 0 && tid == ERR_THREAD) printf("thread %d read byte %x\n", tid, input_buffer[input_buffer_head]);
// #endif
			auto ret = input_buffer[input_buffer_head];
			input_buffer_count -= 1;
			used_bytes += 1;
			input_buffer_head = (input_buffer_head + 1) % DECODE_BUFFER_COUNT;
			return ret;
		};

		auto write_int = [&](int64_t i) {
// #ifdef DEBUG
// if (cid == 0 && tid == ERR_THREAD) printf("thread %d write int %ld at idx %d\n", tid, i, (out_8B - out));
// #endif
			*(out_8B + curr_write_offset)= i; 
            curr_write_offset ++;
			if (curr_write_offset == read_unit) {
                curr_write_offset = 0;
                out_8B += BLK_SIZE * read_unit;
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

		auto read_long = [&](uint8_t fbw) {
			int64_t ret = 0;
			while (fbw-- > 0) {
				ret |= (read_byte() << (fbw * 8));
			}
			return ret;
		};

		auto read_longs = [&]() {
			while (curr_len > 0) {
				while (curr_fbw_left > bits_left) {
					if (input_buffer_count <= 0) return;
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
		};

		const uint32_t t_read_mask = (0xffffffff >> (32 - tid));
        while (used_bytes < mychunk_size || curr_len > 0) {
// #ifdef DEBUG
// iteration ++;
// #endif
            auto mask = __activemask();
            bool read = used_bytes + input_buffer_count < mychunk_size;
			uint32_t read_sync = __ballot_sync(mask, (read));
			auto res = __popc(read_sync);
            if (read) {
				int left_act = __popc(read_sync & t_read_mask);
				*(uint32_t *)(&(input_buffer[input_buffer_tail])) = in_4B[in_4B_off + left_act];  
// #ifdef DEBUG
// if (cid ==0 && tid == ERR_THREAD) {
// 	printf("thread %d read bytes at %u %x%x%x%x\n", tid, (in_4B_off + left_act) * 4, input_buffer[input_buffer_tail], 
// 	input_buffer[input_buffer_tail + 1], 
// 	input_buffer[input_buffer_tail + 2], 
// 	input_buffer[input_buffer_tail + 3]);
// }
// #endif
				input_buffer_tail = (input_buffer_tail + 4) % DECODE_BUFFER_COUNT;
				input_buffer_count += 4;
				in_4B_off += res; 
            } 

			if (!read_first) {
				read_first = true;
				first_byte = read_byte();
        		curr_fbw = get_decoded_bit_width((first_byte >> 1) & 0x1f);
				curr_fbw_left = curr_fbw;
				read_second = false;
			}

			switch(first_byte & HEADER_MASK) {
			case HEADER_SHORT_REPEAT: {
				auto num_bytes = ((first_byte >> 3) & 0x07) + 1;
				if (num_bytes <= input_buffer_count) { 
					int64_t tmp_int = 0;
					while (num_bytes-- > 0) {
						tmp_int |= ((int64_t)read_byte() << (num_bytes * 8));
					}
					auto cnt = (first_byte & 0x07) + MINIMUM_REPEAT;
					while (cnt-- > 0) {
						*(out_8B + curr_write_offset)= tmp_int; 
						curr_write_offset ++;
						if (curr_write_offset == read_unit) {
							curr_write_offset = 0;
							out_8B += BLK_SIZE * read_unit;
						} 
					}
					read_first = false;
				} 
			}	break;
			case HEADER_DIRECT: {
				if (!read_second) {
					read_second = true;
					curr_len = (((first_byte & 0x01) << 8) | read_byte()) + 1;
					bits_left = 0; bits_left_over = 0;
					curr_64 = 0;
				}
				
				read_longs();

				if (curr_len <= 0) {
					read_first = false;
				}
			}	break;
			case HEADER_DELTA: {
				if (!read_second) {
					read_second = true;
					curr_len = (((first_byte & 0x01) << 8) | read_byte()) + 1;
					bits_left = 0;
					bits_left_over = 0;
					curr_64 = 0;
					dal_read_base = false;
				}

				if (!dal_read_base && (input_buffer_count >= 64 + curr_fbw || !read)) {
					//TODO: fbw only ensures delta bits. No simple way to make sure base val bits.
					//TODO: try find a safer way.
					base_val = read_uvarint();
					base_delta = read_svarint();
					write_int(base_val);
					base_val += base_delta;
					write_int(base_val);

					base_out = out_8B;
                    base_write_offset = curr_write_offset;

					dal_read_base = true;
				}

				if (dal_read_base) {
					if (((first_byte >> 1) & 0x1f) != 0) {
						// var delta
						read_longs();
					} else {
						// fixed delta encoding
						while (curr_len > 0) {
							base_val += base_delta;
							write_int(base_val);
							curr_len --;
						}
					}

					if (curr_len <= 0) {
						if (((first_byte >> 1) & 0x1f) != 0) {
							if (base_delta > 0) {
								while (base_out < out_8B) {
									base_val = *(base_out + base_write_offset) += base_val;
									base_write_offset ++;

                                    if (base_write_offset == read_unit) {
                                        base_out += BLK_SIZE * read_unit;
                                        base_write_offset = 0;
                                    } 
								}
							} else {
								while (base_out < out_8B) {
									base_val = *(base_out + base_write_offset) = base_val - *(base_out + base_write_offset);
									base_write_offset ++;

                                    if (base_write_offset == read_unit) {
                                        base_out += BLK_SIZE * read_unit;
                                        base_write_offset = 0;
                                    } 
								}
							}
						}
						
						read_first = false;
					}
				}
			}	break;
			case HEADER_PACTED_BASE: {
// if (tid == ERR_THREAD) {
// 	printf("patched base case\n");
// }
				//TODO Try to guarantee there are at least 4 btyes to read (all headers)
				if (!read_second) {
					if (input_buffer_count < 3) break;
				// Here for patched base, read_second includes third and forth header byte
					curr_len = (((first_byte & 0x01) << 8) | read_byte()) + 1;
					auto third = read_byte();
        			auto forth = read_byte();

					bw = ((third >> 5) & 0x07) + 1;
					pw = get_decoded_bit_width(third & 0x1f);
					pgw = ((forth >> 5) & 0x07) + 1;
					pll = forth & 0x1f;

					bits_left = 0;
					bits_left_over = 0;
					curr_64 = 0;
					patch_gap = 0;
					dal_read_base = false;

					curr_pwb_left = get_closest_bit(pw + pgw);

					read_second = true;
				}
				
				if (!read_second) break;

				if (!dal_read_base && input_buffer_count >= bw) {
					base_val = read_long(bw);
					base_out = out_8B;
                    base_write_offset = curr_write_offset;

					dal_read_base = true;
				}

				if (!dal_read_base) break;

				if (curr_len > 0) {
					read_longs();
				} else {
					auto patch_mask = (static_cast<uint16_t>(1) << pw) - 1;
					while (pll > 0) {
						while (input_buffer_count > 0 && curr_pwb_left > bits_left) {
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
							base_out[(patch_gap / read_unit) * BLK_SIZE + (patch_gap % read_unit)] |= static_cast<int64_t>(curr_64 & patch_mask) << curr_fbw;


							pll --;
							curr_64 = 0;
							curr_pwb_left = get_closest_bit(pw + pgw);
						} 
						if (input_buffer_count <= 0) break;

					}
					if (pll <= 0) {
						while (base_out < out_8B) {
							*base_out += base_val;
							if (base_write_offset == read_unit - 1) {
                                base_out += BLK_SIZE * read_unit;
                                base_write_offset = 0;
                            } else {
                                base_out ++;
                                base_write_offset ++;
                            }
						}
						read_first = false; 
					}
				}
			}	break;
			default:
			printf("something went wrong=================\n");
			break;
			}

			__syncwarp(mask);
// #ifdef DEBUG
// if (iteration > max_iter) {
// printf("break max iter with tid %d %lu(%lu) currlen %d \n", tid, used_bytes, mychunk_size, curr_len);
// break;
// }
// #endif
        }
		

    }

	__host__
	void decompress_gpu(const uint8_t *in, const uint64_t in_n_bytes, const uint64_t n_chunks,
			blk_off_t *blk_off, col_len_t *col_len,
			int64_t *&out, uint64_t &out_n_bytes) {
		initialize_bit_maps();

#ifdef DEBUG
		printf("======> decompress kernerl\n");
#endif
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
		decompress_func_non_template<<<n_chunks, BLK_SIZE>>>(d_in, n_chunks, d_blk_off, d_col_len, d_out);
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