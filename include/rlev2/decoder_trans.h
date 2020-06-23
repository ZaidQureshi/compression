#ifndef _RLEV2_DECODER_TRANPOSE_H_
#define _RLEV2_DECODER_TRANPOSE_H_

#include "utils.h"
#include <stdio.h>

#define WRITE_OFFSET 1

namespace rlev2 {
    void decompress_func_new(const uint8_t* const in, int64_t* out, 
                        const uint64_t in_n_bytes, const uint64_t out_n_bytes, 
                        const uint64_t out_chunk_size, const uint64_t n_chunks, 
                        const uint64_t* const blk_off, const uint64_t* const col_len, const uint8_t* const col_map) {
		// int tid = threadIdx.x;
		int tid = 0;
		// int chunk_idx = blockIdx.x;
		int chunk_idx = 0;
		uint64_t used_bytes = 0;
		// uint64_t mychunk_size = col_len[blockDim.x * chunk_idx + tid];
		uint64_t mychunk_size = col_len[0 * chunk_idx + tid];
		uint64_t in_start_idx = blk_off[chunk_idx];

		uint8_t header_byte = 0;
		uint8_t block = 0;

		uint64_t out_bytes = 0;
		uint64_t out_start_idx = chunk_idx * CHUNK_SIZE;

		uint8_t compress_counter = 0;
		// int col_idx = col_map[blockDim.x * chunk_idx + tid];
		int col_idx = col_map[0 * chunk_idx + tid];

		uint8_t v = 0;
		uint32_t* in_4B = (uint32_t *)(&(in[in_start_idx]));
		uint32_t in_4B_off = 0;


		int data_counter = 0;

		int64_t out_buffer = 0;
		uint8_t* out_buffer_8 = (uint8_t*) &out_buffer;
		uint8_t out_buffer_tail = 0;
		uint64_t out_off = 0;

		int64_t* out_8B = (int64_t*)(&(out[out_start_idx + col_idx*4]));

		uint8_t input_buffer[INPUT_BUFFER_SIZE];
		uint8_t input_buffer_head = 0;
		uint8_t input_buffer_tail = 0;
		uint8_t input_buffer_count = 0;
		// uint8_t input_buffer_read_count = 0;

		uint64_t read_count = 0;

		bool read_first = false, read_second = false, read_third = false, read_forth = false;
		uint8_t first_byte, third_byte, forth_byte;

		// for complicated encoding schemes, input_buffer may not have enough data
		// needs a structure to hold the info of last encoding scheme
		uint16_t curr_len = 0; 

		int64_t curr_64; // current outint, might not be complete
		uint16_t bits_left = 0;
		uint8_t bits_left_over; // leftover from last byte since bit packing

		bool dal_read_base = false;
		int64_t base_val, base_delta, *base_out; // for delta encoding

		uint8_t bw, pw, pgw, pll, patch_gap; // for patched base encoding

#define proceed(x) { \
	input_buffer_count -= x; \
	used_bytes += x; \
	input_buffer_head = (input_buffer_head + x) % INPUT_BUFFER_SIZE; \
}

#define write_varint(i) { \
	*out_8B = i; \
	out_8B += WRITE_OFFSET; \
}

		auto read_byte = [&]() {
			auto ret = input_buffer[input_buffer_head];
			input_buffer_count -= 1;
			used_bytes += 1;
			input_buffer_head = (input_buffer_head + 1) % INPUT_BUFFER_SIZE;
			return ret;
		};

		// by default, lambda captures what was referecnes in lambda
		auto read_uvarint = [&]() {
			uint64_t out = 0;
			int offset = 0;
			uint8_t b = 0;
			do {
				b = input_buffer[input_buffer_head];
				proceed(1);
				out |= (0x7f & b) << offset;
				offset += 7;
			} while (b >= 0x80);
			return out;
		};

		auto read_svarint = [&]() {
			auto ret = static_cast<int64_t>(read_uvarint());
			return ret >> 1 ^ -(ret & 1);
		};

		auto read_long = [&](uint8_t fbw) {
			int64_t ret = 0;
			while (fbw-- > 0) {
				ret |= (input_buffer[input_buffer_head] << (fbw * 8));
				proceed(1);
			}
			return ret;
		};

		auto read_longs = [&](uint8_t fbw) {
			while (curr_len > 0 && (input_buffer_count || bits_left)) {
				auto bits_left_to_read = fbw;
				while (input_buffer_count && bits_left_to_read > bits_left) {
					curr_64 <<= bits_left;
					curr_64 |= bits_left_over & ((1 << bits_left) - 1);
					bits_left_to_read -= bits_left;
					bits_left_over = input_buffer[input_buffer_head];
					proceed(1);
					bits_left = 8;
					
				}

				if (bits_left_to_read <= bits_left) {
				// if not curr_64 is imcomplete
					if (bits_left_to_read > 0) {
						curr_64 <<= bits_left_to_read;
						bits_left -= bits_left_to_read;
						curr_64 |= (bits_left_over >> bits_left) & ((1 << bits_left_to_read) - 1);
					}

					write_varint(curr_64);
					curr_len --;
					curr_64 = 0;
				}
			}
		};



		// printf("chunk size: %lu\n", mychunk_size);

		// bits_left is for 3 other encoding schemes
        while (used_bytes < mychunk_size || curr_len > 0) {
			// printf("loop\n");
            // unsigned mask = __activemask();
            bool read = read_count < mychunk_size;
			// int res = __popc(__ballot_sync(mask, (read)));
            if (read && input_buffer_count + 4 <= INPUT_BUFFER_SIZE) {
                uint32_t* input_buffer_4B = (uint32_t *)(&(input_buffer[input_buffer_tail]));
				input_buffer_4B[0] = in_4B[in_4B_off + tid];

				input_buffer_tail = (input_buffer_tail + 4) % INPUT_BUFFER_SIZE;
				input_buffer_count += 4;
				read_count += 4;
		
				in_4B_off += 1; //TODO: this should be block offset
            } 

			if (!read_first) {
				first_byte = input_buffer[input_buffer_head];
				// printf("read first byte %x\n", first_byte);
				proceed(1);
				read_first = true;
			}

			switch(first_byte & 0xC0) {
			case HEADER_SHORT_REPEAT: {
				// printf("cas	e short repeat\n");
				auto num_bytes = ((first_byte >> 3) & 0x07) + 1;
				if (num_bytes <= input_buffer_count) { 
				//TODO: this is guaranteed since for short repeat encoded bytes are small enough
					curr_64 = 0;
					while (num_bytes-- > 0) {
						curr_64 |= (input_buffer[input_buffer_head] << (num_bytes * 8));
						proceed(1);
					}
					auto cnt = (first_byte & 0x07) + MINIMUM_REPEAT;
					// printf("write %ld %u times\n", curr_64, cnt);
					while (cnt-- > 0) {
						write_varint(curr_64);
					}
				}
				read_first = false;
			}	break;
			case HEADER_DIRECT: {
				// printf("case direct\n");
        		uint8_t curr_fbw = get_decoded_bit_width((first_byte >> 1) & 0x1f);
				if (!read_second) {
					read_second = true;
					curr_len = ((static_cast<uint16_t>(first_byte & 0x01) << 8) | input_buffer[input_buffer_head]) + 1;
					proceed(1);
					bits_left = 0;
					bits_left_over = 0;
					curr_64 = 0;
				}
				read_longs(curr_fbw);
				if (curr_len == 0) {
					read_first = false; read_second = false;
				}
			}	break;
			case HEADER_DELTA: {
        		uint8_t curr_fbw = get_decoded_bit_width((first_byte >> 1) & 0x1f);
				if (!read_second) {
					read_second = true;
					curr_len = ((static_cast<uint16_t>(first_byte & 0x01) << 8) | input_buffer[input_buffer_head]) + 1;
					proceed(1);
					bits_left = 0;
					bits_left_over = 0;
					curr_64 = 0;
					dal_read_base = false;
				}

				// TODO: double check whether curr_fbw represent base length or delta length 
				if (!dal_read_base && input_buffer_count >= 2 * curr_fbw / 8) {//TODO: should be min(fbw, 8)
					base_val = read_uvarint();
					base_delta = read_svarint();
					
					write_varint(base_val);
					base_val += base_delta;
					write_varint(base_val);

					base_out = out_8B;

					curr_len -= 2;
					dal_read_base = true;
				}

				if (dal_read_base) {
					if (curr_fbw != 0) {
						// var delta
						read_longs(curr_fbw);
						
					} else {
						// fixed delta encoding
						while (curr_len-- > 0) {
							base_val += base_delta;
							write_varint(base_val);
						}
					}

					if (curr_len == 0) {
						if (base_delta > 0) {
							while (base_out < out_8B) {
								base_val = *base_out += base_val;
								base_out += WRITE_OFFSET;
							}
						} else {
							while (base_out < out_8B) {
                    			base_val = *base_out = base_val - *base_out;
								base_out += WRITE_OFFSET;
							}
						}
						read_first = false; read_second = false;
					}
				}
			}	break;
			case HEADER_PACTED_BASE: {
				//TODO Try to guarantee there are at least 4 btyes to read (all headers)
				uint8_t curr_fbw = get_decoded_bit_width((first_byte >> 1) & 0x1f);
				if (!read_second) {
					if (input_buffer_count < 3) break;
				// Here for patched base, read_second includes third and forth header byte
					curr_len = ((static_cast<uint16_t>(first_byte & 0x01) << 8) | input_buffer[input_buffer_head]) + 1;
					proceed(1);
					auto third = input_buffer[input_buffer_head];
					proceed(1);
        			auto forth = input_buffer[input_buffer_head];
					proceed(1);

					bw = ((third >> 5) & 0x07) + 1;
					pw = get_decoded_bit_width(third & 0x1f);
					pgw = ((forth >> 5) & 0x07) + 1;
					pll = forth & 0x1f;

					bits_left = 0;
					bits_left_over = 0;
					curr_64 = 0;
					patch_gap = 0;
					dal_read_base = false;

					read_second = true;
				}

				// TODO: double check whether curr_fbw represent base length or delta length 
				if (!dal_read_base && input_buffer_count >= bw) {//TODO: should be min(fbw, 8)
					base_val = read_long(bw);
					base_out = out_8B;
					dal_read_base = true;
				}
				if (curr_len > 0) {
					read_longs(curr_fbw);
				} else {
					// process patched list
					auto cfb = get_closest_bit(pw + pgw);
					auto patch_mask = (static_cast<uint16_t>(1) << pw) - 1;
					while (pll > 0 && (input_buffer_count || bits_left)) {
						auto bits_left_to_read = cfb;
						while (input_buffer_count && bits_left_to_read > bits_left) {
							curr_64 <<= bits_left;
							curr_64 |= bits_left_over & ((1 << bits_left) - 1);
							bits_left_to_read -= bits_left;
							bits_left_over = input_buffer[input_buffer_head];
							proceed(1);
							bits_left = 8;
						}

						if (bits_left_to_read <= bits_left) {
						// if not curr_64 is imcomplete
							if (bits_left_to_read > 0) {
								curr_64 <<= bits_left_to_read;
								bits_left -= bits_left_to_read;
								curr_64 |= (bits_left_over >> bits_left) & ((1 << bits_left_to_read) - 1);
							}

							// write_varint(curr_64);
							patch_gap += curr_64 >> pw;
							base_out[patch_gap * WRITE_OFFSET] |= static_cast<int64_t>(curr_64 & patch_mask) << curr_fbw;


							pll --;
							curr_64 = 0;
						}
					}
					if (pll == 0) {
						while (base_out < out_8B) {
							*base_out += base_val;
							base_out += WRITE_OFFSET;
						}
						read_first = false; read_second = false;
					}
				}
			}	break;
			}
        }
#undef proceed
#undef write_varint

    }
}

#endif