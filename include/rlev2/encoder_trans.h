#ifndef _RLEV2_ENCODER_TRANPOSE_H_
#define _RLEV2_ENCODER_TRANPOSE_H_

#include "utils.h"
#include "encoder.h"

namespace rlev2 {
<<<<<<< HEAD
    template<bool should_write, int READ_UNIT>
    __global__ void block_encode(int64_t* in, const uint64_t in_n_bytes, 
            uint8_t* out, col_len_t* acc_col_len, blk_off_t* blk_off) {

=======
    template<bool should_write, int READ_UNIT, typename COMP_TYPE>
    __global__ void block_encode(INPUT_T* in2, const uint64_t in_n_bytes, 
            uint8_t* out, col_len_t* acc_col_len, blk_off_t* blk_off, uint64_t CHUNK_SIZE) {

	    COMP_TYPE* in = (COMP_TYPE*) in2;
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
__shared__ unsigned long long int blk_len;

        uint32_t tid = threadIdx.x;
        uint32_t cid = blockIdx.x;

<<<<<<< HEAD
=======

>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
if (!should_write) {
    if (tid == 0) {
        blk_len = 0;
        if (cid == 0) {
            blk_off[0] = 0;
        }
    }
    __syncthreads();
}
<<<<<<< HEAD
        int64_t in_start_limit = min((cid + 1) * CHUNK_SIZE, in_n_bytes) / sizeof(int64_t);
        int64_t in_start = cid * CHUNK_SIZE / sizeof(int64_t) + tid * READ_UNIT;
=======
        uint64_t in_start_limit = min((cid + 1) * CHUNK_SIZE, in_n_bytes) / sizeof(COMP_TYPE);
        uint64_t in_start = cid * CHUNK_SIZE / sizeof(COMP_TYPE) + tid * READ_UNIT;
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d

        //TODO: Make this more intelligent
        // uint8_t* out_4B = blk_off[cid] - blk_off[0] + WRITE_UNIT * tid;


        encode_info<> info;
<<<<<<< HEAD
        
if (!should_write) {
    info.output = out + 32 * tid;
} else {
    uint32_t write_off =  (cid + tid == 0) ? 0 : acc_col_len[cid * BLK_SIZE + tid - 1];
    info.output = out + write_off;
}
        

        int64_t prev_delta;

        auto& num_literals = info.num_literals;
        auto& fix_runlen = info.fix_runlen;
        auto& var_runlen = info.var_runlen;
        int64_t *literals = info.literals;
=======
        info.tid = tid; info.cid = cid;
        
// if (!should_write) {
//     info.output = out + 32 * tid;
// } else {
    uint32_t write_off =  (cid + tid == 0) ? 0 : acc_col_len[cid * BLK_SIZE + tid - 1];
    info.output = out + write_off;

// #ifdef DEBUG
// if (should_write  && cid == ERR_CHUNK && tid == ERR_THREAD) 
// printf("chunk %d thread %d without offset %u\n", cid, tid, write_off);
// #endif
// }
        

        INPUT_T prev_delta;

        uint32_t& num_literals = info.num_literals;
        uint32_t& fix_runlen = info.fix_runlen;
        uint32_t& var_runlen = info.var_runlen;
        INPUT_T *literals = info.literals;
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d

        int curr_read_offset = 0;
        // printf("thread %d with chunksize %ld\n", tid, mychunk_size);
        while (true) {
            if (in_start + curr_read_offset >= in_start_limit) break;
<<<<<<< HEAD
            auto val = in[in_start + curr_read_offset]; curr_read_offset ++;
            if (curr_read_offset == READ_UNIT) {
=======
            uint64_t val = in[in_start + curr_read_offset]; curr_read_offset ++;
#ifdef DEBUG
if (should_write  && cid == ERR_CHUNK && tid == ERR_THREAD) printf("thread %u read %u at offset %lu\n", tid, val, in_start + curr_read_offset);
#endif

	    if (curr_read_offset == READ_UNIT) {
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
                in_start += BLK_SIZE * READ_UNIT;
                curr_read_offset = 0;
            }
            if (num_literals == 0) {
                literals[num_literals ++] = val;
                fix_runlen = 1;
                var_runlen = 1;
                continue;
            }

            if (num_literals == 1) {
                literals[num_literals ++] = val; 
                prev_delta = val - literals[0];

                if (val == literals[0]) {
                    fix_runlen = 2; var_runlen = 0;
                } else {
                    fix_runlen = 0; var_runlen = 2;
                }
                continue;
            }

<<<<<<< HEAD
            int64_t curr_delta = val - literals[num_literals - 1];
            if (prev_delta == 0 && curr_delta == 0) {
=======
            INPUT_T curr_delta = val - literals[num_literals - 1];
          //  if(threadIdx.x == 0 && blockIdx.x == 0) printf("cur delta: %llu past delta: %llu \n", (unsigned long long) curr_delta, (unsigned long long) info.deltas[0]);
	    if (prev_delta == 0 && curr_delta == 0) {
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
                // fixed run length
                literals[num_literals ++] = val;
                if (var_runlen > 0) {
                    // fix run len is at the end of literals
                    fix_runlen = 2;
                }
                fix_runlen ++;

                if (fix_runlen >= MINIMUM_REPEAT && var_runlen > 0) {
                    num_literals -= MINIMUM_REPEAT;
                    var_runlen -= (MINIMUM_REPEAT - 1);

                    determineEncoding(info);
<<<<<<< HEAD

                    for (uint32_t ii = 0; ii < MINIMUM_REPEAT; ++ii) {
                        literals[ii] = val;
                    }
                    num_literals = MINIMUM_REPEAT;
                }

=======
                    
                    prev_delta = 0; 
                    for (uint32_t ii = 0; ii < MINIMUM_REPEAT; ++ii) {
                        literals[ii] = val;
                        info.deltas[ii] = 0;
                    }

                    num_literals = MINIMUM_REPEAT;
                }

		else if(fix_runlen >= MINIMUM_REPEAT){
			
			info.deltas[0] = 0;
		}
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
                if (info.fix_runlen == MAX_LITERAL_SIZE) {
                    determineEncoding(info);
                }

                continue;
            }

            if (info.fix_runlen >= MINIMUM_REPEAT) {
                if (info.fix_runlen <= MAX_SHORT_REPEAT_LENGTH) {
<<<<<<< HEAD
                    writeShortRepeatValues(info);
                } else {
                    info.is_fixed_delta = true;
=======
            //          if(threadIdx.x == 0 && blockIdx.x == 0) printf("short val: %lx\n", val);
			writeShortRepeatValues(info);
                } else {
	//		if(threadIdx.x == 0 && blockIdx.x == 0){ printf("delta runlen: %lu val: %lx\n",(unsigned long) info.fix_runlen,  val);
	//			for(int ii = 0 ; ii < info.fix_runlen; ii++){
	//				printf("delta data: %llu\n", (unsigned long long) info.deltas[ii]);
	//			}	
	//		}
                    info.is_fixed_delta = true;
#ifdef DEBUG
if (cid == ERR_CHUNK && tid == ERR_THREAD) {
	printf("thread %u case write delta case 4\n", tid);
}
#endif
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
                    writeDeltaValues(info);
                }
            }

            if (info.fix_runlen > 0 && info.fix_runlen < MINIMUM_REPEAT && val != literals[num_literals - 1]) {
                info.var_runlen = info.fix_runlen;
                info.fix_runlen = 0;
            }

            if (num_literals == 0) {
                literals[num_literals ++] = val;
                fix_runlen = 1;
                var_runlen = 1;
            } else {
                prev_delta = val - literals[num_literals - 1];
                literals[num_literals++] = val;
                info.var_runlen++;

                if (info.var_runlen == MAX_LITERAL_SIZE) {
                    determineEncoding(info);
                }
            }


        }

        if (num_literals != 0) {
            if (info.var_runlen != 0) {
                determineEncoding(info);
            } else if (info.fix_runlen != 0) {
                if (info.fix_runlen < MINIMUM_REPEAT) {
                    info.var_runlen = info.fix_runlen;
                    info.fix_runlen = 0;
                    determineEncoding(info);
                } else if (info.fix_runlen >= MINIMUM_REPEAT
                        && info.fix_runlen <= MAX_SHORT_REPEAT_LENGTH) {
                    writeShortRepeatValues(info);
                } else {
<<<<<<< HEAD
=======
#ifdef DEBUG
if (cid == ERR_CHUNK && tid == ERR_THREAD) {
	printf("thread %u case write delta case 3\n", tid);
}
#endif
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
                    info.is_fixed_delta = true;
                    writeDeltaValues(info);
                }
            }
        }

if (!should_write) {
    acc_col_len[BLK_SIZE * cid + tid] = info.potision;
    auto col_len_4B = static_cast<unsigned long long int>((info.potision + 3) / 4 * 4);
    atomicAdd(&blk_len, col_len_4B);
    
    __syncthreads();
    if (tid == 0) {
        // Block alignment should be 4(decoder's READ_UNIT) * 32 
        blk_len = (blk_len + 127) / 128 * 128;
        blk_off[cid + 1] = blk_len;

    }
}

<<<<<<< HEAD
#ifdef DEBUG
if (should_write && tid == ERR_THREAD) {
    // for (int i=0; i<info.potision; i+=4) {
    //     printf("thread %d write: %u\n", tid, *(uint32_t*)(&info.output[i]));
    // }
    for (int i=0; i<info.potision; i+=4) {
        // printf("thread %d write byte %x%x%x%x\n", tid, info.output[i], info.output[i + 1], info.output[i + 2], info.output[i + 3]);
    }
}
=======
#ifdef DEBUG_MORE
// if (should_write && cid == ERR_CHUNK && tid == ERR_THREAD) {
// if (should_write && cid == ERR_CHUNK && tid == ERR_THREAD) {
//     for (int i=0; i<info.potision; i+=4) {
//         printf("thread %d write byte %x%x%x%x\n", tid, info.output[i], info.output[i + 1], info.output[i + 2], info.output[i + 3]);
//     }
// }
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
#endif

    }

    __global__
    void tranpose_col_len_single(uint8_t* in, col_len_t *acc_col_len, col_len_t *col_len, blk_off_t *blk_off, uint8_t* out) {
        uint32_t cid = blockIdx.x;

<<<<<<< HEAD
        // uint64_t in_idx = (cid + tid == 0) ? 0 : acc_col_len[cid * BLK_SIZE + tid - 1];
        // uint64_t out_idx = blk_off[cid] + tid * DECODE_UNIT;
        // int64_t out_bytes = acc_col_len[cid * BLK_SIZE + tid] - ((cid + tid == 0) ? 0 : acc_col_len[cid * BLK_SIZE + tid - 1]);

        // uint32_t* in_4B = (uint32_t *)(&(in[in_idx]));
        // uint32_t in_4B_idx = in_idx;
        // uint32_t* out_4B = (uint32_t *)(&(out[blk_off[cid]]));
        // uint32_t out_4B_idx = 0;
        // int iter = 0;
=======

// if (cid == ERR_CHUNK) {
// for (int i=498475; i<498475+16; i+=4) {
// printf("out4B pre: %x%x%x%x\n", in[i+0],  
// in[i+1],  in[i+2], in[i+3]);
// }
// }
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d

        col_len_t loc_col_len[32];
        memcpy(loc_col_len, col_len + cid * BLK_SIZE, 32 * sizeof(col_len_t));

        // More space should be saved. TODO: 
        uint64_t curr_iter_off = 0;
        uint64_t out_idx = blk_off[cid];
<<<<<<< HEAD
=======

>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
        while (true) {
            int res = 0;
            for (int tid=0; tid<32; ++tid) {
                if (loc_col_len[tid] > curr_iter_off) {
                    auto tidx = ((cid + tid == 0) ? 0 : acc_col_len[cid * BLK_SIZE + tid - 1]) + curr_iter_off;
                    
                    for (int i=0; i<DECODE_UNIT; ++i) {
                        out[out_idx + i] = in[tidx + i];
                    }
<<<<<<< HEAD
        // if (cid == 0 && tid == ERR_THREAD) 
        // printf("thread %d out4b at %lu: %x%x%x%x\n", tid, out_idx, out[out_idx], 
        // out[out_idx+1], 
        // out[out_idx+2], 
        // out[out_idx + 3]);
=======
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d

                    out_idx += DECODE_UNIT;
                    res ++;
                }
            }
            if (res == 0) break;
            curr_iter_off += DECODE_UNIT;
        }
<<<<<<< HEAD
    }


    __global__ 
    void post_data_tranpose(uint8_t*in, col_len_t *col_len, blk_off_t *blk_off, uint8_t* out) {
        auto cid = blockIdx.x;
        auto tid = threadIdx.x;
        uint64_t out_offset = blk_off[cid];
        col_len_t out_limit = col_len[cid * BLK_SIZE + tid], out_byte = 0;
        in += cid * OUTPUT_CHUNK_SIZE + tid * OUTPUT_CHUNK_SIZE / BLK_SIZE;

		const uint32_t t_write_mask = (0xffffffff << (32 - tid));
        while (true) {
            auto mask = __activemask();
            bool read = out_byte < out_limit;
            if (!read) break;

            auto read_sync = __ballot_sync(mask, read);

            auto active = __popc(read_sync);
            // printf("thread %d active mask at iter %d: %d\n", (int)tid, (int)(out_byte / 4), active);

            auto left_active = __popc(read_sync & t_write_mask);
            // printf("thread %d active read at iter %d: %d\n", (int)tid, (int)(out_byte / 4), left_active);

            for (int i=0; i<DECODE_UNIT; ++i) {
                out[out_offset + left_active * DECODE_UNIT + i] = in[out_byte ++];
            }

// #ifdef DEBUG
// if (cid == ERR_CHUNK && tid == ERR_THREAD) {
//     printf("chunk %d thread %d out4b at %ld with %x%x%x%x\n", cid, tid, 
//     out_offset + left_active * DECODE_UNIT,
//     out[out_offset + left_active * DECODE_UNIT ],
//     out[out_offset + left_active * DECODE_UNIT + 1],
//     out[out_offset + left_active * DECODE_UNIT + 2],
//     out[out_offset + left_active * DECODE_UNIT + 3]);
// }
// #endif
            
            out_offset += DECODE_UNIT * active;
            __syncwarp(mask);
        }
    }

    template <int READ_UNIT>
    __host__
    void compress_gpu_transpose(const int64_t* const in, const uint64_t in_n_bytes, uint8_t*& out, uint64_t& out_n_bytes,
                    uint64_t& out_n_chunks, blk_off_t *&blk_off, col_len_t *&col_len) {
        printf("Calling compress kernel.\n");
        uint32_t n_chunks = (in_n_bytes - 1) / CHUNK_SIZE + 1;
        out_n_chunks = n_chunks;
        
        int64_t *d_in;
        uint8_t *d_out, *d_out_transpose;
        col_len_t *d_col_len, *d_acc_col_len; //accumulated col len 
        blk_off_t *d_blk_off;
        
        // printf("input chunk: %lu\n", CHUNK_SIZE);
        // printf("output chunk: %lu\n", n_chunks * OUTPUT_CHUNK_SIZE);

        // printf("in_n_bytes: %lu\n", in_n_bytes);
        // printf("n_chunks: %u\n", n_chunks);

	    cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
=======

// if (cid == ERR_CHUNK) {
// uint64_t of = blk_off[cid] + ERR_THREAD*4;
// for (int i=0; i<4; ++i) {
// printf("out4B pre: %x%x%x%x\n", out[of+0],  
// out[of+1],  out[of+2], out[of+3]);
// of += 128;
// }
// }
    }

    template <int READ_UNIT, typename COMP_TYPE>
    __host__
    void compress_gpu_transpose(const INPUT_T* const in, const uint64_t in_n_bytes, uint8_t*& out, uint64_t& out_n_bytes, uint64_t& meta_data_size,
                    uint64_t& out_n_chunks,blk_off_t *&blk_off, col_len_t *&col_len, uint64_t CHUNK_SIZE) {

	uint64_t OUTPUT_CHUNK_SIZE = 2* CHUNK_SIZE;
        const uint64_t padded_n_bytes = ((in_n_bytes - 1) / CHUNK_SIZE + 1) * CHUNK_SIZE;

        uint32_t n_chunks = (padded_n_bytes - 1) / CHUNK_SIZE + 1;
        out_n_chunks = n_chunks;
        
        INPUT_T *d_in;
        uint8_t *d_out, *d_out_transpose;
        col_len_t *d_col_len, *d_acc_col_len; //accumulated col len 
        blk_off_t *d_blk_off;
        meta_data_size =  sizeof(col_len_t) * n_chunks * BLK_SIZE +   sizeof(blk_off_t) * (n_chunks + 1);
	    cuda_err_chk(cudaMalloc(&d_in, padded_n_bytes));
	    cuda_err_chk(cudaMemset(d_in, 0, padded_n_bytes));
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
        cuda_err_chk(cudaMalloc(&d_out, n_chunks * OUTPUT_CHUNK_SIZE));
	    cuda_err_chk(cudaMalloc(&d_col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE));
	    cuda_err_chk(cudaMalloc(&d_acc_col_len, sizeof(col_len_t) * n_chunks * BLK_SIZE));
	    cuda_err_chk(cudaMalloc(&d_blk_off, sizeof(blk_off_t) * (n_chunks + 1)));

	    cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
        
        initialize_bit_maps();

<<<<<<< HEAD
        block_encode<false, READ_UNIT><<<n_chunks, BLK_SIZE>>>(d_in, in_n_bytes, 
                d_out, d_col_len,  d_blk_off);
=======
        block_encode<false, READ_UNIT, COMP_TYPE><<<n_chunks, BLK_SIZE>>>(d_in, padded_n_bytes, 
                d_out, d_col_len,  d_blk_off, CHUNK_SIZE);
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d

	    cuda_err_chk(cudaDeviceSynchronize()); 

        thrust::inclusive_scan(thrust::device, d_blk_off, d_blk_off + n_chunks + 1, d_blk_off);
	    cuda_err_chk(cudaDeviceSynchronize()); 
        
        thrust::inclusive_scan(thrust::device, d_col_len, d_col_len + n_chunks * BLK_SIZE, d_acc_col_len);
	    cuda_err_chk(cudaDeviceSynchronize()); 

<<<<<<< HEAD
        // block_encode_new_write<<<n_chunks, BLK_SIZE>>>(d_in, in_n_bytes, 
        //                     d_out, d_acc_col_len,  d_blk_off);
        block_encode<true, READ_UNIT><<<n_chunks, BLK_SIZE>>>(d_in, in_n_bytes, 
                            d_out, d_acc_col_len,  d_blk_off);
=======
        block_encode<true, READ_UNIT, COMP_TYPE><<<n_chunks, BLK_SIZE>>>(d_in, padded_n_bytes, 
                            d_out, d_acc_col_len,  d_blk_off, CHUNK_SIZE);
>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
	    cuda_err_chk(cudaDeviceSynchronize()); 

        
        col_len = new col_len_t[n_chunks * BLK_SIZE];
	    cuda_err_chk(cudaMemcpy(col_len, d_col_len, sizeof(col_len_t) * BLK_SIZE * n_chunks, cudaMemcpyDeviceToHost));

        blk_off = new blk_off_t[n_chunks + 1];
	    cuda_err_chk(cudaMemcpy(blk_off, d_blk_off, sizeof(blk_off_t) * (n_chunks + 1), cudaMemcpyDeviceToHost));

        out_n_bytes = blk_off[n_chunks];
        out = new uint8_t[out_n_bytes];
        blk_off[n_chunks] = in_n_bytes; //use last index of blk_off to store file size.
        
<<<<<<< HEAD
=======
        // printf("out n bytes encoding: %lu\n", out_n_bytes);

>>>>>>> 2521a77ef53923581b24ccb056259d35bc9e5a2d
        cuda_err_chk(cudaMalloc(&d_out_transpose, out_n_bytes));
        tranpose_col_len_single<<<n_chunks, 1>>>(d_out, d_acc_col_len, d_col_len, d_blk_off, d_out_transpose);
        // post_data_tranpose<<<n_chunks, BLK_SIZE>>>(d_out, d_col_len, d_blk_off, d_out_transpose);
        cuda_err_chk(cudaDeviceSynchronize()); 


	    cuda_err_chk(cudaMemcpy(out, d_out_transpose, out_n_bytes, cudaMemcpyDeviceToHost));
        

        /*
        block_encode_transpose<ENCODE_UNIT><<<n_chunks, BLK_SIZE>>>(d_in, in_n_bytes, d_out, d_col_len,  d_blk_off);
        thrust::inclusive_scan(thrust::device, d_blk_off, d_blk_off + n_chunks + 1, d_blk_off);
        cuda_err_chk(cudaDeviceSynchronize()); 
        
        col_len = new col_len_t[n_chunks * BLK_SIZE];
	    cuda_err_chk(cudaMemcpy(col_len, d_col_len, sizeof(col_len_t) * BLK_SIZE * n_chunks, cudaMemcpyDeviceToHost));

        blk_off = new blk_off_t[n_chunks + 1];
	    cuda_err_chk(cudaMemcpy(blk_off, d_blk_off, sizeof(blk_off_t) * (n_chunks + 1), cudaMemcpyDeviceToHost));

        out_n_bytes = blk_off[n_chunks];
        out = new uint8_t[out_n_bytes];
        blk_off[n_chunks] = in_n_bytes; //use last index of blk_off to store file size.
        
        cuda_err_chk(cudaMalloc(&d_out_transpose, out_n_bytes));
        
        post_data_tranpose<<<n_chunks, BLK_SIZE>>>(d_out, d_col_len, d_blk_off, d_out_transpose);
	    cuda_err_chk(cudaDeviceSynchronize()); 

	    cuda_err_chk(cudaMemcpy(out, d_out_transpose, out_n_bytes, cudaMemcpyDeviceToHost));
        */

	    cuda_err_chk(cudaFree(d_in));
	    cuda_err_chk(cudaFree(d_out));
	    cuda_err_chk(cudaFree(d_out_transpose));
	    cuda_err_chk(cudaFree(d_col_len));
	    cuda_err_chk(cudaFree(d_acc_col_len));
	    cuda_err_chk(cudaFree(d_blk_off));
    }
}

#endif
