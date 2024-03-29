
#include <algorithm>
#include <cassert>
#include <common.h>
#include <cub/cub.cuh>
#include <fstream>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <simt/atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

constexpr uint16_t THRDS_SM_() { return (2048); }
constexpr uint16_t BLK_SIZE_() { return (32); }
constexpr uint16_t BLKS_SM_() { return (THRDS_SM_() / BLK_SIZE_()); }
constexpr uint64_t GRID_SIZE_() { return (1024); }
constexpr uint64_t NUM_CHUNKS_() { return (GRID_SIZE_() * BLK_SIZE_()); }
constexpr uint64_t CHUNK_SIZE_() { return (8 * 1024); }
constexpr uint64_t INPUT_BUFFER_SIZE() { return (8); }
constexpr uint64_t CHUNK_SIZE_4() { return (128); }

constexpr uint64_t HEADER_SIZE_() { return (1); }
constexpr uint32_t OVERHEAD_PER_CHUNK_(uint32_t d) {
  return (ceil<uint32_t>(d, (HEADER_SIZE_() * 8)) + 1);
}
constexpr uint32_t HIST_SIZE_() { return 2048; }
constexpr uint32_t LOOKAHEAD_SIZE_() { return 512; }
constexpr uint32_t REF_SIZE_() { return 16; }
constexpr uint32_t REF_SIZE_BYTES_() { return REF_SIZE_() / 8; }
constexpr uint32_t OFFSET_SIZE_() {
  return (bitsNeeded((uint32_t)HIST_SIZE_()));
}
constexpr uint32_t LENGTH_SIZE_() { return (REF_SIZE_() - OFFSET_SIZE_()); }
constexpr uint32_t LENGTH_MASK_(uint32_t d) {
  return ((d > 0) ? 1 | (LENGTH_MASK_(d - 1)) << 1 : 0);
}
constexpr uint32_t MIN_MATCH_LENGTH_() {
  return (ceil<uint32_t>((OFFSET_SIZE_() + LENGTH_SIZE_()), 8) + 1);
}
constexpr uint32_t MAX_MATCH_LENGTH_() {
  return (pow<uint32_t, uint32_t>(2, LENGTH_SIZE_()) + MIN_MATCH_LENGTH_() - 1);
}
constexpr uint8_t DEFAULT_CHAR_() { return ' '; }
constexpr uint32_t HEAD_INTS_() { return 7; }
constexpr uint32_t READ_UNITS_() { return 4; }
constexpr uint32_t LOOKAHEAD_UNITS_() {
  return LOOKAHEAD_SIZE_() / READ_UNITS_();
}
constexpr uint64_t WARP_ID_(uint64_t t) { return t / 32; }
constexpr uint32_t LOOKAHEAD_SIZE_4_BYTES_() {
  return LOOKAHEAD_SIZE_() / sizeof(uint32_t);
}
constexpr uint32_t HIST_SIZE_4_BYTES_() {
  return HIST_SIZE_() / sizeof(uint32_t);
}

constexpr uint32_t CHUNK_SIZE_4_BYTES_MASK_() {
  return LENGTH_MASK_(bitsNeeded(CHUNK_SIZE_4()));
}

#define BLKS_SM BLKS_SM_()
#define THRDS_SM THRDS_SM_()
#define BLK_SIZE BLK_SIZE_()
#define GRID_SIZE GRID_SIZE_()
#define NUM_CHUNKS NUM_CHUNKS_()
#define CHUNK_SIZE CHUNK_SIZE_()
#define HEADER_SIZE HEADER_SIZE_()
#define OVERHEAD_PER_CHUNK(d) OVERHEAD_PER_CHUNK_(d)
#define HIST_SIZE HIST_SIZE_()
#define LOOKAHEAD_SIZE LOOKAHEAD_SIZE_()
#define OFFSET_SIZE OFFSET_SIZE_()
#define LENGTH_SIZE LENGTH_SIZE_()
#define LENGTH_MASK(d) LENGTH_MASK_(d)
#define MIN_MATCH_LENGTH MIN_MATCH_LENGTH_()
#define MAX_MATCH_LENGTH MAX_MATCH_LENGTH_()
#define DEFAULT_CHAR DEFAULT_CHAR_()
#define HEAD_INTS HEAD_INTS_()
#define READ_UNITS READ_UNITS_()
#define LOOKAHEAD_UNITS LOOKAHEAD_UNITS_()
#define WARP_ID(t) WARP_ID_(t)

#define INPUT_BUFFER_SIZE INPUT_BUFFER_SIZE()

#define CHUNK_SIZE_4_BYTES_MASK CHUNK_SIZE_4_BYTES_MASK_()

#define char_len 4

#define READING_WARP_SIZE 32
#define WRITING_WARP_SIZE 16

#define DATA_BUFFER_SIZE 8

#define NUM_THREADS 32

#define DECOMP_READING_WARP_SIZE 16

namespace brle_trans {

//rlev1 reading

__device__ uint64_t roundUpTo(uint64_t input, uint64_t unit) {
  uint64_t val = ((input + unit - 1) / unit) * unit;
  return val;
}

__device__ void write_byte_op(int8_t *out_buffer, uint64_t *out_bytes_ptr,uint8_t write_byte, uint64_t *out_offset_ptr,uint64_t *col_len, uint8_t *col_counter_ptr, int COMP_WRITE_BYTES) {
  out_buffer[*out_offset_ptr] = write_byte;
  (*out_bytes_ptr) = (*out_bytes_ptr) + 1;

  // update the offset
  uint64_t out_offset = (*out_offset_ptr);

  if (( (*out_bytes_ptr) ) % (COMP_WRITE_BYTES) != 0) {
    *out_offset_ptr = out_offset + 1;
  }

  else {
    uint8_t col_counter = *col_counter_ptr;

    for(int i = 0; i < 32; i++){
      if(col_len[i] > 0 && i != threadIdx.x){
        out_offset += COMP_WRITE_BYTES;
        col_len[i] -= COMP_WRITE_BYTES;
      }
    }
    *out_offset_ptr = out_offset + 1;
    // while ((*out_bytes_ptr) >
    //            roundUpTo(col_len[ col_counter], COMP_WRITE_BYTES) &&
    //        col_counter > 0) {
    //   col_counter--;
    // }
    // out_offset += col_counter * COMP_WRITE_BYTES;

    // *col_counter_ptr = col_counter;
    // *out_offset_ptr = out_offset + 1;
  }
}

template <typename INPUT_T>
__device__ void write_varint_op(int8_t *out_buffer, uint64_t *out_bytes_ptr,
                                INPUT_T val, uint64_t *out_offset_ptr,
                                uint64_t *col_len, uint8_t *col_counter_ptr, int COMP_WRITE_BYTES) {
  INPUT_T write_val = val;
  int8_t write_byte = 0;
  do {
    write_byte = write_val & 0x7F;
    if ((write_val & (~0x7f)) != 0)
      write_byte = write_byte | 0x80;

    // write byte
    write_byte_op(out_buffer, out_bytes_ptr, write_byte, out_offset_ptr,
                  col_len, col_counter_ptr, COMP_WRITE_BYTES);
    write_val = write_val >> 7;
  } while (write_val != 0);
}


template <typename INPUT_T, typename READ_T>
__global__ void 
rlev1_compress_multi_reading_init(uint8_t *in, const uint64_t in_chunk_size, const uint64_t n_chunks, uint64_t *col_len, uint8_t *col_map, uint64_t *blk_offset, int COMP_WRITE_BYTES) {

  volatile __shared__ READ_T reading_queue [32][32];
  volatile __shared__ bool read_flag[32];
  volatile __shared__ uint8_t read_off[32];
  __shared__ unsigned long long int block_len;
    
  
  int tid = threadIdx.x;
  int chunk_idx = blockIdx.x;
  uint8_t which = threadIdx.y;
  uint64_t in_start_idx = in_chunk_size * blockIdx.x ;

  //initalize queue information
  if(which == 0){
    for(int i = 0; i < 32; i++){
      read_flag[tid] = false;
      read_off[tid] = 0;
    }
    if(tid == 0){
      block_len = 0;
      if(blockIdx.x == 0)
        blk_offset[0] = 0;
    }
  }
  __syncthreads();


  if(which == 0) {

    __shared__ uint8_t reading_index;
   
    uint32_t used_iterations = 0;
    uint32_t total_iterations = in_chunk_size / 32 / sizeof(READ_T);
    uint32_t row_bytes = 32 * sizeof(READ_T);
    int cur_idx = 0;

    while(used_iterations < total_iterations) {

      if(tid == 0){
        while(1){
          for(int i = 0; i < 32; i++){
            //read request
            if(read_flag[cur_idx]){
              reading_index = cur_idx;
              cur_idx = (cur_idx + 1) % 32;
              goto read_done;
            }
            cur_idx = (cur_idx + 1) % 32;
          }
        }
      }
      //label to break the while loop
      read_done:
       //sync threads to read the data
      __syncwarp();

      uint32_t in_read_off = read_off[reading_index] * 32 * (row_bytes);
      READ_T* inTyped = (READ_T*)(in + in_start_idx + in_read_off + reading_index * row_bytes + tid * sizeof(READ_T));
      __syncwarp();

      //global reading
      READ_T temp_store = *inTyped;
      reading_queue[reading_index][tid] = temp_store;

      __syncwarp();
      //set the flag and increment the offset for next read
      if(tid == 0) {
        read_flag[reading_index] = false;
        read_off[reading_index]++;
      }
      used_iterations++;
    }
 }

  else if(which == 1){

    uint8_t queue_head = 32;
    uint8_t word_head = 0;
    uint8_t num_words = sizeof(READ_T) / sizeof(INPUT_T);
    uint32_t read_bytes = sizeof(INPUT_T);
    uint64_t used_bytes = 0;
    uint64_t my_chunk_size = in_chunk_size / 32;

    INPUT_T data_buffer[2];
    uint8_t data_buffer_head = 0;
    uint8_t data_buffer_count = 0;

    //compression information
    INPUT_T delta_first_val = 0;
    uint16_t delta_count = 0;
    int8_t cur_delta = 0;
    bool delta_flag = false;
    uint16_t lit_count = 0;
    INPUT_T prev_val = 0;

    //output length
    uint64_t out_len = 0;

    //buffer for data tranportation from shared queue to thread reg
    const int buf_size = (sizeof(READ_T)/sizeof(INPUT_T)) * 32;

    while (used_bytes < my_chunk_size) {

      if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len: %llu\n", out_len);
      //if input_buffer is empty, then send a read request
      if(queue_head == 32){
        read_flag[tid] = true;
        while(read_flag[tid]) {
          __nanosleep(100);
        }
        //reset the head;
        queue_head = 0;
        word_head = 0;
      }
      
      //get data from reading queue
      INPUT_T read_data = ((INPUT_T*)&(reading_queue[tid][queue_head]))[word_head];
      word_head++;
      //adjust the cursor
      if(word_head == num_words){
        word_head = 0;
        queue_head++;
      }

      if(lit_count == 127) {
        lit_count = 0;
      }

      //first element
      if (used_bytes == 0) {
        delta_count = 1;
        prev_val = read_data;

        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count++;
        used_bytes += read_bytes;
        continue;
      }

      // second element to fill the buffer
      else if (used_bytes == read_bytes) {

        int64_t temp_diff = read_data - prev_val;

        if (temp_diff > 127 || temp_diff < -128) {
          delta_flag = false;
          out_len++;
          lit_count = 1;

          INPUT_T lit_val = data_buffer[0];
          uint64_t val_bytes = 1;
          lit_val = lit_val / 128;
          while(lit_val != 0){
            val_bytes++;
            lit_val = lit_val / 128;
          }
          out_len += val_bytes;
          data_buffer_count--;
        }

        else {
          delta_flag = true;
          cur_delta = (int8_t)temp_diff;
          delta_count++;
        }

        prev_val = read_data;
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count++;
        used_bytes += read_bytes;

        continue;
      }
      //we read the data before 2 if statements
      used_bytes += read_bytes;

      if (delta_count == 1) {
        int64_t temp_diff = read_data - prev_val;

        if (temp_diff > 127 || temp_diff < -128) {
          delta_flag = false;
          if (lit_count == 0) {
            out_len++;
            lit_count = 1;
          }
          if (lit_count == 0) 
            out_len++;
          
          int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
          INPUT_T lit_val = data_buffer[data_buffer_tail];
          uint64_t val_bytes = 1;
          lit_val = lit_val / 128;
          while(lit_val != 0){
            val_bytes++;
            lit_val = lit_val / 128;
          }
          out_len += val_bytes;
          lit_count++;
          data_buffer_count--;

        } else {
          delta_flag = true;
          cur_delta = (int8_t)temp_diff;
          delta_count++;
        }

        prev_val = read_data;
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count++;
        continue;
      }

     if (prev_val + cur_delta == read_data && delta_flag) {

        delta_count++;
        if (delta_count == 3) {
          delta_first_val = data_buffer[data_buffer_head];
          if (lit_count != 0) {
            lit_count = 0;
          }
        }
       else if(delta_count == 130){
          out_len += 2;
          INPUT_T lit_val = delta_first_val;
          uint64_t val_bytes = 1;
          lit_val = lit_val / 128;
          while(lit_val != 0){
            val_bytes++;
            lit_val = lit_val / 128;
          }
          out_len += val_bytes;
          delta_count = 1;
          data_buffer_count = 0;
          lit_count = 0;
        }

        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count = min(data_buffer_count + 1, 2);
        prev_val = read_data;
      }

      else {
        if (delta_count >= 3) {
          // write count, del, val
          out_len += 2;          
          INPUT_T lit_val = delta_first_val;
          uint64_t val_bytes = 1;
          lit_val = lit_val / 128;
          while(lit_val != 0){
            val_bytes++;
            lit_val = lit_val / 128;
          }

          out_len += val_bytes;
          delta_count = 1;
          data_buffer_count = 0;
          lit_count = 0;

          data_buffer[data_buffer_head] = read_data;
          data_buffer_head = (data_buffer_head + 1) % 2;
          data_buffer_count = min(data_buffer_count + 1, 2);
          prev_val = read_data;
        } 

        else {
          // first lit val
          if (lit_count == 0) {
            out_len++;
          }
          lit_count++;
          // write lit
          INPUT_T lit_val = data_buffer[data_buffer_head];
          //uint64_t val_bytes = (lit_val / 128) + 1;
          uint64_t val_bytes = 1;
          lit_val = lit_val / 128;
          while(lit_val != 0){
            val_bytes++;
            lit_val = lit_val / 128;
          }
          out_len += val_bytes;

          if(lit_count == 127)
            lit_count = 0;

          int64_t temp_diff = read_data - prev_val;

          if (temp_diff > 127 || temp_diff < -128) {
            if(lit_count == 0)
              out_len++;

            delta_flag = false;
            data_buffer_count = 0;

            int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
            INPUT_T lit_val = data_buffer[data_buffer_tail];
            uint64_t val_bytes = 1;
            lit_val = lit_val / 128;
            while(lit_val != 0){
              val_bytes++;
              lit_val = lit_val / 128;
            }
              out_len += val_bytes;
              lit_count++;
              delta_count = 1;
            }

          else {
            data_buffer_count = 1;
            delta_flag = true;
            cur_delta = (int8_t) temp_diff;
            delta_count = 2;
          }

          prev_val = read_data;
          data_buffer[data_buffer_head] = read_data;
          data_buffer_head = (data_buffer_head + 1) % 2;
          data_buffer_count = min(data_buffer_count + 1, 2);
        }
      }
    }
    //while loop ended

    // write remaining elements
    if (delta_count >= 3 && delta_flag) {
      out_len += 2;
      INPUT_T lit_val = delta_first_val;

      uint64_t num_out_bytes = 1;
      lit_val = lit_val / 128;
      while(lit_val != 0){
        num_out_bytes++;
        lit_val = lit_val / 128;
      }
      out_len += num_out_bytes;
            if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len1: %llu\n", out_len);

    }

    else {
      if(data_buffer_count == 1){
        if (lit_count == 127)
          lit_count = 0;

        if (lit_count == 0) 
          out_len++;

        int data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
        INPUT_T lit_val = data_buffer[data_buffer_tail];
        uint64_t num_out_bytes = 1;
        lit_val = lit_val / 128;
        while(lit_val != 0){
          num_out_bytes++;
          lit_val = lit_val / 128;
        }
        out_len += num_out_bytes;
        data_buffer_head = (data_buffer_head + 1) % 2;

        lit_count ++;
	                                    if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len4: %llu\n", out_len);

      }

      if (data_buffer_count == 2) {
         if (lit_count == 127)
          lit_count = 0;

        if (lit_count == 0) 
          out_len++;


        INPUT_T lit_val = data_buffer[data_buffer_head];
        uint64_t num_out_bytes = 1;
        lit_val = lit_val / 128;
        while(lit_val != 0){
          num_out_bytes++;
          lit_val = lit_val / 128;
        }
        lit_count++;
        if (lit_count == 127)
          lit_count = 0;

        if (lit_count == 0) 
          out_len++;        

        out_len += num_out_bytes;
        data_buffer_head = (data_buffer_head + 1) % 2;
        lit_val = data_buffer[data_buffer_head];
        
        num_out_bytes = 1;
        lit_val = lit_val / 128;
        while(lit_val != 0){
          num_out_bytes++;
          lit_val = lit_val / 128;
        }
        out_len += num_out_bytes;
	                                    if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len5: %llu\n", out_len);

      }
              if(threadIdx.x == 24 && blockIdx.x == 93064 ) printf("out len2: %llu\n", out_len);

    }


 

    col_len[BLK_SIZE * chunk_idx + tid] = out_len;
   if(blockIdx.x == 93064)
	      printf("tid: %i col len: %llu\n", threadIdx.x, (unsigned long long) out_len);

    uint64_t out_len_round = roundUpTo(out_len, COMP_WRITE_BYTES);
    
       //if(blockIdx.x == 0)
	 //             printf("tid: %i rounded col len: %llu len: %llu\n", threadIdx.x, (unsigned long long) out_len, COMP_WRITE_BYTES);

    atomicAdd((unsigned long long int *)&block_len,
              (unsigned long long int)out_len_round);
  
    __syncthreads();
    
    if (threadIdx.x == 0) {
      // 128B alignment
      block_len = roundUpTo(block_len, 128);
      blk_offset[chunk_idx + 1] = (uint64_t)block_len;
    }
  } 
} 


template <typename INPUT_T, typename READ_T>
__global__ void 
rlev1_compress_multi_reading(uint8_t * in, int8_t *out_buffer, const uint64_t in_chunk_size, const uint64_t n_chunks, uint64_t *col_len, uint8_t *col_map, uint64_t *blk_offset, int COMP_WRITE_BYTES) {

 volatile  __shared__ READ_T reading_queue [32][32];
 volatile  __shared__ bool read_flag[32];
 volatile  __shared__ uint8_t read_off[32];
  
   uint64_t s_col_len[32];
   for(int i = 0; i < 32; i++){
      s_col_len[i] = ((col_len[blockIdx.x * BLK_SIZE + i] + COMP_WRITE_BYTES - 1) / COMP_WRITE_BYTES) * COMP_WRITE_BYTES ;
   }
   for(int i = 0; i < threadIdx.x; i++){
      s_col_len[i] -= COMP_WRITE_BYTES;
   }

  int tid = threadIdx.x;
  int chunk_idx = blockIdx.x;
  uint8_t which = threadIdx.y;
  uint64_t in_start_idx = in_chunk_size * blockIdx.x ;
  uint8_t col_idx;




  //initalize queue information
  if(which == 0){
    for(int i = 0; i < 32; i++){
      read_flag[tid] = false;
      read_off[tid] = 0;
    }
  //  s_col_len[tid] = col_len[blockIdx.x * BLK_SIZE + tid];
  }

  // for (int i = 0; i < 32; i++) {
  //   if (tid == col_map[BLK_SIZE * chunk_idx + i]) {
  //     col_idx = i;
  //   }
  // }
  col_idx = threadIdx.x;
  __syncthreads();

  if(which == 0) {
    __shared__ uint8_t reading_index;
   
    uint32_t used_iterations = 0;
    uint32_t total_iterations = in_chunk_size / 32 / sizeof(READ_T);
    uint32_t row_bytes = 32 * sizeof(READ_T);
    int cur_idx = 0;

    while(used_iterations < total_iterations) {

      if(tid == 0){
        while(1){
          for(int i = 0; i < 32; i++){
            //read request
            if(read_flag[cur_idx]){
              reading_index = cur_idx;
              cur_idx = (cur_idx + 1) % 32;
              goto read_done;
            }
            cur_idx = (cur_idx + 1) % 32;
          }
        }
      }
      //label to break the while loop
      read_done:
       //sync threads to read the data
      __syncwarp();

      uint32_t in_read_off = read_off[reading_index] * 32 * (row_bytes);
      READ_T* inTyped = (READ_T*)(in + in_start_idx + in_read_off + reading_index * row_bytes +  tid * sizeof(READ_T));
      __syncwarp();

      //global reading
      READ_T temp_store = *inTyped;
      reading_queue[reading_index][tid] = temp_store;

      __syncwarp();
      //set the flag and increment the offset for next read
      if(tid == 0) {
        read_flag[reading_index] = false;
        read_off[reading_index]++;
      }
      used_iterations++;
    }
  }

  else if(which == 1){

    uint8_t queue_head = 32;
    uint8_t word_head = 0;
    uint8_t num_words = sizeof(READ_T) / sizeof(INPUT_T);
    uint32_t read_bytes = sizeof(INPUT_T);

    uint64_t used_bytes = 0;
    uint64_t my_chunk_size = in_chunk_size / 32;

    INPUT_T data_buffer[2];
    uint8_t data_buffer_head = 0;
    uint8_t data_buffer_count = 0;

    //compression information
    INPUT_T delta_first_val = 0;
    uint16_t delta_count = 0;
    int8_t cur_delta = 0;
    bool delta_flag = false;
    uint64_t lit_idx = 0;
    uint16_t lit_count = 0;
    INPUT_T prev_val = 0;

    //information for output
    uint8_t col_counter = 31;
    uint64_t out_bytes = 0;
    uint64_t out_offset = col_idx * COMP_WRITE_BYTES;
    uint64_t out_start_idx = blk_offset[chunk_idx];
    int8_t *out_buffer_ptr = &(out_buffer[out_start_idx]);

    while (used_bytes < my_chunk_size) {
    //if input_buffer is empty, then send a read request
      if(queue_head == 32){
        read_flag[tid] = true;
        while(read_flag[tid]) {
          __nanosleep(100);
        }
        //reset the head;
        queue_head = 0;
        word_head = 0;
      }
      
      //get data from reading queue
      INPUT_T read_data = ((INPUT_T*)&(reading_queue[tid][queue_head]))[word_head];
// if(threadIdx.x == 9 && blockIdx.x == 6) printf("readdata: %x\n", read_data);
      word_head++;
      //adjust the cursor
      if(word_head == num_words){
        word_head = 0;
        queue_head++;
      }

      if(lit_count == 127) {
        out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
        lit_count = 0;
      }

      //first element
      if (used_bytes == 0) {
        delta_count = 1;
        prev_val = read_data;

        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count++;
        used_bytes += read_bytes;
        continue;
      }

      // second element to fill the buffer
      else if (used_bytes == read_bytes) {

        int64_t temp_diff = read_data - prev_val;

        if (temp_diff > 127 || temp_diff < -128) {
          delta_flag = false;
          lit_idx = out_offset;
          write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, 1, &out_offset,
                                   s_col_len, &col_counter, COMP_WRITE_BYTES);
          lit_count = 1;

          INPUT_T lit_val = data_buffer[0];
          write_varint_op<INPUT_T>(out_buffer_ptr,  &out_bytes, lit_val,
                                   &out_offset, s_col_len, &col_counter, COMP_WRITE_BYTES);

          data_buffer_count--;
        }

        else {
          delta_flag = true;
          cur_delta = (int8_t)temp_diff;
          delta_count++;
        }

        prev_val = read_data;
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count++;
        used_bytes += read_bytes;

        continue;
      }
      used_bytes += read_bytes;

      if (delta_count == 1) {
        int64_t temp_diff = read_data - prev_val;
        if (temp_diff > 127 || temp_diff < -128) {
          delta_flag = false;
          if (lit_count == 0) {
            lit_idx = out_offset;
            write_byte_op(out_buffer_ptr,  &out_bytes, (uint8_t)1, &out_offset,
                                     s_col_len, &col_counter, COMP_WRITE_BYTES);
          }

          int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
          INPUT_T lit_val = data_buffer[data_buffer_tail];
          write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, lit_val, 
                                   &out_offset, s_col_len, &col_counter, COMP_WRITE_BYTES);
          lit_count++;
          data_buffer_count--;
        } 
        else {
          delta_flag = true;
          cur_delta = (int8_t)temp_diff;
          delta_count++;
        }
        prev_val = read_data;
        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count++;
        continue;
      }

      if (prev_val + cur_delta == read_data && delta_flag) {
        delta_count++;
        if (delta_count == 3) {
          delta_first_val = data_buffer[data_buffer_head];
          if (lit_count != 0) {
            // update lit counter
            out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
            lit_count = 0;
          }
        }
        else if(delta_count == 130){
          int8_t write_byte = delta_count - 4;
          write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,
                        s_col_len, &col_counter,COMP_WRITE_BYTES);
          write_byte = (uint8_t)cur_delta;
          write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,
                        s_col_len, &col_counter,COMP_WRITE_BYTES);
          
          write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, delta_first_val, 
                                   &out_offset, s_col_len, &col_counter, COMP_WRITE_BYTES);
   
          delta_count = 1;
          data_buffer_count = 0;
          lit_count = 0;
        }

        data_buffer[data_buffer_head] = read_data;
        data_buffer_head = (data_buffer_head + 1) % 2;
        data_buffer_count = min(data_buffer_count + 1, 2);
        prev_val = read_data;
      }

      else {

         if (delta_count >= 3) {
          // write count, del, val
          int8_t write_byte = delta_count - 3;
          write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,
                        s_col_len, &col_counter,COMP_WRITE_BYTES);
          write_byte = (uint8_t)cur_delta;
          write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,
                        s_col_len, &col_counter,COMP_WRITE_BYTES);
          write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, delta_first_val, 
                                   &out_offset, s_col_len, &col_counter, COMP_WRITE_BYTES);
     if(threadIdx.x == 17 && blockIdx.x == 7 ) printf("write_byte: %x cur delta:%x  fv: %x\n", delta_count - 3, cur_delta, delta_first_val); 
      	  delta_count = 1;
          data_buffer_count = 0;
          lit_count = 0;
          data_buffer[data_buffer_head] = read_data;
          data_buffer_head = (data_buffer_head + 1) % 2;
          data_buffer_count = min(data_buffer_count + 1, 2);
          prev_val = read_data;
        }

        else {
          // first lit val
          if (lit_count == 0) {
            lit_idx = out_offset;
            write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3,  &out_offset,
                                     s_col_len, &col_counter, COMP_WRITE_BYTES);
          }
          lit_count++;
          // write lit
          INPUT_T lit_val = data_buffer[data_buffer_head];
          write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, lit_val,
                                   &out_offset, s_col_len, &col_counter, COMP_WRITE_BYTES);
          if(lit_count == 127) {
            out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
            lit_count = 0;
          }

          int64_t temp_diff = read_data - prev_val;
          if (temp_diff > 127 || temp_diff < -128) {

            if (lit_count == 0) {
              lit_idx = out_offset;
              write_byte_op(out_buffer_ptr, &out_bytes, (uint8_t) 3,  &out_offset,
                                       s_col_len, &col_counter, COMP_WRITE_BYTES);
            }

            delta_flag = false;
            data_buffer_count = 0;
            int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
            INPUT_T lit_val = data_buffer[data_buffer_tail];
            write_varint_op<INPUT_T>(out_buffer_ptr,  &out_bytes, lit_val,
                                     &out_offset, s_col_len, &col_counter, COMP_WRITE_BYTES);
            lit_count++;
            delta_count = 1;
          }

          else {
            data_buffer_count = 1;
            delta_flag = true;
            cur_delta = (int8_t) temp_diff;
            delta_count = 2;
          }

          prev_val = read_data;
          data_buffer[data_buffer_head] = read_data;
          data_buffer_head = (data_buffer_head + 1) % 2;
          data_buffer_count = min(data_buffer_count + 1, 2);
        }

      }

      //end of while loop
    }

    // write remaining elements
    if (delta_count >= 3 && delta_flag) {
      int8_t write_byte = delta_count - 3;
      write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset,  s_col_len,
                    &col_counter, COMP_WRITE_BYTES);
      write_byte = (uint8_t)cur_delta;
      write_byte_op(out_buffer_ptr, &out_bytes, write_byte, &out_offset, s_col_len,
                    &col_counter, COMP_WRITE_BYTES);
      write_varint_op<INPUT_T>(out_buffer_ptr,  &out_bytes, delta_first_val,
                               &out_offset, s_col_len, &col_counter, COMP_WRITE_BYTES);
    }

    else {
      // update lit count

      if(data_buffer_count == 1){

           if(lit_count == 127) {
            out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
            lit_count = 0;
          }
                if (lit_count == 0) {
        lit_idx = out_offset;
        write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, 1,  &out_offset,
                                 s_col_len, &col_counter, COMP_WRITE_BYTES);
      }


        int8_t data_buffer_tail = (data_buffer_head == 0) ? 1 : 0;
            INPUT_T lit_val = data_buffer[data_buffer_tail];
          write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, lit_val, &out_offset,
                                 s_col_len, &col_counter, COMP_WRITE_BYTES);
          lit_count++;
      }
      if (data_buffer_count == 2) {

           if(lit_count == 127) {
            out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
            lit_count = 0;
          }
                if (lit_count == 0) {
        lit_idx = out_offset;
        write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, 1,  &out_offset,
                                 s_col_len, &col_counter, COMP_WRITE_BYTES);
      }


        INPUT_T lit_val = data_buffer[data_buffer_head];
        write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, lit_val, &out_offset,
                               s_col_len, &col_counter, COMP_WRITE_BYTES);
        lit_count++;

   if(lit_count == 127) {
            out_buffer_ptr[lit_idx] = static_cast<int8_t>(-lit_count);
            lit_count = 0;
          }
                if (lit_count == 0) {
        lit_idx = out_offset;
        write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, 1,  &out_offset,
                                 s_col_len, &col_counter, COMP_WRITE_BYTES);
      }

        data_buffer_head = (data_buffer_head + 1) % 2;
        lit_val = data_buffer[data_buffer_head];
        write_varint_op<INPUT_T>(out_buffer_ptr, &out_bytes, lit_val, &out_offset,
                                 s_col_len, &col_counter, COMP_WRITE_BYTES);
        lit_count++;
      }
      out_buffer_ptr[lit_idx] = (-lit_count);
    }
    //End of Computation Warp
  }

}

template <typename INPUT_T, typename READ_T>
__global__ void
rlev1_decompress_multi_reading(const int8_t *const in, READ_T* out, const uint64_t in_chunk_size, uint64_t* col_len, uint8_t *col_map, uint64_t *blk_offset, int DECOMP_WRITE_BYTES) {


  __shared__ simt::atomic<uint8_t,  simt::thread_scope_block> in_head[32];
  __shared__ simt::atomic<uint8_t,  simt::thread_scope_block> in_tail[32];
  __shared__ READ_T input_buffer[READING_WARP_SIZE][32];

  __shared__ uint64_t in_start_idx;


  int tid = threadIdx.x;
  int chunk_idx = blockIdx.x;
  uint8_t which = threadIdx.y;
  uint64_t read_chunk_size = 0;



  if(tid == 0 && which == 0){
    in_start_idx = blk_offset[chunk_idx];
  }



 //initalize queue information
  if(which == 0){
    in_head[tid] = 0;
    in_tail[tid] = 0;
    read_chunk_size = col_len[BLK_SIZE * chunk_idx + tid];
 
  }


  __syncthreads();

  //reading warp
  if(which == 0){

    uint64_t used_bytes = 0;
    uint32_t in_read_off = 0;
    uint64_t ele_size = sizeof(READ_T);
    uint64_t read_bytes = 0;

    uint8_t col_idx = col_map[BLK_SIZE * chunk_idx + tid];

    uint64_t counter = 0;
    while(true) {

      bool alive = (read_bytes < read_chunk_size);
      auto alivemask  = __ballot_sync(0xFFFFFFFF, alive);
      int res =  __popc(alivemask);
      read_bytes += ele_size;

      if(res == 0)
        break;

      if(alive){
        READ_T* inTyped = (READ_T*)(in + in_start_idx + in_read_off + tid * ele_size);
        READ_T temp_store = *inTyped;
        in_read_off += (res * ele_size);
   
   	counter++;
    	// for(int i  = 0; i < ele_size; i++){

        r_decomp_read:
            const auto cur_tail = in_tail[tid].load(simt::memory_order_relaxed);
            const auto next_tail = (cur_tail  + 1) % READING_WARP_SIZE;

            if (next_tail != in_head[tid].load(simt::memory_order_acquire)) {

              input_buffer[cur_tail][tid] = temp_store;

              in_tail[tid].store(next_tail, simt::memory_order_release);
              used_bytes += sizeof(READ_T);
            }
            else {
              goto r_decomp_read;
            }
        }

    }
  }

  else if (which == 1){
    uint16_t used_iterations = 0;

    uint8_t read_byte = sizeof(READ_T);
    uint8_t input_byte = sizeof(INPUT_T);

    bool header_read = true;
    bool read_val_flag = false;
    bool comp_flag = true;
    int8_t head_byte = 0;
    uint64_t remaining = 0;

    INPUT_T value = 0;
    uint64_t write_iterations = (CHUNK_SIZE / BLK_SIZE) / input_byte;

    int8_t delta = 0;
    uint64_t out_start_idx = chunk_idx * CHUNK_SIZE / input_byte;
    uint8_t col_idx = col_map[BLK_SIZE * chunk_idx + tid];


    READ_T temp_write_word = 0;
    uint8_t temp_word_count = 0;
    uint8_t read_input_count = sizeof(READ_T) / sizeof(INPUT_T);

    uint32_t out_start_offset = (in_chunk_size / sizeof(READ_T)) * chunk_idx;

    uint8_t line_count = 0;
    uint8_t byte_count = 0;

   unsigned long long write_vec_array[4]; 


    unsigned long long temp_vector = 0;
    int num_ele_in_vec = sizeof(unsigned long long) / sizeof(READ_T);
    uint8_t temp_vector_count = 0;
    uint8_t array_count = 0;

    uint64_t used_read_bytes = read_byte;


    READ_T read_data;
    uint8_t read_data_count = read_byte;


    uint64_t byte_int_factor = 1;

    while (used_iterations < write_iterations) {

      if(read_data_count == read_byte) {
        r_decomp_compute_read:
          const auto cur_head = in_head[tid].load(simt::memory_order_relaxed); 
          if (cur_head == in_tail[tid].load(simt::memory_order_acquire)) {
              goto r_decomp_compute_read;
          }

        read_data = input_buffer[cur_head][tid];
        const auto next_head = (cur_head + 1) % READING_WARP_SIZE;
        in_head[tid].store(next_head, simt::memory_order_release);
        used_read_bytes += read_byte;

        read_data_count = 0;
      }
    
      //need to read a header
      if(header_read){
          int8_t temp_byte = (read_data >> (8 * read_data_count)) & (0xff);
          read_data_count++;
          head_byte = temp_byte;
          header_read = false;


          //literals
          if(head_byte < 0){
            remaining = static_cast<uint64_t>(-head_byte);
            comp_flag = false;
          }
          //compresssed data
          else{
            remaining = static_cast<uint64_t>(head_byte);
            comp_flag = true;

          if(read_data_count == read_byte) {
             r_decomp_compute_read2:
                const auto cur_head = in_head[tid].load(simt::memory_order_relaxed); 
                if (cur_head == in_tail[tid].load(simt::memory_order_acquire)) {
                    goto r_decomp_compute_read2;
                }

              read_data = input_buffer[cur_head][tid];
              const auto next_head = (cur_head + 1) % READING_WARP_SIZE;
              in_head[tid].store(next_head, simt::memory_order_release);
              used_read_bytes += read_byte;
              read_data_count = 0;
            }
            temp_byte = (read_data >> (8 * read_data_count)) & (0xff);
            read_data_count++;

            delta = temp_byte;
            read_val_flag = true;
          
        }

      }

      if(read_val_flag){
        value = 0;
        int64_t offset = 0;
        while(read_val_flag){

        if(read_data_count == read_byte) {

             r_decomp_compute_read3:
                const auto cur_head = in_head[tid].load(simt::memory_order_relaxed); 
                if (cur_head == in_tail[tid].load(simt::memory_order_acquire)) {
                   // __nanosleep(100);
                    goto r_decomp_compute_read3;
                }

              read_data = input_buffer[cur_head][tid];
              const auto next_head = (cur_head + 1) % READING_WARP_SIZE;
              in_head[tid].store(next_head, simt::memory_order_release);
              used_read_bytes += read_byte;
              read_data_count = 0;
        }

           int8_t in_data = (read_data >> (8 * read_data_count)) & (0xff);
            read_data_count++;
 
      
          if(in_data >= 0){
            value |= (static_cast<INPUT_T>(in_data) << offset);
            read_val_flag = false;
          }
          else{
            value |= ((static_cast<INPUT_T>(in_data) & 0x7f) << offset); 
            offset += 7;
            }
          }

        }

      if(comp_flag){
        for(uint64_t i = 0; i < remaining + 3; ++i){
          int64_t out_ele = value + static_cast<int64_t>(i) * delta;

         // ((INPUT_T*)&temp_write_word)[temp_word_count] = static_cast<INPUT_T>(out_ele);
          temp_write_word = temp_write_word | (static_cast<READ_T>(out_ele) << (temp_word_count * sizeof(INPUT_T) * 8));

          //temp_write_word = temp_write_word | (out_ele )
          temp_word_count++;

           if(temp_word_count == read_input_count){

                temp_vector = temp_vector | (static_cast<unsigned long long> (temp_write_word) << (temp_vector_count * sizeof(READ_T) *8));
                temp_vector_count++;

                if(temp_vector_count == num_ele_in_vec) {

                write_vec_array[array_count] = temp_vector;

                array_count ++;

                if(array_count == 4){
                  
                  ulonglong4 write_vector = make_ulonglong4(write_vec_array[0], write_vec_array[1], write_vec_array[2], write_vec_array[3]);

                  ((ulonglong4*)(out + out_start_offset + line_count * 32 * 32 + col_idx * 32 + byte_count))[0] = write_vector;

                  byte_count += (32 / byte_int_factor);
                   //byte_count++;
                   if(byte_count == 32){
                      byte_count = 0;
                      line_count ++;
                   }
                  array_count = 0;
                }
                  temp_vector = 0;
                  temp_vector_count = 0;
                }

              temp_word_count = 0;
              temp_write_word = 0;
            }

          used_iterations += 1;
        }
          header_read = true;
      }

      else{
        for(uint64_t i = 0; i < remaining; ++i){
          value = 0;
          int64_t offset = 0;
          read_val_flag = true;

          while(read_val_flag){


           if(read_data_count == read_byte) {

               r_decomp_compute_read4:
                const auto cur_head = in_head[tid].load(simt::memory_order_relaxed); 
                if (cur_head == in_tail[tid].load(simt::memory_order_acquire)) {
                   // __nanosleep(100);
                    goto r_decomp_compute_read4;
                }

              read_data = input_buffer[cur_head][tid];
              const auto next_head = (cur_head + 1) % READING_WARP_SIZE;
              in_head[tid].store(next_head, simt::memory_order_release);
              used_read_bytes += read_byte;
              read_data_count = 0;

            }

            int8_t in_data = (read_data >> (8 * read_data_count)) & (0xff);
            read_data_count++;

            if(in_data >= 0){
              value |= (static_cast<INPUT_T>(in_data) << offset);
              read_val_flag = false;
            }
            else{
              value |= ((static_cast<INPUT_T>(in_data) & 0x7f) <<offset); 
              offset += 7;
            }
          }
          //end of real val

          //((INPUT_T*)&temp_write_word)[temp_word_count] = static_cast<INPUT_T>(value);
          temp_write_word = temp_write_word | (static_cast<READ_T>(value) << (temp_word_count * sizeof(INPUT_T) * 8));

          temp_word_count++;
          
          
           if(temp_word_count == read_input_count){

              temp_vector = temp_vector | (static_cast<unsigned long long> (temp_write_word) << (temp_vector_count * sizeof(READ_T) * 8 ));

      

                temp_vector_count++;

                if(temp_vector_count == num_ele_in_vec) {
                  //write_vector = write_vector | (temp_vector << (64 * array_count));

                
                  write_vec_array[array_count] = temp_vector;
                  array_count ++;

                  if(array_count == 4){
                  //  ulonglong4 write_vector = make_ulonglong4(write_vec1, write_vec2, write_vec3, write_vec4);
                   // ulonglong2 write_vector = make_ulonglong2(write_vec_array[0], write_vec_array[1]);
                     ulonglong4 write_vector = make_ulonglong4(write_vec_array[0], write_vec_array[1], write_vec_array[2], write_vec_array[3]);

                    ((ulonglong4*)(out + out_start_offset + line_count * 32 * 32 + col_idx * 32 + byte_count))[0] = write_vector;


                    byte_count += (32 / byte_int_factor);
                     if(byte_count == 32){
                        byte_count = 0;
                        line_count ++;
                     }
                    array_count = 0;
                  }
                  temp_vector = 0;
                  temp_vector_count = 0;
                }
              temp_word_count = 0;
              temp_write_word = 0;
            }

            used_iterations += 1;
        }

        header_read = true;
        read_val_flag = false;
      }

    //End of while loop
   }


   //end of computation warp
  }




}



__global__ void data_Read(uint8_t* in){

   //  uint32_t in_read_off = read_off[reading_index] * 32 * (row_bytes);
      uint64_t* inTyped = (uint64_t*)(in);
      uint64_t temp_store = *inTyped;
      printf("uint64_t data: %llx\n", temp_store);

      uint32_t* inTyped2 = (uint32_t*)(in);
      uint32_t temp_store2 = *inTyped;
      printf("uint32_t data: %x\n", temp_store2);


}


__global__ void ExampleKernel(uint64_t *col_len, uint8_t *col_map,
                              uint64_t *out) {
  // Specialize BlockRadixSort for a 1D block of 128 threads owning 4 integer
  // keys and values each
  typedef cub::BlockRadixSort<uint64_t, 32, 1, int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  int tid = threadIdx.x;
  int bid = blockIdx.x;

  uint64_t thread_keys[1];
  int thread_values[1];
  thread_keys[0] = col_len[bid * BLK_SIZE + tid];
  thread_values[0] = tid;

  BlockRadixSort(temp_storage).Sort(thread_keys, thread_values);

  // col_len[bid*BLK_SIZE + tid] = thread_keys[0];
  col_map[bid * BLK_SIZE + BLK_SIZE - 1 - tid] = thread_values[0];
  out[bid * BLK_SIZE + BLK_SIZE - 1 - tid] = thread_keys[0];
}

// change it to actual parallel scan
__global__ void parallel_scan(uint64_t *blk_offset, uint64_t n) {
  for (int i = 1; i <= n; i++) {
    blk_offset[i] += blk_offset[i - 1];
    if(i-1 == 643)
	    printf("blk offset: %llu\n", blk_offset[i-1]);
  }
}

__device__ bool check_f(uint8_t *a, uint8_t *b) {

  for (int i = 0; i < char_len; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}


template <typename INPUT_T, typename READ_T>
__host__ void compress_gpu( uint8_t * in, uint8_t **out,const uint64_t in_n_bytes, uint64_t *out_n_bytes) {
  
  //int COMP_WRITE_BYTES = sizeof(READ_T);
  int COMP_WRITE_BYTES = 4;
  uint8_t *d_in;
  int8_t *d_out;


  uint64_t padded_in_n_bytes =
      in_n_bytes; // + (CHUNK_SIZE-(in_n_bytes % CHUNK_SIZE));
  uint32_t n_chunks = padded_in_n_bytes / CHUNK_SIZE;
  uint32_t chunk_size = padded_in_n_bytes / n_chunks;
  assert((chunk_size % READ_UNITS) == 0);
  uint64_t exp_out_chunk_size = (chunk_size + OVERHEAD_PER_CHUNK_(chunk_size));
  uint64_t exp_data_out_bytes = (n_chunks * exp_out_chunk_size);
 
  printf("in bytes: %llu\n", in_n_bytes);

  uint64_t num_chunk = in_n_bytes / CHUNK_SIZE;
   printf("num chunk: %llu\n", num_chunk);

  // cpu
  uint8_t *cpu_data_out = (uint8_t *)malloc(exp_data_out_bytes);
  uint64_t *col_len =
      (uint64_t *)malloc(sizeof(uint64_t) * BLK_SIZE * num_chunk);
  uint8_t *col_map = (uint8_t *)malloc(BLK_SIZE * num_chunk);
  uint64_t *blk_offset = (uint64_t *)malloc(8 * (num_chunk + 1));
  uint64_t *chunk_offset = (uint64_t *)malloc(8 * (num_chunk + 1));
  uint64_t *col_offset = (uint64_t *)malloc(8 * (BLK_SIZE * num_chunk + 1));

  uint64_t *d_blk_offset;
  uint64_t *d_col_len;
  uint8_t *d_col_map;
  uint64_t *d_col_len_sorted;

  cuda_err_chk(cudaMalloc(&d_in, padded_in_n_bytes));
  cuda_err_chk(cudaMalloc(&d_col_len, sizeof(uint64_t) * BLK_SIZE * num_chunk));
  cuda_err_chk(
      cudaMalloc(&d_col_len_sorted, sizeof(uint64_t) * BLK_SIZE * num_chunk));

  cuda_err_chk(cudaMalloc(&d_col_map, BLK_SIZE * num_chunk));
  cuda_err_chk(cudaMalloc(&d_blk_offset, 8 * (num_chunk + 1)));

  cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));





  rlev1_compress_multi_reading_init<INPUT_T, READ_T><<<n_chunks, dim3(BLK_SIZE,2,1)>>>(
      d_in, chunk_size, n_chunks, d_col_len, d_col_map, d_blk_offset, COMP_WRITE_BYTES);

  cuda_err_chk(cudaDeviceSynchronize());


  data_Read<<<1,1>>>(d_in);

  cuda_err_chk(cudaDeviceSynchronize());



  parallel_scan<<<1, 1>>>(d_blk_offset, n_chunks);
  cuda_err_chk(cudaDeviceSynchronize());

  d_col_len_sorted = d_col_len;

  // ExampleKernel<<<n_chunks, BLK_SIZE>>>(d_col_len, d_col_map, d_col_len_sorted);
  // cuda_err_chk(cudaDeviceSynchronize());

  cuda_err_chk(cudaMemcpy(col_len, d_col_len,
                          sizeof(uint64_t) * BLK_SIZE * num_chunk,
                          cudaMemcpyDeviceToHost));
  cuda_err_chk(cudaMemcpy(blk_offset, d_blk_offset, 8 * (num_chunk + 1),
                          cudaMemcpyDeviceToHost));
  cuda_err_chk(cudaMemcpy(col_map, d_col_map, BLK_SIZE * num_chunk,
                          cudaMemcpyDeviceToHost));

  uint64_t final_out_size = blk_offset[num_chunk];

  printf("final out size: %llu\n", final_out_size);

  *out = new uint8_t[final_out_size];
  
  
  cuda_err_chk(cudaMalloc(&d_out, final_out_size));  
  cuda_err_chk(cudaDeviceSynchronize());


  rlev1_compress_multi_reading <INPUT_T, READ_T> <<<n_chunks, dim3(BLK_SIZE,2,1)>>> (d_in, d_out, chunk_size, n_chunks, d_col_len_sorted, d_col_map, d_blk_offset, COMP_WRITE_BYTES);
  cuda_err_chk(cudaDeviceSynchronize());

   cuda_err_chk(
       cudaMemcpy((*out), d_out, final_out_size, cudaMemcpyDeviceToHost));

  std::ofstream col_len_file("./input_data/col_len.bin", std::ofstream::binary);
  col_len_file.write((const char *)(col_len), BLK_SIZE * num_chunk * 8);
  col_len_file.close();

  std::ofstream blk_off_file("./input_data/blk_offset.bin",
                             std::ofstream::binary);
  blk_off_file.write((const char *)(blk_offset), (num_chunk + 1) * 8);
  blk_off_file.close();

  std::ofstream col_map_file("./input_data/col_map.bin", std::ofstream::binary);
  col_map_file.write((const char *)(col_map), BLK_SIZE * num_chunk);
  col_map_file.close();


  *out_n_bytes = final_out_size;
  cuda_err_chk(cudaFree(d_out));
  cuda_err_chk(cudaFree(d_col_len));
  cuda_err_chk(cudaFree(d_col_map));
  //cuda_err_chk(cudaFree(d_col_len_sorted));
  cuda_err_chk(cudaFree(d_in));
  cuda_err_chk(cudaFree(d_blk_offset));
}

template <typename INPUT_T, typename READ_T>
__host__ void decompress_gpu(const uint8_t *const in, uint8_t **out,
                             const uint64_t in_n_bytes, uint64_t *out_n_bytes) {

  cudaSetDevice(1);
  int DECOMP_WRITE_BYTES = sizeof(READ_T) / sizeof(INPUT_T);

  std::string file_col_len = "./input_data/col_len.bin";
  std::string file_col_map = "./input_data/col_map.bin";
  std::string file_blk_off = "./input_data/blk_offset.bin";

  const char *filename_col_len = file_col_len.c_str();
  const char *filename_col_map = file_col_map.c_str();
  const char *filename_blk_off = file_blk_off.c_str();

  int fd_col_len;
  int fd_col_map;
  int fd_blk_off;

  struct stat sbcol_len;
  struct stat sbcol_map;
  struct stat sbblk_off;

  if ((fd_col_len = open(filename_col_len, O_RDONLY)) == -1) {
    printf("Fatal Error: Col Len read error\n");
    return;
  }

  if ((fd_col_map = open(filename_col_map, O_RDONLY)) == -1) {
    printf("Fatal Error: Col map read error\n");
    return;
  }

  if ((fd_blk_off = open(filename_blk_off, O_RDONLY)) == -1) {
    printf("Fatal Error: Block off read error\n");
    return;
  }

  fstat(fd_col_len, &sbcol_len);
  fstat(fd_col_map, &sbcol_map);
  fstat(fd_blk_off, &sbblk_off);

  void *map_base_col_len;
  void *map_base_col_map;
  void *map_base_blk_off;

  map_base_col_len =
      mmap(NULL, sbcol_len.st_size, PROT_READ, MAP_SHARED, fd_col_len, 0);
  map_base_col_map =
      mmap(NULL, sbcol_map.st_size, PROT_READ, MAP_SHARED, fd_col_map, 0);
  map_base_blk_off =
      mmap(NULL, sbblk_off.st_size, PROT_READ, MAP_SHARED, fd_blk_off, 0);

  uint64_t num_blk = ((uint64_t)sbblk_off.st_size / sizeof(uint64_t)) - 1;
  // uint64_t blk_size = ((uint8_t) sbcol_map.st_size / num_blk);
  uint64_t blk_size = BLK_SIZE;

  // start
  std::chrono::high_resolution_clock::time_point kernel_start =
      std::chrono::high_resolution_clock::now();

  int8_t *d_in;
  uint8_t *d_out;

  uint64_t *d_col_len;
  uint64_t *d_blk_offset;
  uint8_t *d_col_map;

  const uint8_t *const in_ = in;

  // change it later
  uint64_t in_bytes = ((uint64_t *)map_base_blk_off)[num_blk];
  uint64_t out_bytes = CHUNK_SIZE * num_blk;
  *out_n_bytes = out_bytes;

  printf("out_bytes: %llu\n", out_bytes);

  // cuda_err_chk(cudaMalloc(&d_in, in_bytes));
  // cuda_err_chk(cudaMalloc(&d_out, (*out_n_bytes)));

  cuda_err_chk(cudaMalloc(&d_in, in_bytes));
  cuda_err_chk(cudaMalloc(&d_out, (*out_n_bytes)));

  cuda_err_chk(cudaMalloc(&d_col_len, sbcol_len.st_size));
  cuda_err_chk(cudaMalloc(&d_col_map, sbcol_map.st_size));
  cuda_err_chk(cudaMalloc(&d_blk_offset, sbblk_off.st_size));

  cuda_err_chk(cudaMemcpy(d_in, in_, in_bytes, cudaMemcpyHostToDevice));

  cuda_err_chk(cudaMemcpy(d_col_len, map_base_col_len, sbcol_len.st_size,
                          cudaMemcpyHostToDevice));
  cuda_err_chk(cudaMemcpy(d_col_map, map_base_col_map, sbcol_map.st_size,
                          cudaMemcpyHostToDevice));
  cuda_err_chk(cudaMemcpy(d_blk_offset, map_base_blk_off, sbblk_off.st_size,
                          cudaMemcpyHostToDevice));

  printf("cuda malloc finished\n");
  printf("num_blk: %llu, blk_size: %llu\n", num_blk, blk_size);
  printf("chunk size: %llu\n", (uint64_t) CHUNK_SIZE);
 
  READ_T* d_out_Typed = (READ_T*)(d_out);

  rlev1_decompress_multi_reading <INPUT_T, READ_T><<<(num_blk ), dim3(blk_size, 2, 1)>>>(d_in, d_out_Typed, CHUNK_SIZE, d_col_len, d_col_map, d_blk_offset, DECOMP_WRITE_BYTES);




  cudaDeviceSynchronize();
  printf("decomp function done\n");

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  *out = new uint8_t[out_bytes];
  cuda_err_chk(cudaMemcpy((*out), d_out, out_bytes, cudaMemcpyDeviceToHost));

  std::chrono::high_resolution_clock::time_point kernel_end =
      std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> kt =
      std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end -
                                                                kernel_start);
  std::cout << "Decompression time: " << kt.count() << " secs\n";

  if (munmap(map_base_col_len, sbcol_len.st_size) == -1) {
    printf("Mem unmap error");
  }
  if (munmap(map_base_col_map, sbcol_map.st_size) == -1) {
    printf("Mem unmap error");
  }

  if (munmap(map_base_blk_off, sbblk_off.st_size) == -1) {
    printf("Mem unmap error");
  }

  close(fd_col_len);
  close(fd_blk_off);
  close(fd_col_map);

  cuda_err_chk(cudaFree(d_out));
  cuda_err_chk(cudaFree(d_in));
  cuda_err_chk(cudaFree(d_col_len));
  cuda_err_chk(cudaFree(d_col_map));
  cuda_err_chk(cudaFree(d_blk_offset));
}
} // namespace brle_trans
