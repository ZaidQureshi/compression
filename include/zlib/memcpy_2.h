#include <common.h>
#include <fstream>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>
// #include <simt/atomic>
#include <iostream>
// #include <common_warp.h>

#ifndef MEMCPY_2
#define MEMCPY_2

#define WRITE_COL_LEN 0
// #define NMEMCPY 1
// #define NMEMCPYLOG2 0
// #define MEMCPYLARGEMASK 0xffffffff
// #define MEMCPYSMALLMASK 0x00000000
// #define USE_EIGHT 0

// #define NMEMCPY 2
// #define NMEMCPYLOG2 1
// #define MEMCPYLARGEMASK 0xfffffffe
// #define MEMCPYSMALLMASK 0x00000001
// #define FUNNEL_TYPE uint16_t
// #define USE_EIGHT 0

// #define NMEMCPY 4
// #define NMEMCPYLOG2 2
// #define MEMCPYLARGEMASK 0xfffffffc
// #define MEMCPYSMALLMASK 0x00000003
// #define FUNNEL_TYPE uint16_t
// #define USE_EIGHT 0

#define NMEMCPY 2
#define NMEMCPYLOG2 1
#define MEMCPYLARGEMASK 0xfffffffe
#define MEMCPYSMALLMASK 0x00000001
#define FUNNEL_TYPE uint16_t
#define USE_EIGHT 0


template<typename T>
__device__
void write_literal_2(T* out_ptr, uint32_t* counter, uint8_t idx, uint8_t b){
    if(threadIdx.x == idx){
        out_ptr[counter] = b;
        *counter++;
    }
}

template <uint32_t NUM_THREAD = 8, typename T>
// __forceinline__ 
__device__
void memcpy_onebyte_2(T* out_ptr, uint32_t* counter, uint8_t idx, uint16_t len, uint16_t offset, uint8_t div, uint32_t MASK) {
    int tid = threadIdx.x - div * NUM_THREAD;
    uint32_t orig_counter = __shfl_sync(MASK, *counter, idx);


    uint8_t num_writes = ((len - tid + NUM_THREAD - 1) / NUM_THREAD);
    uint32_t start_counter =  orig_counter - offset;


    uint32_t read_counter = start_counter + tid;
    uint32_t write_counter = orig_counter + tid;

    if(read_counter >= orig_counter){
        read_counter = (read_counter - orig_counter) % offset + start_counter;
    }

    uint8_t num_ph =  (len +  NUM_THREAD - 1) / NUM_THREAD;
    //#pragma unroll 
    for(int i = 0; i < num_ph; i++){
        if(i < num_writes){
            // printf("%d\n",out_ptr[read_counter + WRITE_COL_LEN * idx]);
            out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
            read_counter += NUM_THREAD;
            write_counter += NUM_THREAD;
        }
        __syncwarp(MASK);
    }

    //set the counter
    if(tid == idx % NUM_THREAD) {
        *counter += len;
    }
}

template <uint32_t NUM_THREAD = 8, typename T>
// __forceinline__ 
__device__
void memcpy_nbyte_prefix_body_suffix_2(T* out_ptr, uint32_t* counter, uint8_t idx, uint16_t len, uint16_t offset, uint8_t div, uint32_t MASK) {
    // if (len <= 32) {
    int tid;
    uint32_t orig_counter;
    uint8_t num_writes;
    uint32_t start_counter;
    uint32_t read_counter;
    uint32_t write_counter;
    uint8_t num_ph;

    /* Memcpy aligned on write boundaries */
    tid = threadIdx.x - div * NUM_THREAD;
    orig_counter = __shfl_sync(MASK, *counter, idx); // This is equal to the tid's counter value upon entering the function

    // prefix aligning bytes needed
    #if USE_EIGHT == 0
    uint8_t prefix_bytes = (uint8_t) (NMEMCPY - (orig_counter & MEMCPYSMALLMASK));
    if (prefix_bytes == NMEMCPY) prefix_bytes = 0;
    uint8_t suffix_bytes = (uint8_t) ((orig_counter + len) & MEMCPYSMALLMASK);

    #elif USE_EIGHT == 1
    uint8_t prefix_bytes = (uint8_t) (8 - (orig_counter & 0x00000007));
    if (prefix_bytes == 8) prefix_bytes = 0;
    uint8_t suffix_bytes = (uint8_t) ((orig_counter + len) & 0x00000007);

    #endif
    // prefix_bytes = 0;

    // suffix aligning bytes needed
    // prefix_bytes += suffix_bytes;
    // suffix_bytes = 0;

    // TODO: CHANGE
    start_counter =  orig_counter - offset; // The place to start writing. Notice we subtract offset, not add it.

    read_counter = start_counter + tid; // Start reading from the original counter minus the offset
    write_counter = orig_counter + tid; // Start writing from the original counter
    uint32_t element_idx = tid * NMEMCPY + prefix_bytes;
    uint32_t read_offset = offset - (element_idx) % offset;

    if(read_counter >= orig_counter){ // If we are reading from ahead of the place we are writing.
            read_counter = (read_counter - orig_counter) % offset + start_counter; // Don't really know what this line does.
    }

    // Write the prefix and the suffix by performing byte memcpy
    if (tid < prefix_bytes && tid < len) {
        out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
    } 
    // else if (tid < suffix_bytes + prefix_bytes && tid < len) {
    //     out_ptr[write_counter + WRITE_COL_LEN * idx + len - ((uint32_t) prefix_bytes) - ((uint32_t) suffix_bytes)] = out_ptr[read_counter + WRITE_COL_LEN * idx + len - ((uint32_t) prefix_bytes) - ((uint32_t) suffix_bytes)];
    // }

    read_counter = start_counter + tid * NMEMCPY + prefix_bytes; // Start reading from the original counter minus the offset
    write_counter = orig_counter + tid * NMEMCPY + prefix_bytes; // Start writing from the original counter

    if(read_counter >= orig_counter){ // If we are reading from ahead of the place we are writing.
            read_counter = (read_counter - orig_counter) % offset + start_counter; // Don't really know what this line does.
    }
    #if NMEMCPY == 2
    uchar2* out_ptr_temp  = reinterpret_cast<uchar2*>(out_ptr);

    #elif USE_EIGHT == 0
    uchar4* out_ptr_temp  = reinterpret_cast<uchar4*>(out_ptr);

    #elif USE_EIGHT == 1
    ushort4* out_ptr_temp  = reinterpret_cast<ushort4*>(out_ptr);

    #endif

    __syncwarp(MASK);

    num_writes = ((len - prefix_bytes - suffix_bytes - tid * NMEMCPY + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY));
    if (tid * NMEMCPY + prefix_bytes + suffix_bytes > len) num_writes = 0;

    num_ph =  (len - prefix_bytes - suffix_bytes + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY); // The largest amount of times a thread in the warp is writing to memory.
    if (prefix_bytes + suffix_bytes > len) num_ph = 0;

    //#pragma unroll 
    for(int i = 0; i < num_ph; i++){
        if(i < num_writes){ // If this thread should write. 4 bytes
            // TODO: CHANGE
            #if NMEMCPY == 1
            out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
            #elif NMEMCPY == 2
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_uchar2(out_ptr[orig_counter - read_offset], out_ptr[orig_counter - (offset - (element_idx + 1) % offset)]);
            #elif USE_EIGHT == 0
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_uchar4(out_ptr[read_counter + WRITE_COL_LEN * idx], out_ptr[orig_counter - (offset - (element_idx + 1) % offset)],
                                                                                             out_ptr[orig_counter - (offset - (element_idx + 2) % offset)], out_ptr[orig_counter - (offset - (element_idx + 3) % offset)]);
            #elif USE_EIGHT == 1
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_ushort4((((unsigned short)out_ptr[orig_counter - (offset - (element_idx + 1) % offset)] << 8) | out_ptr[read_counter + WRITE_COL_LEN * idx]),
                                                                                              (((unsigned short)out_ptr[orig_counter - (offset - (element_idx + 3) % offset)] << 8) | out_ptr[orig_counter - (offset - (element_idx + 2) % offset)]),
                                                                                              (((unsigned short)out_ptr[orig_counter - (offset - (element_idx + 5) % offset)] << 8) | out_ptr[orig_counter - (offset - (element_idx + 4) % offset)]),
                                                                                              (((unsigned short)out_ptr[orig_counter - (offset - (element_idx + 7) % offset)] << 8) | out_ptr[orig_counter - (offset - (element_idx + 6) % offset)]));

            #endif
            // 1 4 byte transaction | 4 1 byte transactions

            read_counter += NUM_THREAD * NMEMCPY; // Add the number of bytes that we wrote.
            write_counter += NUM_THREAD * NMEMCPY; // Add the number of bytes that we wrote.
            element_idx += NUM_THREAD * NMEMCPY;
            read_offset = offset - (element_idx) % offset;
        } 
        // #endif
        __syncwarp(MASK); // Synchronize.
    }

    /* This is a big source of slowdown. The suffix of the memory operation must be written last if it reads from a value that was written in this function. */
    // TODO: LOOK INTO MORE
    read_counter = start_counter + tid; // Start reading from the original counter minus the offset
    if(read_counter >= orig_counter){ // If we are reading from ahead of the place we are writing.
            read_counter = (read_counter - orig_counter) % offset + start_counter; // Don't really know what this line does.
    }
    if (tid < suffix_bytes && tid < len) {
        out_ptr[orig_counter + tid + WRITE_COL_LEN * idx + len - ((uint32_t) suffix_bytes)] = out_ptr[read_counter + WRITE_COL_LEN * idx + len - ((uint32_t) suffix_bytes)];
    }

    //set the counter
    if(tid == idx % NUM_THREAD)
        *counter += len;
}

template <uint32_t NUM_THREAD = 8, typename T>
// __forceinline__ 
__device__
void memcpy_nbyte_prefix_body_suffix_needs_writing_alignment_2(T* out_ptr, uint32_t* counter, uint8_t idx, uint16_t len, uint16_t offset, uint8_t div, uint32_t MASK) {
    // if (len <= 32) {
    int tid;
    uint32_t orig_counter;
    uint8_t num_writes;
    uint32_t start_counter;
    uint32_t read_counter;
    uint32_t write_counter;
    uint8_t num_ph;
    /* Memcpy aligned on write boundaries */
    tid = threadIdx.x - div * NUM_THREAD;
    orig_counter = __shfl_sync(MASK, *counter, idx); // This is equal to the tid's counter value upon entering the function

    // prefix aligning bytes needed
    uint8_t prefix_bytes;
    prefix_bytes = 0;

    // suffix aligning bytes needed
    uint8_t suffix_bytes;
    suffix_bytes = 0;

    read_counter = start_counter + tid * NMEMCPY + prefix_bytes; // Start reading from the original counter minus the offset
    write_counter = orig_counter + tid * NMEMCPY + prefix_bytes; // Start writing from the original counter
    uint32_t element_idx = tid * NMEMCPY + prefix_bytes;
    uint32_t read_offset = offset - (element_idx) % offset;

    if(read_counter >= orig_counter){ // If we are reading from ahead of the place we are writing.
            read_counter = (read_counter - orig_counter) % offset + start_counter; // Don't really know what this line does.
    }
    #if NMEMCPY == 2
    uchar2* out_ptr_temp  = reinterpret_cast<uchar2*>(out_ptr);

    #elif USE_EIGHT == 0
    uchar4* out_ptr_temp  = reinterpret_cast<uchar4*>(out_ptr);

    #elif USE_EIGHT == 1
    ushort4* out_ptr_temp  = reinterpret_cast<ushort4*>(out_ptr);

    #endif

    __syncwarp(MASK);

    num_writes = ((len - prefix_bytes - suffix_bytes - tid * NMEMCPY + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY));
    if (tid * NMEMCPY + prefix_bytes + suffix_bytes > len) num_writes = 0;

    num_ph =  (len - prefix_bytes - suffix_bytes + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY); // The largest amount of times a thread in the warp is writing to memory.
    if (prefix_bytes + suffix_bytes > len) num_ph = 0;

    //#pragma unroll 
    for(int i = 0; i < num_ph; i++){
        if(i < num_writes){ // If this thread should write. 4 bytes
            #if NMEMCPY == 1
            out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
            #elif NMEMCPY == 2
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_uchar2(out_ptr[orig_counter - read_offset], out_ptr[orig_counter - (offset - (element_idx + 1) % offset)]);
            #elif NMEMCPY == 4
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_uchar4(out_ptr[read_counter + WRITE_COL_LEN * idx], out_ptr[orig_counter - (offset - (element_idx + 1) % offset)],
                                                                                             out_ptr[orig_counter - (offset - (element_idx + 2) % offset)], out_ptr[orig_counter - (offset - (element_idx + 3) % offset)]);
            #elif NMEMCPY == 8
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_ushort4((((unsigned short)out_ptr[orig_counter - (offset - (element_idx + 1) % offset)] << 8) | out_ptr[read_counter + WRITE_COL_LEN * idx]),
                                                                                  (((unsigned short)out_ptr[orig_counter - (offset - (element_idx + 3) % offset)] << 8) | out_ptr[orig_counter - (offset - (element_idx + 2) % offset)]),
                                                                                  (((unsigned short)out_ptr[orig_counter - (offset - (element_idx + 5) % offset)] << 8) | out_ptr[orig_counter - (offset - (element_idx + 4) % offset)]),
                                                                                  (((unsigned short)out_ptr[orig_counter - (offset - (element_idx + 7) % offset)] << 8) | out_ptr[orig_counter - (offset - (element_idx + 6) % offset)]));

            #endif
            // 1 4 byte transaction | 4 1 byte transactions

            read_counter += NUM_THREAD * NMEMCPY; // Add the number of bytes that we wrote.
            write_counter += NUM_THREAD * NMEMCPY; // Add the number of bytes that we wrote.
            element_idx += NUM_THREAD * NMEMCPY;
            read_offset = offset - (element_idx) % offset;
        } 
        // #endif
        __syncwarp(MASK); // Synchronize.
    }

    //set the counter
    if(tid == idx % NUM_THREAD)
        *counter += len;

}

template <uint32_t NUM_THREAD = 8,
            typename T = uint8_t,
              typename TYPE_EXT = uchar4>
    __device__
    void memcpy_nbyte_prefix_body_suffix_funnelshift_shared_2(T* out_ptr, uint32_t* counter, TYPE_EXT* arr_shared, uint8_t idx, uint16_t len, uint16_t offset, uint8_t div, uint32_t MASK) {
// Asynchronous memcpy for load to shared memory.
    /*
    Read min(offset/4, len/4) N byte packs into shared memory first. Also read the N byte pack of memory at the start of the counter.
    'Prefix' can be written as a combination of the N byte pack of memory at the start and the first pack in shared memory.
    'Suffix' can be written as the last N byte pack with anything at the end, unless we are writting the very last N byte pack in the column.
    */
    // Share the counter from thread idx
    int tid = threadIdx.x - div * NUM_THREAD;
    uint32_t orig_counter = __shfl_sync(MASK, *counter, idx);
    uint32_t start_counter = orig_counter - offset;
    // uint8_t shared_offset = (threadIdx.y - 1) * (32 * NMEMCPY);
    uint8_t shared_offset = 0;

    // Calculate how many bytes we are misaligned in both writing and reading.
    uint8_t bits_written_in_first_pack = 8 - ((orig_counter + WRITE_COL_LEN * idx) & 0x00000007);
    // Recast the output array as an N byte array.
    TYPE_EXT* out_ptr_temp = reinterpret_cast<TYPE_EXT*>(out_ptr);

    // Write the prefix from global memory to global memory
    // REPLACE
    if (tid < bits_written_in_first_pack)
        out_ptr[orig_counter + WRITE_COL_LEN * idx + tid] = out_ptr[start_counter + WRITE_COL_LEN * idx + (tid % offset)];

    // Start reading in data. An additional thread is needed to load in the data at the start of our write section.
    uint32_t read_pack_addr = (start_counter + WRITE_COL_LEN * idx + tid * 4);
    uint8_t reading_starting_bit_number = read_pack_addr & 0x00000003;


    // Get the total amount of unique bits that we care about reading.
    uint8_t total_unique_bits = min(offset, len);

    // Calculate the read address for the threads. Load in N bytes based on either the size of offset or the size of len.
    uint8_t num_packs = (total_unique_bits + reading_starting_bit_number + 4 - 1) >> 2;

    // REPLACE
    if (tid < num_packs) {
        arr_shared[tid] = out_ptr_temp[tid + ((start_counter + WRITE_COL_LEN * idx) >> 2)];
    }
    if (tid + NUM_THREAD < num_packs) {
        arr_shared[tid + NUM_THREAD] = out_ptr_temp[tid + NUM_THREAD + ((start_counter + WRITE_COL_LEN * idx) >> 2)];
    }

    uint8_t shift;
    uint8_t idx2;
    TYPE_EXT pack;
    // Load the last 2 control words
    if (tid < 2) {
        idx2 = 0;
        // Starting control word shift is reading starting bit number
        // Ending control word shift is (reading starting bit number + total unique bits) & MASK
        shift = reading_starting_bit_number;
        if (tid == 0) {
            idx2 = num_packs - 1;
            shift += total_unique_bits;
            if (num_packs > 1) {
                idx2 -= 1;
            }
        }
        
        shift = shift & 0x00000003;

        if (tid == 0 && num_packs > 1 && shift == 0) {
            shift = 4;
        }
        
    }
    // WAIT
    __syncwarp(MASK);


    if (tid < 2) {
        pack = __funnelshift_rc(arr_shared[idx2], arr_shared[idx2+1], shift*8);
        arr_shared[num_packs + tid] = pack;
    }

    // Get the amount of bits in the last pack
    uint8_t bits_written_in_last_pack = (len - bits_written_in_first_pack) & 0x00000007;

    // Calculate how many writing operations to perform.
    int8_t num_writes = ((len - bits_written_in_first_pack - bits_written_in_last_pack - tid * NMEMCPY + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY));

    // Calculate the number of times that the thread with the most blocks has to write.
    uint8_t num_ph = (len - bits_written_in_first_pack + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY);

    // Calculate the write address
    uint32_t write_counter = (orig_counter + WRITE_COL_LEN * idx + tid * NMEMCPY + bits_written_in_first_pack);

    // cooperative_groups::wait(tile32);
    __syncwarp(MASK);
    // if (tid == 0) {
    //     printf("Length: %d\n", len);
    //     printf("Offset: %d\n", offset);

    //     printf("Writing: Location: %d\n", (orig_counter + WRITE_COL_LEN * idx));
    //     printf("Writing: Starting bits: %d\n", bits_written_in_first_pack);
    //     printf("Reading: Starting bits: %d\n", reading_starting_bit_number);
    //     printf("Num Packs: %d\n", num_packs);
    //     printf("Shared:\n");

    //     for (int temp = 0; temp < num_packs + 2; temp++) {
    //         printf("%08x\n", arr_shared[temp]);
    //     }
    //     printf("\n");
    // }

    // Finally begin the memcpy operation
    #pragma unroll 1
    for(int i = 0; i < num_ph; i++){
        if(i < num_writes){
            // We shouldn't have to worry about underflow as we manually set the indices later.
            int32_t total_bits_written = bits_written_in_first_pack + (tid) * NMEMCPY + (i * NUM_THREAD * NMEMCPY);
            if (total_bits_written < 0) total_bits_written = 0;

            // Transform this into the current bit that we start writing in this run.
            uint8_t current_bit_in_run = total_bits_written % total_unique_bits;
            uint8_t current_pack_in_run = (current_bit_in_run + reading_starting_bit_number) >> 2;

            // Each thread will load two packs. Get the indices.
            uint8_t first_pack_idx = current_pack_in_run;
            uint8_t second_pack_idx = first_pack_idx + 1;

            // Deal with the control words. First write reads from words 2 and 1. Any writes at the end read from 0 and 1.
            if(current_bit_in_run > total_unique_bits - 4) {
                first_pack_idx = num_packs;
                second_pack_idx = num_packs + 1;
            }

            // Load in the packs from shared memory.
            TYPE_EXT pack1 = arr_shared[first_pack_idx + shared_offset];
            TYPE_EXT pack2 = arr_shared[second_pack_idx + shared_offset];

            // Calculate the shift needed to combine the packs. Special cases above have modified shifts.
            uint8_t shift = (current_bit_in_run + reading_starting_bit_number) & 0x00000003;
            // out_ptr_temp[write_counter >> NMEMCPYLOG2] = shift;

            if (current_bit_in_run > total_unique_bits - 4) {
                shift = 4 + current_bit_in_run - total_unique_bits;
            }

            // Perform the funnel shift
            // pack1 = __funnelshift_rc(pack1, pack2, shift * 8);
            #if NMEMCPY == 2
            // pack = (arr_shared[idx2] >> (shift * 8)) | (arr_shared[idx2+1] << ((NMEMCPY - shift) * 8));
            uint32_t big_pack = pack2;
            big_pack = big_pack << 16;
            big_pack = big_pack | pack1;
            pack1 = big_pack >> (shift * 8);
            #elif USE_EIGHT == 0
            pack1 = __funnelshift_rc(pack1, pack2, shift*8);
            out_ptr_temp[write_counter >> NMEMCPYLOG2] = pack1;
            #elif USE_EIGHT == 1
            pack1 = __funnelshift_rc(pack1, pack2, shift*8);
            TYPE_EXT pack_temp = pack1;

            current_bit_in_run = (total_bits_written + 4) % total_unique_bits;
            current_pack_in_run = (current_bit_in_run + reading_starting_bit_number) >> 2;

            // Each thread will load two packs. Get the indices.
            first_pack_idx = current_pack_in_run;
            second_pack_idx = first_pack_idx + 1;

            // Deal with the control words. First write reads from words 2 and 1. Any writes at the end read from 0 and 1.
            if(current_bit_in_run > total_unique_bits - 4) {
                first_pack_idx = num_packs;
                second_pack_idx = num_packs + 1;
            }

            // Load in the packs from shared memory.
            pack1 = arr_shared[first_pack_idx + shared_offset];
            pack2 = arr_shared[second_pack_idx + shared_offset];

            // Calculate the shift needed to combine the packs. Special cases above have modified shifts.
            shift = (current_bit_in_run + reading_starting_bit_number) & 0x00000003;
            // out_ptr_temp[write_counter >> NMEMCPYLOG2] = shift;

            if (current_bit_in_run > total_unique_bits - 4) {
                shift = 4 + current_bit_in_run - total_unique_bits;
            }

            pack1 = __funnelshift_rc(pack1, pack2, shift*8);
            uint64_t* out_ptr_temp = reinterpret_cast<uint64_t*>(out_ptr);

            out_ptr_temp[write_counter >> 3] = ((uint64_t) pack1 << 32) | pack_temp;

            #endif
            write_counter += NUM_THREAD * NMEMCPY;
        } 

    }
    // Write the suffix.
    if (tid < bits_written_in_last_pack) {
        // We shouldn't have to worry about underflow as we manually set the indices later.
        uint32_t total_bits_written = len - bits_written_in_last_pack + tid;

        // Transform this into the current bit that we start writing in this run.
        uint8_t current_bit_in_run = total_bits_written % total_unique_bits;
        uint8_t current_pack_in_run = (current_bit_in_run + reading_starting_bit_number) >> 2;
        uint8_t shift = (current_bit_in_run + reading_starting_bit_number) & 0x00000003;

        out_ptr[orig_counter + idx * WRITE_COL_LEN + total_bits_written] = (uint8_t) ((arr_shared[current_pack_in_run] >> ((shift) * 8)) & 0x000000ff);
    }
    if(tid == idx % NUM_THREAD)
        *counter += len;
}


template <uint32_t NUM_THREAD = 8, typename T>
// __forceinline__ 
__device__
void memcpy_nbyte_prefix_body_suffix_shared_2(T* out_ptr, uint32_t* counter, T* arr_shared, uint8_t idx, uint16_t len, uint16_t offset, uint8_t div, uint32_t MASK) {
    // if (len <= 32) {
    int tid;
    uint32_t orig_counter;
    uint8_t num_writes;
    uint32_t start_counter;
    uint32_t read_counter;
    uint32_t write_counter;
    uint8_t num_ph;
    /* Memcpy aligned on write boundaries */
    tid = threadIdx.x - div * NUM_THREAD;
    orig_counter = __shfl_sync(MASK, *counter, idx); // This is equal to the tid's counter value upon entering the function

    // prefix aligning bytes needed
    #if USE_EIGHT == 0
    uint8_t prefix_bytes = (uint8_t) (NMEMCPY - (orig_counter & MEMCPYSMALLMASK));
    if (prefix_bytes == NMEMCPY) prefix_bytes = 0;
    uint8_t suffix_bytes = (uint8_t) ((orig_counter + len) & MEMCPYSMALLMASK);

    #elif USE_EIGHT == 1
    uint8_t prefix_bytes = (uint8_t) (8 - (orig_counter & 0x00000007));
    if (prefix_bytes == 8) prefix_bytes = 0;
    uint8_t suffix_bytes = (uint8_t) ((orig_counter + len) & 0x00000007);

    #endif

    // Write to the shared memory.
    if ((tid) < offset) {
        arr_shared[tid] = out_ptr[orig_counter - offset + tid];
    }
    if ((tid) + NUM_THREAD < offset) {
        arr_shared[(tid) + NUM_THREAD] = out_ptr[orig_counter - offset + (tid) + NUM_THREAD];
    }
    __syncwarp(MASK);


    // TODO: CHANGE
    start_counter =  orig_counter - offset; // The place to start writing. Notice we subtract offset, not add it.

    read_counter = tid % offset; // Start reading from the original counter minus the offset
    write_counter = orig_counter + tid; // Start writing from the original counter

    // Write the prefix and the suffix by performing byte memcpy
    if (tid < prefix_bytes && tid < len) {
        out_ptr[write_counter + WRITE_COL_LEN * idx] = arr_shared[read_counter];
    } 

    read_counter = (tid * NMEMCPY + prefix_bytes) % offset; // Start reading from the original counter minus the offset
    write_counter = orig_counter + tid * NMEMCPY + prefix_bytes; // Start writing from the original counter

    #if NMEMCPY == 2
    uchar2* out_ptr_temp  = reinterpret_cast<uchar2*>(out_ptr);

    #elif USE_EIGHT == 0
    uchar4* out_ptr_temp  = reinterpret_cast<uchar4*>(out_ptr);

    #elif USE_EIGHT == 1
    ushort4* out_ptr_temp  = reinterpret_cast<ushort4*>(out_ptr);

    #endif

    __syncwarp(MASK);

    num_writes = ((len - prefix_bytes - suffix_bytes - tid * NMEMCPY + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY));
    if (tid * NMEMCPY + prefix_bytes + suffix_bytes > len) num_writes = 0;

    num_ph =  (len - prefix_bytes - suffix_bytes + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY); // The largest amount of times a thread in the warp is writing to memory.
    if (prefix_bytes + suffix_bytes > len) num_ph = 0;

    //#pragma unroll 
    for(int i = 0; i < num_ph; i++){
        if(i < num_writes){ // If this thread should write. 4 bytes
            #if NMEMCPY == 1
            out_ptr[write_counter + WRITE_COL_LEN * idx] = arr_shared[read_counter];
            #elif NMEMCPY == 2
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_uchar2(arr_shared[read_counter], arr_shared[(read_counter + 1) % offset]);
            #elif NMEMCPY == 4
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_uchar4(arr_shared[read_counter], arr_shared[(read_counter + 1) % offset],
                                                                                             arr_shared[(read_counter + 2) % offset], arr_shared[(read_counter + 3) % offset]);
            #elif NMEMCPY == 8
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_ushort4((((unsigned short)arr_shared[(read_counter + 1) % offset] << 8) | arr_shared[read_counter]),
                                                                                  (((unsigned short)arr_shared[(read_counter + 3) % offset] << 8) | arr_shared[(read_counter + 2) % offset]),
                                                                                  (((unsigned short)arr_shared[(read_counter + 5) % offset] << 8) | arr_shared[(read_counter + 4) % offset]),
                                                                                  (((unsigned short)arr_shared[(read_counter + 7) % offset] << 8) | arr_shared[(read_counter + 6) % offset]));
            #endif

            // 1 4 byte transaction | 4 1 byte transactions

            read_counter = (read_counter + NUM_THREAD * NMEMCPY) % offset; // Add the number of bytes that we wrote.
            write_counter += NUM_THREAD * NMEMCPY; // Add the number of bytes that we wrote.
        } 
        // #endif
        __syncwarp(MASK); // Synchronize.
    }

    /* This is a big source of slowdown. The suffix of the memory operation must be written last if it reads from a value that was written in this function. */
    read_counter = (len + tid - suffix_bytes) % offset; // Start reading from the original counter minus the offset
    if (tid < suffix_bytes && tid < len) {
        out_ptr[orig_counter + tid + WRITE_COL_LEN * idx + len - ((uint32_t) suffix_bytes)] = arr_shared[read_counter];
    }
    __syncwarp(MASK);

    //set the counter
    if(tid == idx % NUM_THREAD)
        *counter += len;

}

template <uint32_t NUM_THREAD = 8, typename T>
// __forceinline__ 
__device__
void memcpy_nbyte_prefix_body_suffix_needs_writing_alignment_shared_2(T* out_ptr, uint32_t* counter, T* arr_shared, uint8_t idx, uint16_t len, uint16_t offset, uint8_t div, uint32_t MASK) {
    int tid;
    uint32_t orig_counter;
    uint8_t num_writes;
    uint32_t start_counter;
    uint32_t read_counter;
    uint32_t write_counter;
    uint8_t num_ph;
    /* Memcpy aligned on write boundaries */
    tid = threadIdx.x - div * NUM_THREAD;
    orig_counter = __shfl_sync(MASK, *counter, idx); // This is equal to the tid's counter value upon entering the function

    uint8_t prefix_bytes = 0;
    uint8_t suffix_bytes = 0;

    // Write to the shared memory.
    if ((tid) < offset) {
        arr_shared[tid] = out_ptr[orig_counter - offset + tid];
    }
    if ((tid) + NUM_THREAD < offset) {
        arr_shared[(tid) + NUM_THREAD] = out_ptr[orig_counter - offset + (tid) + NUM_THREAD];
    }
    __syncwarp(MASK);

    // TODO: CHANGE
    start_counter =  orig_counter - offset; // The place to start writing. Notice we subtract offset, not add it.

    read_counter = (tid * NMEMCPY + prefix_bytes) % offset; // Start reading from the original counter minus the offset
    write_counter = orig_counter + tid * NMEMCPY + prefix_bytes; // Start writing from the original counter

    #if NMEMCPY == 2
    uchar2* out_ptr_temp  = reinterpret_cast<uchar2*>(out_ptr);
    #elif NMEMCPY == 4
    uchar4* out_ptr_temp  = reinterpret_cast<uchar4*>(out_ptr);
    #elif NMEMCPY == 8
    ushort4* out_ptr_temp  = reinterpret_cast<ushort4*>(out_ptr);
    #endif

    __syncwarp(MASK);

    num_writes = ((len - prefix_bytes - suffix_bytes - tid * NMEMCPY + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY));
    if (tid * NMEMCPY + prefix_bytes + suffix_bytes > len) num_writes = 0;

    num_ph =  (len - prefix_bytes - suffix_bytes + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY); // The largest amount of times a thread in the warp is writing to memory.
    if (prefix_bytes + suffix_bytes > len) num_ph = 0;

    //#pragma unroll 
    for(int i = 0; i < num_ph; i++){
        if(i < num_writes){ // If this thread should write. 4 bytes
            #if NMEMCPY == 1
            out_ptr[write_counter + WRITE_COL_LEN * idx] = arr_shared[read_counter];
            #elif NMEMCPY == 2
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_uchar2(arr_shared[read_counter], arr_shared[(read_counter + 1) % offset]);
            #elif NMEMCPY == 4
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_uchar4(arr_shared[read_counter], arr_shared[(read_counter + 1) % offset],
                                                                                             arr_shared[(read_counter + 2) % offset], arr_shared[(read_counter + 3) % offset]);
            #elif NMEMCPY == 8
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_ushort4((((unsigned short)arr_shared[(read_counter + 1) % offset] << 8) | arr_shared[read_counter]),
                                                                                  (((unsigned short)arr_shared[(read_counter + 3) % offset] << 8) | arr_shared[(read_counter + 2) % offset]),
                                                                                  (((unsigned short)arr_shared[(read_counter + 5) % offset] << 8) | arr_shared[(read_counter + 4) % offset]),
                                                                                  (((unsigned short)arr_shared[(read_counter + 7) % offset] << 8) | arr_shared[(read_counter + 6) % offset]));
            #endif

            // 1 4 byte transaction | 4 1 byte transactions

            read_counter = (read_counter + NUM_THREAD * NMEMCPY) % offset; // Add the number of bytes that we wrote.
            write_counter += NUM_THREAD * NMEMCPY; // Add the number of bytes that we wrote.
        } 
        // #endif
        __syncwarp(MASK); // Synchronize.
    }

    __syncwarp(MASK);

    //set the counter
    if(tid == idx % NUM_THREAD)
        *counter += len;

}

template <uint32_t NUM_THREAD = 8, typename T>
// __forceinline__ 
__device__
void memcpy_nbyte_aligned_2(T* out_ptr, uint32_t* counter, uint8_t idx, uint16_t len, uint16_t offset, uint8_t div, uint32_t MASK) {
    int tid = threadIdx.x - div * NUM_THREAD;
    uint32_t orig_counter = __shfl_sync(MASK, *counter, idx);
    uint8_t num_writes = ((len - tid * NMEMCPY + NUM_THREAD * NMEMCPY - 1) / (NUM_THREAD * NMEMCPY));
    uint32_t start_counter =  orig_counter - offset;


    uint32_t read_counter = start_counter + tid * NMEMCPY;
    uint32_t write_counter = orig_counter + tid * NMEMCPY;

    #if NMEMCPY == 1
    T* out_ptr_temp   = out_ptr;
    #elif NMEMCPY == 2
    uint16_t* out_ptr_temp  = reinterpret_cast<uint16_t*>(out_ptr);
    #elif NMEMCPY == 4
    uint32_t* out_ptr_temp  = reinterpret_cast<uint32_t*>(out_ptr);
    #elif NMEMCPY == 8
    uint64_t* out_ptr_temp  = reinterpret_cast<uint64_t*>(out_ptr);
    #endif

    if(read_counter >= orig_counter){
        read_counter = (read_counter - orig_counter) % (offset) + start_counter;
    }

    uint8_t num_ph =  (len +  NUM_THREAD * NMEMCPY - 1) / (NUM_THREAD * NMEMCPY);
    //#pragma unroll 
    for(int i = 0; i < num_ph; i++){
        if(i < num_writes){
            out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = out_ptr_temp[(read_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2];
            read_counter += NUM_THREAD * NMEMCPY;
            write_counter += NUM_THREAD * NMEMCPY;
        }
        __syncwarp(MASK);
    }

    //set the counter
    if(tid == idx % NUM_THREAD) {
        *counter += len;
    }
}







#endif


