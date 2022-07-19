// #ifndef __ZLIB_H__
// #define __ZLIB_H__

#include <common.h>
#include <fstream>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <simt/atomic>
#include <iostream>
#include <common_warp.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#define BUFF_LEN 2


#define UNCOMP 0
#define STATIC 1
#define DYNAMIC 2
#define FULL_MASK 0xFFFFFFFF

#define MASK_2_1  0x0000FFFF
#define MASK_2_2  0xFFFF0000


#define MASK_4_1  0x000000FF
#define MASK_4_2  0x0000FF00
#define MASK_4_3  0x00FF0000
#define MASK_4_4  0xFF000000

#define MASK_8_1  0x0000000F
#define MASK_8_2  0x000000F0
#define MASK_8_3  0x00000F00
#define MASK_8_4  0x0000F000
#define MASK_8_5  0x000F0000
#define MASK_8_6  0x00F00000
#define MASK_8_7  0x0F000000
#define MASK_8_8  0xF0000000

#define MAXBITS 15
#define FIXLCODES 288
#define MAXDCODES 30
#define MAXCODES 316

#define DECODE_MASK_1  0xFFFFFFFF
#define DECODE_MASK_2  0xAAAAAAAA
#define DECODE_MASK_4  0x88888888
#define DECODE_MASK_8  0x80808080
#define DECODE_MASK_16 0x80008000

#define LOG2LENLUT 5
///#define LOG2DISTLUT 8
#define LOG2DISTLUT 8

/* Uncomment one of these N byte write test cases below */
// #define NMEMCPY 1 // Must be power of 2, acceptable = {1,2,4}
// #define NMEMCPYLOG2 0
// #define MEMCPYLARGEMASK 0xffffffff
// #define MEMCPYSMALLMASK 0x00000000

// #define NMEMCPY 2 // Must be power of 2, acceptable = {1,2,4}
// #define NMEMCPYLOG2 1
// #define MEMCPYLARGEMASK 0xfffffffe
// #define MEMCPYSMALLMASK 0x00000001

#define NMEMCPY 4 // Must be power of 2, acceptable = {1,2,4}
#define NMEMCPYLOG2 2
#define MEMCPYLARGEMASK 0xfffffffc
#define MEMCPYSMALLMASK 0x00000003

#define SHARED_MEMCPY NMEMCPY*32*8
// #define MEMCPY_SIZE_THRESH NMEMCPY*26
#define MEMCPY_SIZE_THRESH NMEMCPY*26

#define MEMCPY_TYPE uint32_t
using namespace cooperative_groups; 
static const __device__ __constant__ uint16_t g_lens[29] = {  // Size base for length codes 257..285
  3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
  31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};


static const __device__ __constant__ uint16_t
  g_lext[29] = { 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};


static const __device__ __constant__ uint16_t
  g_dists[30] = {  // Offset base for distance codes 0..29
    1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
    193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};

static const __device__ __constant__ uint16_t g_dext[30] = {  // Extra bits for distance codes 0..29
  0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};


inline __device__ unsigned int bfe(unsigned int source,
                                   unsigned int bit_start,
                                   unsigned int num_bits)
{
  unsigned int bits;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "r"(bit_start), "r"(num_bits));
  return bits;
};



__constant__ int32_t fixed_lengths[FIXLCODES];



struct dynamic_huffman {
    int16_t lensym[FIXLCODES];
    int16_t treelen[MAXCODES];
    //int16_t off[MAXBITS + 1];
}; //__attribute__((aligned(16)));


struct fix_huffman {
    int16_t lencnt[MAXBITS + 1];
    int16_t lensym[FIXLCODES];
    int16_t distcnt[MAXBITS + 1];
    int16_t distsym[MAXDCODES];
};// __attribute__((aligned(16)));


struct s_huffman {
    int16_t lencnt[MAXBITS + 1];
    int16_t off[MAXBITS + 1];
    int16_t distcnt[MAXBITS + 1];
    int16_t distsym[MAXDCODES];
    dynamic_huffman dh;
};

struct inflate_lut{
  int32_t len_lut[1 << LOG2LENLUT];
  int32_t dist_lut[1 << LOG2DISTLUT];
  uint16_t first_slow_len; 
  uint16_t index_slow_len;
  uint16_t first_slow_dist;
  uint16_t index_slow_dist;

};


struct device_space{
    inflate_lut* d_lut;
};


typedef struct __align__(32)
{
    simt::atomic<uint8_t, simt::thread_scope_device>  counter;
    simt::atomic<uint8_t, simt::thread_scope_device>  lock[32];

} __attribute__((aligned (32))) slot_struct;


typedef uint64_t write_queue_ele;
// struct  write_queue_ele{
//     uint32_t data;
//     uint8_t type;
// };

//__forceinline__
 __device__ unsigned warp_id()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret;
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}
//__forceinline__ 
__device__ uint32_t get_smid() {
     uint32_t ret;
     asm  ("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

//__forceinline__
__device__ uint8_t find_slot(uint32_t sm_id, slot_struct* slot_array){

    //bool fail = true;
    //uint64_t count = 0;
    uint8_t page = 0;

    do{
        page = ((slot_array[sm_id]).counter.fetch_add(1, simt::memory_order_acquire)) % 32;

        bool lock = false;
        uint8_t v = (slot_array[sm_id]).lock[page].load(simt::memory_order_acquire);
        if (v == 0)
        {
            lock = (slot_array[sm_id]).lock[page].compare_exchange_strong(v, 1, simt::memory_order_acquire, simt::memory_order_relaxed);

            if(lock){
                //(slot_array[sm_id]).lock[page].store(0, simt::memory_order_release);
                return page;
            }
        }

    } while(true);

    return page;
}

__forceinline__
__device__ void release_slot(uint32_t sm_id, uint32_t page, slot_struct* slot_array){
    (slot_array[sm_id]).lock[page].store(0, simt::memory_order_release);
}



template <size_t WRITE_COL_LEN = 512>
struct decompress_output {

    uint8_t* out_ptr;
    uint32_t counter;

    __device__
    decompress_output(uint8_t* ptr, uint64_t CHUNK_SIZE):
        out_ptr(ptr) {
            counter = 0;
            
    }

    template <uint32_t NUM_THREAD = 8,
              typename TYPE_EXT = uchar4>
    __device__
    void col_memcpy_Nbyte_shared_onepass2(TYPE_EXT* arr_shared, uint8_t idx, uint16_t len, uint16_t offset, uint8_t div, uint32_t MASK) {
// Asynchronous memcpy for load to shared memory.
    /*
    Read min(offset/4, len/4) N byte packs into shared memory first. Also read the N byte pack of memory at the start of the counter.
    'Prefix' can be written as a combination of the N byte pack of memory at the start and the first pack in shared memory.
    'Suffix' can be written as the last N byte pack with anything at the end, unless we are writting the very last N byte pack in the column.
    */
    // cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    // cooperative_groups::thread_group tile32 = cooperative_groups::tiled_partition(block, 32);
    // cuda::barrier<thread_scope_block> bar;
    // init(&bar, 1);

    // Share the counter from thread idx
    // if (threadIdx.x == 0) printf("%d\n", threadIdx.y);
    int tid = threadIdx.x - div * NUM_THREAD;
    uint32_t orig_counter = __shfl_sync(MASK, counter, idx);
    uint32_t start_counter = orig_counter - offset;
    // uint8_t shared_offset = (threadIdx.y - 1) * (32 * NMEMCPY);
    uint8_t shared_offset = 0;


    // Check if we are at the end of a column.
    // if ((orig_counter + len) % WRITE_COL_LEN == 0) {
    //     col_memcpy_div<32>(idx, len, offset, div, MASK);
    //     return;
    // }

    // Calculate how many bytes we are misaligned in both writing and reading.
    uint8_t bits_written_in_first_pack = NMEMCPY - ((orig_counter + WRITE_COL_LEN * idx) & MEMCPYSMALLMASK);

    // Recast the output array as an N byte array.
    TYPE_EXT* out_ptr_temp = reinterpret_cast<TYPE_EXT*>(out_ptr);

    // Write the prefix from global memory to global memory
    // REPLACE
    if (threadIdx.x == 0)
        memcpy(&out_ptr[orig_counter + WRITE_COL_LEN * idx], &out_ptr[start_counter + WRITE_COL_LEN * idx], bits_written_in_first_pack);
    // cooperative_groups::memcpy_async(tile32, &out_ptr[orig_counter + WRITE_COL_LEN * idx], &out_ptr[start_counter + WRITE_COL_LEN * idx], bits_written_in_first_pack);
    // cuda::memcpy_async(&out_ptr[orig_counter + WRITE_COL_LEN * idx], &out_ptr[start_counter + WRITE_COL_LEN * idx], bits_written_in_first_pack, bar);

    // Start reading in data. An additional thread is needed to load in the data at the start of our write section.
    uint32_t read_pack_addr = (start_counter + WRITE_COL_LEN * idx + tid * NMEMCPY);
    uint8_t reading_starting_bit_number = read_pack_addr & MEMCPYSMALLMASK;


    // Get the total amount of unique bits that we care about reading.
    uint8_t total_unique_bits = min(offset, len);

    // Calculate the read address for the threads. Load in N bytes based on either the size of offset or the size of len.
    uint8_t num_packs = (total_unique_bits + reading_starting_bit_number + NMEMCPY - 1) >> NMEMCPYLOG2;

    // REPLACE
    if (threadIdx.x == 0){
        memcpy(arr_shared, &out_ptr_temp[(start_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2], num_packs * NMEMCPY);
        // printf("%d\n", reading_starting_bit_number);
    }
    // cuda::memcpy_async(arr_shared, &out_ptr_temp[read_pack_addr], num_packs * NMEMCPY, bar);
    // cooperative_groups::memcpy_async(tile32, arr_shared, &out_ptr_temp[(start_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2], num_packs * NMEMCPY);

    uint8_t shift;
    uint8_t idx2;
    TYPE_EXT pack;
    // Load the last 2 control words
    // TODO: FIX incorrect control words
    if (threadIdx.x < 2) {
        idx2 = 0;
        // Starting control word shift is reading starting bit number
        // Ending control word shift is (reading starting bit number + total unique bits) & MASK
        shift = reading_starting_bit_number;
        if (threadIdx.x == 0) {
            idx2 = num_packs - 1;
            shift += total_unique_bits;
            if (num_packs > 1) {
                idx2 -= 1;
            }
        }
            
        shift = shift & MEMCPYSMALLMASK;

        if (threadIdx.x == 0 && num_packs > 1 && shift == 0) {
            shift = 4;
            // shift = NMEMCPY - shift;
            // shift += 1;
        }
        
        // if (shift == 0 && threadIdx.x == 0){
        //     shift = NMEMCPY;
        // }
        
        // Funnel Shift?
        // pack = (pack << (shift * 8)) | out_ptr_temp[read_pack_addr+1] >> ((4 - shift) * 8);
        // pack = __funnelshift_lc(out_ptr_temp[read_pack_addr+1], pack, shift*8);
    }
    // WAIT
    // cooperative_groups::wait(tile32);
    __syncwarp();


    if (threadIdx.x < 2) {
        pack = __funnelshift_rc(arr_shared[idx2], arr_shared[idx2+1], shift*8);
        arr_shared[num_packs + threadIdx.x] = pack;
    }

    // Get the amount of bits in the last pack
    uint8_t bits_written_in_last_pack = (len - bits_written_in_first_pack) & MEMCPYSMALLMASK;
    // if (bits_written_in_last_pack == 0) bits_written_in_last_pack = 4;


    // Calculate how many writing operations to perform.
    // TODO: Deal with (+ NMEMCPY in line below)
    int8_t num_writes = ((len - bits_written_in_first_pack - bits_written_in_last_pack - tid * NMEMCPY + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY));
    // if (tid * NMEMCPY + bits_written_in_first_pack > len) {
    //     num_writes = 0;
    // }

    // Calculate the number of times that the thread with the most blocks has to write.
    uint8_t num_ph = (len - bits_written_in_first_pack + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY);

    // Calculate the write address
    uint32_t write_counter = (orig_counter + WRITE_COL_LEN * idx + tid * NMEMCPY + bits_written_in_first_pack);

    // cooperative_groups::wait(tile32);
    __syncwarp();
    // if (threadIdx.x == 0 && div == 79) {
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
    for(int i = 0; i < num_ph; i++){
        if(i < num_writes){
            // We shouldn't have to worry about underflow as we manually set the indices later.
            int32_t total_bits_written = bits_written_in_first_pack + (threadIdx.x) * NMEMCPY + (i * 32 * NMEMCPY);
            if (total_bits_written < 0) total_bits_written = 0;

            // Transform this into the current bit that we start writing in this run.
            uint8_t current_bit_in_run = total_bits_written % total_unique_bits;
            uint8_t current_pack_in_run = (current_bit_in_run + reading_starting_bit_number) >> NMEMCPYLOG2;

            // Each thread will load two packs. Get the indices.
            uint8_t first_pack_idx = current_pack_in_run;
            uint8_t second_pack_idx = first_pack_idx + 1;

            // Deal with the control words. First write reads from words 2 and 1. Any writes at the end read from 0 and 1.
            if(current_bit_in_run > total_unique_bits - NMEMCPY) {
                first_pack_idx = num_packs;
                second_pack_idx = num_packs + 1;
            }

            // Load in the packs from shared memory.
            TYPE_EXT pack1 = arr_shared[first_pack_idx + shared_offset];
            TYPE_EXT pack2 = arr_shared[second_pack_idx + shared_offset];

            // Calculate the shift needed to combine the packs. Special cases above have modified shifts.
            uint8_t shift = (current_bit_in_run + reading_starting_bit_number) & MEMCPYSMALLMASK;
            // out_ptr_temp[write_counter >> NMEMCPYLOG2] = shift;

            if (current_bit_in_run > total_unique_bits - NMEMCPY) {
                shift = NMEMCPY + current_bit_in_run - total_unique_bits;
            }

            // Perform the funnel shift
            // pack1 = __funnelshift_lc(pack2, pack1, shift * 8);
            pack1 = __funnelshift_rc(pack1, pack2, shift * 8);
            // printf("%08x\n", pack1);

            out_ptr_temp[write_counter >> NMEMCPYLOG2] = pack1;

            // Shouldn't need first case
            // if (i == 0) write_counter += (NUM_THREAD - 1) * NMEMCPY + bits_written_in_first_pack;
            write_counter += NUM_THREAD * NMEMCPY;
            // if (div == 79) {
            //     printf("SHIFT: %d\n", shift);
            //     printf("FIRST: %d\n", first_pack_idx);
            // }

            // if (write_counter >> NMEMCPYLOG2 == 5) printf("%d %d\n", offset, len);
        } 

    }
    // Write the suffix.
    if (threadIdx.x < bits_written_in_last_pack) {
        // We shouldn't have to worry about underflow as we manually set the indices later.
        uint32_t total_bits_written = len - bits_written_in_last_pack + threadIdx.x;

        // Transform this into the current bit that we start writing in this run.
        uint8_t current_bit_in_run = total_bits_written % total_unique_bits;
        uint8_t current_pack_in_run = (current_bit_in_run + reading_starting_bit_number) >> NMEMCPYLOG2;
        uint8_t shift = (current_bit_in_run + reading_starting_bit_number) & MEMCPYSMALLMASK;
        // if (current_bit_in_run > total_unique_bits - NMEMCPY) {
        //     shift = NMEMCPY + current_bit_in_run - total_unique_bits;
        // }
        // if (total_unique_bits - current_bit_in_run <= NMEMCPY) {
        //     shift = total_unique_bits - current_bit_in_run - 1;
        // }
        // if (div == 79) {
        //     printf("%d\n", shift);
        //     printf("%08x\n", (arr_shared[current_pack_in_run] >> ((shift) * 8)) & 0x000000ff);
        //     printf("%08x\n", orig_counter + idx * WRITE_COL_LEN + total_bits_written);
        // }

        out_ptr[orig_counter + idx * WRITE_COL_LEN + total_bits_written] = (uint8_t) ((arr_shared[current_pack_in_run] >> ((shift) * 8)) & 0x000000ff);
    }

    // Increment the counter by len. No race condition because we are using the full warp mask.
    if(threadIdx.x == idx)
        counter += len;
    __syncwarp();
    }

    
    template <uint32_t NUM_THREAD = 8>
   // __forceinline__ 
    __device__
    void col_memcpy_div(uint8_t idx, uint16_t len, uint16_t offset, uint8_t div, uint32_t MASK) {
       
        int tid = threadIdx.x - div * NUM_THREAD;
        uint32_t orig_counter = __shfl_sync(MASK, counter, idx);

        uint8_t num_writes = ((len - tid + NUM_THREAD - 1) / NUM_THREAD);
        uint32_t start_counter =  orig_counter - offset;


        uint32_t read_counter = start_counter + tid;
        uint32_t write_counter = orig_counter + tid;

        if(read_counter >= orig_counter){
            read_counter = (read_counter - orig_counter) % offset + start_counter;
        }

        uint8_t num_ph =  (len +  NUM_THREAD - 1) / NUM_THREAD;
        #pragma unroll 
        for(int i = 0; i < num_ph; i++){
            if(i < num_writes){
                out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
                read_counter += NUM_THREAD;
                write_counter += NUM_THREAD;
            }
            __syncwarp();
        }
    
        //set the counter
        if(threadIdx.x == idx)
            counter += len;



       

        // // } else {
        //     int tid = threadIdx.x - div * NUM_THREAD;
        //     uint32_t orig_counter = __shfl_sync(MASK, counter, idx); // This is equal to the tid's counter value upon entering the function

        //     // TODO: CHANGE
        //     uint8_t num_writes = ((((len - tid * NMEMCPY) & MEMCPYLARGEMASK) + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY) * NMEMCPY);
        //     if (((((len - tid * NMEMCPY + NMEMCPY) & MEMCPYLARGEMASK) + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY)) > ((((len - tid * NMEMCPY) & MEMCPYLARGEMASK) + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY)))
        //         num_writes += (len - tid * NMEMCPY) & MEMCPYSMALLMASK;
        //     if (tid * NMEMCPY > len) num_writes = 0;
        //     uint32_t start_counter =  orig_counter - offset; // The place to start writing. Notice we subtract offset, not add it.

        //     uint32_t read_counter = start_counter + tid * NMEMCPY; // Start reading from the original counter minus the offset
        //     uint32_t write_counter = orig_counter + tid * NMEMCPY; // Start writing from the original counter

        //     if(read_counter >= orig_counter){ // If we are reading from ahead of the place we are writing.
        //             read_counter = (read_counter - orig_counter) % offset + start_counter; // Don't really know what this line does.
        //     }

        //     uint8_t num_ph =  (len +  (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY); // The largest amount of times a thread in the warp is writing to memory.
        //     #pragma unroll 
        //     for(int i = 0; i < num_ph; i++){
        //         if(num_writes - i * NMEMCPY >= NMEMCPY){ // If this thread should write. 4 bytes
        //             // TODO: CHANGE
        //             out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
        //             #if NMEMCPY > 1
        //             out_ptr[write_counter + WRITE_COL_LEN * idx + 1] = out_ptr[read_counter + WRITE_COL_LEN * idx + 1];
        //             #endif
        //             #if NMEMCPY > 2
        //             out_ptr[write_counter + WRITE_COL_LEN * idx + 2] = out_ptr[read_counter + WRITE_COL_LEN * idx + 2];
        //             out_ptr[write_counter + WRITE_COL_LEN * idx + 3] = out_ptr[read_counter + WRITE_COL_LEN * idx + 3];
        //             #endif
        //             // 1 4 byte transaction | 4 1 byte transactions
        //             // char4 out_ptr[idx] = // 4 1bytes read

        //             read_counter += NUM_THREAD * NMEMCPY; // Add the number of bytes that we wrote.
        //             write_counter += NUM_THREAD * NMEMCPY; // Add the number of bytes that we wrote.
        //         } 
        //         // #if NMEMCPY > 1
        //         else if (num_writes - i * NMEMCPY > 0) { // Write however many bytes we have left.
        //             out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
        //             #if NMEMCPY > 2
        //             if (num_writes - i * NMEMCPY > 1) {
        //                 out_ptr[write_counter + WRITE_COL_LEN * idx + 1] = out_ptr[read_counter + WRITE_COL_LEN * idx + 1];
        //             }
        //             if (num_writes - i * NMEMCPY > 2) {
        //                 out_ptr[write_counter + WRITE_COL_LEN * idx + 2] = out_ptr[read_counter + WRITE_COL_LEN * idx + 2];
        //             }
        //             #endif
        //             read_counter += NUM_THREAD * NMEMCPY; // Add the number of bytes that we wrote.
        //             write_counter += NUM_THREAD * NMEMCPY; // Add the number of bytes that we wrote.


        //         }
        //         // #endif
        //         __syncwarp(); // Synchronize.
        //     }
        
        //     //set the counter
        //     if(threadIdx.x == idx)
        //         counter += len; // Counter is by thread
        // // }
  

    }
    


    //__forceinline__ 
    __device__
    void write_literal(uint8_t idx, uint8_t b){
        if(threadIdx.x == idx){
            out_ptr[counter + WRITE_COL_LEN * idx] = b;
            counter++;
        }
    }


};





template <typename READ_COL_TYPE, typename COMP_COL_TYPE, uint8_t NUM_SUBCHUNKS>
//__forceinline__ 
__device__
void reader_warp(decompress_input<READ_COL_TYPE, COMP_COL_TYPE>& in, queue<READ_COL_TYPE>& rq, uint8_t active_chunks) {
    //iterate number of chunks for the single reader warp
   int t = 0;
   while(true){
        bool done = true;
        for(uint8_t cur_chunk = 0; cur_chunk < active_chunks; cur_chunk++){
            COMP_COL_TYPE v;
            uint8_t rc = comp_read_data_seq(FULL_MASK, &v, in, cur_chunk);
            if(rc != 0)
                done = false;
            rq.warp_enqueue(&v, cur_chunk, rc);
        }
        __syncwarp();
        if(done)
            break;
    }
}



template <typename READ_COL_TYPE, typename COMP_COL_TYPE, uint8_t NUM_SUBCHUNKS>
//__forceinline__ 
__device__
void sub_reader_warp(decompress_input<READ_COL_TYPE, COMP_COL_TYPE>& in, queue<READ_COL_TYPE>& rq, uint8_t active_chunks) {
    //iterate number of chunks for the single reader warp
   

   // int t = 0;

   // while(true){
   //      bool done = true;
   //      for(uint8_t cur_chunk = 0; cur_chunk < active_chunks; cur_chunk++){
   //          COMP_COL_TYPE v;
   //          uint8_t rc = comp_read_data_seq(FULL_MASK, &v, in, cur_chunk);
   //          if(rc != 0)
   //              done = false;

   //      if(threadIdx.x == 0) printf("rc = %i\n",rc);
   //          rq.warp_enqueue(&v, cur_chunk * 16, rc);
   //      }
   //      __syncwarp();
   //      if(done)
   //          break;
   //  }


   int t = 0;
   uint32_t READ_MASK = (threadIdx.x < 16) ? MASK_2_1 : MASK_2_2;
   while(true){

        COMP_COL_TYPE v;
        uint8_t rc = comp_read_data_seq_sub(READ_MASK, &v, in, (threadIdx.x / 16)*16);

        if(rc == 0) {break;}


         rq.sub_warp_enqueue(&v, (threadIdx.x / 16) * 16 , rc, READ_MASK, threadIdx.x / 16);


         __syncwarp();
    }
}





__device__ void init_length_lut(inflate_lut *s_lut, const int16_t* cnt, const int16_t* orig_symbols, int t, uint32_t NUM_THREAD)
{

  int32_t* lut = s_lut -> len_lut;
  const int16_t* symbols = orig_symbols;

  for (uint32_t bits = t; bits < (1 << LOG2LENLUT); bits += NUM_THREAD) {
    int sym                = -10 << 5;
    unsigned int first     = 0;
    unsigned int rbits     = __brev(bits) >> (32 - LOG2LENLUT);
    symbols = orig_symbols;
    for (unsigned int len = 1; len <= LOG2LENLUT; len++) {
      unsigned int code  = (rbits >> (LOG2LENLUT - len)) - first;
      unsigned int count = cnt[len];
     
      if (code < count) {
        sym = symbols[code];

    /*
        if (sym > 256) {
         printf("sym larger\n");
            int lext = g_lext[sym - 257];
          sym = (256 + g_lens[sym - 257]) | (((1 << lext) - 1) << (16 - 5)) | (len << (24 - 5));
          len += lext;
        }*/
       sym = (sym << 5) | len;
        break;
      }
      symbols += count;  // else update for next length
      first += count;
      first <<= 1;
    }
    lut[bits] = sym;
    
  }
  if (!t) {
    unsigned int first = 0;
    unsigned int index = 0;
    for (unsigned int len = 1; len <= LOG2LENLUT; len++) {
      unsigned int count = cnt[len];
      index += count;
      first += count;
      first <<= 1;
    }
    s_lut->first_slow_len = first;
    s_lut->index_slow_len = index;
  }

}


__device__ void init_length_lut2(inflate_lut *s_lut, const int16_t* cnt, const int16_t* orig_symbols, int t, uint32_t NUM_THREAD)
{

  int32_t* lut = s_lut -> len_lut;

  for (uint32_t bits = t; bits < (1 << LOG2LENLUT); bits += NUM_THREAD) {
    int sym                = -10 << 5;
    unsigned int first     = 0;
    const int16_t* symbols = orig_symbols;

    unsigned int rbits     = __brev(bits) >> (32 - LOG2LENLUT);
    for (unsigned int len = 1; len <= LOG2LENLUT; len++) {
      unsigned int code  = (rbits >> (LOG2LENLUT - len)) - first;
      unsigned int count = cnt[len];
     
      if (code < count) {
        sym = symbols[code];
   
    
        if (sym > 256) {
        // printf("sym larger\n");
            int lext = g_lext[sym - 257];
          sym = (256 + g_lens[sym - 257]) | (((1 << lext) - 1) << (16 - 5)) | (len << (24 - 5));
          len += lext;
        }
    
        sym = (sym << 5) | len;
        break;
      }
      symbols += count;  // else update for next length
      first += count;
      first <<= 1;
    }
    
    lut[bits] = sym;
    
  }
  if (!t) {
    unsigned int first = 0;
    unsigned int index = 0;
    for (unsigned int len = 1; len <= LOG2LENLUT; len++) {
      unsigned int count = cnt[len];
      index += count;
      first += count;
      first <<= 1;
    }
    s_lut->first_slow_len = first;
    s_lut->index_slow_len = index;
  }

}

__device__ void init_distance_lut(inflate_lut *s_lut, const int16_t* cnt, const int16_t* orig_symbols, int t, uint32_t NUM_THREAD)
{

  int32_t* lut = s_lut -> dist_lut;
  const int16_t* symbols = orig_symbols;

  for (uint32_t bits = t; bits < (1 << LOG2DISTLUT); bits += NUM_THREAD) {
    int sym                = 0;
    unsigned int first     = 0;
    unsigned int rbits     = __brev(bits) >> (32 - LOG2DISTLUT);
    symbols = orig_symbols;
    for (unsigned int len = 1; len <= LOG2DISTLUT; len++) {
      unsigned int code  = (rbits >> (LOG2DISTLUT - len)) - first;
      int count = cnt[len];
     
      if (code < count) {
        sym = symbols[code];
        //int dext = g_dext[dist];
        //sym      = g_dists[dist] | (dext << 15);
        sym      = (sym << 5) | len;
        break;
      }
      symbols += count;  // else update for next length
      first += count;
      first <<= 1;
    }
    lut[bits] = sym;  
  }

  if (!t) {
    unsigned int first = 0;
    unsigned int index = 0;
    for (unsigned int len = 1; len <= LOG2DISTLUT; len++) {
      unsigned int count = cnt[len];
      index += count;
      first += count;
      first <<= 1;
    }
    s_lut->first_slow_dist = first;
    s_lut->index_slow_dist = index;
  }

}

__device__ void init_distance_lut2(inflate_lut *s_lut, const int16_t* cnt, const int16_t* orig_symbols, int t, uint32_t NUM_THREAD)
{

  int32_t* lut = s_lut -> dist_lut;
  const int16_t* symbols = orig_symbols;

  for (uint32_t bits = t; bits < (1 << LOG2DISTLUT); bits += NUM_THREAD) {
    int sym                = 0;
    unsigned int first     = 0;
    unsigned int rbits     = __brev(bits) >> (32 - LOG2DISTLUT);
    symbols = orig_symbols;
    for (unsigned int len = 1; len <= LOG2DISTLUT; len++) {
      unsigned int code  = (rbits >> (LOG2DISTLUT - len)) - first;
      int count = cnt[len];
     
      if (code < count) {
        sym = symbols[code];
        int dist = symbols[code];
        int dext = g_dext[dist];
        sym      = g_dists[dist] | (dext << 15);
        sym      = (sym << 5) | len;
        break;
      }
      symbols += count;  // else update for next length
      first += count;
      first <<= 1;
    }
    lut[bits] = sym;  
  }

  if (!t) {
    unsigned int first = 0;
    unsigned int index = 0;
    for (unsigned int len = 1; len <= LOG2DISTLUT; len++) {
      unsigned int count = cnt[len];
      index += count;
      first += count;
      first <<= 1;
    }
    s_lut->first_slow_dist = first;
    s_lut->index_slow_dist = index;
  }

}




template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
int16_t decode (input_stream<READ_COL_TYPE, in_buff_len>&  s, const int16_t* const   counts, const int16_t*   syms){


    uint32_t next32r = 0;
    s.template peek_n_bits<uint32_t>(32, &next32r);

    next32r = __brev(next32r);
    const int16_t*   symbols = syms;


    uint32_t first = 0;
    #pragma no unroll
    for (uint8_t len = 1; len <= MAXBITS; len++) {
        //if(len == LOG2LENLUT + 1) printf("first: %lu\n", first);

        uint32_t code  = (next32r >> (32 - len)) - first;
        
        uint16_t count = counts[len];
    if (code < count) 
    {
        uint32_t temp = 0;
        s.template fetch_n_bits<uint32_t>(len, &temp);
        //printf("code: %lu count: %lu len: %i, off:%i temp:%lx \n", (unsigned long)code, (unsigned long) count ,(int) len, (int) (symbols - syms), (unsigned long) temp);

    return symbols[code];
    }
        symbols += count;  
        first += count;
        first <<= 1;
    }
    return -10;
}

template <typename READ_COL_TYPE, int LUT_LEN = 1, size_t in_buff_len = 4>
//__forceinline__
 __device__
int16_t decode_lut (input_stream<READ_COL_TYPE, in_buff_len>&  s, const int16_t* const   counts, const int16_t*   symbols, inflate_lut* s_lut, uint32_t lut_first = 0){

    //unsigned int len;
    //unsigned int code;
    //unsigned int count;
    uint32_t next32r = 0;
    s.template peek_n_bits<uint32_t>(32, &next32r);
    //if(threadIdx.x == 1)
   // printf("next: %lx\n",(unsigned long)next32r );

    next32r = __brev(next32r);


    uint32_t first = lut_first;
    #pragma no unroll
    for (uint8_t len = LUT_LEN; len <= MAXBITS; len++) {
        uint32_t code  = (next32r >> (32 - len)) - first;

        uint16_t count = counts[len];
    if (code < count)
    {
        uint32_t temp;
        s.template fetch_n_bits<uint32_t>(len, &temp);
        return symbols[code];
    }
        symbols += count;
        first += count;
        first <<= 1;
    }
    return -10;
}


template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
int32_t decode_len_lut (input_stream<READ_COL_TYPE, in_buff_len>&  s, const int16_t* const   counts, const int16_t*   syms, inflate_lut* s_lut, uint32_t next32){


    uint32_t next32r = __brev(next32);
    const int16_t*   symbols = syms;

    uint32_t first = s_lut -> first_slow_len;
    #pragma no unroll
    for (int len = LOG2LENLUT + 1; len <= MAXBITS; len++) {
        uint32_t code  = (next32r >> (32 - len)) - first;

        uint16_t count = counts[len];
    if (code < count)
    {
        //s.template fetch_n_bits<uint32_t>(len, &temp);
        int32_t sym = symbols[code];
        if(sym > 256){
            sym -= 257;
            int lext = g_lext[sym];
          //  printf("g_len: %i ", g_lens[sym]);
            sym  = 256 + g_lens[sym] + bfe(next32, len, lext);
           // printf("el: %lu\n", (unsigned long) bfe(next32, len, lext));
            len += lext;

        }
        s.skip_n_bits(len);

        return sym;
    }
        symbols += count;
        first += count;
        first <<= 1;
    }
    return -10;
}

template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
uint16_t decode_dist_lut (input_stream<READ_COL_TYPE, in_buff_len>&  s, const int16_t* const   counts, const int16_t*   syms, inflate_lut* s_lut,  uint32_t next32){


    // uint32_t next32r = 0;
    // s.template peek_n_bits<uint32_t>(32, &next32r);

    uint32_t next32r  = __brev(next32);
    const int16_t*   symbols = syms;

    uint32_t first = s_lut -> first_slow_dist;
    #pragma no unroll
    for (int len = LOG2DISTLUT + 1; len <= MAXBITS; len++) {
        uint32_t code  = (next32r >> (32 - len)) - first;

        uint16_t count = counts[len];
    if (code < count)
    {
        int dist = symbols[code];
        int dext = g_dext[dist];

        int off = (g_dists[dist] + bfe(next32, len, dext));
        len += dext;
        s.skip_n_bits(len);
        return (uint16_t)off;
    }
        symbols += count;
        first += count;
        first <<= 1;
    }
    return -10;
}




//Construct huffman tree
template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__ 
void construct(input_stream<READ_COL_TYPE, in_buff_len>& __restrict__ s, int16_t* const __restrict__ counts , int16_t* const  __restrict__ symbols, 
    const int16_t* const __restrict__ length,  int16_t* const __restrict__ offs, const int num_codes){


    int len;
    //#pragma  unroll
    for(len = 0; len <= MAXBITS; len++){
        counts[len] = 0;
    }

    #pragma no unroll
    for(len = 0; len < num_codes; len++){
        symbols[len] = 0;
        (counts[length[len]])++;
    }
  

    //int16_t offs[16];
    //offs[0] = 0;
    offs[1] = 0;

    #pragma no unroll
    for (len = 1; len < MAXBITS; len++){
        offs[len + 1] = offs[len] + counts[len];
    }

    #pragma no unroll
    for(int16_t symbol = 0; symbol < num_codes; symbol++){
         if (length[symbol] != 0){
            symbols[offs[length[symbol]]++] = symbol;
            //offs[length[symbol]]++;
        }
    }
        
}




/// permutation of code length codes
static const __device__ __constant__ uint8_t g_code_order[19 + 1] = {
  16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 0xff};


//construct huffman tree for dynamic huffman encoding block
template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__
 __device__
void decode_dynamic(input_stream<READ_COL_TYPE, in_buff_len>& s, dynamic_huffman* huff_tree_ptr, const uint32_t buff_idx,
    int16_t* const s_len, int16_t* const s_distcnt, int16_t* const s_distsym, int16_t* const s_off){



    uint16_t hlit;
    uint16_t hdist;
    uint16_t hclen;


    uint32_t head_temp;
    s.template fetch_n_bits<uint32_t>(14, &head_temp);
    hlit = (head_temp & (0x1F));
    head_temp >>= 5;
    hdist = (head_temp & (0x1F));
    head_temp >>= 5;
    hclen = (head_temp);

    hlit += 257;
    hdist += 1;
    hclen += 4;
    //int32_t lengths[MAXCODES];
    int index = 1;


    //check
    uint32_t temp;
    s.template fetch_n_bits<uint32_t>(12, &temp);
   // printf("idx: %llu\n", (unsigned long long)buff_idx);
    int16_t* lengths = huff_tree_ptr[buff_idx].treelen;

    for (index = 0; index < 4; index++) {
          lengths[g_code_order[index]] = (int16_t)(temp & 0x07);
            temp >>=3;
    }
    //#pragma no unroll
    for (index = 4; index < hclen; index++) {
        s.template fetch_n_bits<uint32_t>(3, &temp);
        lengths[g_code_order[index]] = (int16_t)temp;
    }

   // #pragma no unroll
    for (; index < 19; index++) {
        lengths[g_code_order[index]] = 0;
    }
       
    construct<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym, lengths, s_off, 19);


     index = 0;
     //symbol;
    while (index < hlit + hdist) {
        int32_t symbol =  decode<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym);
  
        //represent code lengths of 0 - 15
        if(symbol < 16){
            lengths[(index++)] = symbol;
        }

        else{
            int16_t len = 0;
            if(symbol == 16) {
                 len = lengths[index - 1];  // last length
                 s.template fetch_n_bits<int32_t>(2, &symbol);
                 symbol += 3;
            }
            else if(symbol == 17){
                s.template fetch_n_bits<int32_t>(3, &symbol);
                symbol += 3;
            }
            else if(symbol == 18) {
                s.template fetch_n_bits<int32_t>(7, &symbol);
                symbol += 11;
            }
       

            while(symbol-- > 0){
                lengths[index++] = len;
            }

        }
    }

    construct<READ_COL_TYPE, in_buff_len>(s, s_len, huff_tree_ptr[buff_idx].lensym, lengths, s_off, hlit);
    construct<READ_COL_TYPE, in_buff_len>(s, s_distcnt, s_distsym, (lengths + hlit), s_off, hdist);


    return;
}





template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__ 
__device__ 
void decode_symbol(input_stream<READ_COL_TYPE, in_buff_len>& s, queue<write_queue_ele>& mq, /*const dynamic_huffman* const huff_tree_ptr, unsigned buff_idx,*/
    const int16_t* const s_len, const int16_t* const lensym_ptr, const int16_t* const s_distcnt, const int16_t* const s_distsym) {


    while(1){

        uint16_t sym = decode<READ_COL_TYPE, in_buff_len>(s,  s_len,  lensym_ptr);

        if(sym <= 255) {
            write_queue_ele qe;
            qe = (uint32_t) sym;
            qe <<= 8;
            mq.enqueue(&qe);
        }

        //end of block
        else if(sym == 256) {
            break;
        }

        //lenght, need to parse
        else{

           // uint16_t extra_bits = g_lext[sym - 257];

            uint32_t extra_len  = 0;
            if( g_lext[sym - 257] != 0){
               s.template fetch_n_bits<uint32_t>( g_lext[sym - 257], &extra_len);
            }

            uint16_t len = extra_len + g_lens[sym - 257];
            //distance, 5bits
            uint16_t sym_dist = decode<READ_COL_TYPE, in_buff_len>(s,  s_distcnt, s_distsym);    
 
            uint32_t extra_len_dist = 0;
            if(g_dext[sym_dist] != 0){
                s.template fetch_n_bits<uint32_t>(g_dext[sym_dist], &extra_len_dist);
            }

            write_queue_ele qe;
            qe = (len << 16) | (extra_len_dist + g_dists[sym_dist]);
            qe <<= 8;
            qe |= 1;
            mq.enqueue(&qe);            
        }
    }

}



template <typename READ_COL_TYPE, size_t in_buff_len = 4>
//__forceinline__ 
__device__ 
void decode_symbol_lut(input_stream<READ_COL_TYPE, in_buff_len>& s, queue<write_queue_ele>& mq, /*const dynamic_huffman* const huff_tree_ptr, unsigned buff_idx,*/
    const int16_t* const s_len, const int16_t* const lensym_ptr, const int16_t* const s_distcnt, const int16_t* const s_distsym, inflate_lut* s_lut) {

    while(1){

       uint32_t next32 = 0;
        s.template peek_n_bits<uint32_t>(32, &next32);
       uint16_t sym = (s_lut -> len_lut)[next32 & ((1 << LOG2LENLUT) - 1)];
       if ((uint32_t)sym < (uint32_t)(0x1 << 15)) {

        uint32_t len = sym & 0x1f;
        int32_t temp;
         s.template fetch_n_bits<int32_t>(len, &temp);
        
        sym = (s_lut -> len_lut)[next32 & ((1 << LOG2LENLUT) - 1)];
        sym >>= 5;


      }


      else{
           sym = decode<READ_COL_TYPE, in_buff_len>(s,  s_len,  lensym_ptr);
       }

        if(sym <= 255) {
            write_queue_ele qe;
           
            qe = (uint32_t) sym;
            qe <<= 8;
            mq.enqueue(&qe);
        }

        //end of block
        else if(sym == 256) {
            break;
        }

        //lenght, need to parse

        else{

           // uint16_t extra_bits = g_lext[sym - 257];

            uint32_t extra_len  = 0;
            if( g_lext[sym - 257] != 0){
               s.template fetch_n_bits<uint32_t>( g_lext[sym - 257], &extra_len);
            }

            uint16_t len = extra_len + g_lens[sym - 257];
            //distance, 5bits
            uint16_t sym_dist;      

        s.template peek_n_bits<uint32_t>(32, &next32);

        int dist = (s_lut -> dist_lut)[next32 & ((1 << LOG2DISTLUT) - 1)];
            uint32_t extra_len_dist = 0;

        if(dist > 0){
        //if(false){
            
            uint32_t lut_len = dist & 0x1f;
          sym_dist = dist >> 5;
        int32_t temp = 0;
        s.template fetch_n_bits<int32_t>(lut_len, &temp);

        }
        else{
            sym_dist = decode<READ_COL_TYPE, in_buff_len>(s,  s_distcnt, s_distsym);    
        }

         if(g_dext[sym_dist] != 0){
                        s.template fetch_n_bits<uint32_t>(g_dext[sym_dist], &extra_len_dist);
                }


            write_queue_ele qe;
            qe = (len << 16) | (extra_len_dist + g_dists[sym_dist]);
            qe <<= 8;
            qe |= 0x1;
            mq.enqueue(&qe);            
        }
    }

}

template <typename READ_COL_TYPE, size_t in_buff_len = 4, size_t WRITE_COL_LEN>
//__forceinline__ 
__device__ 
void decode_symbol_dw(input_stream<READ_COL_TYPE, in_buff_len>& s, decompress_output<WRITE_COL_LEN>& out, /*const dynamic_huffman* const huff_tree_ptr, unsigned buff_idx,*/
    const int16_t* const s_len, const int16_t* const lensym_ptr, const int16_t* const s_distcnt, const int16_t* const s_distsym) {
    __shared__ MEMCPY_TYPE arr_shared[(SHARED_MEMCPY)];

    while(1){

        uint16_t sym = 0;
        if(threadIdx.x == 0) {sym = decode<READ_COL_TYPE, in_buff_len>(s,  s_len,  lensym_ptr);}
        sym = __shfl_sync(FULL_MASK, sym, 0);

        if(sym <= 255) {
            //writing
             if(threadIdx.x == 0) {out.write_literal(0, sym);
              //rintf("sym: %x\n", sym);
          }

            // write_queue_ele qe;
            // qe = (uint32_t) sym;
            // qe <<= 8;
            // mq.enqueue(&qe);
        }

        //end of block
        else if(sym == 256) {
            break;
        }

        //lenght, need to parse
        else{
            uint16_t len = 0;
            uint16_t off = 0;
            if(threadIdx.x == 0){
                uint32_t extra_len  = 0;
                if( g_lext[sym - 257] != 0){
                   s.template fetch_n_bits<uint32_t>( g_lext[sym - 257], &extra_len);
                }

                len = extra_len + g_lens[sym - 257];
                //distance, 5bits
                uint16_t  sym_dist = decode<READ_COL_TYPE, in_buff_len>(s,  s_distcnt, s_distsym);    
     
                uint32_t extra_len_dist = 0;
                if(g_dext[sym_dist] != 0){
                    s.template fetch_n_bits<uint32_t>(g_dext[sym_dist], &extra_len_dist);
                }
                off = extra_len_dist + g_dists[sym_dist];
            }


            //shfl sync
            len = __shfl_sync(FULL_MASK, len, 0);
            off = __shfl_sync(FULL_MASK, off, 0);     

            if (off < NMEMCPY) {
                // out.template col_memcpy_Nbyte_shared_onepack<32, MEMCPY_TYPE>(arr_shared, 0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);
                // out.template col_memcpy_Nbyte<32>(0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);
                out.template col_memcpy_div<32>(0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);
            }
            else if ((off < MEMCPY_SIZE_THRESH || len < MEMCPY_SIZE_THRESH)) {
                out.template col_memcpy_Nbyte_shared_onepass2<32, MEMCPY_TYPE>(&arr_shared[(threadIdx.y-1)*(32)], 0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);
            }
            else {
                // out.template col_memcpy_Nbyte<32>(0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);
                out.template col_memcpy_div<32>(0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);

            }

            // out.template col_memcpy_div<32>(0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);


            //writing       
        }
    }

}

template <typename READ_COL_TYPE, size_t in_buff_len = 4, size_t WRITE_COL_LEN>
//__forceinline__ 
__device__ 
void decode_symbol_dw_lut(input_stream<READ_COL_TYPE, in_buff_len>& s, decompress_output<WRITE_COL_LEN>& out, /*const dynamic_huffman* const huff_tree_ptr, unsigned buff_idx,*/
    const int16_t* const s_len, const int16_t* const lensym_ptr, const int16_t* const s_distcnt, const int16_t* const s_distsym, inflate_lut* s_lut) {
    __shared__ MEMCPY_TYPE arr_shared[(SHARED_MEMCPY)];

    while(1){
        uint32_t next32 = 0;
        uint32_t sym = 0;
        int32_t len_sym = 0;

         if(threadIdx.x == 0){
            s.template peek_n_bits<uint32_t>(32, &next32);
            len_sym = (s_lut -> len_lut)[next32 & ((1 << LOG2LENLUT) - 1)];
            uint32_t len = 0;
            if ((uint32_t)len_sym < (uint32_t)(0x100 << 5)) {
                len = len_sym & 0x1f;

                len_sym >>= 5;
               // printf("len sym: %x\n", len_sym);
                out.write_literal(0, len_sym);
            
                next32 >>= len;
                s.skip_n_bits(len);
                len_sym = (s_lut -> len_lut)[next32 & ((1 << LOG2LENLUT) - 1)];
            }

            if(len_sym > 0){
                len = len_sym & 0x1f;

                s.skip_n_bits(len);
                sym = ((len_sym >> 5) & 0x3ff) + ((next32 >> (len_sym >> 24)) & ((len_sym >> 16) & 0x1f));
            }

            else{
                sym = decode_len_lut<READ_COL_TYPE, in_buff_len>(s,  s_len,  (lensym_ptr) + s_lut -> index_slow_len, s_lut, (next32));
            }

        }            

        sym = __shfl_sync(FULL_MASK, sym, 0);

        if(sym <= 255) {
            //writing
             if(threadIdx.x == 0) {out.write_literal(0, sym);  }
        }

        //end of block
        else if(sym == 256) {
            break;
        }

        //lenght, need to parse
        else{
            uint16_t len = 0;
            uint16_t off = 0;
            if(threadIdx.x == 0){

            //  uint32_t extra_len  = 0;
            // if( g_lext[sym - 257] != 0){
            //    s.template fetch_n_bits<uint32_t>( g_lext[sym - 257], &extra_len);
            // }
            
            // len = extra_len + g_lens[sym - 257];
            // //distance, 5bits
                 uint16_t sym_dist = 0;     
               
                len = (sym & 0x0FFFF) - 256;
                

                next32 = 0;
                s.template peek_n_bits<uint32_t>(32, &next32);
                int dist = (s_lut -> dist_lut)[next32 & ((1 << LOG2DISTLUT) - 1)];
                uint32_t extra_len_dist = 0;


                if(dist > 0){                    
                    uint32_t lut_len = dist & 0x1f;
                    int dext = bfe(dist, 20, 5);
                    dist = bfe(dist, 5, 15);
                    int cur_off = (dist + bfe(next32, lut_len, dext));
                    lut_len += dext;

                    off = (uint16_t) cur_off;
                    s.skip_n_bits(lut_len);

                  // sym_dist = dist >> 5;
                  // int32_t temp = 0;
                  // s.template fetch_n_bits<int32_t>(lut_len, &temp);
                }
                else{
                    off = decode_dist_lut<READ_COL_TYPE, in_buff_len>(s,  s_distcnt, s_distsym  + s_lut -> index_slow_dist, s_lut, (next32));    
                }
     
                // if(g_dext[sym_dist] != 0){
                //     s.template fetch_n_bits<uint32_t>(g_dext[sym_dist], &extra_len_dist);
                // }
                // off = extra_len_dist + g_dists[sym_dist];

                              //  printf("len: %lu off: %lu\n", (unsigned long) len, (unsigned long) off);

            }


            //shfl sync
            len = __shfl_sync(FULL_MASK, len, 0);
            off = __shfl_sync(FULL_MASK, off, 0);     

            // out.template col_memcpy_div<32>(0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);

            if (off < NMEMCPY) {
                // out.template col_memcpy_Nbyte_shared_onepack<32, MEMCPY_TYPE>(arr_shared, 0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);
                // out.template col_memcpy_Nbyte<32>(0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);
                out.template col_memcpy_div<32>(0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);
            }
            else if ((off < MEMCPY_SIZE_THRESH || len < MEMCPY_SIZE_THRESH)) {
                out.template col_memcpy_Nbyte_shared_onepass2<32, MEMCPY_TYPE>(&arr_shared[(threadIdx.y-1)*(32)], 0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);
            }
            else {
                // out.template col_memcpy_Nbyte<32>(0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);
                out.template col_memcpy_div<32>(0, (uint32_t)len, (uint32_t)off, 0, FULL_MASK);

            }


            //writing       
        }
    }

}

template <typename READ_COL_TYPE, size_t in_buff_len, uint8_t NUM_SUBCHUNKS>
//__forceinline__ 
__device__
void decoder_warp(input_stream<READ_COL_TYPE, in_buff_len>& s,  queue<write_queue_ele>& mq, uint32_t col_len, uint8_t* out, dynamic_huffman* huff_tree_ptr,
    slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, int16_t* s_len, int16_t* s_distcnt, int16_t* s_distsym, int16_t* s_off, uint8_t active_chunks) {

    unsigned sm_id;

    uint8_t slot = 0;
    if(threadIdx.x == 0){
       sm_id = get_smid();
       slot = find_slot(sm_id, d_slot_struct);
    }

    slot = __shfl_sync(FULL_MASK, slot, 0);
    sm_id = __shfl_sync(FULL_MASK, sm_id, 0);

    if(threadIdx.x >= active_chunks){}

    else{

        uint8_t blast;
        uint32_t btype;

      
     s.template fetch_n_bits<uint32_t>(16, &btype);
     btype = 0;
        do{

        s.template fetch_n_bits<uint32_t>(3, &btype);

        
        blast =  (btype & 0x01);
        btype >>= 1;
        //fixed huffman
        if(btype == 1) {
            decode_symbol<READ_COL_TYPE, in_buff_len> (s, mq,  fixed_tree -> lencnt, fixed_tree -> lensym, fixed_tree -> distcnt, fixed_tree -> distsym);
        }
        //dyamic huffman
        else if (btype == 0){
            printf("uncomp\n");
             s.template align_bits();
             int32_t uncomp_len;
             s.template fetch_n_bits<int32_t>(16, &uncomp_len);
             s.template fetch_n_bits<uint32_t>(16, &btype);

            for(int32_t c = 0; c < uncomp_len; c++){
                s.template fetch_n_bits<uint32_t>(8, &btype);
            }
             
             break;
        }
        else{

            decode_dynamic<READ_COL_TYPE, in_buff_len>(s, huff_tree_ptr,  (uint32_t)((sm_id * 32 + slot) * 32 + threadIdx.x), s_len, s_distcnt, s_distsym, s_off);
           
        decode_symbol<READ_COL_TYPE, in_buff_len>(s, mq, 
                s_len, huff_tree_ptr[((sm_id * 32 + slot) * 32 + threadIdx.x)].lensym, s_distcnt, s_distsym);

        }

        }while(blast != 1);

    }

    __syncwarp(FULL_MASK);
    if(threadIdx.x == 0){
        release_slot(sm_id, slot, d_slot_struct);
    }

}



template <typename READ_COL_TYPE, size_t in_buff_len, uint8_t NUM_SUBCHUNKS>
//__forceinline__ 
__device__
void decoder_warp_shared(input_stream<READ_COL_TYPE, in_buff_len>& s,  queue<write_queue_ele>& mq, uint32_t col_len, uint8_t* out, dynamic_huffman* huff_tree_ptr,
    slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, int16_t* s_len, int16_t* s_distcnt, int16_t* s_distsym, int16_t* s_off, uint8_t active_chunks, s_huffman* s_tree) {

    if(threadIdx.x >= active_chunks){}

    else{

        uint8_t blast = 0;
        uint32_t btype = 0;
        s.template fetch_n_bits<uint32_t>(16, &btype);
        btype = 0;

        do{

          s.template fetch_n_bits<uint32_t>(3, &btype);
          blast =  (btype & 0x01);
          btype >>= 1;
        //fixed huffman
        if(btype == 1) {
            decode_symbol<READ_COL_TYPE, in_buff_len> (s, mq,  fixed_tree -> lencnt, fixed_tree -> lensym, fixed_tree -> distcnt, fixed_tree -> distsym);
        }
        //dyamic huffman
        else if (btype == 0){
        }
        else{
            decode_dynamic<READ_COL_TYPE, in_buff_len>(s, &(s_tree->dh) ,  0, s_len, s_distcnt, s_distsym, s_off);
            decode_symbol<READ_COL_TYPE, in_buff_len>(s, mq, s_len, (s_tree->dh).lensym, s_distcnt, s_distsym);
        }

        }while(blast != 1);

    }

    __syncwarp(FULL_MASK);

}

template <typename READ_COL_TYPE, size_t in_buff_len, uint8_t NUM_SUBCHUNKS>
__device__
void decoder_warp_lut(input_stream<READ_COL_TYPE, in_buff_len>& s,  queue<write_queue_ele>& mq, uint32_t col_len, uint8_t* out, dynamic_huffman* huff_tree_ptr,
    slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, int16_t* s_len, int16_t* s_distcnt, int16_t* s_distsym, int16_t* s_off, uint8_t active_chunks, s_huffman* s_tree, inflate_lut* s_lut) {

   
    uint32_t MASK = 0;
    if(NUM_SUBCHUNKS == 1)
        MASK = DECODE_MASK_1;
    else if(NUM_SUBCHUNKS == 2)
        MASK = DECODE_MASK_2;
    else if(NUM_SUBCHUNKS == 4)
        MASK = DECODE_MASK_4;
    else if(NUM_SUBCHUNKS == 8)
        MASK = DECODE_MASK_8;
    else if(NUM_SUBCHUNKS == 16)
        MASK = DECODE_MASK_16;


    MASK >>= (threadIdx.x % NUM_SUBCHUNKS);

    MASK = __brev(MASK);

    uint8_t blast = 0;
    uint32_t btype = 0;



    if(threadIdx.x < active_chunks)
        s.template fetch_n_bits<uint32_t>(16, &btype);
    btype = 1;

    do{

        if(threadIdx.x < active_chunks){
            s.template fetch_n_bits<uint32_t>(3, &btype);
        }

        btype = __shfl_sync(MASK, btype, threadIdx.x % NUM_SUBCHUNKS);
        blast =  (btype & 0x01);
        btype >>= 1;

         //fixed huffman
        if(btype == 1) {
            if(threadIdx.x < active_chunks)
                decode_symbol<READ_COL_TYPE, in_buff_len> (s, mq,  fixed_tree -> lencnt, fixed_tree -> lensym, fixed_tree -> distcnt, fixed_tree -> distsym);
        }
        //dyamic huffman
        else if (btype == 0){
            if(threadIdx.x < active_chunks)
                printf("uncomp\n");
       }
        else{
          if(threadIdx.x < active_chunks){
                decode_dynamic<READ_COL_TYPE, in_buff_len>(s, &(s_tree->dh) ,  0, s_len, s_distcnt, s_distsym, s_off);
            }
            __syncwarp(MASK);

            init_length_lut (s_lut,  s_len, (s_tree->dh).lensym, threadIdx.x / NUM_SUBCHUNKS, 32 / NUM_SUBCHUNKS);

            init_distance_lut (s_lut,  s_distcnt, s_distsym, threadIdx.x / NUM_SUBCHUNKS, 32 / NUM_SUBCHUNKS);

            __syncwarp(MASK);

            if(threadIdx.x < active_chunks){
                decode_symbol_lut<READ_COL_TYPE, in_buff_len>(s, mq, s_len, (s_tree->dh).lensym, s_distcnt, s_distsym, s_lut);
            }
            __syncwarp(MASK);

            //clear_length_lut(s_lut);

    }


    }while(blast != 1);

     __syncwarp(MASK);

        
}

template <typename READ_COL_TYPE, size_t in_buff_len, uint8_t NUM_SUBCHUNKS, size_t WRITE_COL_LEN >
//__forceinline__ 
__device__
void decoder_writer_warp_shared(input_stream<READ_COL_TYPE, in_buff_len>& s,  decompress_output<WRITE_COL_LEN>& out_d, uint32_t col_len, uint8_t* out, dynamic_huffman* huff_tree_ptr,
    slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, int16_t* s_len, int16_t* s_distcnt, int16_t* s_distsym, int16_t* s_off, uint8_t active_chunks, s_huffman* s_tree) {



        uint8_t blast = 0;
        uint32_t btype = 0;


        if(threadIdx.x == 0) {
             s.template fetch_n_bits<uint32_t>(16, &btype);
        }
        btype = 0;

        do{
            if(threadIdx.x == 0)    {s.template fetch_n_bits<uint32_t>(3, &btype);}
            btype =  __shfl_sync(FULL_MASK, btype, 0);
              blast =  (btype & 0x01);
              btype >>= 1;



            //fixed huffman
            if(btype == 1) {
                decode_symbol_dw<READ_COL_TYPE, in_buff_len, WRITE_COL_LEN> (s, out_d,  fixed_tree -> lencnt, fixed_tree -> lensym, fixed_tree -> distcnt, fixed_tree -> distsym);
            }
            //dyamic huffman
            else if (btype == 0){
            }
            else{
                if(threadIdx.x == 0){decode_dynamic<READ_COL_TYPE, in_buff_len>(s, &(s_tree->dh) ,  0, s_len, s_distcnt, s_distsym, s_off);}

                decode_symbol_dw<READ_COL_TYPE, in_buff_len, WRITE_COL_LEN>(s, out_d, s_len, (s_tree->dh).lensym, s_distcnt, s_distsym);
            }
        

        }while(blast != 1);

    

    __syncwarp(FULL_MASK);


    // uint8_t blast;
    // uint32_t btype;
    // if(threadIdx.x == 0) s.template fetch_n_bits<uint32_t>(16, &btype);
    // btype =  __shfl_sync(FULL_MASK, btype, 0);

    // do{
    //     if(threadIdx.x == 0) s.template fetch_n_bits<uint32_t>(16, &btype);
    //     btype =  __shfl_sync(FULL_MASK, btype, 0);
    //     blast =  (btype & 0x01);
    //     btype >>= 1;

    //     if(btype == 1) {
    //         decode_symbol_dw<READ_COL_TYPE, in_buff_len> (s, mq,  fixed_tree -> lencnt, fixed_tree -> lensym, fixed_tree -> distcnt, fixed_tree -> distsym);
    //     }
    //     //dyamic huffman
    //     else if (btype == 0){
    //     }
    //     else{
    //         if(threadIdx.x==0) decode_dynamic<READ_COL_TYPE, in_buff_len>(s, &(s_tree->dh) ,  0, s_len, s_distcnt, s_distsym, s_off);
           
    //         printf("dd start\n");
    //         decode_symbol_dw<READ_COL_TYPE, in_buff_len>(s, mq, s_len, (s_tree->dh).lensym, s_distcnt, s_distsym);
    //         __syncwarp();
    //         printf("dd done\n");

    //     }

    // }while(blast != 1);

    //__syncwarp(FULL_MASK);

}

template <typename READ_COL_TYPE, size_t in_buff_len, uint8_t NUM_SUBCHUNKS, size_t WRITE_COL_LEN >
//__forceinline__ 
__device__
void decoder_writer_warp_shared_lut(input_stream<READ_COL_TYPE, in_buff_len>& s,  decompress_output<WRITE_COL_LEN>& out_d, uint32_t col_len, uint8_t* out, dynamic_huffman* huff_tree_ptr,
    slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, int16_t* s_len, int16_t* s_distcnt, int16_t* s_distsym, int16_t* s_off, uint8_t active_chunks, s_huffman* s_tree,
    inflate_lut* s_lut) {



        uint8_t blast = 0;
        uint32_t btype = 0;


        if(threadIdx.x == 0) {
             s.template fetch_n_bits<uint32_t>(16, &btype);
        }
        btype = 0;

        do{
            if(threadIdx.x == 0)    {s.template fetch_n_bits<uint32_t>(3, &btype);}
            btype =  __shfl_sync(FULL_MASK, btype, 0);
              blast =  (btype & 0x01);
              btype >>= 1;



            //fixed huffman
            if(btype == 1) {
                __syncwarp();

                init_length_lut2 (s_lut,   fixed_tree -> lencnt, fixed_tree -> lensym, threadIdx.x, 32);
                init_distance_lut2 (s_lut,  fixed_tree -> distcnt, fixed_tree -> distsym, threadIdx.x, 32);

                __syncwarp();
                decode_symbol_dw_lut<READ_COL_TYPE, in_buff_len, WRITE_COL_LEN> (s, out_d,  fixed_tree -> lencnt, fixed_tree -> lensym, fixed_tree -> distcnt, fixed_tree -> distsym, s_lut);
           
                //decode_symbol_dw<READ_COL_TYPE, in_buff_len, WRITE_COL_LEN> (s, out_d,  fixed_tree -> lencnt, fixed_tree -> lensym, fixed_tree -> distcnt, fixed_tree -> distsym);

            }
            //dyamic huffman
            else if (btype == 0){
            }
            else{
                if(threadIdx.x == 0){decode_dynamic<READ_COL_TYPE, in_buff_len>(s, &(s_tree->dh) ,  0, s_len, s_distcnt, s_distsym, s_off);}

                __syncwarp();

                init_length_lut2 (s_lut,  s_len, (s_tree->dh).lensym, threadIdx.x, 32);
                init_distance_lut2 (s_lut,  s_distcnt, s_distsym, threadIdx.x, 32);

                __syncwarp();
                decode_symbol_dw_lut<READ_COL_TYPE, in_buff_len, WRITE_COL_LEN>(s, out_d, s_len, (s_tree->dh).lensym, s_distcnt, s_distsym, s_lut);
                __syncwarp();
            }
        

        }while(blast != 1);

    

    __syncwarp(FULL_MASK);

}


template <size_t WRITE_COL_LEN, uint8_t NUM_SUBCHUNKS>
__device__
void writer_warp(queue<write_queue_ele>& mq, decompress_output<WRITE_COL_LEN>& out, uint64_t CHUNK_SIZE, const uint8_t active_chunks ) {

    uint32_t done = 0;
    int t = threadIdx.x;
    while (!done) {

        bool deq = false;
        write_queue_ele v;
        if(t < active_chunks){
            mq.attempt_dequeue(&v, &deq);
        }
        uint32_t deq_mask = __ballot_sync(FULL_MASK, deq);
        uint32_t deq_count = __popc(deq_mask);


        for (size_t i = 0; i < deq_count; i++) {
            int32_t f = __ffs(deq_mask);
            write_queue_ele d = __shfl_sync(FULL_MASK, v, f-1);
            uint8_t t = d & 0x0FF;

            //pair
            if(t == 1){
                uint64_t len = d >> 24;
                uint64_t offset = (d >> 8) & 0x0000ffff;
                out.template col_memcpy_div<32>(f-1, (uint32_t)len, (uint32_t)offset, 0, FULL_MASK);
            }
            //literal
            else{
                uint8_t b = (d>>8) & 0x00FF;
                out.write_literal(f-1, b);
            }

            deq_mask >>= f;
            deq_mask <<= f;
            
        }
        __syncwarp();
        bool check = out.counter != (CHUNK_SIZE);
        if(threadIdx.x >= active_chunks ) check = false;
        done = __ballot_sync(FULL_MASK, check) == 0;
    } 

}




template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, uint16_t in_queue_size = 4, size_t out_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512>
__global__ void 
//__launch_bounds__ (96, 13)
//__launch_bounds__ (96, 21)
inflate(uint8_t* comp_ptr, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[NUM_SUBCHUNKS];

    __shared__ write_queue_ele out_queue_[NUM_SUBCHUNKS][out_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_t[NUM_SUBCHUNKS];

    __shared__ READ_COL_TYPE local_queue[NUM_SUBCHUNKS][local_queue_size];

    //MAXDCODES is 30
    __shared__ int16_t s_distsym[NUM_SUBCHUNKS][MAXDCODES];
    __shared__ int16_t s_off[NUM_SUBCHUNKS][16];
    __shared__ int16_t s_lencnt[NUM_SUBCHUNKS][16];
    __shared__ int16_t s_distcnt[NUM_SUBCHUNKS][16];

    //initialize heads and tails to be 0
    if(threadIdx.x < NUM_SUBCHUNKS){
        h[threadIdx.x] = 0;
        t[threadIdx.x] = 0;
        out_h[threadIdx.x] = 0;
        out_t[threadIdx.x] = 0;
    }

    __syncthreads();

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
       //printf("bid: %i actie chunks: %u\n", blockIdx.x, active_chunks);
    }

    int my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS;
    int my_queue = threadIdx.x % NUM_SUBCHUNKS;
    uint64_t col_len = (col_len_ptr[my_block_idx]);

    if (threadIdx.y == 0) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, in_queue_size);
        decompress_input<READ_COL_TYPE, COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
        reader_warp<READ_COL_TYPE, COMP_COL_TYPE, NUM_SUBCHUNKS>(d, in_queue, active_chunks);
    }

    else if (threadIdx.y == 1) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, in_queue_size);
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size);
        input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x < active_chunks);
        decoder_warp<READ_COL_TYPE, local_queue_size, NUM_SUBCHUNKS>(s, out_queue, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, s_lencnt[my_queue], s_distcnt[my_queue], s_distsym[my_queue], s_off[my_queue], active_chunks);
    }

    else {
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size);
        decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * blockIdx.x * NUM_SUBCHUNKS), CHUNK_SIZE);
        writer_warp<WRITE_COL_LEN, NUM_SUBCHUNKS>(out_queue, d, CHUNK_SIZE, active_chunks);

    }

    __syncthreads();


}

template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, uint16_t in_queue_size = 4, size_t out_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512>
__global__ void 
//__launch_bounds__ (96, 13)
__launch_bounds__ (96, 21)
inflate_shared(uint8_t* comp_ptr, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[NUM_SUBCHUNKS];

    __shared__ write_queue_ele out_queue_[NUM_SUBCHUNKS][out_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_t[NUM_SUBCHUNKS];

    __shared__ READ_COL_TYPE local_queue[NUM_SUBCHUNKS][local_queue_size];

    __shared__ s_huffman shared_tree [NUM_SUBCHUNKS];

    //initialize heads and tails to be 0
    if(threadIdx.x < NUM_SUBCHUNKS){
        h[threadIdx.x] = 0;
        t[threadIdx.x] = 0;
        out_h[threadIdx.x] = 0;
        out_t[threadIdx.x] = 0;
    }

    __syncthreads();

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
    }

    int my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS;
    int my_queue = threadIdx.x % NUM_SUBCHUNKS;
    //uint64_t col_len = (col_len_ptr[my_block_idx]);
    uint64_t col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

    if (threadIdx.y == 0) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, in_queue_size);
        decompress_input<READ_COL_TYPE, COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
        reader_warp<READ_COL_TYPE, COMP_COL_TYPE, NUM_SUBCHUNKS>(d, in_queue, active_chunks);
    }

    else if (threadIdx.y == 1) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, in_queue_size);
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size);
        input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x < active_chunks);
        decoder_warp_shared<READ_COL_TYPE, local_queue_size, NUM_SUBCHUNKS>(s, out_queue, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]));

    }

    else {
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size);
        decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * blockIdx.x * NUM_SUBCHUNKS), CHUNK_SIZE);
        writer_warp<WRITE_COL_LEN, NUM_SUBCHUNKS>(out_queue, d, CHUNK_SIZE, active_chunks);
    }

    __syncthreads();


}

template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, uint16_t in_queue_size = 4, size_t out_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512, bool shared_flag = true>
__launch_bounds__ (96, 21)
__global__ void 
inflate_lookup(uint8_t* comp_ptr, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 slot_struct* d_slot_struct, device_space ds, const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[NUM_SUBCHUNKS];

    __shared__ write_queue_ele out_queue_[NUM_SUBCHUNKS][out_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_t[NUM_SUBCHUNKS];

    __shared__ READ_COL_TYPE local_queue[NUM_SUBCHUNKS][local_queue_size];

    __shared__ s_huffman shared_tree [NUM_SUBCHUNKS];
    extern __shared__ inflate_lut test_lut [];

    


    //initialize heads and tails to be 0
    if(threadIdx.x < NUM_SUBCHUNKS){
        h[threadIdx.x] = 0;
        t[threadIdx.x] = 0;
        out_h[threadIdx.x] = 0;
        out_t[threadIdx.x] = 0;
    }

    __syncthreads();

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
    }




    int my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS;
    int my_queue = threadIdx.x % NUM_SUBCHUNKS;
    //int my_queue = threadIdx.x / active_reg;
    //uint64_t col_len = (col_len_ptr[my_block_idx]);
    uint64_t col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

    if (threadIdx.y == 0) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, in_queue_size);
        decompress_input<READ_COL_TYPE, COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
        reader_warp<READ_COL_TYPE, COMP_COL_TYPE, NUM_SUBCHUNKS>(d, in_queue, active_chunks);
    }

    else if (threadIdx.y == 1) {
    
       inflate_lut* s_lut = test_lut;
       unsigned sm_id;
        uint8_t slot = 0;
       if(shared_flag == false){
            if(threadIdx.x == 0){
                sm_id = get_smid();
                slot = find_slot(sm_id, d_slot_struct);
            }
            slot = __shfl_sync(FULL_MASK, slot, 0);
            sm_id = __shfl_sync(FULL_MASK, sm_id, 0);
            s_lut = &(ds.d_lut[((sm_id * 32 + slot) * 32 + my_queue)]);
       }

    
       queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, in_queue_size);
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size);
        input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x < active_chunks);
    
       decoder_warp_lut<READ_COL_TYPE, local_queue_size, NUM_SUBCHUNKS>(s, out_queue, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]), &(s_lut[my_queue]));


        __syncwarp(FULL_MASK);
         if(shared_flag == false){
            if(threadIdx.x == 0){
                    release_slot(sm_id, slot, d_slot_struct);
            }
       }
    }

    else {
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size);
        decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * blockIdx.x * NUM_SUBCHUNKS), CHUNK_SIZE);
        writer_warp<WRITE_COL_LEN, NUM_SUBCHUNKS>(out_queue, d, CHUNK_SIZE, active_chunks);
    }

    __syncthreads();
}

template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, uint16_t in_queue_size = 4, size_t out_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512>
__global__ void 
//__launch_bounds__ (96, 13)
__launch_bounds__ (96, 21)
inflate_volatile(uint8_t* comp_ptr, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[NUM_SUBCHUNKS];

    __shared__ write_queue_ele out_queue_[NUM_SUBCHUNKS][out_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_t[NUM_SUBCHUNKS];

    __shared__ READ_COL_TYPE local_queue[NUM_SUBCHUNKS][local_queue_size];
    __shared__ s_huffman shared_tree [NUM_SUBCHUNKS];

    //initialize heads and tails to be 0
    if(threadIdx.x < NUM_SUBCHUNKS){
        h[threadIdx.x] = 0;
        t[threadIdx.x] = 0;
        out_h[threadIdx.x] = 0;
        out_t[threadIdx.x] = 0;
    }

    __syncthreads();

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
    }

    int my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS;
    int my_queue = threadIdx.x % NUM_SUBCHUNKS;
    uint64_t col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

    if (threadIdx.y == 0) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, in_queue_size);
        decompress_input<READ_COL_TYPE, COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
        reader_warp<READ_COL_TYPE, COMP_COL_TYPE, NUM_SUBCHUNKS>(d, in_queue, active_chunks);
    }

    else if (threadIdx.y == 1) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, in_queue_size);
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size);
        input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x < active_chunks);
        decoder_warp_shared<READ_COL_TYPE, local_queue_size, NUM_SUBCHUNKS>(s, out_queue, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]));

    }

    else {
        //queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size, v_out_queue_[my_queue], v_out_h + my_queue, v_out_t + my_queue, true);
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size);
        decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * blockIdx.x * NUM_SUBCHUNKS), CHUNK_SIZE);
        writer_warp<WRITE_COL_LEN, NUM_SUBCHUNKS>(out_queue, d, CHUNK_SIZE, active_chunks);
    }

    __syncthreads();


}
template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, uint16_t in_queue_size = 4, size_t out_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512, bool shared_flag = true>
__global__ void 
__launch_bounds__ (96, 21)
inflate_volatile2(uint8_t* comp_ptr, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 slot_struct* d_slot_struct, device_space ds, const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[NUM_SUBCHUNKS];

    __shared__ write_queue_ele out_queue_[NUM_SUBCHUNKS][out_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> out_t[NUM_SUBCHUNKS];

    __shared__ READ_COL_TYPE local_queue[NUM_SUBCHUNKS][local_queue_size];

    __shared__ s_huffman shared_tree [NUM_SUBCHUNKS];
    extern __shared__ inflate_lut test_lut [];

    volatile __shared__ READ_COL_TYPE v_in_queue_[NUM_SUBCHUNKS][in_queue_size];
    volatile __shared__ uint8_t v_h[NUM_SUBCHUNKS];
    volatile __shared__ uint8_t v_t[NUM_SUBCHUNKS];

    volatile __shared__ write_queue_ele v_out_queue_[NUM_SUBCHUNKS][out_queue_size];
    volatile __shared__ uint8_t v_out_h[NUM_SUBCHUNKS];
    volatile __shared__ uint8_t v_out_t[NUM_SUBCHUNKS];



    //initialize heads and tails to be 0
    if(threadIdx.x < NUM_SUBCHUNKS){
        h[threadIdx.x] = 0;
        t[threadIdx.x] = 0;
        out_h[threadIdx.x] = 0;
        out_t[threadIdx.x] = 0;

        v_h[threadIdx.x] = 0;
        v_t[threadIdx.x] = 0;
        v_out_h[threadIdx.x] = 0;
        v_out_t[threadIdx.x] = 0;
    }

    __syncthreads();

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
    }




    int my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS;
    int my_queue = threadIdx.x % NUM_SUBCHUNKS;
    //int my_queue = threadIdx.x / active_reg;
    //uint64_t col_len = (col_len_ptr[my_block_idx]);
    uint64_t col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

    if (threadIdx.y == 0) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, in_queue_size, v_in_queue_[my_queue], v_h + my_queue, v_t + my_queue, true);
        decompress_input<READ_COL_TYPE, COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
        reader_warp<READ_COL_TYPE, COMP_COL_TYPE, NUM_SUBCHUNKS>(d, in_queue, active_chunks);
    }

    else if (threadIdx.y == 1) {
    
     inflate_lut* s_lut = test_lut;
     unsigned sm_id;
        uint8_t slot = 0;
     if(shared_flag == false){
        if(threadIdx.x == 0){
              sm_id = get_smid();
              slot = find_slot(sm_id, d_slot_struct);
        }
          slot = __shfl_sync(FULL_MASK, slot, 0);
          sm_id = __shfl_sync(FULL_MASK, sm_id, 0);
        s_lut = &(ds.d_lut[((sm_id * 32 + slot) * 32 + my_queue)]);
     }

  
     queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, in_queue_size,  v_in_queue_[my_queue],  v_h + my_queue, v_t + my_queue, true);
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size,  v_out_queue_[my_queue],  v_out_h + my_queue, v_out_t + my_queue, false);
        input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x < active_chunks);
  
     decoder_warp_lut<READ_COL_TYPE, local_queue_size, NUM_SUBCHUNKS>(s, out_queue, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]), &(s_lut[my_queue]));


      __syncwarp(FULL_MASK);
       if(shared_flag == false){
        if(threadIdx.x == 0){
                release_slot(sm_id, slot, d_slot_struct);
          }
     }
    }

    else {
        queue<write_queue_ele> out_queue(out_queue_[my_queue], out_h + my_queue, out_t + my_queue, out_queue_size,  v_out_queue_[my_queue],  v_out_h + my_queue, v_out_t + my_queue, false);
        decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * blockIdx.x * NUM_SUBCHUNKS), CHUNK_SIZE);
        writer_warp<WRITE_COL_LEN, NUM_SUBCHUNKS>(out_queue, d, CHUNK_SIZE, active_chunks);
    }

    __syncthreads();
}


template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, uint16_t in_queue_size = 4, size_t out_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512>
__global__ void 
//__launch_bounds__ (96, 13)
__launch_bounds__ (64, 32)
inflate_shared_dw(uint8_t* comp_ptr, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[NUM_SUBCHUNKS];


    __shared__ READ_COL_TYPE local_queue[NUM_SUBCHUNKS][local_queue_size];

    __shared__ s_huffman shared_tree [NUM_SUBCHUNKS];

    //initialize heads and tails to be 0
    if(threadIdx.x < NUM_SUBCHUNKS){
        h[threadIdx.x] = 0;
        t[threadIdx.x] = 0;
    }

    __syncthreads();

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
    }

    int my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS;
    int my_queue = threadIdx.x % NUM_SUBCHUNKS;
    //uint64_t col_len = (col_len_ptr[my_block_idx]);
    uint64_t col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

    if (threadIdx.y == 0) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, in_queue_size);
        decompress_input<READ_COL_TYPE, COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
        reader_warp<READ_COL_TYPE, COMP_COL_TYPE, NUM_SUBCHUNKS>(d, in_queue, active_chunks);
    }

    else if (threadIdx.y == 1) {
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, in_queue_size);
        decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * blockIdx.x * NUM_SUBCHUNKS), CHUNK_SIZE);
        input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x < active_chunks);
        decoder_writer_warp_shared<READ_COL_TYPE, local_queue_size, NUM_SUBCHUNKS, WRITE_COL_LEN>(s, d, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]));

    }

    __syncthreads();

}
template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, uint16_t in_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512>
__global__ void 
__launch_bounds__ (64, 32)
//__launch_bounds__ (96, 21)
//__launch_bounds__ (128, 16)

//__launch_bounds__ (160, 12)
inflate_shared_dw_2(uint8_t* comp_ptr, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[NUM_SUBCHUNKS];
    __shared__ READ_COL_TYPE local_queue[NUM_SUBCHUNKS][local_queue_size];
    __shared__ s_huffman shared_tree [NUM_SUBCHUNKS];

   // int bid = blockIdx.x * NUM_SUBCHUNKS;

    //initialize heads and tails to be 0
    if(threadIdx.x < NUM_SUBCHUNKS){
        h[threadIdx.x] = 0;
        t[threadIdx.x] = 0;
    }

    __syncthreads();

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
    }


    // int my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS;
    // int my_queue = threadIdx.x % NUM_SUBCHUNKS;
    // uint64_t col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

    //int my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS;

    // int my_block_idx = threadIdx.y == 0 ? (blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS) : (blockIdx.x * NUM_SUBCHUNKS + threadIdx.y -1) ;
    // int my_queue = threadIdx.y == 0 ? threadIdx.x % NUM_SUBCHUNKS : threadIdx.y - 1;
    // uint64_t col_len= (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

    int my_block_idx = 0;
    uint64_t col_len = 0;
    int my_queue = 0;

    if (threadIdx.y == 0) {
        my_queue =  threadIdx.x % NUM_SUBCHUNKS;
        my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS;
        col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, in_queue_size);

       // queue<READ_COL_TYPE> in_queue(in_queue_[threadIdx.x % NUM_SUBCHUNKS], h +  threadIdx.x % NUM_SUBCHUNKS , t +  threadIdx.x % NUM_SUBCHUNKS, in_queue_size);
        decompress_input<READ_COL_TYPE, COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
        reader_warp<READ_COL_TYPE, COMP_COL_TYPE, NUM_SUBCHUNKS>(d, in_queue, active_chunks);
    }

    else{
       
        my_queue = (threadIdx.y - 1);
        my_block_idx =  (blockIdx.x * NUM_SUBCHUNKS + threadIdx.y -1);
        col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, in_queue_size);

        decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * my_block_idx ), CHUNK_SIZE);
        input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x == 0);

       decoder_writer_warp_shared<READ_COL_TYPE, local_queue_size, 1, WRITE_COL_LEN>(s, d, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]));
    }


    // else if (threadIdx.y == 1) {
    //     my_block_idx = bid;
    //     my_queue = 0;
    //     col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

    //     queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, in_queue_size);
    //     decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * bid ), CHUNK_SIZE);
    //     input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x == 0);
    //     decoder_writer_warp_shared<READ_COL_TYPE, local_queue_size, 1, WRITE_COL_LEN>(s, d, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]));
    // }

    // else if (threadIdx.y == 2) {

    //     my_block_idx = bid+1;
    //     my_queue = 1;
    //     col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);
    //     queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, in_queue_size);
    //     decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * (bid+1) ), CHUNK_SIZE);
    //     input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x == 0);
    //     decoder_writer_warp_shared<READ_COL_TYPE, local_queue_size, 1, WRITE_COL_LEN>(s, d, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]));

    // }

    //__syncthreads();

}

template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, uint16_t in_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512>
__global__ void 
//__launch_bounds__ (64, 32)
//__launch_bounds__ (96, 21)
//__launch_bounds__ (160, 12)
__launch_bounds__ (288, 7)

inflate_shared_dw_2_lut(uint8_t* comp_ptr, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[NUM_SUBCHUNKS];
    __shared__ READ_COL_TYPE local_queue[NUM_SUBCHUNKS][local_queue_size];
    __shared__ s_huffman shared_tree [NUM_SUBCHUNKS];

    __shared__ inflate_lut test_lut [NUM_SUBCHUNKS];
   
   // int bid = blockIdx.x * NUM_SUBCHUNKS;

    //initialize heads and tails to be 0
    if(threadIdx.x < NUM_SUBCHUNKS){
        h[threadIdx.x] = 0;
        t[threadIdx.x] = 0;
    }

    __syncthreads();

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
    }

    int my_block_idx = 0;
    uint64_t col_len = 0;
    int my_queue = 0;

    if (threadIdx.y == 0) {
        my_queue =  threadIdx.x % NUM_SUBCHUNKS;
        my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x % NUM_SUBCHUNKS;
        col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, in_queue_size);

       // queue<READ_COL_TYPE> in_queue(in_queue_[threadIdx.x % NUM_SUBCHUNKS], h +  threadIdx.x % NUM_SUBCHUNKS , t +  threadIdx.x % NUM_SUBCHUNKS, in_queue_size);
        decompress_input<READ_COL_TYPE, COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
        reader_warp<READ_COL_TYPE, COMP_COL_TYPE, NUM_SUBCHUNKS>(d, in_queue, active_chunks);
    }

    else{
    
        my_queue = (threadIdx.y - 1);
        my_block_idx =  (blockIdx.x * NUM_SUBCHUNKS + threadIdx.y -1);
        col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, in_queue_size);

        decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * my_block_idx ), CHUNK_SIZE);
        input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x == 0);

        decoder_writer_warp_shared_lut<READ_COL_TYPE, local_queue_size, 1, WRITE_COL_LEN>(s, d, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]), test_lut + my_queue);
    }

}

template <typename READ_COL_TYPE, typename COMP_COL_TYPE,  uint8_t NUM_SUBCHUNKS, uint16_t in_queue_size = 4, size_t local_queue_size = 4,  size_t WRITE_COL_LEN = 512>
__global__ void 
//__launch_bounds__ (64, 32)
__launch_bounds__ (96, 21)
//__launch_bounds__ (128, 16)
//__launch_bounds__ (160, 12)

inflate_shared_dw_2_lut_sub(uint8_t* comp_ptr, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint8_t*out, dynamic_huffman* huff_tree_ptr,
 slot_struct* d_slot_struct, const fix_huffman* const fixed_tree, uint64_t CHUNK_SIZE, uint64_t num_chunks) {
    
    __shared__ READ_COL_TYPE in_queue_[NUM_SUBCHUNKS][in_queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[NUM_SUBCHUNKS];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[NUM_SUBCHUNKS];
    __shared__ READ_COL_TYPE local_queue[NUM_SUBCHUNKS][local_queue_size];
    __shared__ s_huffman shared_tree [NUM_SUBCHUNKS];

    __shared__ inflate_lut test_lut [NUM_SUBCHUNKS];
   
   // int bid = blockIdx.x * NUM_SUBCHUNKS;

    //initialize heads and tails to be 0
    if(threadIdx.x < NUM_SUBCHUNKS){
        h[threadIdx.x] = 0;
        t[threadIdx.x] = 0;
    }

    __syncthreads();

    uint8_t active_chunks = NUM_SUBCHUNKS;
    if((blockIdx.x+1) * NUM_SUBCHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_SUBCHUNKS;
    }

    int my_block_idx = 0;
    uint64_t col_len = 0;
    int my_queue = 0;

    if (threadIdx.y == 0) {
        my_queue =  threadIdx.x / (32/NUM_SUBCHUNKS);
        my_block_idx = blockIdx.x * NUM_SUBCHUNKS + threadIdx.x / (32/NUM_SUBCHUNKS);
        col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);
        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, in_queue_size);

       // queue<READ_COL_TYPE> in_queue(in_queue_[threadIdx.x % NUM_SUBCHUNKS], h +  threadIdx.x % NUM_SUBCHUNKS , t +  threadIdx.x % NUM_SUBCHUNKS, in_queue_size);
        decompress_input<READ_COL_TYPE, COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
        sub_reader_warp<READ_COL_TYPE, COMP_COL_TYPE, NUM_SUBCHUNKS>(d, in_queue, active_chunks);
        //reader_warp<READ_COL_TYPE, COMP_COL_TYPE, NUM_SUBCHUNKS>(d, in_queue, active_chunks);

    }

    else{
       
        my_queue = (threadIdx.y - 1);
        my_block_idx =  (blockIdx.x * NUM_SUBCHUNKS + threadIdx.y -1);
        col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

        queue<READ_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, in_queue_size);

        decompress_output<WRITE_COL_LEN> d((out + CHUNK_SIZE * my_block_idx ), CHUNK_SIZE);
        input_stream<READ_COL_TYPE, local_queue_size> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x == 0);

        decoder_writer_warp_shared_lut<READ_COL_TYPE, local_queue_size, 1, WRITE_COL_LEN>(s, d, (uint32_t) col_len, out, huff_tree_ptr, d_slot_struct, fixed_tree, shared_tree[my_queue].lencnt ,  shared_tree[my_queue].distcnt, shared_tree[my_queue].distsym , shared_tree[my_queue].off, active_chunks, &(shared_tree[my_queue]), test_lut + my_queue);
    }

}

namespace deflate {

template <typename READ_COL_TYPE, size_t WRITE_COL_LEN, uint16_t queue_depth, uint8_t NUM_SUBCHUNKS, bool lut_flag,  bool shared_tree_flag>
 __host__ void decompress_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes,
  uint64_t* col_len_f, const uint64_t col_n_bytes, uint64_t* blk_offset_f, const uint64_t blk_n_bytes, uint64_t chunk_size) {

    uint64_t num_blk = ((uint64_t) blk_n_bytes / sizeof(uint64_t)) - 2;

    uint64_t data_size = blk_offset_f[0];


    uint8_t* d_in;
    uint64_t* d_col_len;
    uint64_t* d_blk_offset;

    uint64_t* d_comp_histo;
    uint64_t* d_tree_histo;

    device_space d_space;

    int num_sm = 108;
   
    cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
    cuda_err_chk(cudaMalloc(&d_col_len,col_n_bytes));
    cuda_err_chk(cudaMalloc(&d_blk_offset, blk_n_bytes));


    if(shared_tree_flag == false){
            cuda_err_chk(cudaMalloc(&(d_space.d_lut), sizeof(inflate_lut) * 32 * num_sm * 32));
    }

    cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_col_len, col_len_f, col_n_bytes, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_blk_offset, blk_offset_f+1, blk_n_bytes, cudaMemcpyHostToDevice));



    uint64_t out_bytes = chunk_size * num_blk;
    std::cout << (int)NUM_SUBCHUNKS << "\t" << chunk_size << "\t" << WRITE_COL_LEN << "\t" << queue_depth << "\t"  << in_n_bytes << "\t" << blk_n_bytes + col_n_bytes;
    uint8_t* d_out;
    *out_n_bytes = data_size;
    cuda_err_chk(cudaMalloc(&d_out, out_bytes));


    dynamic_huffman* d_tree;
    cuda_err_chk(cudaMalloc(&d_tree, sizeof(dynamic_huffman) * 32*num_sm*32));

    slot_struct* d_slot_struct;
    cuda_err_chk(cudaMalloc(&d_slot_struct, sizeof(slot_struct) * num_sm));
    cuda_err_chk(cudaMemset(d_slot_struct, 0, num_sm * sizeof(slot_struct)));



    uint16_t fix_lencnt[16] = {0,0,0,0,0,0,0,24,152,112,0,0,0,0,0,0};
    uint16_t fix_lensym[FIXLCODES] =
    { 256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,
        36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,
        69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,
        102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,
        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,280,281,282,283,284,285,286,287,
        144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,
        169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,
        194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,
        219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,
        244,245,246,247,248,249,250,251,252,253,254,255};
     uint16_t fix_distcnt[MAXBITS + 1] = 
    {0,0,0,0,0,30,0,0,0,0,0,0,0,0,0,0};
    uint16_t fix_distsym[MAXDCODES] = 
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29};

    fix_huffman f_tree;
    fix_huffman* d_f_tree;

    cuda_err_chk(cudaMalloc(&d_f_tree, sizeof(fix_huffman)));

    //f_tree.lencnt = fix_lencnt;


    memcpy(f_tree.lencnt, fix_lencnt, sizeof(uint16_t)*16);
    memcpy(f_tree.lensym, fix_lensym, sizeof(uint16_t)*FIXLCODES);
    memcpy(f_tree.distcnt, fix_distcnt, sizeof(uint16_t)*(MAXBITS + 1));
    memcpy(f_tree.distsym, fix_distsym, sizeof(uint16_t)*MAXDCODES);

    cuda_err_chk(cudaMemcpy(d_f_tree, &f_tree, sizeof(fix_huffman), cudaMemcpyHostToDevice));


    dim3 blockD(32,3,1);
    dim3 blockD2(32,2,1);

    //num_blk = 1;
    uint64_t num_tblk = (num_blk + NUM_SUBCHUNKS - 1) / NUM_SUBCHUNKS;

    dim3 gridD(num_tblk,1,1);

    cudaDeviceSynchronize();

    size_t shared_mem_size = shared_tree_flag ? NUM_SUBCHUNKS * sizeof(inflate_lut) : 0;

    std::chrono::high_resolution_clock::time_point kernel_start = std::chrono::high_resolution_clock::now();

    if(lut_flag){

        //inflate_volatile<uint64_t, READ_COL_TYPE, NUM_SUBCHUNKS, queue_depth , queue_depth, 4, WRITE_COL_LEN, shared_tree_flag> <<<gridD,blockD,shared_mem_size>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_space, d_f_tree, chunk_size, num_blk);
        //inflate_volatile<uint64_t, READ_COL_TYPE, NUM_SUBCHUNKS, queue_depth , queue_depth, 4, WRITE_COL_LEN> <<<gridD,blockD>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);
        //inflate_volatile2<uint64_t, READ_COL_TYPE, NUM_SUBCHUNKS, queue_depth , queue_depth, 4, WRITE_COL_LEN, shared_tree_flag> <<<gridD,blockD,shared_mem_size>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_space, d_f_tree, chunk_size, num_blk);

        inflate_lookup<uint32_t, READ_COL_TYPE, NUM_SUBCHUNKS, queue_depth , queue_depth, 4, WRITE_COL_LEN, shared_tree_flag> <<<gridD,blockD,shared_mem_size>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_space, d_f_tree, chunk_size, num_blk);
        

    }

    else if(shared_tree_flag){
        dim3 gridD2(num_tblk/2,1,1);
        dim3 gridD4(num_tblk/4,1,1);
        dim3 blockD5(32,5,1);
                dim3 blockD4(32,4,1);
                dim3 blockD9(32,9,1);
                        dim3 gridD8(num_tblk/8,1,1);



        kernel_start = std::chrono::high_resolution_clock::now();
       // inflate_shared<uint64_t, READ_COL_TYPE, NUM_SUBCHUNKS, queue_depth , queue_depth, 4, WRITE_COL_LEN> <<<gridD,blockD>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);
        
        //inflate_shared_dw<uint64_t, READ_COL_TYPE, NUM_SUBCHUNKS, queue_depth , queue_depth, 4, WRITE_COL_LEN> <<<gridD,blockD2>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);
       //inflate_shared_dw_2<uint64_t, READ_COL_TYPE, 1, queue_depth , 4, WRITE_COL_LEN> <<<gridD,blockD2>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);
             
             //   inflate_shared_dw_2<uint64_t, READ_COL_TYPE, 1, queue_depth , 4, WRITE_COL_LEN> <<<1,blockD2>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);

        //inflate_shared_dw_2<uint64_t, READ_COL_TYPE, 2, queue_depth , 4, WRITE_COL_LEN> <<<gridD2,blockD>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);
        
       // inflate_shared_dw_2<uint64_t, READ_COL_TYPE, 4, queue_depth , 4, WRITE_COL_LEN> <<<gridD4,blockD5>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);
        


        inflate_shared_dw_2_lut<uint32_t, READ_COL_TYPE, 8, queue_depth , 8, WRITE_COL_LEN> <<<gridD8,blockD9>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);


       //inflate_shared_dw_2_lut<uint64_t, READ_COL_TYPE, 4, queue_depth , 4, WRITE_COL_LEN> <<<gridD4,blockD5>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);

        //inflate_shared_dw_2_lut<uint64_t, READ_COL_TYPE, 2, queue_depth , 4, WRITE_COL_LEN> <<<gridD2,blockD>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);
     
      // inflate_shared_dw_2_lut<uint64_t, READ_COL_TYPE, 1, queue_depth , 4, WRITE_COL_LEN> <<<gridD,blockD2>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);
       



      // inflate_shared_dw_2_lut_sub<uint64_t, READ_COL_TYPE, 2, queue_depth , 4, WRITE_COL_LEN> <<<gridD2,blockD>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);
      
      //  inflate_shared_dw_2_lut_sub<uint64_t, READ_COL_TYPE, 2, queue_depth , 4, WRITE_COL_LEN> <<<1,blockD>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct, d_f_tree, chunk_size, num_blk);


    }
    else{
        inflate<uint32_t, READ_COL_TYPE, NUM_SUBCHUNKS, queue_depth , queue_depth, 4, WRITE_COL_LEN> <<<gridD,blockD>>> (d_in, d_col_len, d_blk_offset, d_out, d_tree, d_slot_struct,  d_f_tree, chunk_size, num_blk);
    }

    cuda_err_chk(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(kernel_end - kernel_start);
    std::cout << "\t" << total.count() << std::endl;


    cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
      {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
      }


    *out = new uint8_t[data_size];
    cuda_err_chk(cudaMemcpy((*out), d_out, data_size, cudaMemcpyDeviceToHost));
    cuda_err_chk(cudaFree(d_out));
    cuda_err_chk(cudaFree(d_in));
    cuda_err_chk(cudaFree(d_col_len));
    cuda_err_chk(cudaFree(d_blk_offset));
 }


}

//#endif // __ZLIB_H__

