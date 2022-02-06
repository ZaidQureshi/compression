
#include <common.h>
#include <fstream>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <simt/atomic>
#include <iostream>

// #include <common_warp.h>

#define BUFF_LEN 2

#define UNCOMP 0
#define STATIC 1
#define DYNAMIC 2
#define FULL_MASK 0xFFFFFFFF

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


template<typename T>
struct  decomp_write_queue_ele{
    T data;
    int header;
    bool done;
};

struct write_ele{
    int64_t value;
    int16_t delta;
    int16_t run;

    __device__
    write_ele(){
        value = 0;
        delta = 0;
        run = 0;
    }

    //__device__
    // write_ele& operator =(const write_ele& a){
    //     value = a.value;
    //     delta = a.delta;
    //     run = a.run;
    // }
};



template <typename DATA_TYPE>
struct decompress_output {

    DATA_TYPE* out_ptr;
    //uint64_t offset;
    uint32_t counter;

    __device__
    decompress_output(uint8_t* ptr):
        out_ptr((DATA_TYPE*)ptr) {
        counter = 0;
    }


    __device__
    void write_value(uint8_t idx, uint64_t value){
        if(threadIdx.x == idx){
            out_ptr[counter] = (DATA_TYPE) value;
            counter++;
        }
    }

    __device__
    void write_run(uint8_t idx, int64_t value, int16_t delta, int16_t run){
        
        uint64_t ptr = (uint64_t)out_ptr;
        ptr = __shfl_sync(FULL_MASK, ptr, idx);
        DATA_TYPE* out_buf = (DATA_TYPE*) ptr;

        uint32_t idx_count = __shfl_sync(FULL_MASK, counter, idx);

        idx_count += threadIdx.x;

        #pragma unroll
        for(uint64_t i = threadIdx.x; i < run; i += 32, idx_count += 32){    
            int64_t out_ele = value + static_cast<int64_t>(i) * delta;
            out_buf[idx_count] =  static_cast<DATA_TYPE>(out_ele);

        }

        if(threadIdx.x == idx) counter += run;
    }

};


template < typename COMP_COL_TYPE>
//__forceinline__ 
__device__
void reader_warp(decompress_input< COMP_COL_TYPE>& in, queue<COMP_COL_TYPE>& rq) {
    while (true) {
        COMP_COL_TYPE v;
        int8_t rc = in.comp_read_data(FULL_MASK, &v);
        //int8_t rc = comp_read_data2(FULL_MASK, &v, in);

        if (rc == -1)
            break;
        else if (rc > 0){

            rq.enqueue(&(v));
            //comp_enqueue<COMP_COL_TYPE, READ_COL_TYPE>(&v, &rq);
        }
    }
}


template <typename COMP_COL_TYPE, uint8_t NUM_SUBCHUNKS>
//__forceinline__ 
__device__
void reader_warp_orig(decompress_input< COMP_COL_TYPE>& in, queue<COMP_COL_TYPE>& rq, uint8_t active_chunks) {
    //iterate number of chunks for the single reader warp
   int t = 0;
   while(true){
        bool done = true;
        for(uint8_t cur_chunk = 0; cur_chunk < active_chunks; cur_chunk++){
            COMP_COL_TYPE v;
            uint8_t rc = comp_read_data_seq<COMP_COL_TYPE>(FULL_MASK, &v, in, cur_chunk);
            if(rc != 0)
                done = false;
            
            rq.warp_enqueue(&v, cur_chunk, rc);

        }
        __syncwarp();
        if(done)
            break;
    }
}

template <typename COMP_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
//__forceinline__ 
__device__
void decoder_warp(input_stream<COMP_COL_TYPE, in_buff_len>& s,  decompress_output<DATA_TYPE>& out, uint64_t CHUNK_SIZE, DATA_TYPE* out_buf, int COMP_COL_LEN) {

    int test_idx = 40;

    uint32_t input_data_out_size = 0;
    uint64_t num_iterations = (CHUNK_SIZE / 32)  ;


    uint64_t words_in_line = COMP_COL_LEN / sizeof(DATA_TYPE);
    uint64_t out_offset = words_in_line * threadIdx.x;
    uint64_t c= 0;


    while (input_data_out_size < num_iterations) {
    
      //need to read a header
      int32_t temp_byte = 0;
      s.template fetch_n_bits<int32_t>(8, &temp_byte);

      int8_t head_byte = (int8_t)(temp_byte & 0x00FF);


      //literals
      if(head_byte < 0){
        uint64_t remaining = static_cast<uint64_t>(-head_byte);
            //if(threadIdx.x == test_idx && blockIdx.x == 1 ) printf("num_iterations: %llu remaining: %x\n", num_iterations, remaining);

        for(uint64_t i = 0; i < remaining; ++i){

            DATA_TYPE value = 0;
            int64_t offset = 0;

            
            //read var-int value
            bool read_next = true;
            while(read_next){
                temp_byte = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_byte);
                int8_t in_data = 0;
                in_data = (int8_t) (in_data | (temp_byte & 0x00FF));

                if(in_data >= 0){
                    value |= (static_cast<DATA_TYPE>(in_data) << offset);
                    read_next = false;
                }
                else{
                    value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset); 
                    offset += 7;
                }
            }

            decomp_write_queue_ele<DATA_TYPE> qe;
            qe.data = value;

            out_buf[out_offset] = value;

            //if(threadIdx.x == 0 && blockIdx.x == 0) printf(" value: %lx\n", (unsigned long) value);

            c++;
            if(c == words_in_line){
                out_offset += words_in_line * 31;
                c=0;
            }
       
                out_offset++;
            



           // mq.enqueue(&qe);
           //(threadIdx.x == 0 && blockIdx.x == 0 && input_data_out_size <= 20) printf("out: %c\n", value);

            input_data_out_size+=sizeof(DATA_TYPE);

        }

      }
      //compresssed data
      else{
            uint64_t remaining = static_cast<uint64_t>(head_byte);

            temp_byte = 0;
            s.template fetch_n_bits<int32_t>(8, &temp_byte);
            int8_t delta = (int8_t) (temp_byte & 0x00FF);

            DATA_TYPE value = 0;
            int64_t offset = 0;

            
            int32_t in_data;

            while(1){
                temp_byte = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_byte);

                int8_t in_data =  (int8_t) (temp_byte & 0x00FF);

                if(in_data >= 0){
                                   // if(threadIdx.x == test_idx && blockIdx.x == 0) printf("data4: %x\n", in_data);

                    value |= (static_cast<DATA_TYPE>(in_data) << offset);
                    break;
                }
                else{
                                  // if(threadIdx.x == test_idx && blockIdx.x == 0) printf("data5: %x\n", in_data);

                    value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset); 
                    offset += 7;
                }
            }

         //   if(threadIdx.x == 0 && blockIdx.x == 0) printf("comp value: %lx\n", (unsigned long) value);

            //decoding the compresssed stream
            for(uint64_t i = 0; i < remaining + 3; ++i){
                int64_t out_ele = value + static_cast<int64_t>(i) * delta;
                //write out_ele 

                //temp_write_word = temp_write_word | (static_cast<READ_T>(out_ele) << (temp_word_count * sizeof(INPUT_T) * 8));

                decomp_write_queue_ele<DATA_TYPE> qe;
                qe.data = static_cast<DATA_TYPE>(out_ele); 


                out_buf[out_offset] =  static_cast<DATA_TYPE>(out_ele);
                c++;
                 if(c == words_in_line){
                    out_offset += words_in_line * 31;
                    c=0;
                }
                    
                        out_offset++;
                                 
                //mq.enqueue(&qe);

                 input_data_out_size+=sizeof(DATA_TYPE);

            }


        }

    }


   // printf("yeah done!! bid:%i tid: %i\n", blockIdx.x, threadIdx.x);



}

//only one thread in a warp is decoding 
template <typename COMP_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
__device__
void decoder_warp_orig_dw(input_stream<COMP_COL_TYPE, in_buff_len>& s, uint64_t CHUNK_SIZE, DATA_TYPE* out_buf) {

    uint32_t input_data_out_size = 0;
    uint64_t out_offset = 0;
    
    while (input_data_out_size < CHUNK_SIZE) {
    
      //need to read a header
      int32_t temp_byte = 0;
      if(threadIdx.x == 0) s.template fetch_n_bits<int32_t>(8, &temp_byte);

      int8_t head_byte = (int8_t)(temp_byte & 0x00FF);
      head_byte = __shfl_sync(FULL_MASK, head_byte, 0);

      //literals
      if(head_byte < 0){
        uint64_t remaining = static_cast<uint64_t>(-head_byte);

        for(uint64_t i = 0; i < remaining; ++i){

            DATA_TYPE value = 0;
            int64_t offset = 0;

            
            //read var-int value
            bool read_next = true & (threadIdx.x == 0);
            while(read_next){
                temp_byte = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_byte);
                int8_t in_data = 0;
                in_data = (int8_t) (in_data | (temp_byte & 0x00FF));

                if(in_data >= 0){
                    value |= (static_cast<DATA_TYPE>(in_data) << offset);
                    read_next = false;
                }
                else{
                    value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset); 
                    offset += 7;
                }
            }

            if(threadIdx.x == 0) out_buf[out_offset] = value; 
            
            out_offset++;
            input_data_out_size+=sizeof(DATA_TYPE);

        }

      }
      //compresssed data
      else{
            uint64_t remaining = static_cast<uint64_t>(head_byte);

            temp_byte = 0;
            if(threadIdx.x == 0) s.template fetch_n_bits<int32_t>(8, &temp_byte);
            int8_t delta = (int8_t) (temp_byte & 0x00FF);
            delta = __shfl_sync(FULL_MASK, delta, 0);

            DATA_TYPE value = 0;
            int64_t offset = 0;

            
            int32_t in_data;

            while(threadIdx.x == 0){
                temp_byte = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_byte);

                int8_t in_data =  (int8_t) (temp_byte & 0x00FF);

                if(in_data >= 0){

                    value |= (static_cast<DATA_TYPE>(in_data) << offset);
                    break;
                }
                else{

                    value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset); 
                    offset += 7;
                }
            }

            value = __shfl_sync(FULL_MASK, value, 0);


            uint64_t next_out_offset = out_offset + remaining + 3;
            out_offset += min((uint64_t)threadIdx.x, remaining + 3);


            //decoding the compresssed stream
            for(uint64_t i = threadIdx.x; i < remaining + 3; i += 32, out_offset += 32){
                
                int64_t out_ele = value + static_cast<int64_t>(i) * delta;
                out_buf[out_offset] =  static_cast<DATA_TYPE>(out_ele);

            }

            out_offset = next_out_offset;
            input_data_out_size+= (sizeof(DATA_TYPE) * (remaining + 3));


        }

    }


}

//only one thread in a warp is decoding 
template <typename COMP_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
__device__
void decoder_warp_orig_dw_multi(input_stream<COMP_COL_TYPE, in_buff_len>& s, uint64_t CHUNK_SIZE, DATA_TYPE* out_buf) {

    uint32_t input_data_out_size = 0;
    uint64_t out_offset = 0;
    
    while (input_data_out_size < CHUNK_SIZE) {
    
      //need to read a header
      int32_t temp_byte = 0;
      s.template fetch_n_bits<int32_t>(8, &temp_byte);

      int8_t head_byte = (int8_t)(temp_byte & 0x00FF);
      //head_byte = __shfl_sync(FULL_MASK, head_byte, 0);

      //literals
      if(head_byte < 0){
        uint64_t remaining = static_cast<uint64_t>(-head_byte);

        for(uint64_t i = 0; i < remaining; ++i){

            DATA_TYPE value = 0;
            int64_t offset = 0;

            
            //read var-int value
             bool read_next = true;
            while(read_next){
                temp_byte = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_byte);
                int8_t in_data = 0;
                in_data = (int8_t) (in_data | (temp_byte & 0x00FF));

                if(in_data >= 0){
                    value |= (static_cast<DATA_TYPE>(in_data) << offset);
                    read_next = false;
                }
                else{
                    value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset); 
                    offset += 7;
                }
            }

            out_buf[out_offset] = value; 
            
            out_offset++;
            input_data_out_size+=sizeof(DATA_TYPE);

        }

      }
      //compresssed data
      else{
            uint64_t remaining = static_cast<uint64_t>(head_byte);

            temp_byte = 0;
            s.template fetch_n_bits<int32_t>(8, &temp_byte);
            int8_t delta = (int8_t) (temp_byte & 0x00FF);
            //delta = __shfl_sync(FULL_MASK, delta, 0);

            DATA_TYPE value = 0;
            int64_t offset = 0;

            
            int32_t in_data;

            while(1){
                temp_byte = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_byte);

                int8_t in_data =  (int8_t) (temp_byte & 0x00FF);

                if(in_data >= 0){

                    value |= (static_cast<DATA_TYPE>(in_data) << offset);
                    break;
                }
                else{

                    value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset); 
                    offset += 7;
                }
            }


            for(uint64_t i = 0; i < remaining + 3; ++i){
                int64_t out_ele = value + static_cast<int64_t>(i) * delta;
                out_buf[out_offset] =  static_cast<DATA_TYPE>(out_ele);
                out_offset++;                                
            }
  
            input_data_out_size+= (sizeof(DATA_TYPE) * (remaining + 3));


        }

    }


}


//only one thread in a warp is decoding 
template <typename COMP_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
__device__
void decoder_warp_orig_rdw(full_warp_input_stream<COMP_COL_TYPE, in_buff_len>& s, uint64_t CHUNK_SIZE, DATA_TYPE* out_buf) {

    uint32_t input_data_out_size = 0;
    uint64_t out_offset = 0;
    
    while (input_data_out_size < CHUNK_SIZE) {
    
      //need to read a header
      int32_t temp_byte = 0;
      s.template fetch_n_bits<int32_t>(8, &temp_byte);

      int8_t head_byte = (int8_t)(temp_byte & 0x00FF);
      //head_byte = __shfl_sync(FULL_MASK, head_byte, 0);

      //literals
      if(head_byte < 0){
        uint64_t remaining = static_cast<uint64_t>(-head_byte);

        for(uint64_t i = 0; i < remaining; ++i){

            DATA_TYPE value = 0;
            int64_t offset = 0;

            
            //read var-int value
            //bool read_next = true & (threadIdx.x == 0);
            bool read_next = true;

            
            while(read_next){
                temp_byte = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_byte);

                int8_t in_data = 0;
                in_data = (int8_t) (in_data | (temp_byte & 0x00FF));

                if(in_data >= 0){
                    value |= (static_cast<DATA_TYPE>(in_data) << offset);
                    read_next = false;
                }
                else{
                    value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset); 
                    offset += 7;
                }
            }

            if(threadIdx.x == 0) out_buf[out_offset] = value; 
            
            out_offset++;
            input_data_out_size+=sizeof(DATA_TYPE);

        }

      }
      //compresssed data
      else{
            uint64_t remaining = static_cast<uint64_t>(head_byte);

            temp_byte = 0;
            s.template fetch_n_bits<int32_t>(8, &temp_byte);

            int8_t delta = (int8_t) (temp_byte & 0x00FF);
           // delta = __shfl_sync(FULL_MASK, delta, 0);

            DATA_TYPE value = 0;
            int64_t offset = 0;

            
            int32_t in_data;

            while(1){
                temp_byte = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_byte);

                int8_t in_data =  (int8_t) (temp_byte & 0x00FF);

                if(in_data >= 0){

                    value |= (static_cast<DATA_TYPE>(in_data) << offset);
                    break;
                }
                else{

                    value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset); 
                    offset += 7;
                }
            }

            //value = __shfl_sync(FULL_MASK, value, 0);


            uint64_t next_out_offset = out_offset + remaining + 3;
            out_offset += min((uint64_t)threadIdx.x, remaining + 3);


            //decoding the compresssed stream
            for(uint64_t i = threadIdx.x; i < remaining + 3; i += 32, out_offset += 32){
                
                int64_t out_ele = value + static_cast<int64_t>(i) * delta;
                out_buf[out_offset] =  static_cast<DATA_TYPE>(out_ele);

            }

            out_offset = next_out_offset;
            input_data_out_size+= (sizeof(DATA_TYPE) * (remaining + 3));


        }

    }


}

//only one thread in a warp is decoding 
template <typename COMP_COL_TYPE, typename DATA_TYPE, size_t in_buff_len = 4>
__device__
void decoder_warp_orig_multi(input_stream<COMP_COL_TYPE, in_buff_len>& s, queue<write_ele>& wq, uint64_t CHUNK_SIZE, DATA_TYPE* out_buf) {

    uint32_t input_data_out_size = 0;
    uint64_t out_offset = 0;
    
    while (input_data_out_size < CHUNK_SIZE) {
    
      //need to read a header
      int32_t temp_byte = 0;
      s.template fetch_n_bits<int32_t>(8, &temp_byte);

      int8_t head_byte = (int8_t)(temp_byte & 0x00FF);
      //head_byte = __shfl_sync(FULL_MASK, head_byte, 0);

      //literals
      if(head_byte < 0){
        uint64_t remaining = static_cast<uint64_t>(-head_byte);

        for(uint64_t i = 0; i < remaining; ++i){

            DATA_TYPE value = 0;
            int64_t offset = 0;

            
            //read var-int value
             bool read_next = true;
            while(read_next){
                temp_byte = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_byte);
                int8_t in_data = 0;
                in_data = (int8_t) (in_data | (temp_byte & 0x00FF));

                if(in_data >= 0){
                    value |= (static_cast<DATA_TYPE>(in_data) << offset);
                    read_next = false;
                }
                else{
                    value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset); 
                    offset += 7;
                }
            }

            write_ele w_ele;
            w_ele.run = 0;
            w_ele.value = value;

            wq.enqueue(&w_ele);

            //out_buf[out_offset] = value; 
            //out_offset++;
            input_data_out_size+=sizeof(DATA_TYPE);

        }

      }
      //compresssed data
      else{
            uint64_t remaining = static_cast<uint64_t>(head_byte);

            temp_byte = 0;
            s.template fetch_n_bits<int32_t>(8, &temp_byte);
            int8_t delta = (int8_t) (temp_byte & 0x00FF);
            //delta = __shfl_sync(FULL_MASK, delta, 0);

            DATA_TYPE value = 0;
            int64_t offset = 0;

            
            int32_t in_data;

            while(1){
                temp_byte = 0;
                s.template fetch_n_bits<int32_t>(8, &temp_byte);

                int8_t in_data =  (int8_t) (temp_byte & 0x00FF);

                if(in_data >= 0){

                    value |= (static_cast<DATA_TYPE>(in_data) << offset);
                    break;
                }
                else{

                    value |= ((static_cast<DATA_TYPE>(in_data) & 0x7f) << offset); 
                    offset += 7;
                }
            }

            write_ele w_ele;
            w_ele.run = remaining + 3;
            w_ele.value = value;
            w_ele.delta = delta;

            wq.enqueue(&w_ele);

            // for(uint64_t i = 0; i < remaining + 3; ++i){
            //     int64_t out_ele = value + static_cast<int64_t>(i) * delta;
            //     out_buf[out_offset] =  static_cast<DATA_TYPE>(out_ele);
            //     out_offset++;                                
            // }
  
            input_data_out_size+= (sizeof(DATA_TYPE) * (remaining + 3));


        }

    }


}

template <typename COMP_COL_TYPE, typename DATA_TYPE>
__device__
void write_warp(decompress_output<DATA_TYPE>& d, queue<write_ele>& wq,  uint64_t CHUNK_SIZE, uint8_t active_chunks ){

    uint32_t done = 0;
    uint64_t CHUNK_COUNT = CHUNK_SIZE / sizeof(DATA_TYPE);
    int t = threadIdx.x;
    while (!done) {

        bool deq = false;
        write_ele v;

        if(t < active_chunks){
            wq.attempt_dequeue(&v, &deq);
        }
        uint32_t deq_mask = __ballot_sync(FULL_MASK, deq);
        uint32_t deq_count = __popc(deq_mask);


        for (size_t i = 0; i < deq_count; i++) {
            int32_t f = __ffs(deq_mask);

            int16_t run = __shfl_sync(FULL_MASK, v.run, f-1);
            int64_t val = __shfl_sync(FULL_MASK, v.value, f-1);

            //pair
            if(run == 0){
                d.write_value(f-1, val);
            }
            //literal
            else{
                int16_t delta = __shfl_sync(FULL_MASK, v.delta, f-1);
                d.write_run(f-1, val, delta, run);
            }

            deq_mask >>= f;
            deq_mask <<= f;
            
        }
        __syncwarp();
        bool check = d.counter != (CHUNK_COUNT);
        if(threadIdx.x >= active_chunks ) check = false;
        done = __ballot_sync(FULL_MASK, check) == 0;
    } 
}


template <typename COMP_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE, uint16_t queue_size = 4>
__global__ void 
__launch_bounds__ (64, 32)
//__launch_bounds__ (96, 13)
//__launch_bounds__ (128, 11)
inflate(uint8_t* comp_ptr, uint8_t* out, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint64_t CHUNK_SIZE, int COMP_COL_LEN) {

    __shared__ COMP_COL_TYPE in_queue_[32][queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[32];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[32];

    __shared__ COMP_COL_TYPE local_queue[32][2];

    uint64_t col_len = (col_len_ptr[32 * (blockIdx.x) + threadIdx.x]);

    h[threadIdx.x] = 0;
    t[threadIdx.x] = 0;

    //if(blockIdx.x == 1 && threadIdx.x == 0) printf("blk offset: %llu\n",blk_offset_ptr[blockIdx.x ] );

    __syncthreads();

    if (threadIdx.y == 0) {
        queue<COMP_COL_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x , t + threadIdx.x, queue_size);
        uint8_t* chunk_ptr = (comp_ptr +  blk_offset_ptr[blockIdx.x ]);
        decompress_input< COMP_COL_TYPE> d(chunk_ptr, col_len);
        reader_warp<COMP_COL_TYPE>(d, in_queue);
    }

    else if (threadIdx.y == 1) {
    
        queue<COMP_COL_TYPE> in_queue(in_queue_[threadIdx.x], h + threadIdx.x, t + threadIdx.x, queue_size);
        input_stream<COMP_COL_TYPE, 2> s(&in_queue, (uint32_t)col_len, local_queue[threadIdx.x]);
        decompress_output<DATA_TYPE> d((out + CHUNK_SIZE * (blockIdx.x )));
        decoder_warp<COMP_COL_TYPE, DATA_TYPE, 2>(s, d, CHUNK_SIZE, (DATA_TYPE*)(out + CHUNK_SIZE * blockIdx.x), COMP_COL_LEN);

    }

    __syncthreads();
    //if(threadIdx.x == 0) printf("bid: %i done\n", blockIdx.x);
    // else{
    //     queue<write_queue_ele<DATA_TYPE>> out_queue(out_queue_[threadIdx.x], out_h + threadIdx.x, out_t + threadIdx.x, queue_size);
    //     decompress_output<OUT_COL_TYPE> d((out + CHUNK_SIZE * blockIdx.x));
    //     writer_warp<OUT_COL_TYPE, DATA_TYPE, CHUNK_SIZE>(out_queue, d);
    // }
}


template <typename COMP_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE, int NUM_CHUNKS, uint16_t queue_size = 4, int NT = 64, int BT = 32>
__global__ void 
__launch_bounds__ (NT, BT)
inflate_orig_dw(uint8_t* comp_ptr, uint8_t* out, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint64_t CHUNK_SIZE, int COMP_COL_LEN, uint64_t num_chunks) {

    __shared__ COMP_COL_TYPE in_queue_[NUM_CHUNKS][queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[32];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[32];

    __shared__ COMP_COL_TYPE local_queue[NUM_CHUNKS][2];

    //uint64_t col_len = (col_len_ptr[32 * (blockIdx.x) + threadIdx.x]);

    h[threadIdx.x] = 0;
    t[threadIdx.x] = 0;

    //if(blockIdx.x == 1 && threadIdx.x == 0) printf("blk offset: %llu\n",blk_offset_ptr[blockIdx.x ] );

    __syncthreads();
    uint8_t active_chunks = NUM_CHUNKS;
    if((blockIdx.x+1) * NUM_CHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_CHUNKS;
    }

    int my_block_idx = 0;
    uint64_t col_len = 0;
    int my_queue = 0;

    if (threadIdx.y == 0) {
        
        my_queue =  threadIdx.x % NUM_CHUNKS;
        my_block_idx = blockIdx.x * NUM_CHUNKS + threadIdx.x % NUM_CHUNKS;
        
        col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);
        //col_len = col_len_ptr[my_block_idx];

        queue<COMP_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, queue_size);
        decompress_input< COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));

        reader_warp_orig< COMP_COL_TYPE, NUM_CHUNKS>(d, in_queue, active_chunks);
    }

    else {
           
        my_queue = (threadIdx.y - 1);
        my_block_idx =  (blockIdx.x * NUM_CHUNKS + threadIdx.y -1);
       // col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);
        col_len = col_len_ptr[my_block_idx];


        queue<COMP_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, queue_size);
        unsigned int chunk_id = blockIdx.x * NUM_CHUNKS + (threadIdx.y - 1);

        input_stream<COMP_COL_TYPE, 2> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], threadIdx.x == 0);
        decoder_warp_orig_dw<COMP_COL_TYPE, DATA_TYPE, 2>(s, CHUNK_SIZE, (DATA_TYPE*)(out + CHUNK_SIZE * chunk_id));

    }

    __syncthreads();
}


template <typename COMP_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE, int NUM_CHUNKS, uint16_t queue_size = 4, int NT = 64, int BT = 32>
__global__ void 
__launch_bounds__ (NT, BT)
inflate_orig_dw_multi(uint8_t* comp_ptr, uint8_t* out, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint64_t CHUNK_SIZE, int COMP_COL_LEN, uint64_t num_chunks) {

    __shared__ COMP_COL_TYPE in_queue_[NUM_CHUNKS][queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[32];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[32];


    __shared__ COMP_COL_TYPE local_queue[NUM_CHUNKS][2];

    //uint64_t col_len = (col_len_ptr[32 * (blockIdx.x) + threadIdx.x]);

    h[threadIdx.x] = 0;
    t[threadIdx.x] = 0;

    //if(blockIdx.x == 1 && threadIdx.x == 0) printf("blk offset: %llu\n",blk_offset_ptr[blockIdx.x ] );

    __syncthreads();
    uint8_t active_chunks = NUM_CHUNKS;
    if((blockIdx.x+1) * NUM_CHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_CHUNKS;
    }

    int my_block_idx = 0;
    uint64_t col_len = 0;
    int my_queue = 0;

    if (threadIdx.y == 0) {
        
        my_queue =  threadIdx.x % NUM_CHUNKS;
        my_block_idx = blockIdx.x * NUM_CHUNKS + threadIdx.x % NUM_CHUNKS;
        
        col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);
        //col_len = col_len_ptr[my_block_idx];

        queue<COMP_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, queue_size);
        decompress_input< COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));
       // decompress_input< COMP_COL_TYPE> d(comp_ptr, col_len, 0);

        reader_warp_orig< COMP_COL_TYPE, NUM_CHUNKS>(d, in_queue, active_chunks);
    }

    else {
           

       
        my_queue = (threadIdx.x);
        my_block_idx =  (blockIdx.x * NUM_CHUNKS + threadIdx.x );
       // col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);
        col_len = col_len_ptr[my_block_idx];


        queue<COMP_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, queue_size);
        unsigned int chunk_id = blockIdx.x * NUM_CHUNKS + (threadIdx.x);

        input_stream<COMP_COL_TYPE, 2> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], true);
        decoder_warp_orig_dw_multi<COMP_COL_TYPE, DATA_TYPE, 2>(s, CHUNK_SIZE, (DATA_TYPE*)(out + CHUNK_SIZE * chunk_id));



    }

    __syncthreads();
}

template <typename COMP_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE, int NUM_CHUNKS, uint16_t queue_size = 4, int NT = 64, int BT = 32>
__global__ void 
__launch_bounds__ (NT, BT)
inflate_orig_multi(uint8_t* comp_ptr, uint8_t* out, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint64_t CHUNK_SIZE, int COMP_COL_LEN, uint64_t num_chunks) {

    __shared__ COMP_COL_TYPE in_queue_[NUM_CHUNKS][queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> h[32];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> t[32];

    __shared__ write_ele out_queue_[NUM_CHUNKS][queue_size];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> o_h[32];
    __shared__ simt::atomic<uint8_t,simt::thread_scope_block> o_t[32];

    __shared__ COMP_COL_TYPE local_queue[NUM_CHUNKS][2];

    //uint64_t col_len = (col_len_ptr[32 * (blockIdx.x) + threadIdx.x]);

    h[threadIdx.x] = 0;
    t[threadIdx.x] = 0;
    o_h[threadIdx.x] = 0;
    o_t[threadIdx.x] = 0;

    //if(blockIdx.x == 1 && threadIdx.x == 0) printf("blk offset: %llu\n",blk_offset_ptr[blockIdx.x ] );

    __syncthreads();
    uint8_t active_chunks = NUM_CHUNKS;
    if((blockIdx.x+1) * NUM_CHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_CHUNKS;
    }

    int my_block_idx = 0;
    uint64_t col_len = 0;
    int my_queue = 0;

    if (threadIdx.y == 0) {
        
        my_queue =  threadIdx.x % NUM_CHUNKS;
        my_block_idx = blockIdx.x * NUM_CHUNKS + threadIdx.x % NUM_CHUNKS;
        
        col_len = (blk_offset_ptr[my_block_idx+1] - blk_offset_ptr[my_block_idx]);

        queue<COMP_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue , t + my_queue, queue_size);
        decompress_input< COMP_COL_TYPE> d(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE));

        reader_warp_orig< COMP_COL_TYPE, NUM_CHUNKS>(d, in_queue, active_chunks);
    }

    else if(threadIdx.y == 1) {
                  
        my_queue = (threadIdx.x);
        my_block_idx =  (blockIdx.x * NUM_CHUNKS + threadIdx.x );
        col_len = col_len_ptr[my_block_idx];


        queue<COMP_COL_TYPE> in_queue(in_queue_[my_queue], h + my_queue, t + my_queue, queue_size);
        queue<write_ele> out_queue(out_queue_[my_queue], o_h + my_queue, o_t + my_queue, queue_size);

        unsigned int chunk_id = blockIdx.x * NUM_CHUNKS + (threadIdx.x);

        input_stream<COMP_COL_TYPE, 2> s(&in_queue, (uint32_t)col_len, local_queue[my_queue], true);
        decoder_warp_orig_multi<COMP_COL_TYPE, DATA_TYPE, 2>(s, out_queue, CHUNK_SIZE, (DATA_TYPE*)(out + CHUNK_SIZE * chunk_id));
    }

    else {
        my_queue = (threadIdx.x);
        queue<write_ele> out_queue(out_queue_[my_queue], o_h + my_queue, o_t + my_queue, queue_size);

        unsigned int chunk_id = blockIdx.x * NUM_CHUNKS + (threadIdx.x);
        decompress_output<DATA_TYPE> d(out + CHUNK_SIZE * chunk_id);
        write_warp<COMP_COL_TYPE, DATA_TYPE>(d, out_queue, CHUNK_SIZE, active_chunks);

        //(out + CHUNK_SIZE * chunk_id);

    }

    __syncthreads();
}


template <typename COMP_COL_TYPE, typename DATA_TYPE, typename OUT_COL_TYPE, int NUM_CHUNKS, uint16_t queue_size = 4, int NT = 64, int BT = 32>
__global__ void 
__launch_bounds__ (NT, BT)
inflate_orig_rdw(uint8_t* comp_ptr, uint8_t* out, const uint64_t* const col_len_ptr, const uint64_t* const blk_offset_ptr, uint64_t CHUNK_SIZE, int COMP_COL_LEN, uint64_t num_chunks) {

    __shared__ COMP_COL_TYPE in_queue_[NUM_CHUNKS][queue_size];


    uint8_t active_chunks = NUM_CHUNKS;
    if((blockIdx.x+1) * NUM_CHUNKS > num_chunks){
       active_chunks = num_chunks - blockIdx.x * NUM_CHUNKS;
    }
  
    int   my_queue = (threadIdx.y);
    int   my_block_idx =  (blockIdx.x * NUM_CHUNKS + threadIdx.y);
    uint64_t    col_len = col_len_ptr[my_block_idx];
    __syncthreads();

    full_warp_input_stream<COMP_COL_TYPE, queue_size> s(comp_ptr, col_len, blk_offset_ptr[my_block_idx] / sizeof(COMP_COL_TYPE), in_queue_[my_queue]);
    __syncthreads();
    unsigned int chunk_id = blockIdx.x * NUM_CHUNKS + (threadIdx.y);

    decoder_warp_orig_rdw<COMP_COL_TYPE, DATA_TYPE, queue_size>(s, CHUNK_SIZE, (DATA_TYPE*)(out + CHUNK_SIZE * chunk_id));


}














