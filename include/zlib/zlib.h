#ifndef __ZLIB_H__
#define __ZLIB_H__

#include <common.h>
#include <fstream>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <simt/atomic>
#define BUFF_LEN 2


#define UNCOMP 0
#define STATIC 1
#define DYNAMIC 2
#define FULL_MASK 0xFFFFFFFF

#define MAXBITS 15
#define FIXLCODES 288
#define MAXDCODES 30
#define MAXCODES 316




struct dynamic_huffman {
    int16_t lencnt[MAXBITS + 1];
    int16_t lensym[FIXLCODES];
    int16_t distcnt[MAXBITS + 1];
    int16_t distsym[MAXDCODES];
};

template <typename T>
struct queue {
    T* queue_;
    simt::atomic<uint32_t, simt::thread_scope_block>* head;
    simt::atomic<uint32_t, simt::thread_scope_block>* tail;
    uint32_t len;

    __device__
    queue(T* buff, simt::atomic<uint32_t, simt::thread_scope_block>* h, simt::atomic<uint32_t, simt::thread_scope_block>* t, const uint32_t l) {
        queue_ = buff;
        head = h;
        tail = t;
        len = l;

        //dont we need this?

        *t = 0;
        *h = 0;

    }

    __device__
    void enqueue(const T* v) {

        const auto cur_tail = tail->load(simt::memory_order_relaxed);
        const auto next_tail = (cur_tail + 1) % len;

        while (next_tail == head->load(simt::memory_order_acquire))
            __nanosleep(100);

        queue_[cur_tail] = *v;
        tail->store(next_tail, simt::memory_order_release);


    }

    __device__
    void dequeue(T* v) {

        const auto cur_head = head->load(simt::memory_order_relaxed);
        while (cur_head == tail->load(simt::memory_order_acquire))
            __nanosleep(100);

        *v = queue_[cur_head];

        const auto next_head = (cur_head + 1) % len;

        head->store(next_head, simt::memory_order_release);


    }

    __device__
    void attempt_dequeue(T* v, bool* p) {
        const auto cur_head = head->load(simt::memory_order_relaxed);
        if (cur_head == tail->load(simt::memory_order_acquire)) {
            *p = false;
            return;
        }


        *v = queue_[cur_head];

        const auto next_head = (cur_head + 1) % len;

        head->store(next_head, simt::memory_order_release);
        *p = true;

    }
    
};


//producer of input queue
template <typename READ_COL_TYPE>
struct decompress_input {


    uint64_t col_width = sizeof(READ_COL_TYPE);
    uint64_t row_offset;
    uint64_t len;
    uint64_t read_bytes;
    uint32_t t_read_mask;
    uint64_t used_bytes;
    READ_COL_TYPE* pointer;



    //READ_COL_TYPE in_buff[IN_BUFF_LEN];

    __device__
    decompress_input(const uint8_t* ptr, const uint64_t l) :
        pointer((READ_COL_TYPE*) ptr), len(l) {
        uint64_t tid = threadIdx.x;

        t_read_mask = (0xffffffff >> (32 - tid));
        row_offset = 0;
        used_bytes = 0;
        read_bytes = 0;
    }

    __device__
    int32_t read_data(const uint32_t alivemask, READ_COL_TYPE* v) {
        int32_t read_count  = 0;
        bool read = (read_bytes) < len;
        uint32_t read_sync = __ballot_sync(alivemask, read);
      
        if (read_sync == 0)
            read_count = -1;
        
        if (read) {
            *v = pointer[row_offset + __popc(read_sync & t_read_mask)];
            row_offset += __popc(read_sync);
            read_bytes += sizeof(READ_COL_TYPE);
            read_count = sizeof(READ_COL_TYPE);
        }



        __syncwarp(alivemask);

        return read_count;

    }


};


//consumer of input queue
template <typename READ_COL_TYPE, size_t buff_len = 3>
struct input_stream {

    union buff {
        READ_COL_TYPE b[buff_len];
        uint32_t u[(sizeof(READ_COL_TYPE)* buff_len)/sizeof(uint32_t)];
    } b;
    uint32_t head;
    uint32_t count;
    uint32_t bit_offset;
    uint32_t uint_head;
    uint32_t uint_count;
    queue<READ_COL_TYPE>* q;
    uint32_t read_bytes;
    uint32_t expected_bytes;
    uint32_t bu_size = (sizeof(READ_COL_TYPE)* buff_len)/sizeof(uint32_t);

    uint32_t uint_bit_offset;
    uint32_t uint_offset;

    __device__
    input_stream(queue<READ_COL_TYPE>* q_, uint32_t eb) {

        q = q_;
        expected_bytes = eb;

        head = 0;

        uint_bit_offset = 0;
        uint_offset = 0;
        uint_count = 0;
        uint_head = 0;
        read_bytes = 0;
        count = 0;
        for (; (count < buff_len) && (read_bytes < expected_bytes);
             count++, read_bytes += sizeof(READ_COL_TYPE), uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t)) {
            q->dequeue(b.b + count);

        }
    }

    template<typename T>
    __device__
    void get_n_bits(const uint32_t n, T* out) {
        //assert(sizeof(T) < sizeof(uint32_t));
        //is this correct? 
        assert(sizeof(T) >= sizeof(uint32_t));

        uint32_t out_t_n = 1 + ((sizeof(T)-1)/n);
        uint32_t* out_uint = (uint32_t*) out;

        uint32_t out_t_uint = sizeof(T)/sizeof(uint32_t);
        for (size_t i = 0; i < out_t_n; i++) {
            out[i] = 0;
        }
        //uint32_t left = n;
        uint32_t copied = 0;
        uint32_t out_uint_offset = 0;
        uint32_t a = b.u[(uint_head)];
        uint32_t b2;

        do {
            b2 = b.u[(uint_head+1)%bu_size];

            //(b : a) >> min(uint_bit_offset, 32)

            out_uint[out_uint_offset] = __funnelshift_rc(a, b2, uint_bit_offset);
            copied += 32;
            if (copied > n) {
                uint32_t extra = copied - n;
                out_uint[out_uint_offset] <<= extra;
                out_uint[out_uint_offset] >>= extra;
                copied -= extra;

            }


            uint_bit_offset += copied;
            if (uint_bit_offset >= 32) {
                uint_bit_offset = uint_bit_offset % 32;
                uint_head = (uint_head+1) % bu_size;
                if ((uint_head % (sizeof(READ_COL_TYPE)/sizeof(uint32_t))) == 0) {
                    head = (head + 1) % buff_len;
                    count--;
                }

                uint_count--;

            }
            out_uint_offset++;

            a = b2;

        } while(copied < n);



    }
    template<typename T>
    __device__
    void fetch_n_bits(const uint32_t n, T* out) {

        while ((count < buff_len) && (read_bytes < expected_bytes)) {
            q->dequeue(b.b + ((head+count) % buff_len));
            count++;
            uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t);
            read_bytes += sizeof(READ_COL_TYPE);
        }

        get_n_bits(n, out);
    }

    template<typename T>
    __device__
    void peek_n_bits(const uint32_t n, T* out) {
        while ((count < buff_len) && (read_bytes < expected_bytes)) { 
            q->dequeue(b.b + ((head+count) % buff_len));
            count++;
            uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t);
            read_bytes += sizeof(READ_COL_TYPE);
        }
        uint32_t count_ = count;
        uint32_t head_ = head;
        uint32_t uint_count_ = uint_count;
        uint32_t uint_head_ = uint_head;
        uint32_t uint_bit_offset_ = uint_bit_offset;

        get_n_bits(n, out);

        count = count_;
        head = head_;
        uint_count = uint_count_;
        uint_head = uint_head_;
        uint_bit_offset = uint_bit_offset_;
    }
};



// void writer_warp(queue<uint64_t>& mq, decompress_output<size_t WRITE_COL_LEN = 512>& out) {
template <size_t WRITE_COL_LEN = 512, size_t CHUNK_SIZE = 8192>
struct decompress_output {

    uint8_t* out_ptr;
    uint32_t thread_off;
    uint64_t counter;
    uint64_t total;
    size_t write_col_len;

    __device__
    decompress_output(uint8_t* ptr):
        out_ptr(ptr) {
            thread_off = WRITE_COL_LEN * threadIdx.x;
            counter = 0;
            total = CHUNK_SIZE / 32;
            write_col_len = WRITE_COL_LEN;
    }

    __device__
    void get_byte(uint64_t read_counter, uint8_t* b_ptr, uint32_t t_off){
        *b_ptr = out_ptr[read_counter + (read_counter/write_col_len) * (31*write_col_len) + t_off];
    }

    __device__
    void write_byte(uint64_t write_counter, uint8_t b, uint32_t t_off){
        out_ptr[write_counter + (write_counter/write_col_len) * (31*write_col_len) + t_off] = b;
    }

    __device__
    void copy_byte(uint64_t read_counter, uint64_t write_counter, uint32_t t_off){
        out_ptr[write_counter + (write_counter/write_col_len) * (31*write_col_len) + t_off] =
            out_ptr[read_counter + (read_counter/write_col_len) * (31*write_col_len) + t_off];

      
    }

    __device__
    void copy_8byte(uint64_t read_counter, uint64_t write_counter, uint32_t t_off) {
       int w_idx =  (write_counter + (write_counter/write_col_len) * (31*write_col_len) + t_off) / 8;
       int r_idx = (read_counter + (read_counter/write_col_len) * (31*write_col_len) + t_off) / 8;
       ((uint64_t*)out_ptr)[w_idx] = ((uint64_t*)out_ptr)[r_idx];



    }

    __device__ 
    void new_memcpy(uint64_t read_counter, uint64_t write_counter, uint32_t len, uint32_t offset, uint32_t t_off) {
        uint32_t write_len = len;
        uint64_t rc = read_counter;
        uint64_t wc = write_counter; 

        int slow_bytes = min(write_len, (int)((8 -(size_t)(wc)) & 0x7));
        
        //number of bytes to write per thread
        int slow_bytes_t = (max(slow_bytes - threadIdx.x, 0) + 31) / 32;

        if (slow_bytes){
            uint64_t temp_rc = rc + threadIdx.x;
            uint64_t temp_wc = wc + threadIdx.x;
            for (int i = 0; i < slow_bytes_t; i++){
                copy_byte(temp_rc, temp_wc, t_off);
                temp_rc += 32;
                temp_wc += 32;
            }
            write_len -= slow_bytes;
            wc += (slow_bytes);
            rc += (slow_bytes);
        }

        __syncwarp();
        int fast_bytes = write_len;
        int fast_remain = fast_bytes & (~(32 - 1));
        int fast_bytes_t = (max(fast_remain - threadIdx.x*8, 0) + 32*8 - 1) / (32*8);
        if(fast_remain > 0){
            fast_bytes_t = (fast_remain - threadIdx.x*8);
            if(fast_bytes_t < 0)
                fast_bytes_t = 0;

            fast_bytes_t = ((fast_remain - threadIdx.x*8) + 32*8 - 1) / (32*8);
        }

        //8byte copy
        if (fast_remain > 0) {
           // printf("tid: %i fast bytes:%i, slow_bytes:%i wc: %lu, len: %lu\n",threadIdx.x, fast_bytes, slow_bytes, (unsigned long) wc ,(unsigned long)len);
            uint64_t temp_rc = rc + threadIdx.x * 8;
            uint64_t temp_wc = wc + threadIdx.x * 8;
            for(int i = 0; i < fast_bytes_t; i++){
                copy_8byte(temp_rc,temp_wc,t_off);
                temp_rc += 8 * 32;
                temp_wc += 8 * 32;
            }
            write_len -= fast_remain;
            wc += fast_remain;
            rc += fast_remain;
        }
        
        slow_bytes = write_len;

        //slow_copy remainder
        if(write_len != 0){
            //printf("tid: %i write_len:%lu,  wc: %lu, len: %lu\n",threadIdx.x, (unsigned long)write_len, (unsigned long) wc ,(unsigned long)len);

            slow_bytes_t = (max(slow_bytes - threadIdx.x, 0) + 31) / 32;
            //printf("tid: %i write_len:%lu,  slow_bytes_t: %lu, len: %lu\n",threadIdx.x, (unsigned long)write_len, (unsigned long) slow_bytes_t ,(unsigned long)len);
                rc += threadIdx.x;
                wc += threadIdx.x;
              for (int i = 0; i < slow_bytes_t; i++){
                //printf("copy tid: %i write_len:%lu,  slow_bytes_t: %lu, len: %lu\n",threadIdx.x, (unsigned long)write_len, (unsigned long) slow_bytes_t ,(unsigned long)len);
           
                copy_byte(rc, wc, t_off);
                wc += 32;
                rc += 32;
            }
        }

    }


    __device__
    void col_memcpy(uint32_t idx, uint32_t len, uint32_t offset) {
      
        //copy the meta data for the ptr


        uint64_t orig_counter = __shfl_sync(FULL_MASK, counter, idx);
        uint32_t t_off = __shfl_sync(FULL_MASK, thread_off, idx);


        uint32_t num_writes = ((len - threadIdx.x + 31) / 32);
        
        uint64_t start_counter =  0;
        if(orig_counter > offset)
            start_counter = orig_counter - offset;
        uint64_t read_counter = start_counter + threadIdx.x;
        uint64_t write_counter = orig_counter + threadIdx.x;

        if(offset > len){
            new_memcpy(start_counter, orig_counter, len, offset, t_off);
            __syncwarp();
        }

        else {
            for(int i = 0; i < num_writes; i++){

                //check offset
                if(read_counter >= orig_counter){
                    read_counter = (read_counter - orig_counter) % offset + start_counter;
                }

                //uint8_t read_byte = 0;
                copy_byte(read_counter, write_counter, t_off);
                read_counter += 32;
               
                write_counter += 32;
            }
        }
        //set the counter
        if(threadIdx.x == idx)
            counter = counter + len;

    }

    __device__
    void write_literal(uint32_t idx, uint8_t b){
        if(threadIdx.x == idx){
            write_byte(counter, b, thread_off);
            counter++;
        }
    }


};

template <typename READ_COL_TYPE>
__device__
void reader_warp(decompress_input<READ_COL_TYPE>& in, queue<READ_COL_TYPE>& rq) {
    while (true) {
        READ_COL_TYPE v;
        int32_t rc = in.read_data(FULL_MASK, &v);

        if (rc == -1)
            break;
        else if (rc > 0)
            rq.enqueue(&v);
    }
}


template <typename READ_COL_TYPE, size_t in_buff_len = 3>
__device__
int16_t decode (input_stream<READ_COL_TYPE, in_buff_len>& s, const int16_t* counts, const int16_t* symbols){

    unsigned int first;
    unsigned int len;
    unsigned int code;
    unsigned int count;
    uint32_t next32r = 0;
    s.template peek_n_bits<uint32_t>(32, &next32r);

    next32r = __brev(next32r);



    first  = 0;
    for (len = 1; len <= MAXBITS; len++) {
        code  = (next32r >> (32 - len)) - first;
        
        count = counts[len];
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



//Construct huffman tree
template <typename READ_COL_TYPE, size_t in_buff_len = 3>
__device__ 
void construct(input_stream<READ_COL_TYPE, in_buff_len>& s, int16_t* counts, int16_t* symbols, const int32_t *length, int num_codes){

    int symbol;
    int len;
    int left;
    for(len = 0; len <= MAXBITS; len++)
        counts[len] = 0;

    for(symbol = 0; symbol < num_codes; symbol++)
        (counts[length[symbol]])++;

    left = 1;
    for(len = 1; len <= MAXBITS; len++){
        left <<= 1;
        left -= counts[len];       
        if (left < 0) 
            return; 
    }

    {
        //computing offset array for conunts
        int16_t offs[MAXBITS + 1];
        offs[1] = 0;
        for (len = 1; len < MAXBITS; len++)
            offs[len + 1] = offs[len] + counts[len];

        for(symbol = 0; symbol < num_codes; symbol++){
             if (length[symbol] != 0) 
                symbols[offs[length[symbol]]++] = symbol;
        }
    }   
}

/// permutation of code length codes
static const __device__ __constant__ uint8_t g_code_order[19 + 1] = {
  16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 0xff};


//construct huffman tree for dynamic huffman encoding block
template <typename READ_COL_TYPE, size_t in_buff_len = 3>
__device__
int decode_dynamic(input_stream<READ_COL_TYPE, in_buff_len>& s, dynamic_huffman &dt){
    uint32_t hlit, hdist, hclen;

 
    //getting the meta data for the compressed block
    s.template fetch_n_bits<uint32_t>(5, &hlit) ;
    s.template fetch_n_bits<uint32_t>(5, &hdist);
    s.template fetch_n_bits<uint32_t>(4, &hclen);

    //if(threadIdx.x == 0) printf("hlit: %lu, hdist:%lu, hclen:%lu\n", (unsigned long)hlit,  (unsigned long)hdist,  (unsigned long)hclen );
    

    hlit += 257;
    hdist += 1;
    hclen += 4;
    int32_t lengths[MAXCODES];
    int index;
    //check
    for (index = 0; index < hclen; index++) {
        int32_t temp;
        s.template fetch_n_bits<int32_t>(3, &temp);
        lengths[g_code_order[index]] = temp;
    }
    for (; index < 19; index++) {
        lengths[g_code_order[index]] = 0;
    }
    // if(threadIdx.x == 0){
    //     for(int i = 0; i < 19;i++){
    //         printf("length: %lu\n", (unsigned long) lengths[i] );
    //     }
    // }
    construct(s, dt.lencnt, dt.lensym, lengths, 19);

    
    index = 0;
    while (index < hlit + hdist) {
        int32_t symbol = (int32_t) decode(s, dt.lencnt, dt.lensym);
        //error
        if (symbol < 0){ 
            return symbol;}
        //represent code lengths of 0 - 15
        if(symbol < 16){
            lengths[index++] = symbol;
        }

        else{
            int len = 0;
            if(symbol == 16) {
                 len = lengths[index - 1];  // last length
                 s.template fetch_n_bits<int32_t>(2, &symbol);
                 symbol += 3;
            }
            else if(symbol == 17){
                s.template fetch_n_bits<int32_t>(3, &symbol);
                symbol += 3;
            }
            else {
                s.template fetch_n_bits<int32_t>(7, &symbol);
                symbol += 11;
            }


            while(symbol--){
                lengths[index++] = len;
            }
        }
    }

    //check
    if(lengths[256] == 0) return -9;


    construct(s, dt.lencnt, dt.lensym, lengths, hlit);
    
    construct(s, dt.distcnt, dt.distsym, &lengths[hlit], hdist);

    return 0;
}

template <typename READ_COL_TYPE, size_t in_buff_len = 3>
__device__ 
int decode_fixed(input_stream<READ_COL_TYPE, in_buff_len>& s, dynamic_huffman &dt){
    int32_t lengths[MAXCODES];

    int symbol;

    for (symbol = 0; symbol < 144; symbol++) lengths[symbol] = 8;
    for (; symbol < 256; symbol++) lengths[symbol] = 9;
    for (; symbol < 280; symbol++) lengths[symbol] = 7;
    for (; symbol < FIXLCODES; symbol++) lengths[symbol] = 8;
    
    construct(s, dt.lencnt, dt.lensym, lengths, FIXLCODES);
    
    for (symbol = 0; symbol < MAXDCODES; symbol++) lengths[symbol] = 5;
 
    construct(s, dt.distcnt, dt.distsym, lengths, MAXDCODES);

    return 0;
}



//code starts from 257
static const __device__ __constant__ uint16_t g_lens[29] = {  // Size base for length codes 257..285
  3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
  31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};

//code starts from 257
static const __device__ __constant__ uint16_t
  g_lext[29] = { 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};


static const __device__ __constant__ uint16_t
  g_dists[30] = {  // Offset base for distance codes 0..29
    1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
    193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};

static const __device__ __constant__ uint16_t g_dext[30] = {  // Extra bits for distance codes 0..29
  0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};


//decode code for compressed block
template <typename READ_COL_TYPE, size_t in_buff_len = 3>
__device__ 
void decode_symbol(input_stream<READ_COL_TYPE, in_buff_len>& s,dynamic_huffman &dt, uint32_t* out_off, uint8_t*out,  queue<uint64_t>& mq) {

    uint32_t thread_off = threadIdx.x * 32 + blockIdx.x * (1024*8);


    while(1){
              
        uint16_t sym = decode(s,  dt.lencnt, dt.lensym);


        //parse 5 bits
        //not compressed, literal
        if(sym <= 255) {

            uint64_t literal = 0;
            literal = literal | (sym << 1);
            mq.enqueue(&literal);
        }

        //end of block
        else if(sym == 256) {
            break;
        }

        //lenght, need to parse
        else{

            uint16_t extra_bits = g_lext[sym - 257];

            uint32_t extra_len  = 0;
            if(extra_bits != 0)
               s.template fetch_n_bits<uint32_t>(extra_bits, &extra_len);

            uint32_t len = extra_len + g_lens[sym - 257];

            
            //distance, 5bits
            uint16_t sym_dist = decode(s,  dt.distcnt, dt.distsym);    
            uint32_t extra_bits_dist = g_dext[sym_dist];
            
            uint32_t extra_len_dist = 0;
            if(extra_bits_dist != 0){
                s.template fetch_n_bits<uint32_t>(extra_bits_dist, &extra_len_dist);
            }

            uint32_t dist = extra_len_dist + g_dists[sym_dist];

            //(len, dist_len)
            uint64_t pair_code = 1;
            uint64_t long_len = len;
            pair_code = pair_code | (long_len << 17) | (dist << 1);
            mq.enqueue(&pair_code);
          
        }
    }

}





//if dynamic
//build tree
//
//decode length/distance pairs or literal
//
template <typename READ_COL_TYPE, size_t in_buff_len = 3>
__device__
void decoder_warp(input_stream<READ_COL_TYPE, in_buff_len>& s,  queue<uint64_t>& mq, uint32_t col_len, uint8_t* out) {


    uint32_t header;
    s.template fetch_n_bits<uint32_t>(16, &header);

    uint32_t out_b = 0;

    // bool cont = false;
    // uint8_t prev_type = UNCOMP;
    // READ_COL_TYPE local;
    uint32_t blast;
    uint32_t btype;

    do{
    s.template fetch_n_bits<uint32_t>(1, &blast);
    s.template fetch_n_bits<uint32_t>(2, &btype);
    dynamic_huffman tree;

    //not compressed
    //not going to be used
    if(btype == 0){
        uint32_t temp;
        s.template fetch_n_bits<uint32_t>(col_len, &temp);


    }
    //fixed huffman
    else if(btype == 1) {
        decode_fixed<READ_COL_TYPE, in_buff_len>(s, tree);
        decode_symbol<READ_COL_TYPE, in_buff_len>(s, tree, &out_b, out, mq);

    }
    //dyamic huffman
    else if (btype == 2) {
        //build dynamic huffman tree
        int dyn = decode_dynamic<READ_COL_TYPE, in_buff_len>(s, tree);
        //decode LZ
        decode_symbol<READ_COL_TYPE, in_buff_len>(s, tree, &out_b, out, mq);

    }
    //error
    else {
       // printf("error tid:%i \n", threadIdx.x);

    }

    }while(blast != 1);


}

template <size_t WRITE_COL_LEN = 512, size_t CHUNK_SIZE = 8192>
__device__
void writer_warp(queue<uint64_t>& mq, decompress_output<WRITE_COL_LEN, CHUNK_SIZE>& out) {
    
    uint32_t done = 0;
    while (!done) {


        bool deq = false;
        uint64_t v = 0;
        mq.attempt_dequeue(&v, &deq);
        uint32_t deq_mask = __ballot_sync(FULL_MASK, deq);
        uint32_t deq_count = __popc(deq_mask);
        for (size_t i = 0; i < deq_count; i++) {
            int32_t f = __ffs(deq_mask);
            uint64_t vv = __shfl_sync(FULL_MASK, v, f-1);
       
            //pair
            if(__ffsll(vv) == 1){
                uint64_t len = vv >> 17;
                uint64_t offset = (vv>>1) & 0x0000ffff;
                out.col_memcpy(f-1, (uint32_t)len, (uint32_t)offset);
                //if(threadIdx.x == 0) printf("len: %llu offset: %llu\n", len, offset );
                
            }
            //literal
            else{
                uint8_t b = (vv >> 1) & 0x00FF;
                out.write_literal(f-1, b);
            }

            deq_mask >>= f;
            deq_mask <<= f;
        }

        done = __ballot_sync(FULL_MASK, out.counter != out.total) == 0;

    } 


}


template <typename READ_COL_TYPE, size_t WRITE_COL_LEN = 512, size_t CHUNK_SIZE = 8192>
__global__ void 
__launch_bounds__(96, 15) 
inflate(uint8_t* comp_ptr, uint64_t* col_len_ptr, uint64_t* blk_offset_ptr, uint8_t*out) {
    __shared__ READ_COL_TYPE in_queue_[32][16];
    __shared__ simt::atomic<uint32_t,simt::thread_scope_block> h[32];
    __shared__ simt::atomic<uint32_t,simt::thread_scope_block> t[32];


    int lane_id = threadIdx.x % 32;
    queue<READ_COL_TYPE> in_queue(in_queue_[lane_id], &h[lane_id], &t[lane_id], 16);

    __shared__ uint64_t out_queue_[32][4];
    __shared__ simt::atomic<uint32_t,simt::thread_scope_block> out_h[32];
    __shared__ simt::atomic<uint32_t,simt::thread_scope_block> out_t[32];


    queue<uint64_t> out_queue(out_queue_[lane_id], &out_h[lane_id], &out_t[lane_id], 4);


    bool is_reader_warp = (threadIdx.y == 0);
    bool is_decoder_warp = (threadIdx.y == 1);
    bool is_writer_warp = (threadIdx.y == 2);

    uint64_t col_len = col_len_ptr[32 * blockIdx.x + threadIdx.x];


    __syncthreads();

    if (is_reader_warp) {
        uint64_t blk_off = blk_offset_ptr[blockIdx.x];


        uint8_t* chunk_ptr = &(comp_ptr[blk_off]);
        decompress_input<READ_COL_TYPE> d(chunk_ptr, col_len);
        reader_warp<READ_COL_TYPE>(d, in_queue);
    }

    if (is_decoder_warp) {
        input_stream<READ_COL_TYPE, 16> s(&in_queue, (uint32_t)col_len);
        decoder_warp<READ_COL_TYPE>(s, out_queue, (uint32_t) col_len, out);
    }

    if (is_writer_warp) {

        decompress_output<WRITE_COL_LEN, CHUNK_SIZE> d(&(out[CHUNK_SIZE * blockIdx.x]));

         writer_warp<WRITE_COL_LEN, CHUNK_SIZE>(out_queue,d);
    }


}



namespace deflate {

template <typename READ_COL_TYPE>
 __host__ void decompress_gpu(const uint8_t* const in, uint8_t** out, const uint64_t in_n_bytes, uint64_t* out_n_bytes) {


    std::string file_col_len = "./input_data/col_len.bin";
    std::string file_blk_off = "./input_data/blk_offset.bin";

    const char *filename_col_len = file_col_len.c_str();
    const char *filename_blk_off = file_blk_off.c_str();
    int fd_col_len;
    int fd_blk_off;
     
    struct stat sbcol_len;
    struct stat sbblk_off;

    if((fd_col_len = open(filename_col_len, O_RDONLY)) == -1){
        printf("Fatal Error: Col Len read error\n");
        return;
    }
    if((fd_blk_off = open(filename_blk_off, O_RDONLY)) == -1){
        printf("Fatal Error: Block off read error\n");
        return;
    }

    fstat(fd_col_len, &sbcol_len);
    fstat(fd_blk_off, &sbblk_off);

    void* map_base_col_len;
    void* map_base_blk_off;

    map_base_col_len = mmap(NULL, sbcol_len.st_size, PROT_READ, MAP_SHARED, fd_col_len, 0);   
    map_base_blk_off = mmap(NULL, sbblk_off.st_size, PROT_READ, MAP_SHARED, fd_blk_off, 0);
   
    uint64_t num_blk = ((uint64_t) sbblk_off.st_size / sizeof(uint64_t)) - 1;
    //uint64_t num_blk = 2;

    printf("num blk: %llu\n", num_blk);

    uint8_t* d_in;
    uint64_t* d_col_len;
    uint64_t* d_blk_offset;
   
    cuda_err_chk(cudaMalloc(&d_in, in_n_bytes));
    cuda_err_chk(cudaMalloc(&d_col_len, sbcol_len.st_size));
    cuda_err_chk(cudaMalloc(&d_blk_offset, sbblk_off.st_size));

    cuda_err_chk(cudaMemcpy(d_in, in, in_n_bytes, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_col_len, map_base_col_len, sbcol_len.st_size, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(d_blk_offset, map_base_blk_off, sbblk_off.st_size, cudaMemcpyHostToDevice));


    const size_t chunk_size = 8192 * 8;
    uint64_t out_bytes = chunk_size * num_blk;
    printf("out_bytes: %llu\n", out_bytes);

    uint8_t* d_out;
    *out_n_bytes = out_bytes;
    cuda_err_chk(cudaMalloc(&d_out, out_bytes));

    printf("start inflation\n");

    dim3 blockD(32,3,1);
    dim3 gridD(num_blk,1,1);
    inflate<READ_COL_TYPE, 512, chunk_size> <<<gridD,blockD>>> (d_in, d_col_len, d_blk_offset, d_out);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
      {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
      }
    printf("inflation done\n");


    *out = new uint8_t[out_bytes];
    cuda_err_chk(cudaMemcpy((*out), d_out, out_bytes, cudaMemcpyDeviceToHost));

    close(fd_col_len);
    close(fd_blk_off);

    cuda_err_chk(cudaFree(d_out));
    cuda_err_chk(cudaFree(d_in));
    cuda_err_chk(cudaFree(d_col_len));
    cuda_err_chk(cudaFree(d_blk_offset));
 }


}

#endif // __ZLIB_H__
