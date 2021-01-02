#ifndef __ZLIB_H__
#define __ZLIB_H__

#define BUFF_LEN 2


#define UNCOMP 0
#define STATIC 1
#define DYNAMIC 2

struct dynamic_huff {

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
    
};


//producer of input queue
template <typename READ_COL_TYPE>
struct decompress_input {


    uint64_t col_width = sizeof(READ_COL_TYPE);
    uint64_t row_offset;
    uint64_t len;
    uint64_t read_bytes;
    uint32_t t_read_mask;

    READ_COL_TYPE* pointer;



    //READ_COL_TYPE in_buff[IN_BUFF_LEN];

    __device__
    decompress_input(const uint8_t* ptr, const uint64_t l) :
        pointer((READ_COL_TYPE*) ptr), len(l) {



        uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;
        t_read_mask = (0xffffffff >> (32 - tid));
        row_offset = 0;
        used_bytes = 0;
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
template <typename READ_COL_TYPE, size_t BUFF_LEN = 3>
struct input_stream {

    union buff {
        READ_COL_TYPE b[BUFF_LEN];
        uint32_t u[(sizeof(READ_COL_TYPE)* BUFF_LEN)/sizeof(uint32_t)];
    } b;
    uint32_t head;
    uint32_t count;
    uint32_t bit_offset;
    uint32_t uint_head;
    uint32_t uint_count;
    queue<READ_COL_TYPE>* q;
    uint32_t read_bytes;
    uint32_t expected_bytes;
    uint32_t bu_size = (sizeof(READ_COL_TYPE)* BUFF_LEN)/sizeof(uint32_t);


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
        for (; (count < BUFF_LEN) && (read_bytes < expected_bytes);
             count++, read_bytes += sizeof(READ_COL_TYPE), uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t)) {
            q->dequeue(b.b + count);


        }
    }

    template<typename T>
    __device__
    void get_n_bits(const uint32_t n, T* out) {
        assert(sizeof(T) < sizeof(uint32_t));
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
        uint32_t b;

        do {
            b = b.u[(uint_head+1)%bu_size];

            //(b : a) >> min(uint_bit_offset, 32)

            out_uint[out_uint_offset] = __funnelshift_rc(a, b, uint_bit_offset);
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
                    head = (head + 1) % BUFF_LEN;
                    count--;
                }

                uint_count--;

            }
            out_uint_offset++;

            a = b;

        } while (counted < n);




    }
    template<typename T>
    __device__
    void fetch_n_bits(const uint32_t n, T* out) {

        while ((count < BUFF_LEN) && (read_bytes < expected_bytes)) {
            q->dequeue(b.b + ((head+count) % BUFF_LEN));
            count++;
            uint_count += sizeof(READ_COL_TYPE)/sizeof(uint32_t);
            read_bytes += sizeof(READ_COL_TYPE);
        }

        get_n_bits(n, out);


    }

    template<typename T>
    __device__
    void peek_n_bits(const uint32_t n, T* out) {
        while ((count < BUFF_LEN) && (read_bytes < expected_bytes)) {
            q->dequeue(b.b + ((head+count) % BUFF_LEN));
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


//if dynamic
//build tree
//
//decode length/distance pairs or literal
//
template <typename READ_COL_TYPE, size_t IN_BUFF_LEN = 3>
__device__
void decoder_warp(input_stream<READ_COL_TYPE, IN_BUFF_LEN>& s,  queue<uint64_t>& mq) {
    uint8_t hlit;
    uint8_t hdist;
    uint8_t hclen;
    dynamic_huffman tree;
    bool cont = false;
    uint8_t prev_type = UNCOMP;
    READ_COL_TYPE local;



    //decode stuff
    //
    //

    mq.enqueue(val);
    uint32_t v;
    s.get_n_bits(32, &v);





}

// __device__
// void writer_warp(queue<uint64_t>& mq, decompress_output<size_t WRITE_COL_LEN = 512>& out) {
//     uint32_t done = 0;
//     while (!done) {


//         bool deq = false;
//         uint64_t v = 0;
//         mq.attempt_dequeue(&v, &deq);
//         uint32_t deq_mask = __ballot_sync(FULLMASK, deq);
//         uint32_t deq_count = __popc(deq_mask);
//         for (size_t i = 0; i < deq_count; i++) {
//             int32_t f = __ffs(deq_mask);
//             uint64_t vv = __shfl_sync(FULLMASK, v, f-1);
//             uint32_t len = vv >> 32;
//             uint32_t offset = vv;
//             out.col_memcpy(f-1,len, offset);
//             deq_mask >>= f;
//         }
//         done = __ballot_sync(FULLMASK, out.done != out.total) == 0;

//     }


// }

__global__
void inflate() {
    __shared__ READ_COL_TYPE in_queue_[32][5];
    __shared__ simt::atomic<uint32_t> h[32];
    __shared__ simt::atomic<uint32_t> t[32];


    queue in_queue(in_queue_[lane_id], &h[lane_id], &t[lane_id], 5);


    __shared__ uint64_t out_queue_[32][5];
    __shared__ simt::atomic<uint32_t> h_[32];
    __shared__ simt::atomic<uint32_t> t_[32];


    queue out_queue(out_queue_[lane_id], &h[lane_id], &t[lane_id], 5);


    if (reader_warp) {
        decompress_input d(ptr, col_len);
        reader_warp(d, in_queue);
    }
    if (decoder_warp) {
        input_stream<READ_COL_TYPE, IN_BUFF_LEN> s(in_queue, col_len);
        decoder_warp(s,out_queue);
    }
    if (writer_warp) {
        decompress_output<> d();
        writer_warp(d, out_queue);
    }


}


#endif // __ZLIB_H__
