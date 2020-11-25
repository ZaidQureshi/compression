#ifndef __ZLIB_H__
#define __ZLIB_H__

#define BUFF_LEN 2


#define UNCOMP 0
#define STATIC 1
#define DYNAMIC 2

struct dynamic_huff {

};

template <typename T, size_t LEN>
struct queue {
    T* queue;
    simt::atomic<uint32_t, simt::thread_scope_block>* head;
    simt::atomic<uint32_t, simt::thread_scope_block>* tail;

    __device__
    queue(T* buff, simt::atomic<uint32_t, simt::thread_scope_block>* h, simt::atomic<uint32_t, simt::thread_scope_block>* t) {
        queue = buff;
        head = h;
        tail = t;
    }

    __device__
    void enqueue(const T* v) {

        const auto cur_tail = tail->load(simt::memory_order_relaxed);
        const auto next_tail = (cur_tail + 1) % LEN;

        while (next_tail == head->load(simt::memory_order_acquire))
            __nanosleep(100);

        queue[cur_tail] = *v;
        tail->store(next_tail, simt::memory_order_release);


    }

    __device__
    void dequeue(T* v) {

        const auto cur_head = head->load(simt::memory_order_relaxed);
        while (cur_head == tail->load(simt::memory_order_acquire))
            __nanosleep(100);

        *v = queue[cur_head];

        const auto next_head = (cur_head + 1) % LEN;

        head->store(next_head, simt::memory_order_release);


    }
    
};

template <typename READ_COL_TYPE>
struct decompress_input {


    uint64_t col_width = sizeof(READ_COL_TYPE);
    uint64_t row_offset;
    uint64_t len;
    uint64_t read_bytes;
    uint32_t t_read_mask;




    //READ_COL_TYPE in_buff[IN_BUFF_LEN];

    __device__
    decompress_input(const uint8_t* ptr, const uint64_t l, const uint32_t trm) :
        pointer((READ_COL_TYPE*) ptr), len(l), t_read_mask(trm) {

        buf_head = 0;
        buf_count = 0;
        buf_head_bit_offset = 0;
        row_offset = 0;
        used_bytes = 0;
    }

    __device__
    void read_data(const uint32_t alivemask, READ_COL_TYPE* v) {
        #pragma unroll
        for (size_t i = 0; i < IN_BUFF_LEN; i++) {
            bool read = (read_bytes + buf_count) < len;
            uint32_t read_sync = __ballot_sync(alivemask, read);

            if (read) {
                *v = pointer[row_offset + __popc(read_sync & t_read_mask)];
                row_offset += __popc(read_sync);
                read_bytes += sizeof(READ_COL_TYPE);
            }
        }

        __syncwarp(alivemask);

    }


};


void reader_warp(decompress_input<READ_COL_TYPE>& in, queue& rq) {
     while (true) {
        bool alive = in.used_bytes < in.len;
        auto alivemask = __ballot_sync(FULL_MASK, alive);
        if (alive) {
            in.read_data(alivemask);
        }

        if (alivemask == 0)
            break;

    }
}

template <typename READ_COL_TYPE>
struct input_stream {
    READ_COL_TYPE b[2];
    uint32_t count;
    uint32_t bit_offset;
    queue* q;

    template<typename T>
    void fetch_n_bits(const uint32_t n, T* out) {
        *out = 0;

        bit_offset
    }
};


template <typename READ_COL_TYPE, size_t IN_BUFF_LEN>
__device__
void decoder_warp(stream& s,  queue& mq) {
    uint8_t hlit;
    uint8_t hdist;
    uint8_t hclen;
    dynamic_huffman tree;
    bool cont = false;
    uint8_t prev_type = UNCOMP;
    READ_COL_TYPE local;



    stream.fetch_n_bits()





}

__device__
void memcpy_warp() {


}

__device__
void inflate() {

}


#endif // __ZLIB_H__
