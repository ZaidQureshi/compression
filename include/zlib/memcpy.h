template<typename T>
__device__
void write_literal(T* out_ptr, uint32_t* counter, uint8_t idx, uint8_t b){
    if(threadIdx.x == idx){
        out_ptr[counter] = b;
        *counter++;
    }
}

/*
    @brief Copies len bytes into memory index based on offset and written counter.

    @param out_ptr Pointer to memory location to write to. Also contains the data to write.
    @param counter The total amount of bytes written.
    @param idx The leader thread. Determines where to start writing and shares its counter.
    @param len The amount of bytes to run.
    @param offset The amount of bytes offset from counter to start READING from.
    @param MASK bitmask corresponding to the active threads in the warp for this operation.


*/
template<typename T>
__device__
void memcpy_onebyte(T* out_ptr, uint32_t* counter, uint8_t idx, uint16_t len, uint16_t offset, uint32_t MASK, uint32_t counter, T* out_ptr) {
    int tid = threadIdx.x;
    uint32_t orig_counter = __shfl_sync(MASK, *counter, idx);
    uint8_t active_threads = __popc(MASK);

    uint8_t num_writes = (active_threads + len - tid - 1) / active_threads;
    if (tid > len) num_writes = 0;

    uint32_t start_counter =  orig_counter - offset; // The place to start writing. Notice we subtract offset, not add it.

    uint32_t read_counter = start_counter + tid; // Start reading from the original counter minus the offset
    uint32_t write_counter = orig_counter + tid; // Start writing from the original counter

    if (read_counter >= orig_counter) {
        read_counter = (read_counter - orig_counter) % offset + start_counter; // Don't really know what this line does.
    }

    uint8_t num_ph = 1 + (len - 1) / active_threads;

    for (int i = 0; i < num_ph; i++) {
        if (i < num_writes) {
            out_ptr[write_counter] = out_ptr[read_counter];

            read_counter += active_threads;
            write_counter += active_threads;
        }
        _syncwarp();
    }

    if (threadIdx.x == idx)
        *counter += len;
}



/*
    @brief Copies len bytes into memory index based on offset and written counter.
    This starts running into a suffix issue with divergence in the last thread.

    @param out_ptr Pointer to memory location to write to. Also contains the data to write.
    @param counter The total amount of bytes written.
    @param idx The leader thread. Determines where to start writing and shares its counter.
    @param len The amount of bytes to run.
    @param offset The amount of bytes offset from counter to start READING from.
    @param MASK bitmask corresponding to the active threads in the warp for this operation.


*/
template<typename T, uint8_t bytes>
__device__
void memcpy_onebyte_consecutive(T* out_ptr, uint32_t* counter, uint8_t idx, uint16_t len, uint16_t offset, uint32_t MASK) {
    int tid = threadIdx.x;
    uint32_t orig_counter = __shfl_sync(MASK, *counter, idx);
    uint8_t active_threads = __popc(MASK);
    uint16_t bytes_per_stride = active_threads * bytes;

    uint8_t num_writes = (bytes_per_stride + len - (tid * bytes) - 1) / bytes_per_stride;
    if (tid * bytes > len) num_writes = 0;

    uint32_t start_counter =  orig_counter - offset; // The place to start writing. Notice we subtract offset, not add it.

    uint32_t read_counter = start_counter + tid * bytes; // Start reading from the original counter minus the offset
    uint32_t write_counter = orig_counter + tid * bytes; // Start writing from the original counter

    if (read_counter >= orig_counter) {
        read_counter = (read_counter - orig_counter) % offset + start_counter; // Don't really know what this line does.
    }

    uint8_t num_ph = 1 + (len - 1) / active_threads;

    for (int i = 0; i < num_ph; i++) {
        if (i < num_writes) {
            // We have an entire 'bytes' available to write.
            if (len - i * bytes_per_stride - tid * bytes > bytes) {
                for (int c = 0; c < bytes; c++)
                    out_ptr[write_counter + c] = out_ptr[read_counter + c];

            }
            else {
                for (int c = 0; c < len - i * bytes_per_stride - tid * bytes) {
                    out_ptr[write_counter + c] = out_ptr[read_counter + c];

                }
            }
            read_counter += bytes_per_stride;
            write_counter += bytes_per_stride;

        }
        _syncwarp();
    }

    if (threadIdx.x == idx)
        *counter += len;
}

    template <uint32_t NUM_THREAD = 8>
   // __forceinline__ 
    __device__
    void col_memcpy_div(uint8_t idx, uint16_t len, uint16_t offset, uint8_t div, uint32_t MASK) {
        // if (len <= 32) {
        //     int tid = threadIdx.x - div * NUM_THREAD;
        //     uint32_t orig_counter = __shfl_sync(MASK, counter, idx);

        //     uint8_t num_writes = ((len - tid + NUM_THREAD - 1) / NUM_THREAD);
        //     uint32_t start_counter =  orig_counter - offset;


        //     uint32_t read_counter = start_counter + tid;
        //     uint32_t write_counter = orig_counter + tid;

        //     if(read_counter >= orig_counter){
        //             read_counter = (read_counter - orig_counter) % offset + start_counter;
        //     }

        //     uint8_t num_ph =  (len +  NUM_THREAD - 1) / NUM_THREAD;
        //     //#pragma unroll 
        //     for(int i = 0; i < num_ph; i++){
        //         if(i < num_writes){
        //             out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
        //             read_counter += NUM_THREAD;
        //             write_counter += NUM_THREAD;
        //         }
        //         __syncwarp();
        //     }
        
        //     //set the counter
        //     if(threadIdx.x == idx)
        //         counter += len;

        // } else {
            /* Memcpy aligned on write boundaries */
            int tid = threadIdx.x - div * NUM_THREAD;
            uint32_t orig_counter = __shfl_sync(MASK, counter, idx); // This is equal to the tid's counter value upon entering the function

            // prefix aligning bytes needed
            uint8_t prefix_bytes = (uint8_t) (NMEMCPY - (orig_counter & MEMCPYSMALLMASK));
            if (prefix_bytes == NMEMCPY) prefix_bytes = 0;
            // prefix_bytes = 0;

            // suffix aligning bytes needed
            uint8_t suffix_bytes = (uint8_t) ((orig_counter + len) & MEMCPYSMALLMASK);
            // prefix_bytes += suffix_bytes;
            // suffix_bytes = 0;

            // TODO: CHANGE
            uint32_t start_counter =  orig_counter - offset; // The place to start writing. Notice we subtract offset, not add it.

            uint32_t read_counter = start_counter + tid; // Start reading from the original counter minus the offset
            uint32_t write_counter = orig_counter + tid; // Start writing from the original counter

            if(read_counter >= orig_counter){ // If we are reading from ahead of the place we are writing.
                    read_counter = (read_counter - orig_counter) % offset + start_counter; // Don't really know what this line does.
            }

            uint8_t num_writes = ((len - prefix_bytes - suffix_bytes - tid * NMEMCPY + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY));
            if (tid * NMEMCPY + prefix_bytes + suffix_bytes > len) num_writes = 0;

            uint8_t num_ph =  (len - prefix_bytes - suffix_bytes + (NUM_THREAD * NMEMCPY) - 1) / (NUM_THREAD * NMEMCPY); // The largest amount of times a thread in the warp is writing to memory.
            if (prefix_bytes + suffix_bytes > len) num_ph = 0;

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

            #endif
            #if NMEMCPY == 4
            uchar4* out_ptr_temp  = reinterpret_cast<uchar4*>(out_ptr);

            #endif

            __syncwarp();

            //#pragma unroll 
            for(int i = 0; i < num_ph; i++){
                if(i < num_writes){ // If this thread should write. 4 bytes
                    // TODO: CHANGE
                    #if NMEMCPY == 1
                    out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
                    #endif
                    #if NMEMCPY == 2
                    if ((read_counter + WRITE_COL_LEN * idx) & MEMCPYSMALLMASK == 0) {
                        out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = out_ptr_temp[(read_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2];
                    } else {
                        out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_uchar2(out_ptr[read_counter + WRITE_COL_LEN * idx], out_ptr[read_counter + WRITE_COL_LEN * idx + 1]);
                    }
                    // out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
                    // out_ptr[write_counter + WRITE_COL_LEN * idx + 1] = out_ptr[read_counter + WRITE_COL_LEN * idx + 1];
                    #endif
                    #if NMEMCPY == 4
                    if ((read_counter + WRITE_COL_LEN * idx) & MEMCPYSMALLMASK == 0) {
                        out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = out_ptr_temp[(read_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2];
                    } else {
                        out_ptr_temp[(write_counter + WRITE_COL_LEN * idx) >> NMEMCPYLOG2] = make_uchar4(out_ptr[read_counter + WRITE_COL_LEN * idx], out_ptr[read_counter + WRITE_COL_LEN * idx + 1],
                                                                                                         out_ptr[read_counter + WRITE_COL_LEN * idx + 2], out_ptr[read_counter + WRITE_COL_LEN * idx + 3]);
                    }
                    // out_ptr[write_counter + WRITE_COL_LEN * idx] = out_ptr[read_counter + WRITE_COL_LEN * idx];
                    // out_ptr[write_counter + WRITE_COL_LEN * idx + 1] = out_ptr[read_counter + WRITE_COL_LEN * idx + 1];
                    // out_ptr[write_counter + WRITE_COL_LEN * idx + 2] = out_ptr[read_counter + WRITE_COL_LEN * idx + 2];
                    // out_ptr[write_counter + WRITE_COL_LEN * idx + 3] = out_ptr[read_counter + WRITE_COL_LEN * idx + 3];
                    #endif
                    // 1 4 byte transaction | 4 1 byte transactions
                    // char4 out_ptr[idx] = // 4 1 bytes read

                    read_counter += NUM_THREAD * NMEMCPY; // Add the number of bytes that we wrote.
                    write_counter += NUM_THREAD * NMEMCPY; // Add the number of bytes that we wrote.
                } 
                // #endif
                __syncwarp(); // Synchronize.
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
            if(threadIdx.x == idx)
                counter += len; // Counter is by thread
        // }

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




