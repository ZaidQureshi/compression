#include <memcpy_1.h>


template<typename T, uint32_t THREADS_PER = 32>
__global__
void memcpy_onebyte_kernel_1(uint32_t* offsets, uint32_t* lengths, uint32_t* divisions, T* output, size_t num_columns, size_t column_size, size_t initial_bytes) {
    __shared__ uint32_t offsets_s[256];
    __shared__ uint32_t lengths_s[256];
    uint32_t lane_id = threadIdx.x % THREADS_PER;
    size_t col_idx = threadIdx.x / THREADS_PER + blockIdx.x * (blockDim.x / THREADS_PER);
    uint32_t div_min = divisions[col_idx];
    uint32_t div_max = divisions[col_idx+1];
    uint32_t counter = (column_size) * col_idx + initial_bytes;
    uint32_t s_idx = 0;
    uint8_t leader_id;
    uint32_t mask;
    uint8_t shared_offset = (threadIdx.x/THREADS_PER) * THREADS_PER;
    if (threadIdx.x % 32 < THREADS_PER) {
        leader_id = 0;
        if (THREADS_PER == 16)
            mask = 0x0000ffff;
        else
            mask = 0xffffffff;
        // mask = 0xffff0000;

    }
    else {
        leader_id = THREADS_PER;
        mask = 0xffff0000;
        // mask = 0x0000ffff;

    }
    for (uint32_t iter = 0; iter < 30; iter++) {
        s_idx = 0;
        for (uint32_t c = div_min; c < div_max; c++) {
            if (s_idx == 0) {
                if (c + lane_id < div_max) {
                    offsets_s[shared_offset+lane_id] = offsets[c+lane_id];
                    lengths_s[shared_offset+lane_id] = lengths[c+lane_id];
                }
                __syncwarp();
            }
            // Call the memcpy operation
            memcpy_onebyte_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);

            s_idx = (s_idx + 1) % THREADS_PER;
        }
        // __syncthreads();
    }
}


template<typename T, uint32_t THREADS_PER = 32>
__global__
void memcpy_nbyte_prefix_body_suffix_kernel_1(uint32_t* offsets, uint32_t* lengths, uint32_t* divisions, T* output, size_t num_columns, size_t column_size, size_t initial_bytes, int n_byte) {
    __shared__ uint32_t offsets_s[256];
    __shared__ uint32_t lengths_s[256];
    uint32_t lane_id = threadIdx.x % THREADS_PER;
    size_t col_idx = threadIdx.x / THREADS_PER + blockIdx.x * (blockDim.x / THREADS_PER);
    uint32_t div_min = divisions[col_idx];
    uint32_t div_max = divisions[col_idx+1];
    uint32_t counter = (column_size) * col_idx + initial_bytes;
    uint32_t s_idx = 0;
    uint8_t leader_id;
    uint32_t mask;
    uint8_t shared_offset = (threadIdx.x/THREADS_PER) * THREADS_PER;
    if (threadIdx.x % 32 < THREADS_PER) {
        leader_id = 0;
        if (THREADS_PER == 16)
            mask = 0x0000ffff;
        else
            mask = 0xffffffff;
        // mask = 0xffff0000;

    }
    else {
        leader_id = THREADS_PER;
        mask = 0xffff0000;
        // mask = 0x0000ffff;

    }

    for (uint32_t c = div_min; c < div_max; c++) {
        if (s_idx == 0) {
            if (c + lane_id < div_max) {
                offsets_s[shared_offset+lane_id] = offsets[c+lane_id];
                lengths_s[shared_offset+lane_id] = lengths[c+lane_id];
            }
            __syncwarp();
        }
        // Call the memcpy operation
        memcpy_nbyte_prefix_body_suffix_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);
        // if (lane_id == 0)
            // printf("%d\n", counter);
        s_idx = (s_idx + 1) % THREADS_PER;
    }
}


template<typename T, uint32_t THREADS_PER = 32>
__global__
void memcpy_nbyte_prefix_body_suffix_needs_writing_alignment_kernel_1(uint32_t* offsets, uint32_t* lengths, uint32_t* divisions, T* output, size_t num_columns, size_t column_size, size_t initial_bytes, int n_byte) {
    __shared__ uint32_t offsets_s[256];
    __shared__ uint32_t lengths_s[256];
    uint32_t lane_id = threadIdx.x % THREADS_PER;
    size_t col_idx = threadIdx.x / THREADS_PER + blockIdx.x * (blockDim.x / THREADS_PER);
    uint32_t div_min = divisions[col_idx];
    uint32_t div_max = divisions[col_idx+1];
    uint32_t counter = (column_size) * col_idx + initial_bytes;
    uint32_t s_idx = 0;
    uint8_t leader_id;
    uint32_t mask;
    uint8_t shared_offset = (threadIdx.x/THREADS_PER) * THREADS_PER;
    if (threadIdx.x % 32 < THREADS_PER) {
        leader_id = 0;
        if (THREADS_PER == 16)
            mask = 0x0000ffff;
        else
            mask = 0xffffffff;
        // mask = 0xffff0000;

    }
    else {
        leader_id = THREADS_PER;
        mask = 0xffff0000;
        // mask = 0x0000ffff;

    }

    for (uint32_t c = div_min; c < div_max; c++) {
        if (s_idx == 0) {
            if (c + lane_id < div_max) {
                offsets_s[shared_offset+lane_id] = offsets[c+lane_id];
                lengths_s[shared_offset+lane_id] = lengths[c+lane_id];
            }
            __syncwarp();
        }
        // Call the memcpy operation
        memcpy_nbyte_prefix_body_suffix_needs_writing_alignment_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);

        s_idx = (s_idx + 1) % THREADS_PER;
    }
}

template<typename T, uint32_t THREADS_PER = 32>
__global__
void memcpy_nbyte_prefix_body_suffix_funnelshift_shared_kernel_1(uint32_t* offsets, uint32_t* lengths, uint32_t* divisions, T* output, size_t num_columns, size_t column_size, size_t initial_bytes, int n_byte) {
    __shared__ uint32_t offsets_s[256];
    __shared__ uint32_t lengths_s[256];
    __shared__ FUNNEL_TYPE arr_shared[256];
    uint32_t lane_id = threadIdx.x % THREADS_PER;
    size_t col_idx = threadIdx.x / THREADS_PER + blockIdx.x * (blockDim.x / THREADS_PER);
    uint32_t div_min = divisions[col_idx];
    uint32_t div_max = divisions[col_idx+1];
    uint32_t counter = (column_size) * col_idx + initial_bytes;
    uint32_t s_idx = 0;
    uint8_t leader_id;
    uint32_t mask;
    uint8_t shared_offset = (threadIdx.x/THREADS_PER) * THREADS_PER;
    if (threadIdx.x % 32 < THREADS_PER) {
        leader_id = 0;
        if (THREADS_PER == 16)
            mask = 0x0000ffff;
        else
            mask = 0xffffffff;
        // mask = 0xffff0000;

    }
    else {
        leader_id = THREADS_PER;
        mask = 0xffff0000;
        // mask = 0x0000ffff;

    }

    for (uint32_t c = div_min; c < div_max; c++) {
        if (s_idx == 0) {
            if (c + lane_id < div_max) {
                offsets_s[shared_offset+lane_id] = offsets[c+lane_id];
                lengths_s[shared_offset+lane_id] = lengths[c+lane_id];
            }
            __syncwarp();
        }
        // Call the memcpy operation
        if (min(lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset]) < 116)
            memcpy_nbyte_prefix_body_suffix_funnelshift_shared_1<THREADS_PER, T, FUNNEL_TYPE>(output, &counter, arr_shared+threadIdx.x/THREADS_PER*32, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);
        else
            memcpy_nbyte_prefix_body_suffix_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);

        s_idx = (s_idx + 1) % THREADS_PER;
    }
}

template<typename T, uint32_t THREADS_PER = 32>
__global__
void memcpy_hybrid_nbyte_prefix_body_suffix_kernel_1(uint32_t* offsets, uint32_t* lengths, uint32_t* divisions, T* output, size_t num_columns, size_t column_size, size_t initial_bytes, int n_byte, int length_threshold, int offset_threshold) {
    __shared__ uint32_t offsets_s[256];
    __shared__ uint32_t lengths_s[256];
    uint32_t lane_id = threadIdx.x % THREADS_PER;
    size_t col_idx = threadIdx.x / THREADS_PER + blockIdx.x * (blockDim.x / THREADS_PER);
    uint32_t div_min = divisions[col_idx];
    uint32_t div_max = divisions[col_idx+1];
    uint32_t counter = (column_size) * col_idx + initial_bytes;
    uint32_t s_idx = 0;
    uint8_t leader_id;
    uint32_t mask;
    uint8_t shared_offset = (threadIdx.x/THREADS_PER) * THREADS_PER;
    if (threadIdx.x % 32 < THREADS_PER) {
        leader_id = 0;
        if (THREADS_PER == 16)
            mask = 0x0000ffff;
        else
            mask = 0xffffffff;
        // mask = 0xffff0000;

    }
    else {
        leader_id = THREADS_PER;
        mask = 0xffff0000;
        // mask = 0x0000ffff;

    }

    for (uint32_t c = div_min; c < div_max; c++) {
        if (s_idx == 0) {
            if (c + lane_id < div_max) {
                offsets_s[shared_offset+lane_id] = offsets[c+lane_id];
                lengths_s[shared_offset+lane_id] = lengths[c+lane_id];
            }
            __syncwarp();
        }
        // Call the memcpy operation
        if (lengths_s[s_idx + leader_id] <= length_threshold || offsets_s[s_idx + leader_id] <= offset_threshold) {
            memcpy_onebyte_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);
        } else {
            memcpy_nbyte_prefix_body_suffix_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);
        }
        s_idx = (s_idx + 1) % THREADS_PER;
    }
}


template<typename T, uint32_t THREADS_PER = 32>
__global__
void memcpy_hybrid_nbyte_prefix_body_suffix_needs_writing_alignment_kernel_1(uint32_t* offsets, uint32_t* lengths, uint32_t* divisions, T* output, size_t num_columns, size_t column_size, size_t initial_bytes, int n_byte, int length_threshold, int offset_threshold) {
    __shared__ uint32_t offsets_s[256];
    __shared__ uint32_t lengths_s[256];
    uint32_t lane_id = threadIdx.x % THREADS_PER;
    size_t col_idx = threadIdx.x / THREADS_PER + blockIdx.x * (blockDim.x / THREADS_PER);
    uint32_t div_min = divisions[col_idx];
    uint32_t div_max = divisions[col_idx+1];
    uint32_t counter = (column_size) * col_idx + initial_bytes;
    uint32_t s_idx = 0;
    uint8_t leader_id;
    uint32_t mask;
    uint8_t shared_offset = (threadIdx.x/THREADS_PER) * THREADS_PER;
    if (threadIdx.x % 32 < THREADS_PER) {
        leader_id = 0;
        if (THREADS_PER == 16)
            mask = 0x0000ffff;
        else
            mask = 0xffffffff;
        // mask = 0xffff0000;

    }
    else {
        leader_id = THREADS_PER;
        mask = 0xffff0000;
        // mask = 0x0000ffff;

    }

    for (uint32_t c = div_min; c < div_max; c++) {
        if (s_idx == 0) {
            if (c + lane_id < div_max) {
                offsets_s[shared_offset+lane_id] = offsets[c+lane_id];
                lengths_s[shared_offset+lane_id] = lengths[c+lane_id];
            }
            __syncwarp();
        }
        // Call the memcpy operation
        if (lengths_s[s_idx + leader_id] <= length_threshold || offsets_s[s_idx + leader_id] <= offset_threshold) {
            memcpy_onebyte_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);
        } else {
            memcpy_nbyte_prefix_body_suffix_needs_writing_alignment_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);
        }
        s_idx = (s_idx + 1) % THREADS_PER;
    }
}

template<typename T, uint32_t THREADS_PER = 32>
__global__
void memcpy_hybrid_nbyte_prefix_body_suffix_shared_kernel_1(uint32_t* offsets, uint32_t* lengths, uint32_t* divisions, T* output, size_t num_columns, size_t column_size, size_t initial_bytes, int n_byte, int length_threshold, int offset_threshold) {
    __shared__ uint32_t offsets_s[256];
    __shared__ uint32_t lengths_s[256];
    __shared__ T shared_data[1024];

    uint32_t lane_id = threadIdx.x % THREADS_PER;
    size_t col_idx = threadIdx.x / THREADS_PER + blockIdx.x * (blockDim.x / THREADS_PER);
    uint32_t div_min = divisions[col_idx];
    uint32_t div_max = divisions[col_idx+1];
    uint32_t counter = (column_size) * col_idx + initial_bytes;
    uint32_t s_idx = 0;
    uint8_t leader_id;
    uint32_t mask;
    uint8_t shared_offset = (threadIdx.x/THREADS_PER) * THREADS_PER;
    if (threadIdx.x % 32 < THREADS_PER) {
        leader_id = 0;
        if (THREADS_PER == 16)
            mask = 0x0000ffff;
        else
            mask = 0xffffffff;
        // mask = 0xffff0000;

    }
    else {
        leader_id = THREADS_PER;
        mask = 0xffff0000;
        // mask = 0x0000ffff;

    }

    for (uint32_t c = div_min; c < div_max; c++) {
        if (s_idx == 0) {
            if (c + lane_id < div_max) {
                offsets_s[shared_offset+lane_id] = offsets[c+lane_id];
                lengths_s[shared_offset+lane_id] = lengths[c+lane_id];
            }
            __syncwarp();
        }
        // Call the memcpy operation
        if (lengths_s[s_idx + leader_id] <= length_threshold || offsets_s[s_idx + leader_id] <= offset_threshold) {
            memcpy_onebyte_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);
        } else if (offsets_s[s_idx + leader_id] < THREADS_PER*2)  {
            memcpy_nbyte_prefix_body_suffix_shared_1<THREADS_PER, T>(output, &counter, shared_data + (shared_offset * 2), leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);
        } else {
            memcpy_onebyte_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);
        }
        s_idx = (s_idx + 1) % THREADS_PER;
    }
}


template<typename T, uint32_t THREADS_PER = 32>
__global__
void memcpy_hybrid_nbyte_prefix_body_suffix_needs_writing_alignment_shared_kernel_1(uint32_t* offsets, uint32_t* lengths, uint32_t* divisions, T* output, size_t num_columns, size_t column_size, size_t initial_bytes, int n_byte, int length_threshold, int offset_threshold) {
    __shared__ uint32_t offsets_s[256];
    __shared__ uint32_t lengths_s[256];
    __shared__ T shared_data[1024];

    uint32_t lane_id = threadIdx.x % THREADS_PER;
    size_t col_idx = threadIdx.x / THREADS_PER + blockIdx.x * (blockDim.x / THREADS_PER);
    uint32_t div_min = divisions[col_idx];
    uint32_t div_max = divisions[col_idx+1];
    uint32_t counter = (column_size) * col_idx + initial_bytes;
    uint32_t s_idx = 0;
    uint8_t leader_id;
    uint32_t mask;
    uint8_t shared_offset = (threadIdx.x/THREADS_PER) * THREADS_PER;
    if (threadIdx.x % 32 < THREADS_PER) {
        leader_id = 0;
        if (THREADS_PER == 16)
            mask = 0x0000ffff;
        else
            mask = 0xffffffff;
        // mask = 0xffff0000;

    }
    else {
        leader_id = THREADS_PER;
        mask = 0xffff0000;
        // mask = 0x0000ffff;
    }

    for (uint32_t c = div_min; c < div_max; c++) {
        if (s_idx == 0) {
            if (c + lane_id < div_max) {
                offsets_s[shared_offset+lane_id] = offsets[c+lane_id];
                lengths_s[shared_offset+lane_id] = lengths[c+lane_id];
            }
            __syncwarp();
        }
        // Call the memcpy operation
        if (lengths_s[s_idx + leader_id] <= length_threshold || offsets_s[s_idx + leader_id] <= offset_threshold) {
            memcpy_onebyte_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);
        } else if (offsets_s[s_idx + leader_id] < THREADS_PER*2)  {
            memcpy_nbyte_prefix_body_suffix_needs_writing_alignment_shared_1<THREADS_PER, T>(output, &counter, shared_data + (shared_offset * 2), leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);
        } else {
            memcpy_onebyte_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);
        }
        s_idx = (s_idx + 1) % THREADS_PER;
    }
}


template<typename T, uint32_t THREADS_PER = 32>
__global__
void memcpy_nbyte_aligned_kernel_1(uint32_t* offsets, uint32_t* lengths, uint32_t* divisions, T* output, size_t num_columns, size_t column_size, size_t initial_bytes) {
    __shared__ uint32_t offsets_s[256];
    __shared__ uint32_t lengths_s[256];
    uint32_t lane_id = threadIdx.x % THREADS_PER;
    size_t col_idx = threadIdx.x / THREADS_PER + blockIdx.x * (blockDim.x / THREADS_PER);
    uint32_t div_min = divisions[col_idx];
    uint32_t div_max = divisions[col_idx+1];
    uint32_t counter = (column_size) * col_idx + initial_bytes;
    uint32_t s_idx = 0;
    uint8_t leader_id;
    uint32_t mask;
    uint8_t shared_offset = (threadIdx.x/THREADS_PER) * THREADS_PER;
    if (threadIdx.x % 32 < THREADS_PER) {
        leader_id = 0;
        if (THREADS_PER == 16)
            mask = 0x0000ffff;
        else
            mask = 0xffffffff;
        // mask = 0xffff0000;

    }
    else {
        leader_id = THREADS_PER;
        mask = 0xffff0000;
        // mask = 0x0000ffff;

    }

    for (uint32_t c = div_min; c < div_max; c++) {
        if (s_idx == 0) {
            if (c + lane_id < div_max) {
                offsets_s[shared_offset+lane_id] = offsets[c+lane_id];
                lengths_s[shared_offset+lane_id] = lengths[c+lane_id];
            }
            __syncwarp(mask);
        }
        // Call the memcpy operation
        memcpy_nbyte_aligned_1<THREADS_PER, T>(output, &counter, leader_id, lengths_s[s_idx+shared_offset], offsets_s[s_idx+shared_offset], threadIdx.x/THREADS_PER, mask);

        s_idx = (s_idx + 1) % THREADS_PER;
    }
}

