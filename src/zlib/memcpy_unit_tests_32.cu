//#include <brle/brle_trans.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h> 

#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <chrono>
#include <stdlib.h>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream

#include <vector>
#include <string.h>
#include <sstream>
#include <memcpy_kernels_1.h>
#include <memcpy_kernels_2.h>
#include <memcpy_kernels_4.h>
#include <memcpy_kernels_8.h>

#define INITIAL_BYTES 8
#define SEED 50
#define BLOCKS 4
#define THREADS 32
#define SIZE 128
#define OFFSET_MIN 4
#define OFFSET_MAX 8
#define LENGTH_MIN 8
#define LENGTH_MAX 16
#define ALIGNED false
#define ALIGNMENT 4
// #define MULTIPLE 1
#define THREADS_PER 32


#define DEBUG 0
// #define MULTIPLE 1024
#define MULTIPLE 1024*1024


#define DEFAULT_FILE "output.dat"

size_t initial_bytes_global;
size_t reading_alignment_global;
size_t reading_aligned_global;
int length_threshold_global;
int offset_threshold_global;
int n_byte_global;

using namespace std;

// DONE: Change default type of data/output to byte?
// DONE: Make function for test/nsight and function for profiling.
// TODO: Change the file naming to give info about parameters.
// TODO: Make small changes to memcpy to fit these functions.
// TODO: Run with the existing memcpy functions and profile.
// TODO: Fix the default memcpy functions
// TODO: Change the funnelshift to allow more than 32 threads in a warp
// TODO: Change the amount of elements in the shared arrays.

// Profile with read alignment? Some changes in code may be possible for this.


template<typename T>
void save_into_file(string name, vector<T>* output) {
    ofstream outfile (name, ofstream::binary);
    for(auto iter = output->begin(); iter!=output->end(); iter++)
    {
        auto product = *iter;
        char out[sizeof(T)];
        memcpy(&out, &product, sizeof(T));
        outfile.write(out, sizeof(T));
    }
    outfile.close();
}

// Generate the golden solution for the memcpy. Will be slow!
template <typename T>
void memcpy_test_cpu(string fp, uint32_t* offsets, uint32_t* lengths, uint32_t* divisions, 
    uint32_t num_columns, size_t total_elements) {

    vector<T> output;
    for (int col = 0; col < num_columns; col++) {
        uint32_t min_idx = divisions[col];
        uint32_t max_idx = divisions[col+1];
        // printf("Min idx: %d, Max idx: %d \n", min_idx, max_idx);

        for (T c = 0; c < initial_bytes_global; c++) {
            output.push_back(c);
        }

        for (int idx = min_idx; idx < max_idx; idx++) {
            uint32_t offset = offsets[idx];
            uint32_t length = lengths[idx];

            uint32_t cur_idx = output.size() - offset;

            for (int c = 0; c < length; c++) {
                output.push_back(output[cur_idx+c]);
            }
        }
    }
    #if DEBUG == 1
    for (int c = 0; c < total_elements; c++) {
        printf("%d ", output[c]);
    }
    printf("\n");
    #endif

    // Save the output to a file
    save_into_file<T>(fp, &output);

    // Read the file
    // char out[4];
    // uint32_t character;
    // ifstream infile ("golden.dat", ifstream::binary);
    // infile.seekg (0,infile.end);
    // size_t characters = infile.tellg();
    // infile.seekg(0);
    // for (size_t c = 0; c < characters; c+=sizeof(uint32_t)) {
    //     infile.get(out, sizeof(uint32_t));
    //     memcpy(&out, &character, sizeof(uint32_t));
    //     cout << character;
    // }
    // infile.close();

}


// https://stackoverflow.com/questions/15118661/in-c-whats-the-fastest-way-to-tell-whether-two-string-or-binary-files-are-di
bool compare_files(const std::string& filename1, const std::string& filename2)
{
    std::ifstream file1(filename1, std::ifstream::ate | std::ifstream::binary); //open file at the end
    std::ifstream file2(filename2, std::ifstream::ate | std::ifstream::binary); //open file at the end
    const std::ifstream::pos_type fileSize = file1.tellg();

    if (fileSize != file2.tellg()) {
        return false; //different file size
    }

    file1.seekg(0); //rewind
    file2.seekg(0); //rewind

    std::istreambuf_iterator<char> begin1(file1);
    std::istreambuf_iterator<char> begin2(file2);

    return std::equal(begin1,std::istreambuf_iterator<char>(),begin2); //Second argument is end-of-range iterator
}

/* Memcpy test
 * 
 * blocks: Number of blocks
 * threads: Number of threads per block
 * total_elements: The size of the decompressed output
 * offset_min: The minimum allowed offset when generating a new sample point.
 * offset_max: The maximum allowed offset when generating a new sample point.
 * length_min: The minimum allowed length when generating a new sample point.
 * length_max: The maximum allowed length when generating a new sample point.
 * aligned: Whether the data should be forced along a K-Byte alignment in writing.
 * alignment: The amount of bytes the data should be forced to be aligned on.
 * 
 */
template <typename T>
float memcpy_test(vector<uint32_t>* offsets_pointer, vector<uint32_t>* lengths_pointer, vector<uint32_t>* divisions_pointer, 
    size_t blocks, size_t threads, size_t total_elements, uint32_t offset_min, 
 uint32_t offset_max, uint32_t length_min, uint32_t length_max, bool aligned, uint32_t alignment, uint32_t memcpy_function,
 bool validate = true) {

    srand (SEED);

    uint32_t num_columns = (threads / THREADS_PER) * blocks;
    size_t column_size = total_elements / num_columns - initial_bytes_global;

    vector<uint32_t> offsets = *offsets_pointer;
    vector<uint32_t> lengths = *lengths_pointer;
    vector<uint32_t> divisions = *divisions_pointer; // num_columns+1 elements. [start, end)
    
    // Generate new data.
    if (offsets.size() == 0) {
        divisions.push_back(0);

        // Generate a set of offsets and lengths for each column.
        for (int col = 0; col < num_columns; col++) {
            size_t cur_size = 0;
            while (cur_size < column_size) {
                uint32_t cur_length = length_max;
                if (length_max != length_min)
                    cur_length = rand() % (length_max - length_min + 1) + length_min;
                if (aligned) {
                    cur_length = (cur_length / alignment) * alignment;
                }
                // printf("Column Size: %d, Cur Size: %d, Cur Length: %d \n", column_size, cur_size, cur_length);
                cur_length = min(cur_length, (uint32_t) (column_size - cur_size));

                uint32_t cur_offset = offset_max;
                if (offset_max != offset_min)
                    cur_offset = rand() % (offset_max - offset_min + 1) + offset_min;
                if (reading_aligned_global)
                    // cur_offset = cur_offset - (cur_offset % reading_alignment_global);
                    cur_offset = (cur_offset / reading_alignment_global) * reading_alignment_global;
                    // cur_offset = cur_offset + ((cur_size + initial_bytes_global - cur_offset) % reading_alignment_global);
                cur_offset = min(cur_offset, (uint32_t) (cur_size + initial_bytes_global));

                offsets.push_back(cur_offset);
                lengths.push_back(cur_length);
                #if DEBUG == 1
                printf("Offset: %d, Length: %d, CurSize: %d \n", cur_offset, cur_length, cur_size);
                #endif

                cur_size += cur_length;
            }
            divisions.push_back(offsets.size());
        }
    }

    // Get the list versions of our indexes.
    uint32_t* offsets_h = offsets.data();
    uint32_t* lengths_h = lengths.data();
    uint32_t* divisions_h = divisions.data();

    // Run the CPU version to generate the golden file.
    if (validate) {
        auto stream = std::stringstream{};
        stream << total_elements/MULTIPLE << "_" << blocks << "," << threads << "_" << offset_min << "," << offset_max << "_" << length_min << "," << length_max 
        << "_" << aligned << "_" << alignment << "_" << initial_bytes_global << ".dat";
        string fp = stream.str();
        memcpy_test_cpu<T>(fp, offsets_h, lengths_h, divisions_h, num_columns, total_elements);
        if (DEBUG)
            printf("Done with CPU validation step.\n");

    }

    // Pass list into function to run tests. Each function will generate its own timing.
    // Compare with golden to ensure correctness.
    uint32_t* offsets_d;
    uint32_t* lengths_d;
    uint32_t* divisions_d;
    T* output_d;

    cuda_err_chk(cudaMalloc(&offsets_d, sizeof(uint32_t) * offsets.size()));
    cuda_err_chk(cudaMalloc(&lengths_d, sizeof(uint32_t) * lengths.size()));
    cuda_err_chk(cudaMalloc(&divisions_d, sizeof(uint32_t) * divisions.size()));
    cuda_err_chk(cudaMalloc(&output_d, sizeof(T) * total_elements));

    cuda_err_chk(cudaMemcpy(offsets_d, offsets_h, sizeof(uint32_t) * offsets.size(), cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(lengths_d, lengths_h, sizeof(uint32_t) * lengths.size(), cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(divisions_d, divisions_h, sizeof(uint32_t) * divisions.size(), cudaMemcpyHostToDevice));

    // Copy the initial bytes for each column
    T* initial = (T*) malloc(sizeof(T) * initial_bytes_global);
    for (int c = 0; c < initial_bytes_global; c++) {
        initial[c] = c;
    }

    for (int col = 0; col < num_columns; col++) {
        cuda_err_chk(cudaMemcpy(output_d + col * (column_size + initial_bytes_global), initial, sizeof(T) * initial_bytes_global, cudaMemcpyHostToDevice));
    }

    dim3 blockDim(threads, 1, 1);
    dim3 gridDim(blocks, 1, 1);

    // Set up timing
    // https://stackoverflow.com/questions/7876624/timing-cuda-operations
    float time;
    cudaEvent_t start, stop;

    cuda_err_chk( cudaEventCreate(&start) );
    cuda_err_chk( cudaEventCreate(&stop) );
    cuda_err_chk( cudaEventRecord(start, 0) );
    switch (memcpy_function) {
        case (0) :
            if (DEBUG)
                printf("Running One Byte Kernel. \n");
            switch (n_byte_global) {
                case (1):
                    memcpy_onebyte_kernel_1<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global);
                    break;
                case (2):
                    memcpy_onebyte_kernel_2<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global);
                    break;
                case (4):
                    memcpy_onebyte_kernel_4<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global);
                    break;
                case (8):
                    memcpy_onebyte_kernel_8<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global);
                    break;
                default:
                    break;
            }
            break;
        case (1) :
            if (DEBUG)
                printf("Running NByte kernel: prefix, body, suffix. N=%d \n", NMEMCPY);
            switch (n_byte_global) {
                case (1):
                    memcpy_nbyte_prefix_body_suffix_kernel_1<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global);
                    break;
                case (2):
                    memcpy_nbyte_prefix_body_suffix_kernel_2<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global);
                    break;
                case (4):
                    memcpy_nbyte_prefix_body_suffix_kernel_4<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global);
                    break;
                case (8):
                    memcpy_nbyte_prefix_body_suffix_kernel_8<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global);
                    break;
                default:
                    break;
            }
            break;
        case (2) :
            if (DEBUG)
                printf("Running NByte kernel: prefix, body, suffix. Needs writing alignment. N=%d \n", NMEMCPY);
            switch (n_byte_global) {
                case (1):
                    memcpy_nbyte_prefix_body_suffix_needs_writing_alignment_kernel_1<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global);
                    break;
                case (2):
                    memcpy_nbyte_prefix_body_suffix_needs_writing_alignment_kernel_2<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global);
                    break;
                case (4):
                    memcpy_nbyte_prefix_body_suffix_needs_writing_alignment_kernel_4<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global);
                    break;
                case (8):
                    memcpy_nbyte_prefix_body_suffix_needs_writing_alignment_kernel_8<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global);
                    break;
                default:
                    break;
            }
            break;
        case (3) :
            if (DEBUG)
                printf("Running NByte kernel: prefix, body, suffix. Funnelshift with shared N=%d \n", NMEMCPY);
            switch (n_byte_global) {
                case (1):
                    memcpy_nbyte_prefix_body_suffix_funnelshift_shared_kernel_1<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global);
                    break;
                case (2):
                    memcpy_nbyte_prefix_body_suffix_funnelshift_shared_kernel_2<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global);
                    break;
                case (4):
                    memcpy_nbyte_prefix_body_suffix_funnelshift_shared_kernel_4<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global);
                    break;
                case (8):
                    memcpy_nbyte_prefix_body_suffix_funnelshift_shared_kernel_8<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global);
                    break;
                default:
                    break;
            }
            break;
        case (4) :
            if (DEBUG)
                printf("Running Hybrid NByte kernel: prefix, body, suffix. N=%d, L=%d, O=%d \n", NMEMCPY, length_threshold_global, offset_threshold_global);
            switch (n_byte_global) {
                case (1):
                    memcpy_hybrid_nbyte_prefix_body_suffix_kernel_1<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                case (2):
                    memcpy_hybrid_nbyte_prefix_body_suffix_kernel_2<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                case (4):
                    memcpy_hybrid_nbyte_prefix_body_suffix_kernel_4<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                case (8):
                    memcpy_hybrid_nbyte_prefix_body_suffix_kernel_8<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                default:
                    break;
            }
            break;
        case (5) :
            if (DEBUG)
                printf("Running Hybrid NByte kernel: prefix, body, suffix. Needs writing alignment. N=%d, L=%d, O=%d \n", NMEMCPY, length_threshold_global, offset_threshold_global);
            switch (n_byte_global) {
                case (1):
                    memcpy_hybrid_nbyte_prefix_body_suffix_needs_writing_alignment_kernel_1<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                case (2):
                    memcpy_hybrid_nbyte_prefix_body_suffix_needs_writing_alignment_kernel_2<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                case (4):
                    memcpy_hybrid_nbyte_prefix_body_suffix_needs_writing_alignment_kernel_4<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                case (8):
                    memcpy_hybrid_nbyte_prefix_body_suffix_needs_writing_alignment_kernel_8<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                default:
                    break;
            }
            break;
        case (6) :
            if (DEBUG)
                printf("Running Hybrid NByte kernel: prefix, body, suffix. Shared. N=%d, L=%d, O=%d \n", NMEMCPY, length_threshold_global, offset_threshold_global);
            switch (n_byte_global) {
                case (1):
                    memcpy_hybrid_nbyte_prefix_body_suffix_shared_kernel_1<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                case (2):
                    memcpy_hybrid_nbyte_prefix_body_suffix_shared_kernel_2<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                case (4):
                    memcpy_hybrid_nbyte_prefix_body_suffix_shared_kernel_4<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                case (8):
                    memcpy_hybrid_nbyte_prefix_body_suffix_shared_kernel_8<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                default:
                    break;
            }
            break;
        case (7) :
            if (DEBUG)
                printf("Running Hybrid NByte kernel: prefix, body, suffix. Shared. Needs writing alignment. N=%d, L=%d, O=%d \n", NMEMCPY, length_threshold_global, offset_threshold_global);
            switch (n_byte_global) {
                case (1):
                    memcpy_hybrid_nbyte_prefix_body_suffix_needs_writing_alignment_shared_kernel_1<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                case (2):
                    memcpy_hybrid_nbyte_prefix_body_suffix_needs_writing_alignment_shared_kernel_2<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                case (4):
                    memcpy_hybrid_nbyte_prefix_body_suffix_needs_writing_alignment_shared_kernel_4<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                case (8):
                    memcpy_hybrid_nbyte_prefix_body_suffix_needs_writing_alignment_shared_kernel_8<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global, n_byte_global, length_threshold_global, offset_threshold_global);
                    break;
                default:
                    break;
            }
            break;
        case (8) :
            if (DEBUG)
                printf("Running NByte kernel: Needs total alignment. N=%d \n", NMEMCPY);
            switch (n_byte_global) {
                case (1):
                    memcpy_nbyte_aligned_kernel_1<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global);
                    break;
                case (2):
                    memcpy_nbyte_aligned_kernel_2<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global);
                    break;
                case (4):
                    memcpy_nbyte_aligned_kernel_4<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global);
                    break;
                case (8):
                    memcpy_nbyte_aligned_kernel_8<T, THREADS_PER><<<gridDim, blockDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size + initial_bytes_global, initial_bytes_global);
                    break;
                default:
                    break;
            }
            break;
        case (9) :
            break;
        case (10):
            break;
        case (11):
            break;
        default :
            printf("Invalid memcpy function number \n");
    }
    
    cuda_err_chk( cudaEventRecord(stop, 0) );
    cuda_err_chk( cudaEventSynchronize(stop) );
    cuda_err_chk( cudaEventElapsedTime(&time, start, stop));


    // Write the output file
    if (validate) {
        T* output_h = (T*) malloc(sizeof(T) * total_elements);
        cuda_err_chk(cudaDeviceSynchronize());
        cuda_err_chk(cudaMemcpy(output_h, output_d, sizeof(T) * total_elements, cudaMemcpyDeviceToHost));

        #if DEBUG == 1
        for (int c = 0; c < total_elements; c++) {
            printf("%d ", output_h[c]);
        }
        printf("\n");
        #endif
        auto stream = std::stringstream{};
        stream << memcpy_function << "_" << total_elements/MULTIPLE << "_" << blocks << "," << threads << "_" << offset_min << "," << offset_max << "_" << length_min << "," << length_max 
        << "_" << aligned << "_" << alignment << "_" << initial_bytes_global << ".dat";
        string fp = stream.str();

        auto stream2 = std::stringstream{};
        stream2 << total_elements/MULTIPLE << "_" << blocks << "," << threads << "_" << offset_min << "," << offset_max << "_" << length_min << "," << length_max 
        << "_" << aligned << "_" << alignment << "_" << initial_bytes_global << ".dat";
        string golden = stream2.str();

        vector<T> output_vector = vector<T>(output_h, output_h + total_elements);
        save_into_file<T>(fp, &output_vector);
        if (compare_files(fp, golden) == false) {
            std::cout << fp << std::endl;
            std::cout << "Error: Data mismatch!" << std::endl;
        }
        delete output_h;

    }

    cuda_err_chk(cudaFree(offsets_d));
    cuda_err_chk(cudaFree(lengths_d));
    cuda_err_chk(cudaFree(divisions_d));
    cuda_err_chk(cudaFree(output_d));

    // Write the time taken
    return time;
}

int main(int argc, char* argv[]) {
    string output_file = DEFAULT_FILE;
    int profiling = 0;
    int memcpy_function = 0;
    int blocks = 32;
    int threads = 128;
    int megabytes = 128;
    int offset_min = 4;
    int offset_max = 8;
    int length_min = 8;
    int length_max = 512;
    int aligned = 0;
    int alignment = 4;
    int reading_aligned = 0;
    int reading_alignment = 4;
    int initial_bytes = 8;
    int length_threshold = 32;
    int offset_threshold = 3;
    int n_byte = 4;
    int iterations = 1;
    int validate = 1;

    if (argc > 2) {
        for (int idx = 2; idx < argc; idx = idx + 2) {
            char* flag = argv[idx-1];
            char* arg = argv[idx];
            if (strcmp(flag, "-fp") == 0) {
                output_file = arg;
            }
            if (strcmp(flag, "-p") == 0) {
                profiling = atoi(arg);
            }
            if (strcmp(flag, "-mcf") == 0) {
                memcpy_function = atoi(arg);
            }
            if (strcmp(flag, "-b") == 0) {
                blocks = atoi(arg);
            }
            if (strcmp(flag, "-t") == 0) {
                threads = atoi(arg);
            }
            if (strcmp(flag, "-mb") == 0) {
                megabytes = atoi(arg);
            }
            if (strcmp(flag, "-omn") == 0) {
                offset_min = atoi(arg);
            }
            if (strcmp(flag, "-omx") == 0) {
                offset_max = atoi(arg);
            }
            if (strcmp(flag, "-lmn") == 0) {
                length_min = atoi(arg);
            }
            if (strcmp(flag, "-lmx") == 0) {
                length_max = atoi(arg);
            }
            if (strcmp(flag, "-a") == 0) {
                aligned = atoi(arg);
            }
            if (strcmp(flag, "-amt") == 0) {
                alignment = atoi(arg);
            }
            if (strcmp(flag, "-ra") == 0) {
                reading_aligned = atoi(arg);
            }
            if (strcmp(flag, "-ramt") == 0) {
                reading_alignment = atoi(arg);
            }
            if (strcmp(flag, "-inb") == 0) {
                initial_bytes = atoi(arg);
            }
            if (strcmp(flag, "-lth") == 0) {
                length_threshold = atoi(arg);
            }
            if (strcmp(flag, "-oth") == 0) {
                offset_threshold = atoi(arg);
            }
            if (strcmp(flag, "-n") == 0) {
                n_byte = atoi(arg);
            }
            if (strcmp(flag, "-it") == 0) {
                iterations = atoi(arg);
            }
            if (strcmp(flag, "-v") == 0) {
                validate = atoi(arg);
            }
        }
    }

    vector<uint32_t> offsets;
    vector<uint32_t> lengths;
    vector<uint32_t> divisions;
    initial_bytes_global = initial_bytes;
    reading_aligned_global = reading_aligned;
    reading_alignment_global = reading_alignment;
    length_threshold_global = length_threshold;
    offset_threshold_global = offset_threshold;
    n_byte_global = n_byte;

    // Run the test function
    if (profiling == 0) {
        float time = 0;
        for (int c = 0; c < iterations; c++) {
        time += memcpy_test<uint8_t>(&offsets, &lengths, &divisions, (size_t) blocks, (size_t) threads, (size_t) (megabytes*MULTIPLE), (uint32_t) offset_min,
            (uint32_t) offset_max, (uint32_t) length_min, (uint32_t) length_max, (aligned != 0), (uint32_t) alignment, memcpy_function, (c == 0 & validate > 0));
        }
        time = time / iterations;

        // float time = memcpy_test<uint8_t>(&offsets, &lengths, &divisions, (size_t) blocks, (size_t) threads, (size_t) (megabytes*MULTIPLE), (uint32_t) offset_min,
            // (uint32_t) offset_max, (uint32_t) length_min, (uint32_t) length_max, (aligned != 0), (uint32_t) alignment);
        fstream file;

        file.open(output_file, std::ios_base::app | std::ios_base::in);
        
        auto stream = std::stringstream{};
        stream << memcpy_function << " | " << blocks << " | " << threads << " | " << megabytes << " | " << offset_min << " | " << offset_max
          << " | " << length_min << " | " << length_max << " | " << (aligned != 0) << " | " << alignment << " | " << (reading_aligned != 0) << " | " << reading_alignment << " | " << initial_bytes << " | " << n_byte_global << std::endl << "Time taken (ms)," << time << std::endl;
        std::cout << stream.str();

        if (file.is_open())
            file << stream.str();
    } else {
        std::cout << memcpy_function << " | " << blocks << " | " << threads << " | " << megabytes << " | " << (aligned != 0) << " | " << alignment << " | " << initial_bytes << " | " << n_byte_global << std::endl;
        // Need to choose the variables to profile over. This function iterates over offset min/max and length min/max
        for (int off_max = offset_max; off_max <= offset_max * 8; off_max *= 2) {
            for (int off_min = offset_min; off_min <= off_max; off_min *= 2) {
                for (int len_max = length_max; len_max <= length_max * 8; len_max *= 2) {
                    for (int len_min = length_min; len_min <= len_max; len_min *= 2) {
                        offsets.clear();
                        lengths.clear();
                        divisions.clear();
                        float time = memcpy_test<uint8_t>(&offsets, &lengths, &divisions, (size_t) blocks, (size_t) threads, (size_t) megabytes*MULTIPLE, (uint32_t) off_min,
                            (uint32_t) off_max, (uint32_t) len_min, (uint32_t) len_max, (aligned != 0), (uint32_t) alignment, memcpy_function, false);
                        std::cout << off_min << "," << off_max << " | " << len_min << "," << len_max << " | " << memcpy_function << " | " << initial_bytes_global << std::endl;
                        std::cout << "Time taken (ms)|" << time;

                        time = memcpy_test<uint8_t>(&offsets, &lengths, &divisions, (size_t) blocks, (size_t) threads, (size_t) megabytes*MULTIPLE, (uint32_t) off_min,
                            (uint32_t) off_max, (uint32_t) len_min, (uint32_t) len_max, (aligned != 0), (uint32_t) alignment, memcpy_function, false);
                        std::cout << "," << time;

                        time = memcpy_test<uint8_t>(&offsets, &lengths, &divisions, (size_t) blocks, (size_t) threads, (size_t) megabytes*MULTIPLE, (uint32_t) off_min,
                            (uint32_t) off_max, (uint32_t) len_min, (uint32_t) len_max, (aligned != 0), (uint32_t) alignment, memcpy_function, false);
                        std::cout << "," << time << std::endl;

                    }
                }
            }
        }
    }

}






