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
#include <zlib/memcpy.h>

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

#define DEFAULT_FILE "output.dat"

using namespace std;

// DONE: Change default type of data/output to byte?
// DONE: Make function for test/nsight and function for profiling.
// TODO: Make small changes to memcpy to fit these functions.
// TODO: Run with the existing memcpy functions and profile.

// Generate the golden solution for the memcpy. Will be slow!
template <typename T>
void memcpy_test_cpu(uint32_t* offsets, uint32_t* lengths, uint32_t* divisions, 
    uint32_t num_columns, size_t total_elements) {

    vector<uint32_t> output;
    for (int col = 0; col < num_columns; col++) {
        uint32_t min_idx = divisions[col];
        uint32_t max_idx = divisions[col+1];

        for (int c = 0; c < INITIAL_BYTES; c++) {
            output.push_back(c);
        }

        for (int idx = min_idx; idx < max_idx; idx++) {
            uint32_t offset = offsets[idx];
            uint32_t length = lengths[idx];

            uint32_t cur_idx = output.size() - 1 - offset;

            for (int c = 0; c < length; c++) {
                output.push_back(output[cur_idx+c]);
            }
        }
    }

    // Save the output to a file
    save_into_file<T>("golden.dat", output);

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

template<typename T>
void save_into_file(string name, vector<T>* output) {
    ofstream outfile (name, ofstream::binary);
    for(auto iter = output.begin(); iter!=output.end(); iter++)
    {
        auto product = *iter;
        char out[sizeof(T)];
        memcpy(&out, &product, sizeof(T));
        outfile.write(out, sizeof(T));
    }
    outfile.close();
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

template<typename T>
__device__
void memcpy_kernel(uint32_t* offsets, uint32_t* lengths, uint32_t* divisions, T* output, size_t num_columns, size_t column_size) {
    __shared__ uint32_t offsets_s[32];
    __shared__ uint32_t lengths_s[32];
    uint32_t lane_id = threadIdx.x % 32;
    size_t col_idx = threadIdx.x / 32 + blockIdx.x * (blockDim.x * 32);
    uint32_t div_min = divisions[col_idx];
    uint32_t div_max = divisions[col_idx+1];
    uint32_t counter = column_size * col_idx;
    uint32_t s_idx = 0;

    for (uint32_t c = div_min, c < div_max; c++) {
        if (s_idx == 0) {
            offsets_s[lane_id] = offsets[c+lane_id];
            lengths[lane_id] = lengths[c+lane_id];
            __syncwarp();
        }

        // Call the memcpy operation
        // TODO: Memcpy functions need a small change here to include counter and output as vars.
        memcpy_onebyte<32>(0, lengths_s[s_idx], offsets_s[s_idx], 0, 0xFFFFFFFF, counter, output);
        s_idx = (s_idx + 1) % 32;
    }

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
 uint32_t offset_max, uint32_t length_min, uint32_t length_max, bool aligned, uint32_t alignment,
 bool validate = true) {

    srand (SEED);

    uint32_t num_columns = (threads / 32) * blocks;
    size_t column_size = total_elements / num_columns - INITIAL_BYTES;

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
                cur_length = min(cur_length, (uint32_t) (column_size - cur_size));

                uint32_t cur_offset = offset_max;
                if (offset_max != offset_min)
                    cur_offset = rand() % (offset_max - offset_min + 1) + offset_min;
                cur_offset = min(cur_offset, (uint32_t) cur_size + INITIAL_BYTES);

                offsets.push_back(cur_offset);
                lengths.push_back(cur_length);

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
    if (validate)
        memcpy_test_cpu<T>(offsets_h, lengths_h, divisions_h, num_columns, total_elements);

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

    dim3 blockDim(threads, 1, 1);
    dim3 gridDim(blocks, 1, 1);

    // Set up timing
    // https://stackoverflow.com/questions/7876624/timing-cuda-operations
    float time;
    cudaEvent_t start, stop;

    cuda_err_chk( cudaEventCreate(&start) );
    cuda_err_chk( cudaEventCreate(&stop) );
    cuda_err_chk( cudaEventRecord(start, 0) );
    memcpy_kernel<<<blockDim, gridDim>>>(offsets_d, lengths_d, divisions_d, output_d, num_columns, column_size);
    cuda_err_chk( cudaEventRecord(stop, 0) );
    cuda_err_chk( cudaEventSynchronize(stop) );
    cuda_err_chk( cudaEventElapsedTime(&time, start, stop));

    T* output_h;
    cuda_err_chk(cudaMemcpy(output_h, output_d, sizeof(uint32_t) * total_elements, cudaMemcpyDeviceToHost));

    // Write the output file
    if (validate) {
        save_into_file<T>("memcpy.dat", vector<uint32_t>(output_h));
        if (compare_files("memcpy.dat", "golden.dat") == false) {
            std::cout << "Error: Data mismatch!" << std::endl;
        }
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
    int initial_bytes = 8;

    if (argc > 2) {
        for (int idx = 2; idx < argc; idx = idx + 2) {
            char* flag = argv[idx-1];
            char* arg = argv[idx];
            if (flag == "-fp")
                fp = arg;
            if (flag == "-p")
                profiling = atoi(arg);
            if (flag == "-mcp")
                memcpy_function = atoi(arg);
            if (flag == "-b")
                blocks = atoi(arg);
            if (flag == "-t")
                threads = atoi(arg);
            if (flag == "-mb")
                megabytes = atoi(arg);
            if (flag == "-omn")
                offset_min = atoi(arg);
            if (flag == "-omx")
                offset_max = atoi(arg);
            if (flag == "-lmn")
                length_min = atoi(arg);
            if (flag == "-omx")
                length_max = atoi(arg);
            if (flag == "-a")
                aligned = atoi(arg);
            if (flag == "-amt")
                alignment = atoi(arg);
            if (flag == "-inb")
                initial_bytes = atoi(arg);
        }
    }

    vector<uint32_t> offsets;
    vector<uint32_t> lengths;
    vector<uint32_t> divisions;


    // Run the test function
    if (profiling == 0) {
        float time = memcpy_test<uint8_t>(offsets, lengths, divisions, (size_t) blocks, (size_t) threads, (size_t) megabytes*1024*1024, (uint32_t) offset_min,
            (uint32_t) offset_max, (uint32_t) length_min, (uint32_t) length_max, aligned != 0, (uint32_t) alignment);
        std::cout << memcpy_function << " | " << blocks << " | " << threads << " | " << megabytes << " | " << offset_min << "," << offset_max
          << " | " << length_min << "," << length_max << " | " << aligned != 0 << " | " << alignment << " | " << initial_bytes << std::endl;
        std::cout << "Time taken (ms)," << time << std::endl;
    } else {
        std::cout << memcpy_function << " | " << blocks << " | " << threads << " | " << megabytes << " | " << aligned != 0 << " | " << alignment << " | " << initial_bytes << std::endl;
        // Need to choose the variables to profile over. This function iterates over offset min/max and length min/max
        for (int off_max = offset_max; off_max <= offset_max * 8; off_max *= 2) {
            for (int off_min = offset_min; off_min <= off_max; off_min *= 2) {
                for (int len_max = length_max; len_max <= length_max * 8; len_max *= 2) {
                    for (int len_min = length_min; len_min <= len_max; len_min *= 2) {
                        offsets.clear();
                        lengths.clear();
                        divisions.clear();
                        float time = memcpy_test<uint8_t>(offsets, lengths, divisions, (size_t) blocks, (size_t) threads, (size_t) megabytes*1024*1024, (uint32_t) offset_min,
                            (uint32_t) offset_max, (uint32_t) length_min, (uint32_t) length_max, aligned != 0, (uint32_t) alignment, false);
                        std::cout << offset_min << "," << offset_max << " | " << length_min << "," << length_max << " | " << " | " << initial_bytes << std::endl;
                        std::cout << "Time taken (ms)," << time << std::endl;

                    }
                }
            }
        }
    }

}






