#include <common.h>
//#include <brle/brle_trans.h>
#include <zlib/zlib.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <sys/types.h> 

#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 5) {
      std::cerr << "Please provide arguments input output input_bytes read_bytes \n";
      exit(1);
    }

    bool decomp = (strcmp(argv[1],"-d")==0);
    if (decomp && (argc < 6)) {
      std::cerr << "Please provide arguments -d input output  col_len blk_offset input_bytes read_bytes \n";
      exit(1);
    }

    const char* input = decomp ? argv[2] : argv[1];
    const char* output = decomp ? argv[3] : argv[2];
    const char* col_f = decomp ? argv[4] : argv[3];
    const char* blk_f = decomp ? argv[5] : argv[4];
   // const char* s_input_bytes = decomp ? argv[6] : argv[5];
   // const char* s_read_bytes = decomp ? argv[7] : argv[6];

   // int input_bytes = std::atoi(s_input_bytes);
   // int read_bytes = std::atoi(s_read_bytes);

    std::chrono::high_resolution_clock::time_point total_start = std::chrono::high_resolution_clock::now();
    int in_fd;
    struct stat in_sb;
    void* in;

    int out_fd;
    struct stat out_sb;
    void* out;

    int col_fd;
    struct  stat col_sb;
    void* col;


    int blk_fd;
    struct  stat blk_sb;
    void* blk;

    if((in_fd = open(input, O_RDONLY)) == 0) {
        printf("Fatal Error: INPUT File open error\n");
        return -1;
    }
    if((out_fd = open(output, O_RDWR | O_TRUNC | O_CREAT, S_IRWXU | S_IRGRP | S_IROTH)) == 0) {
        printf("Fatal Error: OUTPUT File open error\n");
        return -1;
    }
    if((col_fd = open(col_f, O_RDONLY)) == 0) {
        printf("Fatal Error: COL_LEN File open error\n");
        return -1;
    }
    if((blk_fd = open(blk_f, O_RDONLY)) == 0) {
        printf("Fatal Error: BLK_OFFSET File open error\n");
        return -1;
    }

    fstat(in_fd, &in_sb);

    in = mmap(nullptr, in_sb.st_size, PROT_READ, MAP_PRIVATE, in_fd, 0);

    if(in == (void*)-1){
        printf("Fatal Error: INPUT Mapping error\n");
        return -1;
    }
    uint8_t* in_ = (uint8_t*) in;
    

    fstat(col_fd, &col_sb);
    col = mmap(nullptr, col_sb.st_size,  PROT_READ, MAP_PRIVATE, col_fd, 0);
    if(col == (void*)-1){
        printf("Fatal Error: COL_LEN Mapping error\n");
        return -1;
    }
    uint64_t*col_ = (uint64_t*) col;

    fstat(blk_fd, &blk_sb);
    blk = mmap(nullptr, blk_sb.st_size,  PROT_READ, MAP_PRIVATE, blk_fd, 0);
    if(blk == (void*)-1){
        printf("Fatal Error: BLK_OFF Mapping error\n");
        return -1;
    }
    uint64_t*blk_ = (uint64_t*) blk;

    uint8_t* out_;
    uint64_t out_size;
    std::chrono::high_resolution_clock::time_point compress_start = std::chrono::high_resolution_clock::now();
    if (!decomp) {
                      std::cerr << "read bytes should be multiple of input bytes \n";
        //brle_trans::compress_gpu<uint8_t, uint32_t>(in_, &out_, in_sb.st_size, &out_size);
    }
    else {

            // if((input_bytes) == 1 && (read_bytes) == 1){
            //     brle_trans::decompress_gpu<uint8_t, uint8_t>(in_, &out_, in_sb.st_size, &out_size);
            // }

            // else if((input_bytes) == 1 && (read_bytes) == 2){
            //     brle_trans::decompress_gpu<uint8_t, uint16_t>(in_, &out_, in_sb.st_size, &out_size);
            // }

            // else if((input_bytes) == 1 && (read_bytes) == 4){
            //     brle_trans::decompress_gpu<uint8_t, uint32_t>(in_, &out_, in_sb.st_size, &out_size);
            // }


            // else if((input_bytes) == 2 && (read_bytes) == 2){
            //     brle_trans::decompress_gpu<uint16_t, uint16_t>(in_, &out_, in_sb.st_size, &out_size);
            // }

            // else if((input_bytes) == 2 && (read_bytes) == 4){
            //     brle_trans::decompress_gpu<uint16_t, uint32_t>(in_, &out_, in_sb.st_size, &out_size);
            // }

            // else if((input_bytes) == 4 && (read_bytes) == 4){
            //     brle_trans::decompress_gpu<uint32_t, uint32_t>(in_, &out_, in_sb.st_size, &out_size);
            // }
            
            // else {
            //           std::cerr << "read bytes should be multiple of input bytes \n";
            // }

            deflate::decompress_gpu<ulonglong4>(in_, &out_, in_sb.st_size, &out_size, col_, col_sb.st_size, blk_, blk_sb.st_size);


        //brle_trans::decompress_gpu<uint8_t, uint32_t>(in_, &out_, in_sb.st_size, &out_size);
    }

    std::chrono::high_resolution_clock::time_point compress_end = std::chrono::high_resolution_clock::now();

    fstat(out_fd, &out_sb);
    if (out_sb.st_size != out_size) {
        if(ftruncate(out_fd, out_size) == -1) PRINT_ERROR;
    }
    out = mmap(nullptr, out_size, PROT_WRITE | PROT_READ, MAP_SHARED, out_fd, 0);

    memcpy(out, out_, out_size);

    if(munmap(in, in_sb.st_size) == -1) PRINT_ERROR;
    if(munmap(out, out_size) == -1) PRINT_ERROR;
    if(munmap(col, col_sb.st_size) == -1) PRINT_ERROR;
    if(munmap(blk, blk_sb.st_size) == -1) PRINT_ERROR;


    close(in_fd);
    close(out_fd);
    close(col_fd);
    close(blk_fd);

    std::chrono::high_resolution_clock::time_point total_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> total = std::chrono::duration_cast<std::chrono::duration<double>>(total_end - total_start);
    std::chrono::duration<double> comp = std::chrono::duration_cast<std::chrono::duration<double>>(compress_end - compress_start);
    std::chrono::duration<double> wrt = std::chrono::duration_cast<std::chrono::duration<double>>(total_end - compress_end);
	std::cout << "Done\n";
 //    std::cout << "Total time: " << total.count() << " secs\n";
//    std::cout << "Compute time: " << comp.count() << " secs\n";
//    std::cout << "Write time: " << wrt.count() << " secs\n";
    
}
