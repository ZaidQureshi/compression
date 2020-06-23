#include "decoder_trans.h"
#include <stdio.h>
#include <cassert>

void test_SHORTREPEAT() {
    printf("================ TEST SHORT REPEAT =================\n");
    int64_t ll[] =  {10000, 10000, 10000, 10000, 10000, 10001, 10001, 10001, 10001, 10001, 10001};
    uint8_t in[] = {0x0a, 0x27, 0x10, 0x0b, 0x27, 0x11};
    
    uint64_t blk_off[32] = {0};
    uint64_t col_len[32] = {sizeof(in)};
    uint8_t col_map[32] = {0};
    int64_t out[1024] = {0};

    rlev2::decompress_func_new(in, out,
                0, 0, 
                0, 0,
                blk_off, col_len, col_map);

    for (int i=0; i<sizeof(ll) / sizeof(int64_t); ++i) {
        assert(out[i] == ll[i]);
    }
    printf("================ PASS SHORT REPEAT =================\n\n");
}

void test_DIRECT() {
    printf("================ TEST DIRECT =================\n");
    int64_t ll[] =  {23713, 43806, 57005, 48879,23713, 43806, 57005, 48879};
    uint8_t in[] = {0x5e, 0x03, 0x5c, 0xa1, 0xab, 0x1e, 0xde, 0xad, 0xbe, 0xef,0x5e, 0x03, 0x5c, 0xa1, 0xab, 0x1e, 0xde, 0xad, 0xbe, 0xef};
    
    uint64_t blk_off[32] = {0};
    uint64_t col_len[32] = {sizeof(in)};
    uint8_t col_map[32] = {0};
    int64_t out[1024] = {0};

    rlev2::decompress_func_new(in, out,
                0, 0, 
                0, 0,
                blk_off, col_len, col_map);

    for (int i=0; i<sizeof(ll) / sizeof(int64_t); ++i) {
        assert(out[i] == ll[i]);
        // printf("out[%d]: %ld\n", i, out[i]);
    }
    printf("================ PASS DIRECT =================\n\n");
}

void test_MULTIPLE() {
    printf("================ TEST MULTIPLE =================\n");

    int64_t ll[] =  {23713, 43806, 57005, 48879,10000, 10000, 10000, 10000, 10000, 10001, 10001, 10001, 10001, 10001, 10001, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        2030, 2000, 2020, 1000000, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120, 2130, 2140, 2150, 2160, 2170, 2180, 2190,
        23713, 43806, 57005, 48879,23713, 43806, 57005, 48879,
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    uint8_t in[] = {0x5e, 0x03, 0x5c, 0xa1, 0xab, 0x1e, 0xde, 0xad, 0xbe, 0xef, 0x0a, 0x27, 0x10, 0x0b, 0x27, 0x11, 0xc6, 0x09, 0x02, 0x02, 0x22, 0x42, 0x42, 0x46,
        0x8e, 0x13, 0x2b, 0x21, 0x07, 0xd0, 0x1e, 0x00, 0x14, 0x70, 0x28, 0x32, 0x3c, 0x46, 0x50, 0x5a, 0x64, 0x6e, 0x78, 0x82, 0x8c, 0x96, 0xa0, 0xaa, 0xb4, 0xbe, 0xfc, 0xe8,
        0x5e, 0x03, 0x5c, 0xa1, 0xab, 0x1e, 0xde, 0xad, 0xbe, 0xef,0x5e, 0x03, 0x5c, 0xa1, 0xab, 0x1e, 0xde, 0xad, 0xbe, 0xef,
        0xc6, 0x09, 0x02, 0x02, 0x22, 0x42, 0x42, 0x46};
    
    uint64_t blk_off[32] = {0};
    uint64_t col_len[32] = {sizeof(in)};
    uint8_t col_map[32] = {0};
    int64_t out[1024] = {0};

    rlev2::decompress_func_new(in, out,
                0, 0, 
                0, 0,
                blk_off, col_len, col_map);

    // for (int i=0; i<sizeof(in); ++i) {
    //     printf("[%d]: %x\n", i, in[i]);
    // }


    for (int i=0; i<sizeof(ll) / sizeof(int64_t); ++i) {
        // printf("out[%d]: %ld\n", i, out[i]);
        assert(out[i] == ll[i]);
    }
    printf("================ PASS MULTIPLE =================\n\n");

}

void test_DELTA() {
    printf("================ TEST DELTA =================\n");

    int64_t ll[] =  {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    uint8_t in[] = {0xc6, 0x09, 0x02, 0x02, 0x22, 0x42, 0x42, 0x46};

    uint64_t blk_off[32] = {0};
    uint64_t col_len[32] = {sizeof(in)};
    uint8_t col_map[32] = {0};
    int64_t out[1024] = {0};

    rlev2::decompress_func_new(in, out,
                0, 0, 
                0, 0,
                blk_off, col_len, col_map);

    for (int i=0; i<sizeof(ll) / sizeof(int64_t); ++i) {
        // printf("out[%d]: %ld\n", i, out[i]);
        assert(out[i] == ll[i]);
    }
    printf("================ PASS DELTA =================\n\n");
}

void test_PTACHED_BASE() {
    printf("================ TEST PATCHED BASE =================\n");

    int64_t ll[] =  {2030, 2000, 2020, 1000000, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120, 2130, 2140, 2150, 2160, 2170, 2180, 2190};
    uint8_t in[] = {0x8e, 0x13, 0x2b, 0x21, 0x07, 0xd0, 0x1e, 0x00, 0x14, 0x70, 0x28, 0x32, 0x3c, 0x46, 0x50, 0x5a, 0x64, 0x6e, 0x78, 0x82, 0x8c, 0x96, 0xa0, 0xaa, 0xb4, 0xbe, 0xfc, 0xe8};

    uint64_t blk_off[32] = {0};
    uint64_t col_len[32] = {sizeof(in)};
    uint8_t col_map[32] = {0};
    int64_t out[1024] = {0};

    rlev2::decompress_func_new(in, out,
                0, 0, 
                0, 0,
                blk_off, col_len, col_map);

    for (int i=0; i<sizeof(ll) / sizeof(int64_t); ++i) {
        // printf("out[%d]: %ld\n", i, out[i]);
        assert(out[i] == ll[i]);
    }
    printf("================ PASS PATCHED BASE =================\n\n");
}

int main() {
    test_SHORTREPEAT();
    test_DIRECT();
    test_DELTA();
    test_PTACHED_BASE();
    test_MULTIPLE();
    return 0;
}