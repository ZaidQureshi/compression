./build/exe/src/zlib/no_stride.cu.exe -d ~/input_datasets/col0_comp.bin decomp.bin  ~/input_datasets/col0_col_len.bin  ~/input_datasets/col0_blk_offset.bin 131072
# ./build/exe/src/zlib/no_stride.cu.exe -d ~/input_datasets/col0_comp.bin decomp2.bin  ~/input_datasets/col0_col_len.bin  ~/input_datasets/col0_blk_offset.bin 131072
head -c 102400 ./decomp.bin > ./decomp_short.bin
head -c 102400 ./golden.bin > ./golden_short.bin
cmp ./decomp_short.bin ./golden_short.bin
cmp ./decomp.bin ./golden.bin