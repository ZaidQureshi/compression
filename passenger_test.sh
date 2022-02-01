./build/exe/src/zlib/no_stride.cu.exe -d ~/input_datasets/passenger_count_comp.bin decomp.bin  ~/input_datasets/passenger_count_col_len.bin  ~/input_datasets/passenger_count_blk_offset.bin 131072
head -c 10120000 ./decomp.bin | tail -c 200000 > ./decomp_short.bin
head -c 10120000 ./golden.bin | tail -c 200000 > ./golden_short.bin
cmp ./decomp_short.bin ./golden_short.bin
cmp ./decomp.bin ./golden.bin