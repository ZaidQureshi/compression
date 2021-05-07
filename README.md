# compression

1. Checkout the deflate decompressor for GDS application
`git checkout GDS_deflate`

2. Fetch the most recent version
`git pull origin GDS_deflate`

3. Submodule for freestanding libraries 
`git submodule update --init --remote --recursive -- freestanding/`

4. Compile the decompressor
`make clean && make build/exe/src/zlib/zlib.cu.exe`

5. Download a dataset from https://rapidsai.github.io/demos/datasets/mortgage-data then unzip the file. There should be a file like 'mortgage/perf/Performance_2000Q1.txt'.

6. Convert txt file to binary file. The arguments for the text_to_binary is in this format
`python text_to_binary.py <text file> <column number> <data type> <output file> '|'`

```
python text_to_binary.py mortgage/perf/Performance_2000Q1.txt 0 long column_data.bin '|'
```

7. Create input and output directories and move the binary input file to the input directory
```
mkdir data
mkdir output
mv column_data.bin ./data/column_data.bin
```

8. Run the compressor to compress in the right format 
`python compressor.py`

9. Run the decompressor
`
./build/exe/src/zlib/zlib.cu.exe -d ./output/compressed_file.bin decompressed_file.bin ./output/col_len.bin ./output/blk_offset.bin`
