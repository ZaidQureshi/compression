# CODAG
Source code for the paper CODAG

<!-- # README
We will update the read me with the instructions in coming weeks. We are cleaning up the codebase and adding scripts for easy execution. 
 -->
 
# Dataset
The datasets used in the paper CODAG can be downloaded from this link (https://drive.google.com/drive/folders/1q_3hCQz2dBGQw7VowBNANcVJ-6qvmDha?usp=sharing). Make sure both <dataset_name>_comp.bin and <dataset_name>_blk_offset.bin are downloaded for Deflate

# Instruction

First, CODAG decompressors for RLE v1, RLE v2, and Deflate can be simply generated by calling
`make clean && make`

Once all decompressors are generated, you can use the following commend to run RLE v1 or RLE v2 decompression.
`./build/exe/src/<ENCODING_TECHNIQUE>/<ENCODING_TECHNIQUE>.cu.exe -f <ORIG_DATASET> -t <DATATYPE BYTE SIZE> `.

For example, to run RLE v1 decompression on MC0 dataset with `uint64_t` datatype, the commend would be.
./build/exe/src/rlev1/rlev1.cu.exe -f col0.bin -t 8

To run Deflate decompression, the datasets are firsted compressed by `deflate_compressor.py` or simply use the compressed datasets from the link for the dataset.

Once, compressed datasets are ready, use the following commend to call CODAG Deflate decompression.
`./build/exe/src/zlib/zlib.cu.exe  <DATASET_NAME>_comp.bin (decompressed file)  <DATASET_NAME>_blk_offset.bin <CHYNK_SIZE>`.
For the provided compressed dataset, CHUNK_SIZE should be 131072 (128KB). 