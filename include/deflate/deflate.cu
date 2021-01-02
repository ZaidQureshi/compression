#include <stdio.h>
#include <cstdint>

#define MAXBITS 15
#define FIXLCODES 288
#define MaxDCodes 30

struct huffman_tree {

	int16_t lencnt[MAXBITS + 1];
	int16_t lensym[FIXLCODES];
	int16_t distcnt[MAXBITS + 1];
	int16_t distsym[MaxDCodes];

};

__device__
int16_t decode (uint32_t test, const int16_t* counts, const int16_t* symbols){

	unsigned int first;
	unsigned int len;
	unsigned int code;
	unsigned int count;
	uint32_t next32r = __brev(test);


	first  = 0;
  	for (len = 1; len <= MAXBITS; len++) {
    	code  = (next32r >> (32 - len)) - first;
    	
    	count = counts[len];
    if (code < count) 
    {
      	//skipbits(s, len);
      	return symbols[code];
    }
	    symbols += count;  
	    first += count;
	    first <<= 1;
  	}

  return -10;
}

__global__
void d_test_decode_testcase(uint32_t test, const int16_t* counts, const int16_t* symbols){

	int16_t out = decode(test, counts, symbols);
	char c = 'A' + out;
	printf("symbol: %c\n", c);

}

void test_decode_testcase(uint32_t* tests, int num_test, const int16_t* counts, const int16_t* symbols){

	int16_t* d_counts;
	int16_t* d_symbols;

	cudaMalloc(&d_counts, (MAXBITS + 1) * sizeof(int16_t));
	cudaMalloc(&d_symbols, (FIXLCODES) * sizeof(int16_t));

	cudaMemcpy(d_counts, counts, (MAXBITS + 1) * sizeof(int16_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_symbols, symbols, (FIXLCODES) * sizeof(int16_t), cudaMemcpyHostToDevice);

	for(int i = 0; i < num_test; i++) {
		uint32_t test = tests[i];
		d_test_decode_testcase<<<1,1>>>(test, d_counts, d_symbols);
	}
	
	cudaFree(d_counts);
	cudaFree(d_symbols);
}

//Construct huffman tree
__device__ 
void construct(int16_t* counts, int16_t* symbols, const int16_t *length, int num_codes){

	int symbol;
	int len;
	int left;
	for(len = 0; len <= MAXBITS; len++)
		counts[len] = 0;

	for(symbol = 0; symbol < num_codes; symbol++)
		(counts[length[symbol]])++;

	left = 1;
	for(len = 1; len <= MAXBITS; len++){
		left <<= 1;
		left -= counts[len];       
    	if (left < 0) 
    		return; 
	}

	{
		//computing offset array for conunts
		int16_t offs[MAXBITS + 1];
		offs[1] = 0;
		for (len = 1; len < MAXBITS; len++)
			offs[len + 1] = offs[len] + counts[len];

		for(symbol = 0; symbol < num_codes; symbol++){
			 if (length[symbol] != 0) 
			 	symbols[offs[length[symbol]]++] = symbol;
		}
	}	
}

__global__ 
void d_test_construct(int16_t* counts, int16_t* symbols, int16_t* length, int num_codes) {

	construct(counts, symbols, length, num_codes);
}

void test_construct_testcase(int16_t* length, int num_codes, int16_t* counts, int16_t* symbols){

	int16_t* d_length;
	cudaMalloc(&d_length, num_codes * sizeof(int16_t));
	cudaMemcpy(d_length, length, num_codes * sizeof(int16_t), cudaMemcpyHostToDevice);

	int16_t* d_counts;
	int16_t* d_symbols;

	cudaMalloc(&d_counts, (MAXBITS + 1) * sizeof(int16_t));
	cudaMalloc(&d_symbols, (FIXLCODES) * sizeof(int16_t));

	printf("test construct kernel launch\n");
	d_test_construct<<<1,1>>>(d_counts, d_symbols, d_length, num_codes);

	cudaMemcpy(counts, d_counts, (MAXBITS + 1) * sizeof(int16_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(symbols, d_symbols, (FIXLCODES) * sizeof(int16_t), cudaMemcpyDeviceToHost);

	cudaFree(d_counts);
	cudaFree(d_symbols);
}



void test_construct(){

	int16_t counts [MAXBITS + 1];
	int16_t symbols [FIXLCODES];



	printf("test1 starts\n");
	//test 1
	int num_codes = 4;
	int16_t length[4] = {2, 1, 3, 3};
	uint32_t tests1[4] = {1, 0, 3, 7};
	test_construct_testcase(length, num_codes, counts, symbols);

	for(int i = 0; i < 10; i++){
		printf("counts: %i \t symbols: %i \n", counts[i], symbols[i]);
	}

	test_decode_testcase(tests1, 4, counts, symbols);

	printf("test2 starts\n");
	//test 2
	num_codes = 8;
	int16_t length2[8] = {3, 3, 3, 3, 3, 2, 4, 4};
	test_construct_testcase(length2, num_codes, counts, symbols);
	
	for(int i = 0; i < 10; i++){
		printf("counts: %i \t symbols: %i \n", counts[i], symbols[i]);
	}

}



//construct huffman tree for dynamic huffman encoding block
__device__
int decode_dynamic(){
	int hlit, hdist, hclen;

	//getting the meta data for the compressed block
	fetch_n_bits<int>(5, hlit);
	fetch_n_bits<int>(5, hdist);
	fetch_n_bits<int>(4, hclen);
	hlit += 257;
	hdist += 1;
	hclen += 4;
 
 	//check
 	for (index = 0; index < ncode; index++) lengths[g_code_order[index]] = getbits(s, 3);
  	for (; index < 19; index++) lengths[g_code_order[index]] = 0;


	construct(s, s->lencnt, s->lensym, lengths, 19);

	
	int index = 0;
	while (index < nlen + ndist) {
		int symbol = decode(s, s->lencnt, s->lensym);
		if (symbol < 0) return symbol; //error

		//represent code lengths of 0 - 15
		if(symbol < 16)
			lengths[index++] = symbol;

		else{
			int len = 0;
			if(symbol == 16) {
				 len = lengths[index - 1];  // last length
				 fetch_n_bits<int>(2, symbol);
				 symbol += 3;
			}
			else if(symbol == 17){
				fetch_n_bits<int>(3, symbol);
				symbol += 3;
			}
			else {
				fetch_n_bits<int>(7, symbol);
				symbol += 11;
			}

			while(symbol--)
				lenghts[index++] = len;
		}
	}

	//check
	if(lengts[256] == 0) return -9;


	construct(s, s->lencnt, s->lensym, lengths, nlen);
    
    construct(s, s->distcnt, s->distsym, &lengths[nlen], ndist);

	return 0;
}


//code starts from 257
static const __device__ __constant__ uint16_t g_lens[29] = {  // Size base for length codes 257..285
  3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
  31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};

//code starts from 257
static const __device__ __constant__ uint16_t
  g_lext[29] = { 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};


static const __device__ __constant__ uint16_t
  g_dists[30] = {  // Offset base for distance codes 0..29
    1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
    193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};

static const __device__ __constant__ uint16_t g_dext[30] = {  // Extra bits for distance codes 0..29
  0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};



//decode code for compressed block
__device__ decode_symbol() {


	uint16_t sym = decode(test,  s->lencnt, s->lensym);


	//parse 5 bits

	//not compressed, literal
	if(sym < 255) {

	}

	//end of block
	else if(sym == 256) {

	}

	//lenght, need to parse
	else{
		uint16_t extra_bits = g_lext[sym - 257];

		uint16_t extra_len  = 0;
		if(extra_bits != 0)
			fetch_n_bits<uint16_t>(extra_bits, extra_len);

		uint16_t len = extra_len + g_lens[sym - 257];


		//distance, 5bits
		uint16_t sym_dist = decode(test,  s->distcnt, s->distsym);
		uint16_t extra_bits_dist = g_dext[sym_dist];
		
		uint16_t extra_len_dist = 0;
		if(extra_bits_dist != 0)
			fetch_n_bits<uint16_t>(extra_bits_dist, extra_len_dist);

		uint16_t dist = extra_len_dist + g_dists[dist];

		//(len, dist_len)

	}

}



int main(int argc, char** argv) {

	test_construct();
	return 0;

}