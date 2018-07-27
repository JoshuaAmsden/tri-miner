/**
 * X16R algorithm (X16 with Randomized chain order)
 *
 * tpruvot 2018 - GPL code
 */

#include <stdio.h>
#include <memory.h>
#include <unistd.h>
#include <string.h>

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"

#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_sha2.h"
#include "sph/sph_haval.h"
}

#include "uint256.h"
#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x16.h"
//#include "hash_selection.h"

using namespace std;

static uint32_t *d_hash[MAX_GPUS];

enum Algo {
	BLAKE = 0,
	BMW,
	GROESTL,
	JH,
	KECCAK,
	SKEIN,
	LUFFA,
	CUBEHASH,
	SHAVITE,
	SIMD,
	ECHO,
	HAMSI,
	FUGUE,
	SHABAL,
	WHIRLPOOL,
	SHA512,
	HAVAL,
	HASH_FUNC_COUNT
};

static const char* algo_strings[] = {
	"blake",
	"bmw512",
	"groestl",
	"jh512",
	"keccak",
	"skein",
	"luffa",
	"cube",
	"shavite",
	"simd",
	"echo",
	"hamsi",
	"fugue",
	"shabal",
	"whirlpool",
	"sha512",
	"haval",
	NULL
};

static __thread uint32_t s_ntime = UINT32_MAX;
static __thread bool s_implemented = false;
static __thread char hashOrder[HASH_FUNC_COUNT + 1] = { 0 };

static bool isScrambleHash(const uint256& blockHash) {
	#define START_OF_LAST_35_NIBBLES_OF_HASH 29
	int last35Nibble = blockHash.GetNibble(START_OF_LAST_35_NIBBLES_OF_HASH);
	return (last35Nibble % 2 == 0);
}

static uint256 scrambleHash(const uint256& blockHash) {
	// Cliffnotes: use last 34 of PrevBlockHash to shuffle
	// a list of all algos and append that to PrevBlockHash and pass to hasher
	//////

	std::string hashString = blockHash.GetHex(); // uint256 to string
	std::string list = "0123456789abcdef";
	std::string order = list;
	std::string order2 = list;

	std::string hashFront = hashString.substr(0,30); // preserve first 30 chars
	std::string sixteen2 = hashString.substr(30,46); // extract last 19-34 chars
	std::string sixteen = hashString.substr(46,62); // extract last 3-18 chars
	std::string last2 = hashString.substr(62,64); // extract last 2 chars
	for(int i=0; i<16; i++){
	  int offset = list.find(sixteen[i]); // find offset of 16 char

	  order.insert(0, 1, order[offset]); // insert the nth character at the beginning
	  order.erase(offset+1, 1);  // erase the n+1 character (was nth)
	}

	for(int j=0; j<16; j++){
	  int offset = list.find(sixteen2[j]); // find offset of 16 char

	  order2.insert(0, 1, order2[offset]); // insert the nth character at the beginning
	  order2.erase(offset+1, 1);  // erase the n+1 character (was nth)
	}
	int offset = list.find(last2[0]); // find offset of 16 char
	order2.insert(0, 1, order2[offset]);
	offset = list.find(last2[1]); // find offset of 16 char
	order2.insert(0, 1, order2[offset]);
	uint256 scrambleHash = uint256(hashFront + order2 + order); // uint256 with length of hash and shuffled last seventeen
	return scrambleHash;
}

static uint8_t GetSelection(const uint256& blockHash, const int index) {
	//assert(index >= 0);
	///assert(index < 17);

	#define START_OF_LAST_17_NIBBLES_OF_HASH 47
	uint8_t hashSelection = blockHash.GetNibble(START_OF_LAST_17_NIBBLES_OF_HASH + index);
	#define START_OF_LAST_34_NIBBLES_OF_HASH 30
	uint8_t additionalSelection = blockHash.GetNibble(START_OF_LAST_34_NIBBLES_OF_HASH + index);
	hashSelection += (additionalSelection % 2);
	return(hashSelection);
}

/*
static void getAlgoScrambleString(const uint32_t* prevblock, char *output)
{
	uint8_t* data = (uint8_t*)prevblock;

	strcpy(output, "000123456789ABCDEF0123456789ABCDEF");

	for(int i = 2; i < 18; i++){
		uint8_t b = (17 - i) >> 1; // 16 ascii hex chars, reversed
		uint8_t algoDigit = (i & 1) ? data[b] & 0xF : data[b] >> 4;
		int offset = algoDigit + 2;
		// insert the nth character at the front
		char oldVal = output[offset];
		for(int j=offset; j-->0;)
			output[j+1] = output[j];
		output[2] = oldVal;
	}
	for(i < 34; i++){
		uint8_t b = (33 - i) >> 1; // 16 ascii hex chars, reversed
		uint8_t algoDigit = (i & 1) ? data[b] & 0xF : data[b] >> 4;
		int offset = algoDigit + 16;
		// insert the nth character at the front
		char oldVal = output[offset];
		for(int j=offset; j-->0;)
			output[j+1] = output[j];
		output[18] = oldVal;
	}
	uint8_t algoDigit = data[1] & 0F;
	output[0] = algoDigit;
	algoDigit = data[0] >> 4;
	output[1]; = algoDigit;
}*/

static void getAlgoString(const uint32_t* prevblock, char *output)
{

	uint256 prevHash;
	prevHash.setUint32t(prevblock);
	applog(LOG_NOTICE, "prevhash %s\n", prevHash.GetHex().c_str());
	bool toBeScamble = isScrambleHash(prevHash);
	uint256 hash;
	if(toBeScamble) {
		hash = scrambleHash(prevHash);
	} else {
		hash = prevHash;
	}
	char *sptr = output;
	printf("hash selection %s:",  hash.GetHex().c_str());
	for(int i = 0; i < 17; i ++) {
		uint8_t hashSelection =  GetSelection(hash, i);
		printf("%u,", hashSelection);
		if (hashSelection >= 10) {
			//printf("%c", 'A' + (hashSelection - 10));
			sprintf(sptr, "%c", 'A' + (hashSelection - 10));
		}
		else {
			//printf("%d", hashSelection);
			sprintf(sptr, "%u", (uint32_t) hashSelection);
		}
		sptr++;
	}
	*sptr = '\0';
	printf("----%s\n", output);
}

// Trihash CPU Hash (Validation)
extern "C" void trihash(void *output, const void *input)
{
	unsigned char _ALIGN(64) hash[136];

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;
	sph_hamsi512_context ctx_hamsi;
	sph_fugue512_context ctx_fugue;
	sph_shabal512_context ctx_shabal;
	sph_whirlpool_context ctx_whirlpool;
	sph_sha512_context ctx_sha512;
	sph_haval256_5_context   ctx_haval;

	void *in = (void*) input;
	int size = 80;

	uint32_t *in32 = (uint32_t*) input;
	getAlgoString(&in32[1], hashOrder);

	for (int i = 0; i < 17; i++)
	{
		const char elem = hashOrder[i];
		const uint8_t algo = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

		switch (algo) {
		case BLAKE:
			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, in, size);
			sph_blake512_close(&ctx_blake, hash);
			break;
		case BMW:
			sph_bmw512_init(&ctx_bmw);
			sph_bmw512(&ctx_bmw, in, size);
			sph_bmw512_close(&ctx_bmw, hash);
			break;
		case GROESTL:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, in, size);
			sph_groestl512_close(&ctx_groestl, hash);
			break;
		case SKEIN:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, in, size);
			sph_skein512_close(&ctx_skein, hash);
			break;
		case JH:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, in, size);
			sph_jh512_close(&ctx_jh, hash);
			break;
		case KECCAK:
			sph_keccak512_init(&ctx_keccak);
			sph_keccak512(&ctx_keccak, in, size);
			sph_keccak512_close(&ctx_keccak, hash);
			break;
		case LUFFA:
			sph_luffa512_init(&ctx_luffa);
			sph_luffa512(&ctx_luffa, in, size);
			sph_luffa512_close(&ctx_luffa, hash);
			break;
		case CUBEHASH:
			sph_cubehash512_init(&ctx_cubehash);
			sph_cubehash512(&ctx_cubehash, in, size);
			sph_cubehash512_close(&ctx_cubehash, hash);
			break;
		case SHAVITE:
			sph_shavite512_init(&ctx_shavite);
			sph_shavite512(&ctx_shavite, in, size);
			sph_shavite512_close(&ctx_shavite, hash);
			break;
		case SIMD:
			sph_simd512_init(&ctx_simd);
			sph_simd512(&ctx_simd, in, size);
			sph_simd512_close(&ctx_simd, hash);
			break;
		case ECHO:
			sph_echo512_init(&ctx_echo);
			sph_echo512(&ctx_echo, in, size);
			sph_echo512_close(&ctx_echo, hash);
			break;
		case HAMSI:
			sph_hamsi512_init(&ctx_hamsi);
			sph_hamsi512(&ctx_hamsi, in, size);
			sph_hamsi512_close(&ctx_hamsi, hash);
			break;
		case FUGUE:
			sph_fugue512_init(&ctx_fugue);
			sph_fugue512(&ctx_fugue, in, size);
			sph_fugue512_close(&ctx_fugue, hash);
			break;
		case SHABAL:
			sph_shabal512_init(&ctx_shabal);
			sph_shabal512(&ctx_shabal, in, size);
			sph_shabal512_close(&ctx_shabal, hash);
			break;
		case WHIRLPOOL:
			sph_whirlpool_init(&ctx_whirlpool);
			sph_whirlpool(&ctx_whirlpool, in, size);
			sph_whirlpool_close(&ctx_whirlpool, hash);
			break;
		case SHA512:
			sph_sha512_init(&ctx_sha512);
			sph_sha512(&ctx_sha512,(const void*) in, size);
			sph_sha512_close(&ctx_sha512,(void*) hash);
			break;
		case HAVAL:
			printf("hashing havals\n");
			sph_haval256_5_init(&ctx_haval);
			sph_haval256_5(&ctx_haval,(const void*) in, size);
			sph_haval256_5_close(&ctx_haval,hash);
		   break;
		}
		in = (void*) hash;
		size = 64;
	}
	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

//#define _DEBUG
#define _DEBUG_PREFIX "trihash-"
#include "cuda_debug.cuh"

// #define GPU_HASH_CHECK_LOG

#ifdef GPU_HASH_CHECK_LOG
	static int algo80_tests[HASH_FUNC_COUNT] = { 0 };
	static int algo64_tests[HASH_FUNC_COUNT] = { 0 };
#endif
static int algo80_fails[HASH_FUNC_COUNT] = { 0 };

extern "C" int scanhash_trihash(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 20 : 19;
	if (strstr(device_name[dev_id], "GTX 1080")) intensity = 19;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		//quark_blake512_cpu_init(thr_id, throughput); // Redundant
		//quark_bmw512_cpu_init(thr_id, throughput); // Redundant
		quark_groestl512_cpu_init(thr_id, throughput);
		//quark_skein512_cpu_init(thr_id, throughput); // Redundant
		//quark_jh512_cpu_init(thr_id, throughput); // Redundant
		quark_keccak512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput); // 64
		x13_hamsi512_cpu_init(thr_id, throughput);
		x16_fugue512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);
		x16_whirlpool512_init(thr_id, throughput);
		x17_sha512_cpu_init(thr_id, throughput);
		x17_haval256_cpu_init(thr_id, throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	if (opt_benchmark) {
		((uint32_t*)ptarget)[7] = 0x003f;
		((uint32_t*)pdata)[1] = 0xEFCDAB89;
		((uint32_t*)pdata)[2] = 0x67452301;
	}
	uint32_t _ALIGN(64) endiandata[20];

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	uint32_t ntime = swab32(pdata[17]);
	if (s_ntime != ntime) {
		getAlgoString(&endiandata[1], hashOrder);
		s_ntime = ntime;
		s_implemented = true;
		if (!thr_id) applog(LOG_INFO, "hash order1 %s (%08x)\n", hashOrder, ntime);
	}

	if (!s_implemented) {
	 	applog(LOG_INFO, "s_implemented is false, wait 1 min to terminate %s \n", hashOrder);
		sleep(1);
		return -1;
	}

	cuda_check_cpu_setTarget(ptarget);

	char elem = hashOrder[0];
	const uint8_t algo80 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

	switch (algo80) {
		case BLAKE:
			quark_blake512_cpu_setBlock_80(thr_id, endiandata);
			break;
		case BMW:
			quark_bmw512_cpu_setBlock_80(endiandata);
			break;
		case GROESTL:
			groestl512_setBlock_80(thr_id, endiandata);
			break;
		case JH:
			jh512_setBlock_80(thr_id, endiandata);
			break;
		case KECCAK:
			keccak512_setBlock_80(thr_id, endiandata);
			break;
		case SKEIN:
			skein512_cpu_setBlock_80((void*)endiandata);
			break;
		case LUFFA:
			qubit_luffa512_cpu_setBlock_80_alexis((void*)endiandata);
			break;
		case CUBEHASH:
			cubehash512_setBlock_80(thr_id, endiandata);
			break;
		case SHAVITE:
			x11_shavite512_setBlock_80((void*)endiandata);
			break;
		case SIMD:
			x16_simd512_setBlock_80((void*)endiandata);
			break;
		case ECHO:
			x11_echo512_setBlock_80_alexis((void*)endiandata);
			break;
		case HAMSI:
			x16_hamsi512_setBlock_80((void*)endiandata);
			break;
		case FUGUE:
			x16_fugue512_setBlock_80((void*)pdata);
			break;
		case SHABAL:
			x16_shabal512_setBlock_80((void*)endiandata);
			break;
		case WHIRLPOOL:
			x16_whirlpool512_setBlock_80((void*)endiandata);
			break;
		case SHA512:
			x16_sha512_setBlock_80(endiandata);
			break;
		case HAVAL: //TODO: implement setblock_80 for haval256
			//x17_haval256_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], 256); order++;
			break;
		default: {
			if (!thr_id)
				applog(LOG_WARNING, "kernel %s %c unimplemented, order %s", algo_strings[algo80], elem, hashOrder);
			s_implemented = false;
			applog(LOG_INFO, "s_implemented is false, wait 5 min to terminate %s \n", hashOrder);
			sleep(5);
			return -1;
		}
	}

	int warn = 0;

	do {
		int order = 0;

		// Hash with CUDA

		switch (algo80) {
			case BLAKE:
				quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("blake80:");
				break;
			case BMW:
				quark_bmw512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
				TRACE("bmw80  :");
				break;
			case GROESTL:
				groestl512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("grstl80:");
				break;
			case JH:
				jh512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("jh51280:");
				break;
			case KECCAK:
				keccak512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("kecck80:");
				break;
			case SKEIN:
				skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], 1); order++;
				TRACE("skein80:");
				break;
			case LUFFA:
				qubit_luffa512_cpu_hash_80_alexis(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("luffa80:");
				break;
			case CUBEHASH:
				cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("cube 80:");
				break;
			case SHAVITE:
				x11_shavite512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
				TRACE("shavite:");
				break;
			case SIMD:
				x16_simd512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("simd512:");
				break;
			case ECHO:
				x11_echo512_cpu_hash_80_alexis(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("echo   :");
				break;
			case HAMSI:
				x16_hamsi512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("hamsi  :");
				break;
			case FUGUE:
				x16_fugue512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("fugue  :");
				break;
			case SHABAL:
				x16_shabal512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("shabal :");
				break;
			case WHIRLPOOL:
				x16_whirlpool512_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("whirl  :");
				break;
			case SHA512:
				x16_sha512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("sha512 :");
				break;
			case HAVAL:
				order++;
				//x17_haval256_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], 256); order++;
				TRACE("haval256");
				break;
		}

		for (int i = 1; i < 17; i++)
		{
			const char elem = hashOrder[i];
			const uint8_t algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

			switch (algo64) {
			case BLAKE:
				quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("blake  :");
				break;
			case BMW:
				quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("bmw    :");
				break;
			case GROESTL:
				quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("groestl:");
				break;
			case JH:
				quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("jh512  :");
				break;
			case KECCAK:
				quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("keccak :");
				break;
			case SKEIN:
				quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("skein  :");
				break;
			case LUFFA:
				x11_luffa512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
				TRACE("luffa  :");
				break;
			case CUBEHASH:
				x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("cube   :");
				break;
			case SHAVITE:
				x11_shavite512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
				TRACE("shavite:");
				break;
			case SIMD:
				x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("simd   :");
				break;
			case ECHO:
				x11_echo512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
				TRACE("echo   :");
				break;
			case HAMSI:
				x13_hamsi512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
				TRACE("hamsi  :");
				break;
			case FUGUE:
				x13_fugue512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
				TRACE("fugue  :");
				break;
			case SHABAL:
				x14_shabal512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
				TRACE("shabal :");
				break;
			case WHIRLPOOL:
				x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("shabal :");
				break;
			case SHA512:
				x17_sha512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("sha512 :");
				break;
			case HAVAL:
				x17_haval256_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], 256); order++;
				TRACE("haval256");
				break;
			}
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
#ifdef _DEBUG
		uint32_t _ALIGN(64) dhash[8];
		be32enc(&endiandata[19], pdata[19]);
		trihash(dhash, endiandata);
		applog_hash(dhash);
		return -1;
#endif
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			trihash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					trihash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				#ifdef GPU_HASH_CHECK_LOG
				gpulog(LOG_INFO, thr_id, "hash found with %s 80!", algo_strings[algo80]);

				algo80_tests[algo80] += work->valid_nonces;
				char oks64[128] = { 0 };
				char oks80[128] = { 0 };
				char fails[128] = { 0 };
				for (int a = 0; a < HASH_FUNC_COUNT; a++) {
						const char elem = hashOrder[a];
						const uint8_t algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';
						if (a > 0) algo64_tests[algo64] += work->valid_nonces;
						sprintf(&oks64[strlen(oks64)], "|%X:%2d", a, algo64_tests[a] < 100 ? algo64_tests[a] : 99);
						sprintf(&oks80[strlen(oks80)], "|%X:%2d", a, algo80_tests[a] < 100 ? algo80_tests[a] : 99);
						sprintf(&fails[strlen(fails)], "|%X:%2d", a, algo80_fails[a] < 100 ? algo80_fails[a] : 99);
				}
				applog(LOG_INFO, "K64: %s", oks64);
				applog(LOG_INFO, "K80: %s", oks80);
				applog(LOG_ERR,  "F80: %s", fails);
				#endif
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				// x11+ coins could do some random error, but not on retry
				gpu_increment_reject(thr_id);
				algo80_fails[algo80]++;
				if (!warn) {
					warn++;
					pdata[19] = work->nonces[0] + 1;
					continue;
				} else {
					if (!opt_quiet)	gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU! %s %s",
						work->nonces[0], algo_strings[algo80], hashOrder);
					warn = 0;
				}
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_trihash(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);
	x16_fugue512_cpu_free(thr_id); // to merge with x13_fugue512 ?
	x15_whirlpool_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
