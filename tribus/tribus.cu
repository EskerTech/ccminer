/**
 * Tribus Algo for Denarius
 *
 * tpruvot@github 09 2017 - GPLv3
 *
 */
extern "C" {
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_echo.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "x11/cuda_x11.h"

#define AS_U64(addr) *((uint64_t*)(addr))

#define NBN 2

void jh512_setBlock_80(int thr_id, uint32_t *endiandata);
void jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash);
void tribus_echo512_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

// cpu hash

extern "C" void tribus_hash(void *state, const void *input)
{
	uint8_t _ALIGN(64) hash[64];

	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_echo512_context ctx_echo;

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, input, 80);
	sph_jh512_close(&ctx_jh, (void*) hash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, (const void*) hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*) hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, (void*) hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_tribus(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{	
	uint32_t _ALIGN(64) endiandata[20];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	int8_t intensity = is_windows() ? 20 : 23;
	uint32_t throughput =  cuda_default_throughput(thr_id, 1 << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00FF;

	if (!init[thr_id])
	{
		int dev_id = device_map[thr_id];
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		x11_simd_echo_512_cpu_init(thr_id, throughput);

		cuda_get_arch(thr_id);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		h_resNonce[thr_id] = (uint32_t*)malloc(NBN * sizeof(uint32_t));
		if (h_resNonce[thr_id] == NULL) {
			gpulog(LOG_ERR, thr_id, "Host memory allocation failed");
			exit(EXIT_FAILURE);
		}
		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	jh512_setBlock_80(thr_id, endiandata);
	cudaMemset(d_resNonce[thr_id], 0xff, NBN * sizeof(uint32_t));

	do
	{
		jh512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		tribus_echo512_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id], AS_U64(&ptarget[6]));

		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (h_resNonce[thr_id][0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], startNounce + h_resNonce[thr_id][0]);
			tribus_hash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				work_set_target_ratio(work, vhash64);
				pdata[19] = startNounce + h_resNonce[thr_id][0];
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
					//					if(!opt_quiet)
					//						gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %08x", h_resNonce[thr_id][1]);
					be32enc(&endiandata[19], startNounce + h_resNonce[thr_id][1]);
					tribus_hash(vhash64, endiandata);
					pdata[21] = startNounce + h_resNonce[thr_id][1];
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0]) {
						work_set_target_ratio(work, vhash64);
						xchg(pdata[19], pdata[21]);
					}
					res++;
				}
				return res;
			}
			else {
				gpulog(LOG_WARNING, dev_id, "result for %08x does not validate on CPU!", h_resNonce[thr_id][0]);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN * sizeof(uint32_t));
			}
		}

		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && (((uint64_t)pdata[19] + (uint64_t)throughput) < (uint64_t)max_nonce));

	*hashes_done = pdata[19] - first_nonce;

	return 0;
}

// ressources cleanup
extern "C" void free_tribus(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
