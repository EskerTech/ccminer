#include <string.h>
#include "cuda_helper.h"
#include "miner.h"

extern void neoscrypt_setBlockTarget(int thr_id, uint32_t* pdata, const void *target);
extern void neoscrypt_cpu_init_2stream(int thr_id, uint32_t threads);
extern void neoscrypt_cpu_hash_k4_2stream(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *result);
//extern void neoscrypt_cpu_hash_k4_52(int stratum, int thr_id, int threads, uint32_t startNounce, int order, uint32_t* foundnonce);
void neoscrypt_init(int thr_id, uint32_t threads);
void neoscrypt_setBlockTarget_tpruvot(uint32_t* const pdata, uint32_t* const target);
void neoscrypt_hash_tpruvot(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNonces);

int scanhash_neoscrypt(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t* pdata = work->data;
	uint32_t* ptarget = work->target;

	const uint32_t first_nonce = pdata[19];
	uint32_t throughput;
	static THREAD uint32_t throughputmax;

	static THREAD volatile bool init = false;
	static THREAD uint32_t hw_errors = 0;
	static THREAD uint32_t *foundNonce = nullptr;
	static THREAD bool use_tpruvot = false;

	if (opt_benchmark)
	{
		ptarget[7] = 0x01ff;
	}

	if (!init)
	{
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));

		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, device_map[thr_id]);
		unsigned int cc = props.major * 10 + props.minor;
		if (cc <= 30)
		{
			applog(LOG_ERR, "GPU #%d: this gpu is not supported", device_map[thr_id]);
			//mining_has_stopped[thr_id] = true;
			proper_exit(2);
		}
		if (cc == 30)
			use_tpruvot = true;

		unsigned int intensity = (256 * 64 * 1); // -i 14
		if (strstr(props.name, "1080 Ti"))
		{
			intensity = 256 * 64 * 5;
			use_tpruvot = true;
		}
		else if (strstr(props.name, "1080"))
		{
			intensity = 256 * 64 * 5;
			use_tpruvot = true;
		}
		else if (strstr(props.name, "1070"))
		{
			intensity = 256 * 64 * 5;
		}
		else if (strstr(props.name, "970"))
		{
			intensity = (256 * 64 * 5);
		}
		else if (strstr(props.name, "980"))
		{
			intensity = (256 * 64 * 5);
		}
		else if (strstr(props.name, "980 Ti"))
		{
			intensity = (256 * 64 * 5);
		}
		else if (strstr(props.name, "750 Ti"))
		{
			intensity = (256 * 64 * 3);
		}
		else if (strstr(props.name, "750"))
		{
			intensity = (256 * 64 * 1);
		}
		else if (strstr(props.name, "960"))
		{
			intensity = (256 * 64 * 2);
		}
		else if (strstr(props.name, "950"))
		{
			intensity = (256 * 64 * 2);
		}


		throughputmax = device_intensity(device_map[thr_id], __func__, intensity) / 2;
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		//		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);	
		CUDA_SAFE_CALL(cudaMallocHost(&foundNonce, 2 * 4));

#if defined WIN32 && !defined _WIN64
		// 2GB limit for cudaMalloc
		if (throughputmax > 0x7fffffffULL / (32 * 128 * sizeof(uint64_t)))
		{
			applog(LOG_ERR, "intensity too high");
			mining_has_stopped[thr_id] = true;
			proper_exit(2);
		}
#endif
		if (use_tpruvot)
			neoscrypt_init(thr_id, throughputmax);
		else
			neoscrypt_cpu_init_2stream(thr_id, throughputmax);
		//mining_has_stopped[thr_id] = false;
		init = true;
	}

	throughput = (uint32_t)((throttle / 100) * throughputmax);
	throughput = min(throughput, (max_nonce - first_nonce) / 2) & 0xffffff00;

	uint32_t endiandata[20];
	for (int k = 0; k < 20; k++)
	{
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);
	}
	if (use_tpruvot)
		neoscrypt_setBlockTarget_tpruvot(endiandata, ptarget);
	else
		neoscrypt_setBlockTarget(thr_id, endiandata, ptarget);


	do
	{
		if(throttle < 100) usleep((100.0f - throttle) * 500);

		if (use_tpruvot)
			neoscrypt_hash_tpruvot(thr_id, throughput, pdata[19], foundNonce);
		else
			neoscrypt_cpu_hash_k4_2stream(thr_id, throughput, pdata[19], foundNonce);
		/*if (stop_mining)
		{
			mining_has_stopped[thr_id] = true; pthread_exit(nullptr);
		}*/
		if (foundNonce[0] != 0xffffffff)
		{
			uint32_t vhash64[8] = { 0 };

			be32enc(&endiandata[19], foundNonce[0]);
			neoscrypt((unsigned char*)endiandata, (unsigned char*)vhash64, 0x80000620);

			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget))
			{
				*hashes_done = pdata[19] - first_nonce + throughput;
				int res = 1;
				if (opt_benchmark)
					applog(LOG_INFO, "GPU #%d Found nonce %08x", device_map[thr_id], foundNonce[0]);
				pdata[19] = foundNonce[0];
				if (foundNonce[1] != 0xffffffff)
				{
					be32enc(&endiandata[19], foundNonce[1]);
					neoscrypt((unsigned char*)endiandata, (unsigned char*)vhash64, 0x80000620);

					if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget))
					{
						pdata[21] = foundNonce[1];
						res++;
						if (opt_benchmark)
							applog(LOG_INFO, "GPU #%d: Found second nonce %08x", device_map[thr_id], foundNonce[1]);
					}
					else
					{
						if (vhash64[7] != ptarget[7])
						{
							applog(LOG_WARNING, "GPU #%d: Second nonce $%08X does not validate on CPU!", device_map[thr_id], foundNonce[1]);
							hw_errors++;
						}
					}

				}
				return res;
			}
			else
			{
				if (vhash64[7] != ptarget[7])
				{
					applog(LOG_WARNING, "GPU #%d: Nonce $%08X does not validate on CPU!", device_map[thr_id], foundNonce[0]);
					hw_errors++;
				}
			}
			//						if(hw_errors > 0) applog(LOG_WARNING, "Hardware errors: %u", hw_errors);
		}
		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));
	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

void free_neoscrypt(int thr_id)
{
	// Nothing to do
}