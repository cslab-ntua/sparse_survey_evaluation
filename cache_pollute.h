#ifndef CACHE_POLLUTE_H
#define CACHE_POLLUTE_H

// Optional cache pollution between the timed iterations of the benchmarks (spmm_bench, sddmm_bench, attention_bench), to
// model a kernel that runs between other work (e.g. the other layers of an LLM inference step) instead of back to back
// with warm caches. Off by default.
//
//   CACHE_POLLUTE=1           enable, 512 MB buffer
//   CACHE_POLLUTE_MB=<n>      enable with an n MB buffer (0 = off). Use at least 2x the L3 the threads use: on the AMD
//                             EPYC 7402 (24 threads on one socket, 8 x 16 MB L3 slices) the slowdown is complete from
//                             256 MB on (diag_cache_sweep.sh, 2026-10-27).
//   CACHE_POLLUTE_MODE=       write  (default) read and write one value per cache line: the caches are full of unrelated
//                                    dirty data, the kernel then also pays their write-backs (pessimistic bound)
//                             read   read one value per cache line: same eviction, clean lines (optimistic bound)
//                             x_warm write pollution, then the dense inputs are read once (x for SpMM, x and y for SDDMM,
//                                    Q, K, V for attention): inputs just produced by the previous layer
//
// Before every timed iteration (outside the timed region) the same OpenMP threads that run the kernel each touch their own
// static share of the buffer, so every core's private L1 / L2, the shared L3 and the TLB are polluted.
//
// Usage in a benchmark:
//     long mb = cache_pollute_init();                         // 0 = off
//     if (mb > 0) printf("Cache pollution between iterations: %ld MB (%s)\n", mb, cache_pollute_mode_name());
//     for (...) {
//         if (mb > 0) { cache_pollute(); cache_pollute_warm(x, x_n); ... }      // not timed
//         time += time_it(1, kernel(...););
//     }
//     cache_pollute_free();

#include <stdlib.h>
#include <string.h>

static float * cache_pollute_buf = NULL;
static long cache_pollute_n = 0;           // floats in the buffer
static int cache_pollute_mode = 0;         // 0 write, 1 read, 2 x_warm
static volatile float cache_pollute_sink;  // keeps the reads from being optimized away

static inline const char * cache_pollute_mode_name(void)
{
	static const char * names[] = {"write", "read", "x_warm"};
	return names[cache_pollute_mode];
}

// Reads the environment, allocates and first-touches the buffer. Returns the buffer size in MB (0 = pollution off).
static inline long cache_pollute_init(void)
{
	long mb = 0;
	const char * env_mb = getenv("CACHE_POLLUTE_MB");
	const char * env_on = getenv("CACHE_POLLUTE");
	const char * env_mode = getenv("CACHE_POLLUTE_MODE");
	if (env_mb != NULL)
		mb = atol(env_mb);
	else if (env_on != NULL && atoi(env_on))
		mb = 512;
	if (mb <= 0)
		return 0;
	if (env_mode != NULL && strcmp(env_mode, "read") == 0)
		cache_pollute_mode = 1;
	else if (env_mode != NULL && strcmp(env_mode, "x_warm") == 0)
		cache_pollute_mode = 2;
	else
		cache_pollute_mode = 0;
	cache_pollute_n = mb * 1024L * 1024L / sizeof(*cache_pollute_buf);
	cache_pollute_buf = (float *) aligned_alloc(64, cache_pollute_n * sizeof(*cache_pollute_buf));
	#pragma omp parallel for schedule(static)
	for (long i = 0; i < cache_pollute_n; i++)      // first touch by the same threads and in the same split as cache_pollute()
		cache_pollute_buf[i] = 0;
	return mb;
}

// Pollutes the caches (read or write, depending on the mode).
static inline void cache_pollute(void)
{
	const long stride = 64 / sizeof(*cache_pollute_buf);     // one value per 64-byte cache line
	if (cache_pollute_mode == 1) {
		float sum = 0;
		#pragma omp parallel for schedule(static) reduction(+:sum)
		for (long i = 0; i < cache_pollute_n; i += stride)
			sum += cache_pollute_buf[i];
		cache_pollute_sink = sum;
	} else {
		#pragma omp parallel for schedule(static)
		for (long i = 0; i < cache_pollute_n; i += stride)
			cache_pollute_buf[i] += 1.0f;
	}
}

// x_warm mode: reads one value per cache line of a dense input after the pollution. No-op in the other modes.
static inline void cache_pollute_warm(const ValueType * a, long n)
{
	if (cache_pollute_mode != 2)
		return;
	const long stride = 64 / sizeof(*a);
	float sum = 0;
	#pragma omp parallel for schedule(static) reduction(+:sum)
	for (long i = 0; i < n; i += stride)
		sum += a[i];
	cache_pollute_sink = sum;
}

static inline void cache_pollute_free(void)
{
	free(cache_pollute_buf);
	cache_pollute_buf = NULL;
	cache_pollute_n = 0;
}

#endif /* CACHE_POLLUTE_H */
