// Read-only streaming bandwidth. Compile: zig cc -O3 -mavx2 -pthread ...
// A fixed 1 GiB buffer is split among workers; setup is outside the timed region.
#define _GNU_SOURCE
#include <errno.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "stream-timing.h"

enum { PASSES = 32 };
static pthread_barrier_t ready, start, done;
static double duration, deadline;
struct worker {
    uint64_t *data;
    size_t words;
    int cpu;
    uint64_t checksum;
    uint64_t passes_done;
};

static void check(int error, const char *operation) {
    if (error) {
        fprintf(stderr, "%s: %s\n", operation, strerror(error));
        exit(1);
    }
}

static double now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(1); }
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *read_worker(void *arg) {
    struct worker *w = arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(w->cpu, &set);
    check(pthread_setaffinity_np(pthread_self(), sizeof(set), &set), "pin worker");
    for (size_t i = 0; i < w->words; i++) w->data[i] = 1;
    pthread_barrier_wait(&ready);
    pthread_barrier_wait(&start);
    __m256i a = _mm256_setzero_si256(), b = a, c = a, d = a;
    do {
        // Force fresh loads each pass even under aggressive optimization.
        __asm__ volatile("" ::: "memory");
        for (size_t i = 0; i < w->words; i += 16) {
            a = _mm256_add_epi64(a, _mm256_load_si256((const __m256i *)(w->data + i)));
            b = _mm256_add_epi64(b, _mm256_load_si256((const __m256i *)(w->data + i + 4)));
            c = _mm256_add_epi64(c, _mm256_load_si256((const __m256i *)(w->data + i + 8)));
            d = _mm256_add_epi64(d, _mm256_load_si256((const __m256i *)(w->data + i + 12)));
        }
        w->passes_done++;
    } while (duration ? now() < deadline : w->passes_done < PASSES);
    uint64_t sums[4];
    _mm256_storeu_si256((__m256i *)sums, _mm256_add_epi64(_mm256_add_epi64(a,b), _mm256_add_epi64(c,d)));
    w->checksum = sums[0] + sums[1] + sums[2] + sums[3];
    pthread_barrier_wait(&done);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 65) {
        fprintf(stderr, "Usage: %s CPU_ID [CPU_ID ...] (up to 64 workers)\n", argv[0]);
        return 2;
    }
    const size_t bytes = (size_t)1 << 30, blocks = bytes / 128;
    int count = argc - 1;
    duration = stream_seconds();
    struct worker workers[64] = {0};
    pthread_t threads[64];
    uint64_t *data;
    check(posix_memalign((void **)&data, 4096, bytes), "allocate buffer");
    check(pthread_barrier_init(&ready, NULL, count + 1), "ready barrier");
    check(pthread_barrier_init(&start, NULL, count + 1), "start barrier");
    check(pthread_barrier_init(&done, NULL, count + 1), "done barrier");
    for (int i = 0; i < count; i++) {
        char *end;
        errno = 0;
        long cpu = strtol(argv[i + 1], &end, 10);
        if (errno || !*argv[i + 1] || *end || cpu < 0 || cpu >= CPU_SETSIZE) {
            fprintf(stderr, "Invalid CPU: %s\n", argv[i + 1]);
            return 2;
        }
        size_t begin = blocks * i / count, finish = blocks * (i + 1) / count;
        workers[i] = (struct worker){.data = data + begin * 16, .words = (finish - begin) * 16, .cpu = cpu};
        check(pthread_create(&threads[i], NULL, read_worker, &workers[i]), "start worker");
    }
    pthread_barrier_wait(&ready);
    stream_wait_start();
    double before = now();
    deadline = before + duration;
    pthread_barrier_wait(&start);
    pthread_barrier_wait(&done);
    double after = now(), seconds = after - before;
    uint64_t read_bytes = 0;
    for (int i = 0; i < count; i++) {
        check(pthread_join(threads[i], NULL), "join worker");
        if (workers[i].checksum != workers[i].words * workers[i].passes_done) {
            fprintf(stderr, "Read checksum failed\n");
            return 1;
        }
        read_bytes += workers[i].words * sizeof(uint64_t) * workers[i].passes_done;
    }
    printf("{\"workers\":%d,\"buffer_bytes\":%zu,\"read_bytes\":%llu,\"seconds\":%.9f,\"start_s\":%.9f,\"end_s\":%.9f,\"GB_s\":%.6f}\n",
           count, bytes, (unsigned long long)read_bytes, seconds, before, after, read_bytes / seconds / 1e9);
    free(data);
    return 0;
}
