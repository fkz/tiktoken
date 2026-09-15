#include <math.h>
static double stream_seconds(void) {
    const char *value = getenv("STREAM_SECONDS");
    if (!value) return 0;
    char *end;
    double seconds = strtod(value, &end);
    if (!*value || *end || !isfinite(seconds) || seconds <= 0 || seconds > 30) {
        fprintf(stderr, "STREAM_SECONDS must be in (0, 30]\n"); exit(2);
    }
    return seconds;
}
static void stream_wait_start(void) {
    if (!getenv("STREAM_SYNC")) return;
    puts("READY"); fflush(stdout);
    double target;
    if (scanf("%lf", &target) != 1 || !isfinite(target) || target <= 0) exit(2);
    struct timespec t = {.tv_sec = (time_t)target,
                         .tv_nsec = (long)((target - (time_t)target) * 1e9)};
    int error;
    do { error = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL); } while (error == EINTR);
    if (error) { fprintf(stderr, "clock_nanosleep failed: %d\n", error); exit(1); }
}
