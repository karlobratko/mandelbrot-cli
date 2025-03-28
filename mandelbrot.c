#include <assert.h>
#include <complex.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
#include <time.h>
#include <xmmintrin.h>

// lib

#define TODO(message) assert(0 && message)

#define ignored (void)

#define var __auto_type

#define array(T, N) __typeof__(T[N])

#define defer(f) __attribute__ ((cleanup(f)))

#define bool_fmt "%s"

#define bool_arg(x)                                \
    ( (x) == true  ? "true"                        \
    : (x) == false ? "false"                       \
    : (assert(0 && "Unsupported bool value."), "") )

typedef const char *cstr;

typedef size_t   usize;
typedef intptr_t isize;

static_assert(sizeof(char)     == 1,  "sizeof(char) != 1");
static_assert(sizeof(short)    == 2,  "sizeof(short) != 2");
static_assert(sizeof(int)      == 4,  "sizeof(int) != 4");
static_assert(sizeof(long)     == 8,  "sizeof(long) != 8");
static_assert(sizeof(__int128) == 16, "sizeof(__int128) != 16");

typedef char              i8;
typedef unsigned char     u8;
typedef short             i16;
typedef unsigned short    u16;
typedef int               i32;
typedef unsigned int      u32;
typedef long              i64;
typedef unsigned long     u64;
typedef __int128          i128;
typedef unsigned __int128 u128;

#define U32_MAX UINT_MAX

static_assert(sizeof(float)       == 4,  "sizeof(float) != 4");
static_assert(sizeof(double)      == 8,  "sizeof(double) != 8");
static_assert(sizeof(long double) == 16, "sizeof(long double) != 16");

typedef float       f32;
typedef double      f64;
typedef long double f128;

#define F32_MIN FLT_MIN
#define F32_MAX FLT_MAX

#define MM_ALIGN 16

#define cast_vector(T, x) __builtin_convertvector(x, T)

//typedef array(f32, 2) f32x2;
typedef f32 f32x2 __attribute__ ((vector_size(2 * sizeof(f32)), aligned(MM_ALIGN)));

#define f32x2_fmt "[%f,%f]"

#define f32x2_arg(x) (x)[0], (x)[1]

typedef __m128 f32x4;

#define f32x4_fmt "[%f,%f,%f,%f]"

#define f32x4_arg(x) (x)[0], (x)[1], (x)[2], (x)[3]

//typedef array(u32, 2) u32x2;
typedef u32 u32x2 __attribute__ ((vector_size(2 * sizeof(u32)), aligned(MM_ALIGN)));

#define u32x2_fmt "[%u,%u]"

#define u32x2_arg(x) (x)[0], (x)[1]

typedef __m128i u32x4;

#define u32x4_fmt "[%u,%u,%u,%u]"

#define u32x4_arg(x) (x)[0], (x)[1], (x)[2], (x)[3]

typedef complex float complex_f32;

#define Re(x) __real__ x
#define Im(x) __imag__ x

static int strtoi(const char *str, char **endptr, int base) {
    const long res = strtol(str, endptr, base);
    if (res < INT_MIN) {
        errno = ERANGE;
        return INT_MIN;
    }
    if (res > INT_MAX) {
        errno = ERANGE;
        return INT_MAX;
    }

    return (int) res;
}

static unsigned int strtoui(const char *str, char **endptr, int base) {
    const unsigned long res = strtoul(str, endptr, base);
    if (res > UINT_MAX) {
        errno = ERANGE;
        return UINT_MAX;
    }

    return res;
}

#define str2dec(T, str, endptr) _Generic((*(T*)0),                    \
    double:             strtod  ((str), (endptr)),                    \
    long double:        strtold ((str), (endptr)),                    \
    float:              strtof  ((str), (endptr)),                    \
    int:                strtoi  ((str), (endptr), 10),                \
    unsigned int:       strtoui ((str), (endptr), 10),                \
    long:               strtol  ((str), (endptr), 10),                \
    unsigned long:      strtoul ((str), (endptr), 10),                \
    long long:          strtoll ((str), (endptr), 10),                \
    unsigned long long: strtoull((str), (endptr), 10),                \
    default:            (assert(0 && "Unsupported str2dec type."), 0) \
)

#define max(a, b) ({   \
    const var _a = a;        \
    const var _b = b;        \
    _a > _b ? _a : _b; \
})

#define min(a, b) ({   \
    const var _a = a;        \
    const var _b = b;        \
    _a < _b ? _a : _b; \
})

#define minmax(x, min_val, max_val) _Generic((x),                    \
    char:               max  (min_val, min  (x, max_val)),           \
    unsigned char:      max  (min_val, min  (x, max_val)),           \
    short:              max  (min_val, min  (x, max_val)),           \
    unsigned short:     max  (min_val, min  (x, max_val)),           \
    int:                max  (min_val, min  (x, max_val)),           \
    unsigned int:       max  (min_val, min  (x, max_val)),           \
    long:               max  (min_val, min  (x, max_val)),           \
    unsigned long:      max  (min_val, min  (x, max_val)),           \
    long long:          max  (min_val, min  (x, max_val)),           \
    unsigned long long: max  (min_val, min  (x, max_val)),           \
    float:              fmaxf(min_val, fminf(x, max_val)),           \
    double:             fmax (min_val, fmin (x, max_val)),           \
    long double:        fmaxl(min_val, fminl(x, max_val)),           \
    default:            (assert(0 && "Unsupported minmax type."), 0) \
)

static void defer_free(void *ptr) {
    const var p = (void**)ptr;
    free(*p);
}

static void defer_fclose(void *ptr) {
    const var fp = (FILE**)ptr;
    fclose(*fp);
}

#define fill(buf, len, v) ({          \
    for (usize i = 0; i < len; i++) { \
        buf[i] = v;                   \
    }                                 \
    buf;                              \
})

#define swap(a, b) ({ \
    const var tmp = *(a);   \
    *(a) = *(b);      \
    *(b) = tmp;       \
})

f64 get_time_s(void) {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        clock_gettime(CLOCK_REALTIME, &ts);
    }

    return (f64) ts.tv_sec + (f64) ts.tv_nsec * 1.e-9;
}

// mandelbrot

#define MAX_W     4096
#define DEFAULT_W 800

#define MAX_H     2160
#define DEFAULT_H 600

#define MAX_MAX_ITERATIONS     100000
#define DEFAULT_MAX_ITERATIONS 1000

#define MIN_ZOOM     0.0f
#define DEFAULT_ZOOM 1.0f

#define MAX_OUTPUT_LENGTH 256

#define DEFAULT_CENTER_X (-0.5f)
#define DEFAULT_CENTER_Y   0.0f

#define X_SCALE_FACTOR 3.5f

#define ESCAPE_RADIUS_2 4.0f

#define N_THREADS 4

#define dimens_fmt "%ux%u"

#define dimens_arg(v) ((v)[0]), ((v)[1])

#define dimens_length(v) (((v)[0]) * ((v)[1]))

enum color_mode {
    COLOR_MODE_GRAYSCALE,
    COLOR_MODE_SMOOTH,
    COLOR_MODE_HSV
};

#define color_mode_fmt "%s"

#define color_mode_arg(x) ({                                          \
    cstr s = NULL;                                                    \
    switch (x) {                                                      \
        case COLOR_MODE_GRAYSCALE: s = "COLOR_MODE_GRAYSCALE"; break; \
        case COLOR_MODE_SMOOTH:    s = "COLOR_MODE_SMOOTH";    break; \
        case COLOR_MODE_HSV:       s = "COLOR_MODE_HSV";       break; \
        default: assert(0 && "Unsupported color_mode value."); break; \
    }                                                                 \
    s;                                                                \
})

enum processing_mode {
    PROCESSING_MODE_SISD,
    PROCESSING_MODE_SIMD
};

#define processing_mode_fmt "%s"

#define processing_mode_arg(x) ({                                          \
    cstr s = NULL;                                                         \
    switch (x) {                                                           \
        case PROCESSING_MODE_SISD: s = "PROCESSING_MODE_SISD";      break; \
        case PROCESSING_MODE_SIMD: s = "PROCESSING_MODE_SIMD";      break; \
        default: assert(0 && "Unsupported processing_mode value."); break; \
    }                                                                      \
    s;                                                                     \
})

void print_usage(cstr target) {
    printf("Usage: %s [options]\n", target);
    printf("Options:\n");
    printf("  --width N          Set image width (default: %d)\n", DEFAULT_W);
    printf("  --height N         Set image height (default: %d)\n", DEFAULT_H);
    printf("  --iterations N     Set maximum iterations (default: %d)\n", DEFAULT_MAX_ITERATIONS);
    printf("  --center X Y       Set center coordinates (default: %.2f, %.2f)\n", DEFAULT_CENTER_X, DEFAULT_CENTER_Y);
    printf("  --zoom N           Set zoom level (default: %.2f)\n", DEFAULT_ZOOM);
    printf("  --color MODE       Set coloring mode: grayscale, smooth, hsv (default: smooth)\n");
    printf("  --no-smooth        Disable smooth coloring\n");
    printf("  --output FILE      Set output filename (default: mandelbrot.ppm)\n");
    printf("  --scalar           Use scalar computation\n");
    printf("  --vector           Use vectorized computation (default)\n");
    printf("  --benchmark        Run benchmark comparing scalar and vector methods\n");
    printf("  --debug            Run program in debug mode (with additional logging)\n");
    printf("  --locations        Show predefined interesting locations\n");
    printf("  --help             Show this help message\n");
}

void print_locations(void) {
    printf("Predefined interesting locations:\n");
    printf("1. Main Mandelbrot view: --center -0.5 0 --zoom 1\n");
    printf("2. Seahorse Valley: --center -0.745 0.1 --zoom 80\n");
    printf("3. Elephant Valley: --center 0.3 -0.01 --zoom 30\n");
    printf("4. Triple Spiral: --center -0.041 0.682 --zoom 200\n");
    printf("5. Quad Spiral: --center -1.25 0.02 --zoom 100\n");
    printf("6. Mini Mandelbrot: --center -1.77 0 --zoom 50\n");
    printf("7. Feigenbaum Point: --center -1.401155 0 --zoom 10000\n");
}

struct program_args {
    u32x2 dimens;
    u32 max_iterations;
    f32x2 center;
    f32 zoom;
    enum color_mode color_mode;
    array(i8, MAX_OUTPUT_LENGTH) output_filename;
    enum processing_mode processing_mode;
};

void parse_program_args(cstr *argv, i32 argc, struct program_args *args) {
    const var target = argv[0];

    args->dimens = (u32x2){DEFAULT_W, DEFAULT_H};
    args->max_iterations = DEFAULT_MAX_ITERATIONS;
    args->center = (f32x2){DEFAULT_CENTER_X, DEFAULT_CENTER_Y};
    args->zoom = DEFAULT_ZOOM;
    args->color_mode = COLOR_MODE_SMOOTH;
    strcpy(args->output_filename, "mandelbrot.ppm");
    args->processing_mode = PROCESSING_MODE_SISD;

    for (i32 i = 1; i < argc; i++) {
        const var value = argv[i];
        if (strcmp(value, "--width") == 0 && i + 1 < argc) {
            args->dimens[0] = str2dec(u32, argv[i + 1], NULL);
            if (errno == ERANGE || args->dimens[0] > MAX_W) {
                errno = 0;
                fprintf(stderr, "--width value outside of valid range [0, %u]\n", MAX_W);
                print_usage(target);
                exit(EXIT_FAILURE);
            }
            i++;
        } else if (strcmp(value, "--height") == 0 && i + 1 < argc) {
            args->dimens[1] = str2dec(u32, argv[i + 1], NULL);
            if (errno == ERANGE || args->dimens[1] > MAX_H) {
                errno = 0;
                fprintf(stderr, "--height value outside of valid range [0, %u]\n", MAX_H);
                print_usage(target);
                exit(EXIT_FAILURE);
            }
            i++;
        } else if (strcmp(value, "--iterations") == 0 && i + 1 < argc) {
            args->max_iterations = str2dec(u32, argv[i + 1], NULL);
            if (errno == ERANGE || args->max_iterations > MAX_MAX_ITERATIONS) {
                errno = 0;
                fprintf(stderr, "--iterations value outside of valid range [0, %u]\n", MAX_MAX_ITERATIONS);
                print_usage(target);
                exit(EXIT_FAILURE);
            }
            i++;
        } else if (strcmp(value, "--center") == 0 && i + 2 < argc) {
            args->center[0] = str2dec(f32, argv[i + 1], NULL);
            if (errno == ERANGE) {
                errno = 0;
                fprintf(stderr, "--center X value outside of valid range [%f, %f]\n", F32_MIN, F32_MAX);
                print_usage(target);
                exit(EXIT_FAILURE);
            }

            args->center[1] = str2dec(f32, argv[i + 2], NULL);
            if (errno == ERANGE) {
                errno = 0;
                fprintf(stderr, "--center Y value outside of valid range [%f, %f]\n", F32_MIN, F32_MAX);
                print_usage(target);
                exit(EXIT_FAILURE);
            }

            i += 2;
        } else if (strcmp(value, "--zoom") == 0 && i + 1 < argc) {
            args->zoom = str2dec(f32, argv[i + 1], NULL);
            if (errno == ERANGE || args->zoom < MIN_ZOOM) {
                errno = 0;
                fprintf(stderr, "--zoom value outside of valid range [%f, %f]\n", MIN_ZOOM, F32_MAX);
                print_usage(target);
                exit(EXIT_FAILURE);
            }
            i++;
        } else if (strcmp(value, "--color") == 0 && i + 1 < argc) {
            const var next_value = argv[i + 1];
            if (strcmp(next_value, "grayscale") == 0) {
                args->color_mode = COLOR_MODE_GRAYSCALE;
            } else if (strcmp(next_value, "smooth") == 0) {
                args->color_mode = COLOR_MODE_SMOOTH;
            } else if (strcmp(next_value, "hsv") == 0) {
                args->color_mode = COLOR_MODE_HSV;
            } else {
                fprintf(stderr, "--color value outside of valid range [grayscale, smooth, hsv]\n");
                print_usage(target);
                exit(EXIT_FAILURE);
            }
            i++;
        } else if (strcmp(value, "--output") == 0 && i + 1 < argc) {
            const var next_value = argv[i + 1];
            const var next_value_len = strlen(next_value);
            if (next_value_len >= MAX_OUTPUT_LENGTH) {
                fprintf(stderr, "--output length outside of valid range [0, %d]\n", MAX_OUTPUT_LENGTH);
                print_usage(target);
                exit(EXIT_FAILURE);
            }

            strcpy(args->output_filename, next_value);
            i++;
        } else if (strcmp(value, "--scalar") == 0) {
            args->processing_mode = PROCESSING_MODE_SISD;
        } else if (strcmp(value, "--vector") == 0) {
            args->processing_mode = PROCESSING_MODE_SIMD;
        } else if (strcmp(value, "--locations") == 0) {
            print_locations();
            exit(EXIT_SUCCESS);
        } else if (strcmp(value, "--help") == 0) {
            print_usage(target);
            exit(EXIT_SUCCESS);
        } else {
            fprintf(stderr, "Unknown option: %s\n", value);
            print_usage(target);
            exit(EXIT_FAILURE);
        }
    }
}

void mandelbrot_compute_scalar(u32 *iterations,
                               u32 max_iterations,
                               u32x2 dimens,
                               u32x2 pixel_tl,
                               u32x2 pixel_br,
                               f32x2 fract_tl,
                               f32x2 fract_br) {
    const var x_scale = (fract_br[0] - fract_tl[0]) / ((f32)pixel_br[0] - (f32)pixel_tl[0]);
    const var y_scale = (fract_br[1] - fract_tl[1]) / ((f32)pixel_br[1] - (f32)pixel_tl[1]);

    for (u32 y = pixel_tl[1]; y < pixel_br[1]; y++) {
        f32 y_pos = fract_tl[1] + (y - pixel_tl[1]) * y_scale;

        for (u32 x = pixel_tl[0]; x < pixel_br[0]; x++) {
            f32 x_pos = fract_tl[0] + (x - pixel_tl[0]) * x_scale;

            complex_f32 c = x_pos + y_pos * I;
            var z = 0.f + 0.fi;
            var z2 = z * z;

            u32 iteration = 0;
            while (Re(z2) + Im(z2) < ESCAPE_RADIUS_2 && iteration < max_iterations) {
                z = z * z + c;
                z2 = z * z;
                iteration++;
            }

            iterations[y * dimens[0] + x] = iteration;
        }
    }
}

void mandelbrot_compute_vector(u32 *iterations,
                               u32 max_iterations,
                               u32x2 dimens,
                               u32x2 pixel_tl,
                               u32x2 pixel_br,
                               f32x2 fract_tl,
                               f32x2 fract_br) {
    f32x4 _er2 = _mm_set1_ps(ESCAPE_RADIUS_2);
    u32x4 _one = _mm_set1_epi32(1);
    f32x4 _two = _mm_set1_ps(2);

    u32x4 _max_iterations = _mm_set1_epi32(max_iterations);

    const var x_scale = (fract_br[0] - fract_tl[0]) / ((f32)pixel_br[0] - (f32)pixel_tl[0]);
    const var y_scale = (fract_br[1] - fract_tl[1]) / ((f32)pixel_br[1] - (f32)pixel_tl[1]);

    const var width            = pixel_br[0] - pixel_tl[0];
    const var complete_blocks  = width / 4;
    const var remaining_pixels = width % 4;

    f32x4 _x_pos_offsets = _mm_set_ps(3, 2, 1, 0);
    f32x4 _x_scale       = _mm_set1_ps(x_scale);
    f32x4 _fract_tl_x    = _mm_set1_ps(fract_tl[0]);

    for (u32 y = pixel_tl[1]; y < pixel_br[1]; y++) {
        const var y_pos = fract_tl[1] + (y - pixel_tl[1]) * y_scale;

        for (u32 block = 0; block < complete_blocks; block++) {
            const var x_offset = block * 4;

            f32x4 _x_pos = _mm_set1_ps(x_offset);
                  _x_pos = _mm_add_ps(_x_pos, _x_pos_offsets);
                  _x_pos = _mm_mul_ps(_x_pos, _x_scale);
                  _x_pos = _mm_add_ps(_x_pos, _fract_tl_x);

            f32x4 _cr = _x_pos;
            f32x4 _ci = _mm_set1_ps(y_pos);

            f32x4 _zr = _mm_setzero_ps();
            f32x4 _zi = _mm_setzero_ps();

            u32x4 _iterations = _mm_setzero_si128();

            loop: {

                f32x4 _zr2 = _mm_mul_ps(_zr, _zr);
                f32x4 _zi2 = _mm_mul_ps(_zi, _zi);

                f32x4 _mag = _mm_add_ps(_zr2, _zi2);

                f32x4 _a = _mm_sub_ps(_zr2, _zi2);
                      _a = _mm_add_ps(_a, _cr);

                f32x4 _b = _mm_mul_ps(_zr, _zi);
                      _b = _mm_mul_ps(_b, _two);
                      _b = _mm_add_ps(_b, _ci);

                _zr = _a;
                _zi = _b;

                f32x4 _mask1 = _mm_cmplt_ps(_mag, _er2);
                u32x4 _mask2 = _mm_cmplt_epi32(_iterations, _max_iterations);
                      _mask2 = _mm_and_si128(_mm_castps_si128(_mask1), _mask2);

                _iterations = _mm_add_epi32(_iterations, _mm_and_si128(_one, _mask2));

                if (_mm_movemask_ps(_mm_castsi128_ps(_mask2)) > 0) goto loop;
            }

            _mm_storeu_si128((__m128i*)(iterations + y * dimens[0] + pixel_tl[0] + x_offset), _iterations);
        }

        if (remaining_pixels > 0) {
            const var x_start = pixel_tl[0] + complete_blocks * 4;

            for (u32 x = x_start; x < pixel_br[0]; x++) {
                var x_pos = fract_tl[0] + (x - pixel_tl[0]) * x_scale;

                complex_f32 c = x_pos + y_pos * I;
                var z = 0.f + 0.fi;
                var z2 = z * z;
                u32 iteration = 0;

                while (Re(z2) + Im(z2) < ESCAPE_RADIUS_2 && iteration < max_iterations) {
                    z = z * z + c;
                    z2 = z * z;
                    iteration++;
                }

                iterations[y * dimens[0] + x] = iteration;
            }
        }
    }
}

struct compute_worker_props {
    enum processing_mode processing_mode;
    u32 *iterations;
    u32 max_iterations;
    u32x2 dimens;
    u32x2 pixel_tl;
    u32x2 pixel_br;
    f32x2 fract_tl;
    f32x2 fract_br;
};

i32 mandelbrot_compute_worker(void *arg) {
    const var props = (struct compute_worker_props*)arg;

    switch (props->processing_mode) {
        case PROCESSING_MODE_SISD: {
            mandelbrot_compute_scalar(props->iterations, props->max_iterations, props->dimens,
                                      props->pixel_tl, props->pixel_br,
                                      props->fract_tl, props->fract_br);
            break;
        }
        case PROCESSING_MODE_SIMD: {
            mandelbrot_compute_vector(props->iterations, props->max_iterations, props->dimens,
                                      props->pixel_tl, props->pixel_br,
                                      props->fract_tl, props->fract_br);
            break;
        }
    }

    return EXIT_SUCCESS;
}

void init_mandelbrot_compute_workers(thrd_t *threads,
                                     struct compute_worker_props *thread_props,
                                     u32 n_threads,
                                     enum processing_mode processing_mode,
                                     u32 *iterations,
                                     u32 max_iterations,
                                     u32x2 dimens,
                                     f32x2 center,
                                     f32 zoom) {
    const var x_range = X_SCALE_FACTOR / zoom;
    const var y_range = x_range * dimens[1] / dimens[0];
    f32x2 scale = {x_range / dimens[0], y_range / dimens[1]};

    const var half_dimens = cast_vector(f32x2, dimens) / 2.0f;

    u32 batch_rows  = dimens[1] / n_threads;
    u32 excess_rows = dimens[1] % n_threads;
    for (u32 i = 0; i < n_threads; i++) {
        u32 start_row = i * batch_rows;
        u32 end_row = i == n_threads - 1 ? start_row + batch_rows + excess_rows
                                         : start_row + batch_rows;

        u32x2 pixel_tl = {0, start_row};
        u32x2 pixel_br = {dimens[0], end_row};

        const var fract_tl = center + (cast_vector(f32x2, pixel_tl) - half_dimens) * scale;
        const var fract_br = center + (cast_vector(f32x2, pixel_br) - half_dimens) * scale;

        thread_props[i] = (struct compute_worker_props) {
            .processing_mode = processing_mode,
            .iterations = iterations,
            .max_iterations = max_iterations,
            .dimens = dimens,
            .pixel_tl = pixel_tl,
            .pixel_br = pixel_br,
            .fract_tl = fract_tl,
            .fract_br = fract_br
        };
        if (thrd_create(threads + i, mandelbrot_compute_worker, thread_props + i) != thrd_success) {
            fprintf(stderr, "Failed to create worker thread\n");
            exit(EXIT_FAILURE);
        }
    }
}

void join_mandelbrot_compute_workers(thrd_t *threads, u32 n_threads) {
    for (u32 i = 0; i < n_threads; i++) {
        if (thrd_join(threads[i], NULL) != thrd_success) {
            fprintf(stderr, "Failed to join worker thread\n");
            exit(EXIT_FAILURE);
        }
    }
}

void mandelbrot_apply_coloring_grayscale(const u32 *iterations,
                                         u32 *colors,
                                         u32x2 dimens,
                                         u32 max_iterations) {
    for (u32 y = 0; y < dimens[1]; y++) {
        for (u32 x = 0; x < dimens[0]; x++) {
            const var i = y * dimens[0] + x;
            var iteration = iterations[i];

            u32 color = 0x000000FF;

            if (iteration < max_iterations) {
                const var t = iteration / (f32)max_iterations;

                const var v = (u8)(255.0f * (1.0f - t));

                color = 0x000000FF | (v << 24) | (v << 16) | (v << 8);
            }

            colors[i] = color;
        }
    }
}

void mandelbrot_apply_coloring_smooth(const u32 *iterations,
                                      u32 *colors,
                                      u32x2 dimens,
                                      u32 max_iterations) {
    for (u32 y = 0; y < dimens[1]; y++) {
        for (u32 x = 0; x < dimens[0]; x++) {
            const var i = y * dimens[0] + x;
            var iteration = iterations[i];

            u32 color = 0x000000FF;

            if (iteration < max_iterations) {
                const var t = iteration / (f32)max_iterations;

                const var r = (u8)(9.0f * (1.0f - t) * t * t * t * 255.0f);
                const var g = (u8)(15.0f * (1.0f - t) * (1.0f - t) * t * t * 255.0f);
                const var b = (u8)(8.5f * (1.0f - t) * (1.0f - t) * (1.0f - t) * t * 255.0f);

                color = 0x000000FF | (r << 24) | (g << 16) | (b << 8);
            }

            colors[i] = color;
        }
    }
}

void mandelbrot_apply_coloring_hsv(const u32 *iterations,
                                   u32 *colors,
                                   u32x2 dimens,
                                   u32 max_iterations) {
    for (u32 y = 0; y < dimens[1]; y++) {
        for (u32 x = 0; x < dimens[0]; x++) {
            const var i = y * dimens[0] + x;
            var iteration = iterations[i];

            u32 color = 0x000000FF;

            if (iteration < max_iterations) {
                const var hue = 360.0f * (iteration / (f32)max_iterations);
                const var sat = 1.0f;
                const var value = 1.0f;

                i32 hi = (i32)(hue / 60.0f) % 6;
                f32 f = hue / 60.0f - hi;
                f32 p = value * (1.0f - sat);
                f32 q = value * (1.0f - f * sat);
                f32 t = value * (1.0f - (1.0f - f) * sat);

                u8 r, g, b;
                switch (hi) {
                    case 0:  r = value * 255; g = t * 255;     b = p * 255;     break;
                    case 1:  r = q * 255;     g = value * 255; b = p * 255;     break;
                    case 2:  r = p * 255;     g = value * 255; b = t * 255;     break;
                    case 3:  r = p * 255;     g = q * 255;     b = value * 255; break;
                    case 4:  r = t * 255;     g = p * 255;     b = value * 255; break;
                    default: r = value * 255; g = p * 255;     b = q * 255;     break;
                }

                color = 0x000000FF | (r << 24) | (g << 16) | (b << 8);
            }

            colors[i] = color;
        }
    }
}

void mandelbrot_apply_coloring(enum color_mode color_mode,
                               const u32 *iterations,
                               u32 *colors,
                               u32x2 dimens,
                               u32 max_iterations) {
    switch (color_mode) {
        case COLOR_MODE_GRAYSCALE: mandelbrot_apply_coloring_grayscale(iterations, colors, dimens, max_iterations); break;
        case COLOR_MODE_SMOOTH:    mandelbrot_apply_coloring_smooth(iterations, colors, dimens, max_iterations);    break;
        case COLOR_MODE_HSV:       mandelbrot_apply_coloring_hsv(iterations, colors, dimens, max_iterations);       break;
    }
}

void fwrite_rgba_to_ppm(cstr output_filename, const u32 *rgba_colors, u32x2 dimens) {
    var file defer(defer_fclose) = fopen(output_filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing: %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    fprintf(file, "P6\n%d %d\n255\n", dimens[0], dimens[1]);

    const var n_points = dimens_length(dimens);

    const var rgb_size = n_points * 3 * sizeof(u8);
    u8 *rgb_colors defer(defer_free) = malloc(rgb_size);
    if (rgb_colors == NULL) {
        fprintf(stderr, "Failed to allocate %zu bytes\n", rgb_size);
        exit(EXIT_FAILURE);
    }


    for (u32 i = 0; i < n_points; i++) {
        const var rgba_value = rgba_colors[i];
        rgb_colors[3 * i + 0] = (rgba_value >> 24);
        rgb_colors[3 * i + 1] = (rgba_value >> 16);
        rgb_colors[3 * i + 2] = (rgba_value >> 8);
    }

    fwrite(rgb_colors, n_points, 3 * sizeof(u8), file);
}

i32 main(i32 argc, cstr *argv) {
    struct program_args args;
    parse_program_args(argv, argc, &args);

    const var dimens          = args.dimens;
    const var max_iterations  = args.max_iterations;
    const var center          = args.center;
    const var zoom            = args.zoom;
    const var color_mode      = args.color_mode;
    const var output_filename = args.output_filename;
    const var processing_mode = args.processing_mode;

    const var n_points = dimens_length(dimens);

    const var iterations_size = n_points * sizeof(u32);
    u32 *iterations defer(defer_free) = aligned_alloc(MM_ALIGN, iterations_size);
    if (iterations == NULL) {
        fprintf(stderr, "Failed to allocate %zu bytes\n", iterations_size);
        return EXIT_FAILURE;
    }
    fill(iterations, n_points, 0);

    array(thrd_t, N_THREADS) threads;
    array(struct compute_worker_props, N_THREADS) thread_props;
    init_mandelbrot_compute_workers(threads, thread_props, N_THREADS,
                                    processing_mode,
                                    iterations, max_iterations,
                                    dimens, center, zoom);
    join_mandelbrot_compute_workers(threads, N_THREADS);

    const var colors_size = n_points * sizeof(u32);
    u32 *colors defer(defer_free) = malloc(colors_size);
    if (colors == NULL) {
        fprintf(stderr, "Failed to allocate %zu bytes\n", colors_size);
        return EXIT_FAILURE;
    }

    mandelbrot_apply_coloring(color_mode, iterations, colors, dimens, max_iterations);

    fwrite_rgba_to_ppm(output_filename, colors, dimens);

    return EXIT_SUCCESS;
}
