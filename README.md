# Mandelbrot Set Renderer

A high-performance Mandelbrot set renderer written in C that supports both scalar and SIMD vector processing.

## Features

- Generate Mandelbrot set images with customizable dimensions, iteration limits, and zoom levels
- Multiple coloring modes (grayscale, smooth, HSV)
- SIMD-accelerated computation with Intel SSE intrinsics
- Multithreaded rendering for faster processing
- Predefined interesting locations in the Mandelbrot set
- Output to PPM image format

## Usage

```
./mandelbrot [options]
```

### Options

- `--width N`: Set image width (default: 800)
- `--height N`: Set image height (default: 600)
- `--iterations N`: Set maximum iterations (default: 1000)
- `--center X Y`: Set center coordinates (default: -0.5, 0.0)
- `--zoom N`: Set zoom level (default: 1.0)
- `--color MODE`: Set coloring mode: grayscale, smooth, hsv (default: smooth)
- `--output FILE`: Set output filename (default: mandelbrot.ppm)
- `--scalar`: Use scalar computation
- `--vector`: Use vectorized computation (default)
- `--locations`: Show predefined interesting locations
- `--help`: Show help message

## Examples

Generate a default view of the Mandelbrot set:
```
./mandelbrot
```

Explore the Seahorse Valley with HSV coloring:
```
./mandelbrot --center -0.745 0.1 --zoom 80 --color hsv
```

Generate a high-resolution image using scalar computation:
```
./mandelbrot --width 1920 --height 1080 --iterations 5000 --scalar
```

## Interesting Locations

The program includes several predefined interesting locations that can be viewed with the `--locations` flag:

1. Main Mandelbrot view: `--center -0.5 0 --zoom 1`
2. Seahorse Valley: `--center -0.745 0.1 --zoom 80`
3. Elephant Valley: `--center 0.3 -0.01 --zoom 30`
4. Triple Spiral: `--center -0.041 0.682 --zoom 200`
5. Quad Spiral: `--center -1.25 0.02 --zoom 100`
6. Mini Mandelbrot: `--center -1.77 0 --zoom 50`
7. Feigenbaum Point: `--center -1.401155 0 --zoom 10000`

## Building

The project uses modern C features and requires a C compiler with support for:
- C11 standard
- `_Generic` expressions
- Complex number support
- SIMD intrinsics (SSE/SSE2)
- Thread support (C11 threads)

> Project was successfully compiled with gcc 14.2.1 and clang 18.1.8 on Fedora Linux.

### Using Make

The project includes a makefile for easy building:

```
make
```

To clean the project (remove object files, executables, and PPM images):

```
make clean
```

## Technical Details

- Uses multithreading to distribute the rendering workload
- Implements both scalar and vectorized SIMD computation methods
- Optimized memory management with aligned allocations for SIMD operations
- Custom defer mechanism for resource cleanup
- Implements multiple coloring algorithms for different visual effects
