# ASYNCIO -- support for formatted write in CUDA Fortran

This package makes it possible to use an equivalent of `write` command from within CUDA Fortran kernels, with format and/or unit specified. By some reason, PGI never adds support for formatted writes (only `write(*,*)` is possible), greatly limiting GPU portability, as they are actively used in many legacy Fortran codes.

## Overview

Since `write` command could not be overridden for standard types, the best bet is to create a runtime library with a closely resembling interface. Thus, we've created the following API:

```
call asyncio_begin('*','*')
call asyncio_write(arr(length / 8))
call asyncio_end()
```

In the first example write outputs single array element to stdout.

```
integer :: funcptr
attributes(device) :: funcptr
bind(C,name="__FUNC_kernel2") :: funcptr

character(5), bind(C,name="__GPUFMT_kernel2_123") :: _123 = '(I8)'C

call asyncio_begin(0, funcptr, 123)
call asyncio_write(i8)
call asyncio_end()
```

In the second example write is performed to *unit* `0` (stderr), using format string identified by the name of *caller* `kernel2` and *label* `123`. In addition to two versions above, `asyncio_begin` supports all possible combinations of default and non-default formats and unit numbers.

Function handle `funcptr` has to be defined in a module visible to the caller on GPU, while format string `_123` is a host variable. Function handle and format string are getting linked at the time when writing is actually being performed with `async_flush()` command:

```
call asyncio_flush()
```

Similarly to CUDA builtin `printf`, out API does not output anything from the running GPU kernel while it is executing (although, it's potentially possible to automatically flush the buffer to host and print it with higher granularity). Unlike builtin `printf`, flush is not being performed automatically. Instead, user has to call `asyncio_flush` explicitly.

Finally, ASYNCIO implements both GPU and host writes -- the same code could be executed on GPU and host.

## Methodology

The GPU part of ASYNCIO implementation is straight-forward: on device side data is getting serialized into large linear buffer. Host part is trickier, because we need to deserialize buffer contents according to the specified format. Instead of doing this explicitly, which would undoubtedly be a very large work, ASYNCIO simply deploys libgfortran's internal I/O runtime functions on the buffered data.

For compatibility with its own writes, ASYNCIO also implements reads through libgfortran. The user has to refactor reads similarly to writes, if the reader program is not compiled by gfortran. Currently reads are supported on host only, however it seems to be possible to implement unformatted reads on GPU by reading the transaction buffer directly.

## Package contents

This package contains 3 source files with the core functionality:

* `asyncio.CUF` -- Fortran / CUDA Fortran interfaces
* `asyncio.cu` -- host / CUDA implementations of device-side buffer writes and host-side buffer reads
* `hooks.f90` -- hooks for libgfortran runtime functions, to make our implementation portable across multiple GCC compiler versions

Other files are user code examples:

* `kernel1.CUF` -- first Fortran / CUDA Fortran example kernel performing unformatted writes
* `kernel2.CUF` -- second Fortran / CUDA Fortran example kernel performing formatted writes
* `main_gpu.CUF` -- main entry calling 2 GPU kernels above
* `main_cpu.f90` -- main entry calling 2 kernels above as CPU functions

Note CUDA Fortran sources have to be compiled in a specific way (see `makefile`).

