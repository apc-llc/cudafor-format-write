# ASYNCIO -- support for formatted write in CUDA Fortran

This package makes it possible to use an equivalent of `write` command from within CUDA Fortran kernels, with format and/or unit specified. By some reason, PGI never adds support for formatted writes (only `write(*,*)` is possible), however they are actively used in many legacy Fortran codes, therefore limiting porting onto GPUs.

## Methodology

Since `write` command could not be overriden for standard types, the best bet is to create a runtime library with closely resembling interface. Thus, we've created the following API:

```

```
