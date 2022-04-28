# OpenNGP
With explosion of NeRF-based method and its promising performance to capture/render reality, we believe NGP (neural graphics primitive) has potential to become next generation data type in CG world. This project aims to make NGP first class citizen as with polygon mesh in current DCC (Digital Content Creation) toochain.

Implemented Features:
1. Configurable render pipeline and NGP arch (Primitive as base class)
2. NeRF (comparable result to original baseline)
3. Instant NGP (limited features with HashEncoder|SHEncoder|cuda ray marching)
4. Per-data optimization for exposure|whilte balance|pose (not fully validated yet)

TODO:
1. dev in cpp for better integratability to current DCC toolchain?
2. concept model ([wiki](https://github.com/openNGP/openNGP/wiki))

Acknowledgements:\
[nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)|[torch-ngp](https://github.com/ashawkey/torch-ngp)
