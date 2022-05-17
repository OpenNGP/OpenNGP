# OpenNGP
With explosion of NeRF-based method and its promising performance to capture/render reality, we believe NGP (neural graphics primitive) has potential to become next generation data type in CG world. This project aims to make NGP first class citizen as with polygon mesh in current DCC (Digital Content Creation) toochain.

Implemented Features:
1. Configurable render pipeline and NGP arch (Primitive as base class)
2. NeRF (comparable result to original baseline)
3. Instant NGP (limited features with HashEncoder|SHEncoder|cuda ray marching)
4. Per-data optimization for exposure|whilte balance|pose (not fully validated yet)

TODO:
1. dev in cpp for better integratability to current DCC toolchain?

Class Design
![OpenNGP_class](https://user-images.githubusercontent.com/7394919/167256775-8b7a004f-7f1b-4239-b611-40b1ce97491b.jpg)

Data Flow
![OpenNGP_dataflow](https://user-images.githubusercontent.com/7394919/165758547-b5b39fec-7045-44dc-9cb8-2f574dd442d5.jpg)

[wiki](https://github.com/openNGP/openNGP/wiki)

Acknowledgements:\
[nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)|[torch-ngp](https://github.com/ashawkey/torch-ngp)
