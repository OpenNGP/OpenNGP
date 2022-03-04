# openNGP
With explosion of NeRF-based method and its promising performance to capture/render reality, we believe NGP (neural graphics primitive) has potential to become next generation data type in CG world. This project aims to make NGP first class citizen as with polygon mesh in current DCC (Digital Content Creation) toochain.

TODO:
0. dev in cpp for better integratability to current DCC toolchain?
1. concept model ([wiki](https://github.com/openNGP/openNGP/wiki))
- NGP: a continuous volume to represent a 3D object (including geometry and appearance), it can be seemless integrated into volumetric rendering (or ray tracing?), yet problem arises when working with conventional forward rendering pipeline.
  - Encoder: map query pt to feature vector
  - Regressor: map feature vector to certain field property (density/SDF/color/etc)
