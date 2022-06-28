#ifndef DELOG_H
#define DELOG_H

//#define DELOG_DISABLE_ALL
//#define DELOG_DISABLE_LOG
//#define DELOG_DISABLE_PAUSE
//#define DELOG_DISABLE_TIMER
//#define DELOG_DISABLE_TYPE_LOG

//#define DELOG_ENABLE_EIGEN
//#define DELOG_ENABLE_OPENCV

#include "delog/delog.hpp"
#include "delog/basics.hpp"

#ifdef DELOG_ENABLE_EIGEN
#include "delog/eigen.hpp"
#endif

#ifdef DELOG_ENABLE_OPENCV
#include "delog/opencv.hpp"
#endif

#include "delog/stl.hpp"

#endif // DELOG_H