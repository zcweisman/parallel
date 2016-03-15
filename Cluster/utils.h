#ifndef __UTILS_H__
#define __UTILS_H__

#include "types.h"
#include <string>

extern "C" void bindTriangles(float* dev_triangle_p, unsigned int num_triangles);

extern "C" void RayTraceImage(unsigned int*, int, int, int, float3, float3, float3,
                                float3, float3, float3, float3, float3);

void loadObj(const std::string filename, TriangleMesh &mesh, int scale);

#endif
