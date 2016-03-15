#include <vector>
#include <vector_types.h>
#include <vector_functions.h>

struct TriangleFace {
    int v[3];
};

struct TriangleMesh {
    std::vector<float3> verts;
    std::vector<TriangleFace> faces;
    float3 bounding_box[2];
};
