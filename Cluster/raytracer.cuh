struct Ray {
    __device__ Ray() {}
    __device__ Ray(const float3 &o, const float3 &d) {
        ori = o;
        dir = normalize(d);
        inv_dir = make_float3(1.0/dir.x, 1.0/dir.y, 1.0/dir.z);
    } //cudaReadMode

    float3 ori;
    float3 dir;
    float3 inv_dir;
};

struct HitRecord {
    __device__ HitRecord() {t = UINT_MAX; hit_index = -1; color.x = 0; color.y = 0; color.z = 0;}
    __device__ void resetT() {t = UINT_MAX; hit_index = -1;}

    float t;
    float3 color;
    float3 normal;
    int hit_index;
};
