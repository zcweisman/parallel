#include <vector_types.h>
#include <vector_functions.h>
#include <math_functions.h>

#include "cutil_math.h"
#include "raytracer.cuh"

texture<float4, 1, cudaReadModeElementType> triangle_texture;

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b) {
	r = clamp(r, 0.0f, 255.0f);
	g = clamp(g, 0.0f, 255.0f);
	b = clamp(b, 0.0f, 255.0f);
	return (int(r)<<16) | (int(g)<<8) | int(b); // notice switch red and blue to counter the GL_BGRA
}

// Intersection - Done
__device__ int RayBoxIntersection(const float3 &BBMin, const float3 &BBMax, const float3 &RayOrg, const float3 &RayDirInv, float &tmin, float &tmax) {
	float l1 = (BBMin.x - RayOrg.x) * RayDirInv.x;
	float l2 = (BBMax.x - RayOrg.x) * RayDirInv.x;
	tmin = fminf(l1, l2);
	tmax = fmaxf(l1, l2);

	l1 = (BBMin.y - RayOrg.y) * RayDirInv.y;
	l2 = (BBMax.y - RayOrg.y) * RayDirInv.y;
	tmin = fmaxf(fminf(l1, l2), tmin);
	tmax = fminf(fmaxf(l1, l2), tmax);

	l1 = (BBMin.z - RayOrg.z) * RayDirInv.z;
	l2 = (BBMax.z - RayOrg.z) * RayDirInv.z;
	tmin = fmaxf(fminf(l1, l2), tmin);
	tmax = fminf(fmaxf(l1, l2), tmax);

	return ((tmax >= tmin) && (tmax >= 0.0f));
}

__device__ float RayTriangleIntersection(const Ray &r, const float3 &v0, const float3 &edge1, const float3 &edge2) {
	float3 tvec = r.ori - v0;
	float3 pvec = cross(r.dir, edge2);
	float det = dot(edge1, pvec);

	det = 1.0f / det;

	float u = dot(tvec, pvec) * det;

	if (u < 0.0f || u > 1.0f) return -1.0f;

	float3 qvec = cross(tvec, edge1);
	float v = dot(r.dir, qvec) * det;

	if (v < 0.0f || (u + v) > 1.0f) return -1.0f;

	return dot(edge2, qvec) * det;
}

__device__ int RaySphereIntersection(const Ray &ray, const float3 sphere_center, const float sphere_radius, float &t) {
	float b, c, d;

	float3 sr = ray.ori - sphere_center;
	b = dot(sr, ray.dir);
	c = dot(sr, sr) - (sphere_radius*sphere_radius);
	d = b*b - c;

	if (d > 0) {
		float e = sqrt(d);
		float t0 = -b-e;
		if (t0 < 0) t = -b+e;
		else t = min(-b-e, -b+e);
		return 1;
	}
	return 0;
}

__global__ void raytrace(unsigned int* out_data,
							const int w,
							const int h,
							const int num_triangles,
							const float3 a, const float3 b, const float3 c,
							const float3 cam_pos,
							const float3 light_pos,
							const float3 light_color,
							const float3 scene_aabb_min,
							const float3 scene_aabb_max) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float xf = (x-0.5)/((float)w);
	float yf = (y-0.5)/((float)h);

	int ray_depth = 0;
	bool continue_path = true;

	float3 t1 = c+(a*xf);
	float3 t2 = b*yf;
	float3 image_pos = t1 + t2;
	Ray r(image_pos,image_pos-cam_pos);
	HitRecord hit_r;

	float t_min,t_max;
	continue_path = RayBoxIntersection(scene_aabb_min, scene_aabb_max, r.ori, r.inv_dir,t_min, t_max);
	hit_r.color = make_float3(0,0,0);

	// hack to display the light source we simple make a ray sphere intersection and
	// compare the depth with the found t value from the triangles
	float sphere_t;
	bool sphere_hit = RaySphereIntersection(r,light_pos,2.0,sphere_t);

	if(sphere_hit && sphere_t > 0.001) {
		if(!continue_path) {
			hit_r.color = light_color;
		}
		sphere_hit = true;
	}

	while(continue_path && ray_depth < 4) {
		// search through the triangles and find the nearest hit point
		for(int i = 0; i < num_triangles; i++) {
			float4 v0 = tex1Dfetch(triangle_texture,i*3);
			float4 e1 = tex1Dfetch(triangle_texture,i*3+1);
			float4 e2 = tex1Dfetch(triangle_texture,i*3+2);

			float t = RayTriangleIntersection(r, make_float3(v0.x,v0.y,v0.z),make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));

			if(t < hit_r.t && t > 0.001) {
				hit_r.t = t;
				hit_r.hit_index = i;
			}
		}

		if(sphere_hit && sphere_t < hit_r.t) {
			hit_r.color += light_color;
			continue_path = false;
			break;
		}

		if(hit_r.hit_index >= 0) {
			ray_depth++;

			// create the normal
			float4 e1 = tex1Dfetch(triangle_texture,hit_r.hit_index*3+1);
			float4 e2 = tex1Dfetch(triangle_texture,hit_r.hit_index*3+2);

			hit_r.normal = cross(make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));
			hit_r.normal = normalize(hit_r.normal);

			// calculate simple diffuse light
			float3 hitpoint = r.ori + r.dir *hit_r.t;
			float3 L = light_pos - hitpoint;
			float dist_to_light = length(L);

			L = normalize(L);
			float diffuse_light = max( dot(L,hit_r.normal), 0.0);
			diffuse_light = min( (diffuse_light),1.0);
			//calculate simple specular light
			float3 H = L + (-r.dir);
			H = normalize(H);
			float specular_light = powf(max(dot(H,hit_r.normal),0.0),25.0f);

			diffuse_light  *=  16.0/dist_to_light;
			specular_light *=  16.0/dist_to_light;

			clamp(diffuse_light, 0.0f, 1.0f);
			clamp(specular_light, 0.0f, 1.0f);

			hit_r.color += light_color * diffuse_light + make_float3(1.0,1.0,1.0)*specular_light*0.2 + make_float3(0.2,0.2,0.2);

			// create a shadow ray
			Ray shadow_ray(hitpoint, L);
			for(int i = 0; i < num_triangles; i++) {
				float4 v0 = tex1Dfetch(triangle_texture,i*3);
				float4 e1 = tex1Dfetch(triangle_texture,i*3+1);
				float4 e2 = tex1Dfetch(triangle_texture,i*3+2);
				float t = RayTriangleIntersection(shadow_ray, make_float3(v0.x,v0.y,v0.z),make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));

				if(t > 0.025) // there is a blocker on the path to the light
				{
					hit_r.color *= 0.25;
					break;
				}
			}

			if(e1.w > 0) { // this is also a little hack to include a specular material
				hit_r.resetT();
				r = Ray(hitpoint, reflect(r.dir,hit_r.normal));
			} else {
				continue_path = false;
			}
		}
		else {
			continue_path = false;
			hit_r.color += make_float3(0.5,0.5,0.95*yf+0.3);
		}
	}

	if(ray_depth >= 1 || sphere_hit) {
		ray_depth = max(ray_depth,1);
		hit_r.color /= ray_depth; // normalize the colors
	}
	else {
		hit_r.color = make_float3(0.5,0.5,yf+0.3);
	}

	int val = rgbToInt(hit_r.color.x*255,hit_r.color.y*255,hit_r.color.z*255);
	out_data[y * w + x] = val;
}

extern "C" {
	void RayTraceImage(unsigned int *pbo_out, int w, int h, int num_triangles,
		               float3 a, float3 b, float3 c,
		               float3 cam_pos,
					   float3 light_pos,
					   float3 light_color,
					   float3 scene_aabbox_min, float3 scene_aabbox_max)
	{

		dim3 block(8,8,1);
		dim3 grid(w/block.x,h/block.y, 1);
		raytrace<<<grid, block>>>(pbo_out,w,h,num_triangles,a,b,c,cam_pos,light_pos,light_color,scene_aabbox_min,scene_aabbox_max);

        return;
	}

	void bindTriangles(float *dev_triangle_p, unsigned int num_triangles)	{
		triangle_texture.normalized = false;                      // access with normalized texture coordinates
		triangle_texture.filterMode = cudaFilterModePoint;        // Point mode, so no
		triangle_texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4)*num_triangles*3;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0,triangle_texture,dev_triangle_p,channelDesc,size);

        return;
	}
}
