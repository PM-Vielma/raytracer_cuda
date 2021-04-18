/*Portion used from CUDA's sample files*/
#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <vector_types.h>
typedef unsigned int  uint;
typedef unsigned char uchar;

#pragma pack(push,4)
struct Image
{
	void* h_data;
	cudaExtent              size;
	cudaResourceType        type;
	cudaArray_t             dataArray;
	cudaMipmappedArray_t    mipmapArray; //mipmaps were not utilized in my raytracer. I'm sure there are plenty of redundancies like this.
	cudaTextureObject_t     textureObject;

	Image()
	{
		memset(this, 0, sizeof(Image));
	}
};
#pragma pack(pop)
#endif

/*Portion I wrote*/
class Vec3 {
	float x, y, z;
public:
	__host__ __device__ Vec3() {
		x = 0;
		y = 0;
		z = 0;
	}
	__host__ __device__ Vec3(float i, float j, float k) {
		x = i;
		y = j;
		z = k;
	}
	__host__ __device__ void set_values(float i, float j, float k) {
		x = i;
		y = j;
		z = k;
	}
	__host__ __device__ float getX() {
		return x;
	}
	__host__ __device__ float getY() {
		return y;
	}
	__host__ __device__ float getZ() {
		return z;
	}
	__host__ __device__ float r() {
		return x;
	}
	__host__ __device__ float g() {
		return y;
	}
	__host__ __device__ float b() {
		return z;
	}
	__host__ __device__ float r(float red) {
		x = red;
		return x;
	}
	__host__ __device__ float g(float green) {
		y = green;
		return y;
	}
	__host__ __device__ float b(float blue) {
		z = blue;
		return z;
	}
	__host__ __device__ Vec3 mult(float f) {
		return Vec3(f * x, f * y, f * z);
	}
};

__host__ __device__ inline Vec3 add(Vec3& a, Vec3& b) {
	float x = a.getX() + b.getX();
	float y = a.getY() + b.getY();
	float z = a.getZ() + b.getZ();
	return Vec3(x, y, z);
}

__host__ __device__ inline Vec3 subtract(Vec3& a, Vec3& b) {
	float x = a.getX() - b.getX();
	float y = a.getY() - b.getY();
	float z = a.getZ() - b.getZ();
	return Vec3(x, y, z);
}

__host__ __device__ inline float dot(Vec3& a, Vec3& b) {
	float x = a.getX() * b.getX();
	float y = a.getY() * b.getY();
	float z = a.getZ() * b.getZ();
	return (x + y + z);
}
__host__ __device__ inline Vec3 cross(Vec3& a, Vec3& b) {
	float x = a.getY() * b.getZ() - a.getZ() * b.getY();
	float y = a.getZ() * b.getX() - a.getX() * b.getZ();
	float z = a.getX() * b.getY() - a.getY() * b.getX();
	return Vec3(x, y, z);
}

__host__ __device__ inline Vec3 normalize(Vec3& a) {
	float x = a.getX();
	float y = a.getY();
	float z = a.getZ();
	float s = sqrt(x * x + y * y + z * z);
	return Vec3(x / s, y / s, z / s);
}
__host__ __device__ inline float crop(float min, float max, float x) {
	x = (x - min) / (max - min);
	if (x > 1)
		x = 1;
	if (x < 0)
		x = 0;
	return x * x * (3 - 2 * x);
}

__host__ __device__ inline Vec3 rotateX(Vec3& a, float angle) {
	float x = a.getX();
	float y = a.getY() * cos(angle) - a.getZ() * sin(angle);
	float z = a.getY() * sin(angle) + a.getZ() * cos(angle);

	return Vec3(x, y, z);
}
__host__ __device__ inline Vec3 rotateY(Vec3& a, float angle) {
	float x = a.getX() * cos(angle) + a.getZ() * sin(angle);
	float y = a.getY();
	float z = -a.getX() * sin(angle) + a.getZ() * cos(angle);

	return Vec3(x, y, z);
}

class SceneParams {
public:
	Vec3 Pe;
	Vec3 n0;
	Vec3 n1;
	Vec3 n2;
	Vec3 PL;
	float sx;
	float sy;
	Vec3 Pc;
	Vec3 P00;
	Vec3 PiY;
	//Vec3 Plane_v;
	Vec3 npiY;
	Vec3 PiX;
	//Vec3 PlaneX_v;
	Vec3 npiX;
	Vec3 Pcirc;
	float r;
	float Light_m;
	float Light_n;

	__host__ __device__ SceneParams(int g_camX, int g_camY, int g_camZ, float g_rotY, float g_rotX, float t) {
		float d = 50;
		Pe = Vec3((float)g_camX, (float)g_camY, (float)g_camZ - d); //cam\eye position
		Vec3 Vview(0, 0, 1); //forward vector 
		Vec3 Vup(0, 1, 0);
		Vview = rotateY(Vview, g_rotY);
		Vup = rotateY(Vup, g_rotY);
		Vview = rotateX(Vview, g_rotX);
		Vup = rotateX(Vup, g_rotX);
		n0 = normalize(cross(Vview, Vup));
		n1 = normalize(cross(n0, Vview));
		n2 = normalize(Vview);
		sx = 100;
		sy = 100;
		Pc = add(Pe, n2.mult(d));
		P00 = subtract(subtract(Pc, n0.mult(sx / 2.0f)), n1.mult(sy / 2.0f));

		//Planes
		//Planes
		PiY = Vec3(-200, -200, 300);
		Vec3 Plane_v = Vec3(0, 2, -.2);
		npiY = normalize(Plane_v);
		PiX = Vec3(-200, -200, 300);
		Vec3 PlaneX_v = Vec3(2, 0, -.2);
		npiX = normalize(PlaneX_v);

		//Sphere
		Pcirc = Vec3(50, -150, 150);
		r = 100;

		//Area Light Starting Point and size

		PL = Vec3(cos(t / 5) * 200 + 150, sin(t / 5) * 200 + 150, -200);
		Light_m = 30;
		Light_n = 30;
	}
	__device__ SceneParams(float t) {
		float d = 50;

		//Planes
		PiY = Vec3(-200, -200, 300);
		Vec3 Plane_v = Vec3(0, 2, -.2);
		npiY = normalize(Plane_v);
		PiX = Vec3(-200, -200, 300);
		Vec3 PlaneX_v = Vec3(2, 0, -.2);
		npiX = normalize(PlaneX_v);

		//Sphere
		Pcirc = Vec3(50, -150, 150);
		r = 100;

		//Area Light Starting Point and size

		PL = Vec3(cos(t / 5) * 200 + 150, sin(t / 5) * 200 + 150, -200);
	}
};