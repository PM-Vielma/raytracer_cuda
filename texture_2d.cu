/*
 *Repurposed NVIDIA's CUDA Samples demo code.
 *NVIDIA's code originally painted a 2D texture with a moving red/green hatch pattern on a strobing blue background.
 *I repurposed it to code a ray tracer.
 *The only parts here I did not write include: 
 *		the method to populate threads for each pixel.
 *		lines after 413, which are Frankenstein'ed and edited portions of other CUDA sample code in order to handle more inputs and to read texture input data. It also communicates with simpleD3D9Texture.cpp, which I did not write (though I tweaked a bit to work for my purposes).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Vec3.h>
#include <vector>

#define PI 3.1415926536f
Image               atlasImage;
std::vector<Image>  contentImages;
__device__ Vec3 raytracer_recursive(Vec3&, Vec3&, float, cudaTextureObject_t, cudaTextureObject_t, Vec3);

__device__ float fract(float x) {
	return x - floorf(x);
}

__device__ float p_rand(float x, float y) {
	return fract(sin(dot(Vec3(x, y, 0),
		Vec3(12.9898, 78.233, 0))) *
		43758.5453123);
}

__device__ float getSpecular(Vec3& npe, Vec3& nLh, Vec3& Ph, Vec3& PL, Vec3& nH) {
	Vec3 PH_PL = subtract(Ph, PL);
	Vec3 nL2 = nLh;
	Vec3 reflection = add(nL2.mult(-1), nH.mult(2.0f * dot(nL2, nH)));
	float S = -dot(npe, normalize(reflection));

	return S;
}
__device__ Vec3 getReflection(Vec3& npe, Vec3& Ph, Vec3& PL, Vec3& nH, Vec3& Color, float t, cudaTextureObject_t tex, cudaTextureObject_t tex2) {
	Vec3 transmitted;
	float ior = 1.33f;
	float alpha = -1.0f / ior;
	Vec3 V = normalize(subtract(npe, Ph));
	float C = dot(V,nH);
	float term = (C * C - 1.0f)/(ior*ior) + 1.0f;
	if (false) {
		transmitted = normalize(add(V.mult(alpha), nH.mult(C/ior - sqrt(term))));
	}
	else {
		transmitted = normalize(add(V.mult(-1), nH.mult(2 * dot(V, nH))));
	}

	return raytracer_recursive(transmitted, Ph, t, tex, tex2, Color);
	
}
__device__ bool getObjPlane(Vec3& PL, Vec3& A_Ph, Vec3& A_n, Vec3& Color, Vec3& npe , float s, float t, float p0u, float p0v, float p1u, float p1v, float p2u, float p2v, cudaTextureObject_t tex, cudaTextureObject_t tex2, float time){

	Color = getReflection(npe,A_Ph,PL,A_n,Color,time,tex,tex2);
	return true;
}
__device__ bool getPlaneIntersections_PointLight(Vec3& PiY, Vec3& npiY, Vec3& PiX, Vec3& npiX, Vec3& npe, Vec3& Pe, Vec3& Color, Vec3& PL, Vec3& Pcirc, float r, cudaTextureObject_t tex) {

	float T = 0;
	float S = 0;
	float hit_planeY = dot(npiY, subtract(PiY, Pe)) / dot(npiY, npe);
	float hit_planeX = dot(npiX, subtract(PiX, Pe)) / dot(npiX, npe);
	float hit_plane;
	Vec3* npi;


	if (hit_planeY > 0 && hit_planeX > 0) {
		if (hit_planeY < hit_planeX) {
			hit_plane = hit_planeY;
			npi = &npiY;
		}
		else {
			hit_plane = hit_planeX;
			npi = &npiX;
		}
	}
	else if (hit_planeY > 0) {
		hit_plane = hit_planeY;
		npi = &npiY;
	}
	else if (hit_planeX > 0) {
		hit_plane = hit_planeX;
		npi = &npiX;
	}
	else {
		return 0;
	}

	Vec3 Ph = add(Pe, npe.mult(hit_plane));
	Vec3 nLh = normalize(subtract(PL, Ph));
	
	Vec3 np0 = normalize(cross((Vec3(0, 1, 0)),*npi));
	Vec3 np1 = normalize(cross((np0), *npi));

	float Px = dot(np0.mult(1.0f/500.0f), Ph.mult(-1));
	float Py = dot(np1.mult(1.0f/500.0f), Ph.mult(-1));
	float Pu = Px - floorf(Px);

	float Pv = Py - floorf(Py);

	S = getSpecular(npe, nLh, Ph, PL, *npi);
	T = dot(nLh, *npi);


	T = crop(0, 1, T);
	S = crop(0.7, 1, S);

	//Sphere Cast Shadow!
	float4 tex_color = tex2D<float4>(tex, Pu, 1 - Pv);
	Vec3 Pminus = subtract(PL, Pcirc);
	float b = dot(nLh, Pminus);
	float c = dot(Pminus, Pminus) - (r * r);
	float delta = b * b - c;

	if ((b >= 0 && delta >= 0)) {
		T = 0;
		S = 0;
	}

	float ks = 1;
	Color.r( 0.2f + 0.6f * T); // red
	Color.r((1 - S * ks) * Color.r() + S * 0.9f);
	Color.g(0.2f + 0.6f * T); // green
	Color.g((1 - S * ks) * Color.g() + S * 0.9f);
	Color.b(0.2f + 0.6f * T); // blue
	Color.b((1 - S * ks) * Color.b() + S);

	Color.r(Color.r() * tex_color.x);
	Color.g(Color.g() * tex_color.y);
	Color.b(Color.b() * tex_color.z);

	return 1;
}

__device__ Vec3 raytracer_recursive(Vec3& npe, Vec3& Pe, float t, cudaTextureObject_t tex, cudaTextureObject_t tex2, Vec3 Color) {

			//Set AreaLight Current Pos!
			//Vec3 PL_mn = add(PL, Vec3(light_M*(nn / N + (p_rand((float)x * nn + nn, (float)y * nn + nn)) / N), light_N*(mm / M + (p_rand((float)x * mm + mm, (float)y * mm + mm)) / M), 0)  );

	SceneParams win(t);

	Vec3& PL = win.PL; // point light/light position
	Vec3& PiY = win.PiY; // vertical plane position
	Vec3& npiY = win.npiY; // vertical plane normalized forward direction
	Vec3& PiX = win.PiX; // horizontal plane pos
	Vec3& npiX = win.npiX; // horizontal plane forward vector
	Vec3& Pcirc = win.Pcirc; // sphere origin
	float r = win.r;

	//Vec3 Color(0, 0, 0);
	float hit_plane_A = -1;
	Vec3 A_Ph_closest;
	Vec3 A_n_closest;
	float ss;
	float tt;
	float p0u;
	float p0v;
	float p1u;
	float p1v;
	float p2u;
	float p2v;

	Vec3 Pminus = subtract(Pe, Pcirc);
	float b = dot(npe, Pminus);
	float c = dot(Pminus, Pminus) - (r * r);
	float delta = b * b - c;

	float hit_plane = dot(npiY, subtract(PiY, Pe)) / dot(npiY, npe); //vertical plane ray intersection distance
	float hit_planeX = dot(npiX, subtract(PiX, Pe)) / dot(npiX, npe); //horizontal plane ray intersection distance
	if (false)
	{

		float hit_t = -b - sqrt(delta);
		if ((hit_t <= hit_plane || hit_plane < 0) && (hit_t <= hit_planeX || hit_planeX < 0) && (hit_t > 0)) { //
			Vec3 Ph = add(Pe, npe.mult(hit_t));
			Vec3 nLh = normalize(subtract(PL, Ph));
			Vec3 Ncirc = subtract(Ph, Pcirc).mult(1 / r);

			//diffuse
			float T = dot(nLh, Ncirc);
			float S = getSpecular(npe, nLh, Ph, PL, Ncirc);
			T = crop(0, 1, T);
			S = crop(0, 1, S);

			//T = T*face[0][1];

			float x_circ = dot(Vec3(1, 0, 0), Ncirc);
			float y_circ = dot(Vec3(0, 0, 1), Ncirc);
			float z_circ = dot(Vec3(0, -1, 0), Ncirc);
			float phi = acosf(z_circ);
			float v_circ = phi / PI;
			float theta = acosf(y_circ / sin(phi));
			if (x_circ < 0)
				theta = 2 * PI - theta;
			float u_circ = theta / (2 * PI);

			float4 tex_color = tex2D<float4>(tex2, u_circ, 1 - v_circ);


			float ks = 1;
			// populate it
			Color.r(0.1f + 0.9f * T); // red
			Color.r((1.0f - S * ks) * Color.r() + S);
			Color.g(0.05f); // green
			Color.g((1 - S * ks) * Color.g() + 0.5f*S);
			Color.b(0.05f); // blue
			Color.b((1 - S * ks) * Color.b() + 0.5f*S);
		}
		else {
			getPlaneIntersections_PointLight(PiX, npiX, PiY, npiY, npe, Pe, Color, PL, Pcirc, r, tex);

		}
	}
	else {

		if (!getPlaneIntersections_PointLight(PiX, npiX, PiY, npiY, npe, Pe, Color, PL, Pcirc, r, tex)) {
			float env_x = npe.getX();
			float env_y = npe.getY();
			float env_z = npe.getZ();
			float phi = acos(env_z);
			float env_v = phi / PI;
			float theta = acos(env_y / sin(phi));
			float env_u = theta / (2 * PI);
			if (env_x < 0)
				env_u = 1 - env_u;

			float4 tex_color = tex2D<float4>(tex2, env_u, 1 - env_v);

			//Color.r(1.0f * ii / (float)width); // red
			//Color.g(1.0f * jj / (float)height); // green
			//Color.b(0.5f + 0.5f * cos(t / 16.0f)); // blue
			Color.r(tex_color.x);
			Color.g(tex_color.y);
			Color.b(tex_color.z);
		}
	}
	return Color;
}

/*
 * Paint a 2D texture with a moving red/green hatch pattern on a
 * strobing blue background.  Note that this kernel reads to and
 * writes from the texture, hence why this texture was not mapped
 * as WriteDiscard.
 */
__global__ void cuda_kernel_texture_2d(unsigned char* surface, int width, int height, size_t pitch, float t, int g_camX, int g_camY, int g_camZ, float g_rotY, float g_rotX, bool g_aaliasing, cudaTextureObject_t tex, cudaTextureObject_t tex2, cudaTextureObject_t tex3)//cuda_kernel_texture_2d(unsigned char *surface, int width, int height, size_t pitch, float t, int g_camX, int g_camY, int g_camZ, float g_rotY, float g_rotX)//, Vec3& Pe, float& d, float& sx, float& sy, Vec3& n0, Vec3& n1, Vec3& n2, Vec3& P00)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float* pixel;
	float u = x / (float)width;
	float v = y / (float)height;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the pixel at (x,y)
	pixel = (float*)(surface + y * pitch) + 4 * x;




	
	//Set Raycasting input
	SceneParams win(g_camX, g_camY, g_camZ, g_rotY, g_rotX, t);

	Vec3& Pe = win.Pe;
	Vec3& n0 = win.n0;
	Vec3& n1 = win.n1;
	Vec3& n2 = win.n2;
	Vec3& PL = win.PL;
	float sx = win.sx;
	float sy = win.sy;
	Vec3& Pc = win.Pc;
	Vec3& P00 = win.P00;
	Vec3& PiY = win.PiY;
	Vec3& npiY = win.npiY;
	Vec3& PiX = win.PiX;
	Vec3& npiX = win.npiX;
	Vec3& Pcirc = win.Pcirc;
	float r = win.r;
	float light_M = win.Light_m;
	float light_N = win.Light_n;



	//ordered jittering antialiasing

	float M = 6;
	float N = 6;
	bool aalias = g_aaliasing;
	if (aalias == false) {
		M = 1;
		N = 1;
	}
	Vec3 Color(0, 0, 0);
	Vec3 ColorTotal(0, 0, 0);
	for (float mm = 0; mm < M; mm++) {
		for (float nn = 0; nn < N; nn++) {
			//float ii = (float)x + nn / N + 0.5f / N;
			//float jj = (float)y + mm / N + 0.5f / N;
			float ii = (float)x + nn / N + (p_rand((float)x * nn + nn, (float)y * nn + nn)) / N; //rand function does not work in CUDA! and cuda's implementation of random generation is confusing
			float jj = (float)y + mm / M + (p_rand((float)x * mm + mm, (float)y * mm + mm)) / M; //rand function does not work in CUDA! and cuda's implementation of random generation is confusing
			if (!aalias) {
				ii = x;
				jj = y;
			}
			//Begin Raycast!!!!!!!!!!!!!
			Vec3 Pp = add(add(P00, n0.mult(ii / (float)width * sx)), n1.mult(jj / (float)height * sy));
			Vec3 npe = normalize(subtract(Pp, Pe));

			Vec3 Pminus = subtract(Pe, Pcirc);
			float b = dot(npe, Pminus);
			float c = dot(Pminus, Pminus) - (r * r);
			float delta = b * b - c;

			float hit_plane = dot(npiY, subtract(PiY, Pe)) / dot(npiY, npe);
			float hit_planeX = dot(npiX, subtract(PiX, Pe)) / dot(npiX, npe);
			if (b <= 0 && delta >= 0)
			{

				float hit_t = -b - sqrt(delta);
				if ((hit_t <= hit_plane || hit_plane < 0) && (hit_t <= hit_planeX || hit_planeX < 0) && (hit_t > 0)) {
					Vec3 Ph = add(Pe, npe.mult(hit_t));
					Vec3 nLh = normalize(subtract(PL, Ph));
					Vec3 Ncirc = subtract(Ph, Pcirc).mult(1 / r);

					//diffuse
					float T = dot(nLh, Ncirc);
					//float B =  dot(Ncirc,npe);
					float S = getSpecular(npe, nLh, Ph, PL, Ncirc);
					T = crop(0, 1, T);
					//B = crop(-0.3,-0.3, B);
					S = crop(0, 1, S);

					//T = T*face[0][1];

					float x_circ = dot(Vec3(1, 0, 0), Ncirc);
					float y_circ = dot(Vec3(0, 0, 1), Ncirc);
					float z_circ = dot(Vec3(0, -1, 0), Ncirc);
					float phi = acosf(z_circ);
					float v_circ = phi /PI;
					float theta = acosf(y_circ / sin(phi));
					if (x_circ < 0)
						theta = 2 * PI - theta;
					float u_circ = theta / (2 * PI);

					float4 tex_color = tex2D<float4>(tex2, u_circ, 1 - v_circ);


					float ks = 0.8;

					Vec3 R = getReflection(Pe, Ph, PL, Ncirc, Color, t, tex, tex2);
					// populate it
					Color.r(0.2f + 0.9f * T); // red
					Color.r(Color.r()* tex_color.x);
					Color.r((1.0f - ks)* Color.r() + ks*R.r());
					//Color.r((1.0f - B) * Color.r() + B * 0.2f);
					Color.g(0.2f + 0.9f * T); // green
					Color.g(Color.g()* tex_color.y);
					Color.g((1 -  ks)* Color.g() + ks*R.g());
					//Color.g((1.0f - B) * Color.g() + B * 1.0f);
					Color.b(0.2f + 0.9f * T); // blue
					Color.b(Color.b()* tex_color.z);
					Color.b((1 -  ks)* Color.b() + ks*R.b());
					//Color.b((1.0f - B) * Color.b() + B * 0.4f);

				}
				else {
					getPlaneIntersections_PointLight(PiX, npiX, PiY, npiY, npe, Pe, Color, PL, Pcirc, r, tex);

				}
			}
			else {

				if (!getPlaneIntersections_PointLight(PiX, npiX, PiY, npiY, npe, Pe, Color, PL, Pcirc, r, tex)) {
					float env_x = npe.getX();
					float env_y = npe.getY();
					float env_z = npe.getZ();
					float phi = acos(env_z);
					float env_v = phi / PI;
					float theta = acos(env_y / sin(phi));
					float env_u = theta / (2 * PI);
					if (env_x < 0)
						env_u = 1 - env_u;

					float4 tex_color = tex2D<float4>(tex2, env_u, 1 - env_v);

					//Color.r(1.0f * ii / (float)width); // red
					//Color.g(1.0f * jj / (float)height); // green
					//Color.b(0.5f + 0.5f * cos(t / 16.0f)); // blue
					Color.r(tex_color.x);
					Color.g(tex_color.y);
					Color.b(tex_color.z);
				}
			}
			ColorTotal = add(ColorTotal, Color);
		}
	}
	pixel[0] = ColorTotal.r() / (M * N);
	pixel[1] = ColorTotal.g() / (M * N);
	pixel[2] = ColorTotal.b() / (M * N);
	pixel[3] = 1;
	//end Raycast!!!!!!!!!
}

extern "C"
void cuda_texture_2d(void* surface, int width, int height, size_t pitch, float t, int g_camX, int g_camY, int g_camZ, float g_rotY, float g_rotX, bool g_aaliasing)//cuda_texture_2d(void *surface, int width, int height, size_t pitch, float t, int g_camX, int g_camY, int g_camZ, float g_rotY, float g_rotX)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);



	cuda_kernel_texture_2d << <Dg, Db >> > ((unsigned char*)surface, width, height, pitch, t, g_camX, g_camY, g_camZ, g_rotY, g_rotX, g_aaliasing, contentImages[0].textureObject, contentImages[1].textureObject, contentImages[2].textureObject);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
	}
}
extern "C"
void deinitAtlasAndImages()
{
	for (size_t i = 0; i < contentImages.size(); i++)
	{
		Image& image = contentImages[i];

		if (image.h_data)
		{
			free(image.h_data);
		}

		if (image.textureObject)
		{
			checkCudaErrors(cudaDestroyTextureObject(image.textureObject));
		}

		if (image.mipmapArray)
		{
			checkCudaErrors(cudaFreeMipmappedArray(image.mipmapArray));
		}
	}

	if (atlasImage.h_data)
	{
		free(atlasImage.h_data);
	}

	if (atlasImage.textureObject)
	{
		checkCudaErrors(cudaDestroyTextureObject(atlasImage.textureObject));
	}

	if (atlasImage.dataArray)
	{
		checkCudaErrors(cudaFreeArray(atlasImage.dataArray));
	}
}

extern "C"
void initAtlasAndImages(const Image* images, size_t numImages, cudaExtent atlasSize)
{
	// create individual textures
	contentImages.resize(numImages);
	for (size_t i = 0; i < numImages; i++)
	{
		Image& image = contentImages[i];
		image.size = images[i].size;
		image.size.depth = 0;
		image.type = cudaResourceTypeMipmappedArray;

		// how many mipmaps we need

		cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
		checkCudaErrors(cudaMallocMipmappedArray(&image.mipmapArray, &desc, image.size, 1));

		// upload level 0
		cudaArray_t level0;
		checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, image.mipmapArray, 0));

		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(images[i].h_data, image.size.width * sizeof(uchar4), image.size.width, image.size.height);
		copyParams.dstArray = level0;
		copyParams.extent = image.size;
		copyParams.extent.depth = 1;
		copyParams.kind = cudaMemcpyHostToDevice;
		checkCudaErrors(cudaMemcpy3D(&copyParams));

		// compute rest of mipmaps based on level 0
		//generateMipMaps(image.mipmapArray, image.size);

		// generate bindless texture object

		cudaResourceDesc            resDescr;
		memset(&resDescr, 0, sizeof(cudaResourceDesc));

		resDescr.resType = cudaResourceTypeMipmappedArray;
		resDescr.res.mipmap.mipmap = image.mipmapArray;

		cudaTextureDesc             texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));

		texDescr.normalizedCoords = 1;
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.mipmapFilterMode = cudaFilterModeLinear;

		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;

		texDescr.maxMipmapLevelClamp = float(0);

		texDescr.readMode = cudaReadModeNormalizedFloat;

		checkCudaErrors(cudaCreateTextureObject(&image.textureObject, &resDescr, &texDescr, NULL));
	}

}
