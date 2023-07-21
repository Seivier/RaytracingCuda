#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "rt/vector.cuh"
#include "rt/ray.cuh"
#include "rt/sphere.cuh"
#include "rt/hittable_list.cuh"
#include "rt/camera.cuh"
#include "rt/material.cuh"


#define CUDA_CALL(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into rayColor() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vector rayColor(const ray& r, hittable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vector cur_attenuation = vector(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vector attenuation;
            if(rec.matPtr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vector(0.0,0.0,0.0);
            }
        }
        else {
            vector unit_direction = unitVector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vector c = (1.0f-t)*vector(1.0, 1.0, 1.0) + t*vector(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vector(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vector *fb, int max_x, int max_y, int ns, camera **cam, hittable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vector col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->getRay(u, v, &local_rand_state);
        col += rayColor(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vector(0,-1000.0,-1), 1000,
                               new lambertian(vector(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vector center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vector(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vector(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vector(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vector(-4, 1, 0), 1.0, new lambertian(vector(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vector(4, 1, 0),  1.0, new metal(vector(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hittable_list(d_list, 22*22+1+3);

        vector lookfrom(13,2,3);
        vector lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vector(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)d_list[i])->matPtr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main() {
    int nx = 1200;
    int ny = 800;
    int ns = 10;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vector);

    // allocate FB
    vector *fb;
    CUDA_CALL(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    CUDA_CALL(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    CUDA_CALL(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // make our world of hittables & the camera
    hittable **d_list;
    int num_hittables = 22*22+1+3;
    CUDA_CALL(cudaMalloc((void **)&d_list, num_hittables*sizeof(hittable *)));
    hittable **d_world;
    CUDA_CALL(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    camera **d_camera;
    CUDA_CALL(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    CUDA_CALL(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaFree(d_camera));
    CUDA_CALL(cudaFree(d_world));
    CUDA_CALL(cudaFree(d_list));
    CUDA_CALL(cudaFree(d_rand_state));
    CUDA_CALL(cudaFree(d_rand_state2));
    CUDA_CALL(cudaFree(fb));

    cudaDeviceReset();
}