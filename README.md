# raytracer_cuda
This code was taken and repurposed from NVIDIA's CUDA Sample files running on visual studio. 
I am not claiming this program to be my own. Just the raytracer parts of it.
This was done for a class project for educational purposes and it was okay with my professor.
The only thing expected from the class was to write the raytracer, not for it to be done in CUDA or perform well.
Therefore, the fact that it ran in realtime was just a personal bonus for having control over the camera as well
as animate lights and other variables to create an interesting video in realtime.

The three files included are everything with relevant code or information for the sake of evaluation. 
It is not enough to run as it does not include project files, input data, project settings, etc.

The raytracer code I wrote is found in texture_2d.cu. 
The top comment in that file describes what parts I used from NVIDIA's original code and what parts I wrote. 
I wrote anything related to the actual raytracer. The code is completely repurposed from what originally produced.

Vec3.h contains helper functions and classes. 
Due to how Inheritance and classes work in CUDA, instead of applying proper classes in a way that
can make the program easy to expand upon and more efficient, my lack of understanding on top of time constraints
resulted in me taking a more brute-force approach. 
The result is messy code that doesn't run as well as it potentially could, but can run well enough to run in realtime.

simpleD3D9Texture.cpp is almost completely taken from NVIDIA.
