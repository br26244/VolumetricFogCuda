Volumetric fog is typically used in animation and video game development, as it provides real
time light scattering and particle movement against animated models. In comparison to Depth-Based Fog,
which modifies pixels in relation to their distance from the camera, volumetric fog is a true 3D effect that
renders how light interacts with particles in real time. This project is a
volumetric fog renderer with implemented simulated light scattering, light extinction, shadow rays,
animated basic fluid simulation, and colored lighting. Part of this project assignment's requirements was
to also involve CUDA and parallel optimization. I involved CUDA’s api
to highly optimize the rendering and simulation of this graphics project.


#Requirement:
--CUDA
--OpenGL
--Cmake
--GLEW
--GLM

Currently, the output is a series of png taken at each frame. The PNGs are formatted in index form so you can easily convert them into a playable MP4.

#How to set up:
mkdir -p png
mkdir build
cd build
cmake ..
Make
./volumetric

You can simply download the png folder and upload it to any PNG to MP4 converter.

#customization
If you wish to change the background color, simply edit the rgb values assigned to float3 background =
make
_
float3(1.0f, 0.50f, 0.0f); line 89 of rayMarcher.cu.
const int maxSteps = 500;
const float stepSize = 0.008f;
const float extinction = 5.0f;
Changing these values will change the quality of the fog. Increase the extinction value to increase the fog
density, increase max steps, and reduce step size for a much more detailed fog


On lighting.cu
At each index you can edit the color at light.color to change any of the three colored lights. To increase
the strength or intensity of these colors by increasing the particular light intensity and the sun intensity.
