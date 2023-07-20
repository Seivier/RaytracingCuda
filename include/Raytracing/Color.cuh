//
// Created by vigb9 on 20/07/2023.
//

#ifndef _COLOR_CUH_
#define _COLOR_CUH_

#include "Raytracing/Vector.cuh"
#include <iostream>

void writeColor(std::ostream& out, Color pixel_color, int samples_per_pixel);

#endif //_COLOR_CUH_
