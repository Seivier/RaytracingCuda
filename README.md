# RaytracingCuda
Este proyecto requiere CUDA.

## Instrucciones de uso
Asegurarse que los submodulos se encuentren actualizados, esto no es mandatorio, ya que el programa no requiere de librerías externas de momento:
```bash
git submodule update --init --recursive
```

Para compilar el programa se debe ejecutar el siguiente comando:
```bash
mkdir build && cd build
cmake -B . -S ..
cmake --build .
```

El ejecutable se encontrará en la carpeta distintas carpetas dependiendo del generado usado en CMake, 
el proyecto fue construido en CLion con el generador por defecto de CMake, por lo que el ejecutable se encontrará en la carpeta `build/src/Debug/`.

Para ejecutar el programa se debe ejecutar el siguiente comando:
```bash
.\path\to\exe\rtcuda.exe width height tx ty samples depth
``` 

Donde:
- `width` es el ancho de la imagen a generar.
- `height` es el alto de la imagen a generar.
- `tx` es la cantidad de threads por bloque en la dim x.
- `ty` es la cantidad de threads por bloque en la dim y.
- `samples` es la cantidad de rayos lanzados por pixel.
- `depth` es la cantidad de rebotes que puede tener un rayo.

Se recomienda usar el ejecutable de la siguiente manera:
```bash
.\path\to\exe\rtcuda.exe 800 800 16 16 10 50 > out.ppm
```

Para imágenes más nítidas se recomienda usar más samples, pero esto aumenta el tiempo de ejecución.

Página para visualizar la imagen generada: https://www.cs.rhodes.edu/welshc/COMP141_F16/ppmReader.html