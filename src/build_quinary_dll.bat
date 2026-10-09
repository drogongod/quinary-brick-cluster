call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /O2 /openmp /arch:AVX2 /LD quinary_simd_kernel.c /Fe:quinary_simd_kernel.dll
