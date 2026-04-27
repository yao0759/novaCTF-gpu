#pragma once

#ifdef USE_CUDA

#include <cstddef>
#include <string>
#include <vector>

namespace novaCTF
{

class CudaFftRoutines
{
public:
	static bool isRuntimeAvailable(std::string* reason = nullptr);
	static bool real1DTransform(size_t size, const std::vector<float>& data, std::vector<float>& fft, std::string* error = nullptr);
	static bool real2DTransform(size_t sizeX, size_t sizeY, std::vector<float>& data, const std::vector<float>& filter, std::string* error = nullptr);
	static bool complex2DTransform(size_t sizeX, size_t sizeY, std::vector<float>& data, const std::vector<float>& filter, std::string* error = nullptr);
	static bool many1DTransform(float* data, int sizeX, int sizeY, int direction, std::string* error = nullptr);
};

}

#endif