#ifdef USE_CUDA

#include "cudaFftRoutines.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <sstream>
#include <vector>

namespace
{

bool checkCuda(cudaError_t status, const char* operation, std::string* error)
{
	if (status == cudaSuccess)
		return true;

	if (error != nullptr)
	{
		std::ostringstream stream;
		stream << operation << " failed: " << cudaGetErrorString(status);
		*error = stream.str();
	}
	return false;
}

bool checkCuFft(cufftResult status, const char* operation, std::string* error)
{
	if (status == CUFFT_SUCCESS)
		return true;

	if (error != nullptr)
	{
		std::ostringstream stream;
		stream << operation << " failed with cuFFT status " << static_cast<int>(status);
		*error = stream.str();
	}
	return false;
}

template<typename T>
void releaseCudaBuffer(T*& buffer)
{
	if (buffer != nullptr)
	{
		cudaFree(buffer);
		buffer = nullptr;
	}
}

void destroyPlan(cufftHandle& plan, bool& hasPlan)
{
	if (hasPlan)
	{
		cufftDestroy(plan);
		hasPlan = false;
	}
}

void inversefftshift(std::vector<float>& out, const std::vector<float>& in, size_t xdim, size_t ydim)
{
	for(size_t i=0; i<ydim/2;i++)
	{
		for(size_t j=0; j<xdim/2;j++)
		{
			size_t outIndex1 = (j+xdim/2)+(i+ydim/2)*xdim;
			size_t inIndex1 = j+i*xdim;
			size_t outIndex2 = j+i*xdim;
			size_t inIndex2 = (j+xdim/2)+(i+ydim/2)*xdim;
			out[inIndex1]=in[outIndex1];
			out[inIndex2]=in[outIndex2];
		}
	}

	for(size_t i=0; i<ydim/2;i++)
	{
		for(size_t j=xdim/2; j<xdim;j++)
		{
			size_t outIndex1 = (j-xdim/2)+(i+ydim/2)*xdim;
			size_t inIndex1 = j+i*xdim;
			size_t outIndex2 =j+i*xdim;
			size_t inIndex2 = (j-xdim/2)+(i+ydim/2)*xdim;
			out[inIndex1]=in[outIndex1];
			out[inIndex2]=in[outIndex2];
		}
	}
}

__global__ void multiplyDoubleComplexByFloat(cufftDoubleComplex* values, const float* factors, size_t count)
{
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count)
		return;

	double factor = static_cast<double>(factors[index]);
	if (factor != 1.0)
	{
		values[index].x *= factor;
		values[index].y *= factor;
	}
}

__global__ void normalizeFloatBuffer(float* values, float factor, size_t count)
{
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count)
		values[index] *= factor;
}

bool launchComplexMultiply(cufftDoubleComplex* deviceSpectrum, float* deviceFilter, size_t count, std::string* error)
{
	const int threadsPerBlock = 256;
	const int blocks = static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock);
	multiplyDoubleComplexByFloat<<<blocks, threadsPerBlock>>>(deviceSpectrum, deviceFilter, count);
	return checkCuda(cudaGetLastError(), "frequency-domain multiply kernel", error);
}

bool launchNormalize(float* deviceValues, float factor, size_t count, std::string* error)
{
	const int threadsPerBlock = 256;
	const int blocks = static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock);
	normalizeFloatBuffer<<<blocks, threadsPerBlock>>>(deviceValues, factor, count);
	return checkCuda(cudaGetLastError(), "normalization kernel", error);
}

std::vector<float> packFilterForSpectrumOrder(const std::vector<float>& shiftedFilter, size_t sizeX, size_t sizeYOrNyh)
{
	std::vector<float> packedFilter(sizeX * sizeYOrNyh);
	for (size_t j = 0; j < sizeYOrNyh; j++)
	{
		for (size_t i = 0; i < sizeX; i++)
		{
			packedFilter[j + i * sizeYOrNyh] = shiftedFilter[i + j * sizeX];
		}
	}
	return packedFilter;
}

}

namespace novaCTF
{

bool CudaFftRoutines::isRuntimeAvailable(std::string* reason)
{
	int deviceCount = 0;
	cudaError_t status = cudaGetDeviceCount(&deviceCount);
	if (status != cudaSuccess)
	{
		if (reason != nullptr)
			*reason = cudaGetErrorString(status);
		return false;
	}

	if (deviceCount <= 0)
	{
		if (reason != nullptr)
			*reason = "no CUDA device detected";
		return false;
	}

	return true;
}

bool CudaFftRoutines::real1DTransform(size_t size, const std::vector<float>& data, std::vector<float>& fft, std::string* error)
{
	fft.clear();
	fft.reserve(size);

	double* deviceInput = nullptr;
	cufftDoubleComplex* deviceOutput = nullptr;
	cufftHandle plan = 0;
	bool hasPlan = false;

	std::vector<double> hostInput(size);
	for (size_t i = 0; i < size; i++)
		hostInput[i] = static_cast<double>(data[i]);

	std::vector<cufftDoubleComplex> hostOutput((size / 2) + 1);
	bool ok = checkCuda(cudaMalloc(&deviceInput, sizeof(double) * size), "cudaMalloc real1D input", error)
		&& checkCuda(cudaMalloc(&deviceOutput, sizeof(cufftDoubleComplex) * hostOutput.size()), "cudaMalloc real1D output", error)
		&& checkCuda(cudaMemcpy(deviceInput, &hostInput[0], sizeof(double) * size, cudaMemcpyHostToDevice), "cudaMemcpy real1D input", error)
		&& checkCuFft(cufftPlan1d(&plan, static_cast<int>(size), CUFFT_D2Z, 1), "cufftPlan1d real1D", error);

	if (ok)
		hasPlan = true;

	if (ok)
		ok = checkCuFft(cufftExecD2Z(plan, deviceInput, deviceOutput), "cufftExecD2Z real1D", error);

	if (ok)
		ok = checkCuda(cudaMemcpy(&hostOutput[0], deviceOutput, sizeof(cufftDoubleComplex) * hostOutput.size(), cudaMemcpyDeviceToHost), "cudaMemcpy real1D output", error);

	if (ok)
	{
		if (size % 2 == 0)
			fft.push_back(static_cast<float>(hostOutput[size / 2].x));

		for (int k = static_cast<int>((size + 1) / 2) - 1; k >= 1; k--)
			fft.push_back(static_cast<float>(hostOutput[k].x));

		for (int k = 1; k < static_cast<int>((size + 1) / 2); k++)
			fft.push_back(static_cast<float>(hostOutput[k].x));

		if (size % 2 == 0)
			fft.push_back(static_cast<float>(hostOutput[size / 2].x));
	}

	destroyPlan(plan, hasPlan);
	releaseCudaBuffer(deviceOutput);
	releaseCudaBuffer(deviceInput);
	return ok;
}

bool CudaFftRoutines::real2DTransform(size_t sizeX, size_t sizeY, std::vector<float>& data, const std::vector<float>& filter, std::string* error)
{
	const size_t realCount = sizeX * sizeY;
	const size_t complexCount = sizeX * ((sizeY / 2) + 1);

	std::vector<double> hostReal(realCount);
	for (size_t i = 0; i < realCount; i++)
		hostReal[i] = static_cast<double>(data[i]);

	std::vector<float> shiftedFilter(filter.size());
	inversefftshift(shiftedFilter, filter, sizeX, sizeY);
	std::vector<float> packedFilter = packFilterForSpectrumOrder(shiftedFilter, sizeX, (sizeY / 2) + 1);

	double* deviceReal = nullptr;
	cufftDoubleComplex* deviceSpectrum = nullptr;
	float* deviceFilter = nullptr;
	cufftHandle forwardPlan = 0;
	cufftHandle backwardPlan = 0;
	bool hasForwardPlan = false;
	bool hasBackwardPlan = false;

	bool ok = checkCuda(cudaMalloc(&deviceReal, sizeof(double) * realCount), "cudaMalloc real2D real", error)
		&& checkCuda(cudaMalloc(&deviceSpectrum, sizeof(cufftDoubleComplex) * complexCount), "cudaMalloc real2D spectrum", error)
		&& checkCuda(cudaMalloc(&deviceFilter, sizeof(float) * complexCount), "cudaMalloc real2D filter", error)
		&& checkCuda(cudaMemcpy(deviceReal, &hostReal[0], sizeof(double) * realCount, cudaMemcpyHostToDevice), "cudaMemcpy real2D real", error)
		&& checkCuda(cudaMemcpy(deviceFilter, &packedFilter[0], sizeof(float) * complexCount, cudaMemcpyHostToDevice), "cudaMemcpy real2D filter", error)
		&& checkCuFft(cufftPlan2d(&forwardPlan, static_cast<int>(sizeX), static_cast<int>(sizeY), CUFFT_D2Z), "cufftPlan2d forward real2D", error);

	if (ok)
		hasForwardPlan = true;

	if (ok)
		ok = checkCuFft(cufftPlan2d(&backwardPlan, static_cast<int>(sizeX), static_cast<int>(sizeY), CUFFT_Z2D), "cufftPlan2d inverse real2D", error);

	if (ok)
		hasBackwardPlan = true;

	if (ok)
		ok = checkCuFft(cufftExecD2Z(forwardPlan, deviceReal, deviceSpectrum), "cufftExecD2Z real2D", error);

	if (ok)
		ok = launchComplexMultiply(deviceSpectrum, deviceFilter, complexCount, error);

	if (ok)
		ok = checkCuFft(cufftExecZ2D(backwardPlan, deviceSpectrum, deviceReal), "cufftExecZ2D real2D", error);

	if (ok)
		ok = checkCuda(cudaMemcpy(&hostReal[0], deviceReal, sizeof(double) * realCount, cudaMemcpyDeviceToHost), "cudaMemcpy real2D output", error);

	if (ok)
	{
		for (size_t i = 0; i < realCount; i++)
			data[i] = static_cast<float>(hostReal[i] / static_cast<double>(data.size()));
	}

	destroyPlan(backwardPlan, hasBackwardPlan);
	destroyPlan(forwardPlan, hasForwardPlan);
	releaseCudaBuffer(deviceFilter);
	releaseCudaBuffer(deviceSpectrum);
	releaseCudaBuffer(deviceReal);
	return ok;
}

bool CudaFftRoutines::complex2DTransform(size_t sizeX, size_t sizeY, std::vector<float>& data, const std::vector<float>& filter, std::string* error)
{
	const size_t complexCount = sizeX * sizeY;
	std::vector<cufftDoubleComplex> hostComplex(complexCount);
	for (size_t i = 0; i < complexCount; i++)
	{
		hostComplex[i].x = static_cast<double>(data[i]);
		hostComplex[i].y = 0.0;
	}

	std::vector<float> shiftedFilter(filter.size());
	inversefftshift(shiftedFilter, filter, sizeX, sizeY);
	std::vector<float> packedFilter = packFilterForSpectrumOrder(shiftedFilter, sizeX, sizeY);

	cufftDoubleComplex* deviceComplex = nullptr;
	float* deviceFilter = nullptr;
	cufftHandle plan = 0;
	bool hasPlan = false;

	bool ok = checkCuda(cudaMalloc(&deviceComplex, sizeof(cufftDoubleComplex) * complexCount), "cudaMalloc complex2D values", error)
		&& checkCuda(cudaMalloc(&deviceFilter, sizeof(float) * complexCount), "cudaMalloc complex2D filter", error)
		&& checkCuda(cudaMemcpy(deviceComplex, &hostComplex[0], sizeof(cufftDoubleComplex) * complexCount, cudaMemcpyHostToDevice), "cudaMemcpy complex2D values", error)
		&& checkCuda(cudaMemcpy(deviceFilter, &packedFilter[0], sizeof(float) * complexCount, cudaMemcpyHostToDevice), "cudaMemcpy complex2D filter", error)
		&& checkCuFft(cufftPlan2d(&plan, static_cast<int>(sizeX), static_cast<int>(sizeY), CUFFT_Z2Z), "cufftPlan2d complex2D", error);

	if (ok)
		hasPlan = true;

	if (ok)
		ok = checkCuFft(cufftExecZ2Z(plan, deviceComplex, deviceComplex, CUFFT_FORWARD), "cufftExecZ2Z complex2D forward", error);

	if (ok)
		ok = launchComplexMultiply(deviceComplex, deviceFilter, complexCount, error);

	if (ok)
		ok = checkCuFft(cufftExecZ2Z(plan, deviceComplex, deviceComplex, CUFFT_INVERSE), "cufftExecZ2Z complex2D inverse", error);

	if (ok)
		ok = checkCuda(cudaMemcpy(&hostComplex[0], deviceComplex, sizeof(cufftDoubleComplex) * complexCount, cudaMemcpyDeviceToHost), "cudaMemcpy complex2D output", error);

	if (ok)
	{
		for (size_t i = 0; i < complexCount; i++)
			data[i] = static_cast<float>(hostComplex[i].x / static_cast<double>(data.size()));
	}

	destroyPlan(plan, hasPlan);
	releaseCudaBuffer(deviceFilter);
	releaseCudaBuffer(deviceComplex);
	return ok;
}

bool CudaFftRoutines::many1DTransform(float* data, int sizeX, int sizeY, int direction, std::string* error)
{
	if (direction != 0 && direction != 1)
	{
		if (error != nullptr)
			*error = "unsupported FFT direction";
		return false;
	}

	const int nxpad = 2 * (sizeX / 2 + 1);
	const size_t valueCount = static_cast<size_t>(nxpad) * static_cast<size_t>(sizeY);
	const float invScale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(sizeX)));

	float* deviceData = nullptr;
	cufftHandle plan = 0;
	bool hasPlan = false;
	int n[1] = { sizeX };
	int realEmbed[1] = { nxpad };
	int complexEmbed[1] = { nxpad / 2 };
	int* inembed = direction == 0 ? realEmbed : complexEmbed;
	int* onembed = direction == 0 ? complexEmbed : realEmbed;
	int idist = direction == 0 ? nxpad : nxpad / 2;
	int odist = direction == 0 ? nxpad / 2 : nxpad;

	bool ok = checkCuda(cudaMalloc(&deviceData, sizeof(float) * valueCount), "cudaMalloc many1D values", error)
		&& checkCuda(cudaMemcpy(deviceData, data, sizeof(float) * valueCount, cudaMemcpyHostToDevice), "cudaMemcpy many1D values", error)
		&& checkCuFft(cufftPlanMany(&plan, 1, n, inembed, 1, idist, onembed, 1, odist, direction == 0 ? CUFFT_R2C : CUFFT_C2R, sizeY), "cufftPlanMany many1D", error);

	if (ok)
		hasPlan = true;

	if (ok && direction == 0)
		ok = checkCuFft(cufftExecR2C(plan, deviceData, reinterpret_cast<cufftComplex*>(deviceData)), "cufftExecR2C many1D", error);

	if (ok && direction == 1)
		ok = checkCuFft(cufftExecC2R(plan, reinterpret_cast<cufftComplex*>(deviceData), deviceData), "cufftExecC2R many1D", error);

	if (ok)
		ok = launchNormalize(deviceData, invScale, valueCount, error);

	if (ok)
		ok = checkCuda(cudaMemcpy(data, deviceData, sizeof(float) * valueCount, cudaMemcpyDeviceToHost), "cudaMemcpy many1D output", error);

	destroyPlan(plan, hasPlan);
	releaseCudaBuffer(deviceData);
	return ok;
}

}

#endif
