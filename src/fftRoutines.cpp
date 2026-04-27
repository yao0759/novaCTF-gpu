#include "fftRoutines.h"
#include <fftw3.h>
#include "common.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include "volumeIO.h"

#ifdef USE_CUDA
#include "cudaFftRoutines.h"
#endif

using namespace std;

namespace
{

bool isEnabledEnvironmentFlag(const char* variableName)
{
	const char* value = std::getenv(variableName);
	if (value == nullptr)
		return false;

	return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0;
}

#ifdef USE_CUDA
bool shouldUseCuda(std::string* reason)
{
	if (isEnabledEnvironmentFlag("NOVACTF_DISABLE_CUDA"))
	{
		if (reason != nullptr)
			*reason = "disabled by NOVACTF_DISABLE_CUDA";
		return false;
	}

	return novaCTF::CudaFftRoutines::isRuntimeAvailable(reason);
}

void logCudaFallbackOnce(const std::string& reason)
{
	if (reason.empty() || reason == "disabled by NOVACTF_DISABLE_CUDA")
		return;

	static bool hasLoggedFallback = false;
	if (!hasLoggedFallback)
	{
		cerr << "CUDA FFT backend unavailable (" << reason << "), falling back to FFTW." << endl;
		hasLoggedFallback = true;
	}
}
#endif

double logarithmizeValueCpu(double value, bool logarithmizeData)
{
	if(logarithmizeData && value>=1.0)
		return log(value);
	else
		return value;
}

void inversefftshift(std::vector<float>& out, std::vector<float>& in, size_t xdim, size_t ydim)
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

void fftshift(std::vector<float>& out, std::vector<float>& in, size_t xdim, size_t ydim)
{
	for(size_t i=0; i<ydim/2;i++)
	{
		for(size_t j=0; j<xdim/2;j++)
		{
			size_t outIndex1 = (j+xdim/2)+(i+ydim/2)*xdim;
			size_t inIndex1 = j+i*xdim;
			size_t outIndex2 = j+i*xdim;
			size_t inIndex2 = (j+xdim/2)+(i+ydim/2)*xdim;
			out[outIndex1]=in[inIndex1];
			out[outIndex2]=in[inIndex2];
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
			out[outIndex1]=in[inIndex1];
			out[outIndex2]=in[inIndex2];
		}
	}
}

void computePowerSpectrumCpu(std::vector<float>& powerSpectrum, fftw_complex* fftOut, size_t sizeX, size_t nyh, bool logarithmizeData)
{
	size_t k=0;
	for(size_t j=0; j<nyh; j++)
	{
		for(size_t i = 0; i<sizeX; i++)
		{
			powerSpectrum[k] = logarithmizeValueCpu(fftOut[j+i*nyh][0]*fftOut[j+i*nyh][0]+fftOut[j+i*nyh][1]*fftOut[j+i*nyh][1],logarithmizeData);
			k++;
		}
	}

	for(int j=(int)nyh-2; j>0; j--)
	{
		powerSpectrum[k]= fftOut[j][0]*fftOut[j][0]+fftOut[j][1]*fftOut[j][1];
		k++;
		for(int i = (int)sizeX-1; i>0; i--)
		{
			powerSpectrum[k] = logarithmizeValueCpu(fftOut[j+i*nyh][0]*fftOut[j+i*nyh][0]+fftOut[j+i*nyh][1]*fftOut[j+i*nyh][1],logarithmizeData);
			k++;
		}
	}
}

void maskFFTCpu(fftw_complex* fftOut, std::vector<double>& mask, size_t sizeX, size_t nyh)
{
	size_t k=0;
	for(size_t j=0; j<nyh; j++)
	{
		for(size_t i = 0; i<sizeX; i++)
		{
			if(mask[k]!=1.0f)
			{
				fftOut[j+i*nyh][0] = fftOut[j+i*nyh][0]*mask[k];
				fftOut[j+i*nyh][1] = fftOut[j+i*nyh][1]*mask[k];
			}
			k++;
		}
	}
}

void filterFFTCpu(fftw_complex* fftOut, std::vector<float>& filter, size_t sizeX, size_t nyh)
{
	size_t k=0;
	for(size_t j=0; j<nyh; j++)
	{
		for(size_t i = 0; i<sizeX; i++)
		{
			if(filter[k]!=1.0f)
			{
				fftOut[j+i*nyh][0] = fftOut[j+i*nyh][0]*filter[k];
				fftOut[j+i*nyh][1] = fftOut[j+i*nyh][1]*filter[k];
			}
			k++;
		}
	}
}

void normalizeValuesCpu(std::vector<float>& normalizedData, std::vector<double>& originalData, size_t dataSize, novaCTF::DataStats& dataStats)
{
	for ( size_t i = 0; i < dataSize; i++ )
	{
		normalizedData[i]=originalData[i]/(dataSize);
		dataStats.mean+=normalizedData[i];
		dataStats.max = max(dataStats.max, normalizedData[i]);
		dataStats.min = min(dataStats.min, normalizedData[i]);
	}
}

void normalizeCpu(float *array, float scale, size_t dataSize)
{
	for (size_t i = 0; i < dataSize; i++)
		array[i] *= scale;
}

void real1DTransformCpu(size_t size, std::vector<float>& data, std::vector<float>& fft)
{
	double* fftIn = (double*) fftw_malloc ( sizeof ( double ) * size );
	for(size_t i=0; i<size; i++)
	{
		fftIn[i] = (double)data[i];
	}

	size_t fftSize = ( size / 2 ) + 1;
	fftw_complex* fftOut = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * fftSize );
	fftw_plan plan_forward = fftw_plan_dft_r2c_1d((int)size, fftIn, fftOut, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	if (size % 2 == 0)
		fft.push_back((float)fftOut[size/2][0]);

	for (int k = (int)((size+1)/2)-1; k >=1 ; k--)
		fft.push_back((float)fftOut[k][0]);

	for (int k = 1; k < (int)(size+1)/2; k++)
		fft.push_back((float)fftOut[k][0]);

	if (size % 2 == 0)
		fft.push_back((float)fftOut[size/2][0]);

	fftw_destroy_plan(plan_forward);
	fftw_free(fftOut);
	fftw_free(fftIn);
}

void complex2DTransformCpu(size_t sizeX, size_t sizeY, std::vector<float>& data, std::vector<float>& filter)
{
	fftw_complex* fftOut = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sizeX * sizeY);
	fftw_complex* fftIn = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sizeX * sizeY);

	for ( size_t i = 0; i < sizeX*sizeY; i++ )
	{
		fftIn[i][0] = (double)data[i];
		fftIn[i][1] = 0.0;
	}

	fftw_plan plan_forward = fftw_plan_dft_2d((int)sizeX, (int)sizeY, fftIn, fftOut, -1, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	std::vector<float> isFilter(filter.size());
	inversefftshift(isFilter, filter, sizeX, sizeY);
	filterFFTCpu(fftOut, isFilter, sizeX, sizeY);

	fftw_plan plan_backward = fftw_plan_dft_2d((int)sizeX, (int)sizeY, fftOut, fftIn, 1, FFTW_ESTIMATE);
	fftw_execute(plan_backward);

	fftw_destroy_plan(plan_backward);
	fftw_destroy_plan(plan_forward);
	fftw_free(fftOut);

	for(size_t i= 0; i < data.size(); i++)
	{
		data[i]=(float)(fftIn[i][0]/data.size());
	}

	fftw_free(fftIn);
}

void real2DTransformCpu(size_t sizeX, size_t sizeY, std::vector<float>& data, std::vector<float>& filter)
{
	std::vector<double> fftIn(sizeX*sizeY);
	size_t sliceSize = sizeX*sizeY;

	for ( size_t i = 0; i < sliceSize; i++ )
	{
		fftIn[i] = (double)data[i];
	}

	size_t nyh = ( sizeY / 2 ) + 1;
	fftw_complex* fftOut = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * sizeX * nyh );
	fftw_plan plan_forward = fftw_plan_dft_r2c_2d((int)sizeX, (int)sizeY, &fftIn[0], fftOut, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	std::vector<float> isFilter(filter.size());
	inversefftshift(isFilter, filter, sizeX, sizeY);
	filterFFTCpu(fftOut, isFilter, sizeX, nyh);

	fftw_plan plan_backward = fftw_plan_dft_c2r_2d((int)sizeX, (int)sizeY, fftOut, &fftIn[0], FFTW_ESTIMATE);
	fftw_execute(plan_backward);

	fftw_destroy_plan(plan_backward);
	fftw_destroy_plan(plan_forward);
	fftw_free(fftOut);

	for(size_t i= 0; i < data.size(); i++)
	{
		data[i]=(float)(fftIn[i]/data.size());
	}
}

void real2DMaskTransformCpu(size_t sizeX, size_t sizeY, std::vector<float>& data, std::vector<double>& mask, novaCTF::DataStats& dataStats)
{
	std::vector<double> fftIn(sizeX*sizeY);
	size_t sliceSize = sizeX*sizeY;

	for ( size_t i = 0; i < sliceSize; i++ )
	{
		fftIn[i] = (double)data[i];
	}

	size_t nyh = ( sizeY / 2 ) + 1;
	fftw_complex* fftOut = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * sizeX * nyh );
	fftw_plan plan_forward = fftw_plan_dft_r2c_2d((int)sizeX, (int)sizeY, &fftIn[0], fftOut, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	maskFFTCpu(fftOut, mask, sizeX, nyh);

	fftw_plan plan_backward = fftw_plan_dft_c2r_2d((int)sizeX, (int)sizeY, fftOut, &fftIn[0], FFTW_ESTIMATE);
	fftw_execute(plan_backward);

	normalizeValuesCpu(data, fftIn, sliceSize, dataStats);

	fftw_destroy_plan(plan_backward);
	fftw_destroy_plan(plan_forward);
	fftw_free(fftOut);
}

void many1DTransformCpu(float* data, int sizeX, int sizeY, int direction)
{
	size_t nx = sizeX;
	size_t ny = sizeY;
	int nxpad = 2 * (int)(nx / 2 + 1);
	float invScale = (float)(1.0f / sqrt((double)nx));

	fftwf_plan plan;

	if (direction == 0)
	{
		plan = fftwf_plan_many_dft_r2c(1, &sizeX, (int)ny, data, NULL, 1, nxpad, (fftwf_complex *)data, NULL, 1, nxpad / 2, FFTW_ESTIMATE);
	}
	else if (direction == 1)
	{
		plan = fftwf_plan_many_dft_c2r(1, &sizeX, (int)ny, (fftwf_complex *)data, NULL, 1, nxpad / 2, data, NULL, 1, nxpad, FFTW_ESTIMATE);
	}
	else
	{
		return;
	}

	fftwf_execute(plan);
	normalizeCpu(data, invScale, (size_t)nxpad * ny);
	fftwf_destroy_plan(plan);
}

}

void FFTRoutines::real1DTransform(size_t size, std::vector<float>& data, std::vector<float>& fft)
{
	fft.clear();

#ifdef USE_CUDA
	std::string error;
	if (shouldUseCuda(&error))
	{
		if (novaCTF::CudaFftRoutines::real1DTransform(size, data, fft, &error))
			return;
	}
	logCudaFallbackOnce(error);
#endif

	real1DTransformCpu(size, data, fft);
}

double FFTRoutines::logarithmizeValue(double value, bool logarithmizeData)
{
	return logarithmizeValueCpu(value, logarithmizeData);
}

void FFTRoutines::computePowerSpectrum(std::vector<float>& powerSpectrum, fftw_complex* fftOut, size_t sizeX, size_t nyh, bool logarithmizeData)
{
	computePowerSpectrumCpu(powerSpectrum, fftOut, sizeX, nyh, logarithmizeData);
}

void FFTRoutines::maskFFT(fftw_complex* fftOut, std::vector<double>& mask, size_t sizeX, size_t nyh)
{
	maskFFTCpu(fftOut, mask, sizeX, nyh);
}

void FFTRoutines::filterFFT(fftw_complex* fftOut, std::vector<float>& filter, size_t sizeX, size_t nyh)
{
	filterFFTCpu(fftOut, filter, sizeX, nyh);
}

void FFTRoutines::normalizeValues(std::vector<float>& normalizedData, std::vector<double>& originalData, size_t dataSize, novaCTF::DataStats& dataStats)
{
	normalizeValuesCpu(normalizedData, originalData, dataSize, dataStats);
}

void FFTRoutines::complex2DTransform(size_t sizeX, size_t sizeY, std::vector<float>& data, std::vector<float>& filter)
{
#ifdef USE_CUDA
	std::string error;
	if (shouldUseCuda(&error))
	{
		if (novaCTF::CudaFftRoutines::complex2DTransform(sizeX, sizeY, data, filter, &error))
			return;
	}
	logCudaFallbackOnce(error);
#endif

	complex2DTransformCpu(sizeX, sizeY, data, filter);
}

void FFTRoutines::real2DTransform(size_t sizeX, size_t sizeY, std::vector<float>& data, std::vector<float>& filter)
{
#ifdef USE_CUDA
	std::string error;
	if (shouldUseCuda(&error))
	{
		if (novaCTF::CudaFftRoutines::real2DTransform(sizeX, sizeY, data, filter, &error))
			return;
	}
	logCudaFallbackOnce(error);
#endif

	real2DTransformCpu(sizeX, sizeY, data, filter);
}

void FFTRoutines::real2DTransform(size_t sizeX, size_t sizeY, std::vector<float>& data, std::vector<double>& mask, novaCTF::DataStats& dataStats)
{
	real2DMaskTransformCpu(sizeX, sizeY, data, mask, dataStats);
}

void FFTRoutines::many1DTransform(float* data, int sizeX, int sizeY, int direction)
{
#ifdef USE_CUDA
	std::string error;
	if (shouldUseCuda(&error))
	{
		if (novaCTF::CudaFftRoutines::many1DTransform(data, sizeX, sizeY, direction, &error))
			return;
	}
	logCudaFallbackOnce(error);
#endif

	many1DTransformCpu(data, sizeX, sizeY, direction);
}

void FFTRoutines::normalize(float *array, float scale, size_t dataSize)
{
	for (size_t i = 0; i < dataSize; i++)
		array[i] *= scale;
}
