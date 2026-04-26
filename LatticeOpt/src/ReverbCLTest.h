#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#include "CL/cl.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace ReverbCLTest
{
	constexpr int NumLayers = 6;
	constexpr int NumParams = 8 * NumLayers + 4;
	constexpr int NumTasks = 1000;
	constexpr int MaxDelay = 8192;
	constexpr int DelayLines = 14;
	constexpr int ScratchFloatsPerTask = DelayLines * MaxDelay;

	inline bool ReadTextFile(const char* path, std::string& out)
	{
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in)
			return false;

		std::ostringstream ss;
		ss << in.rdbuf();
		out = ss.str();
		return true;
	}

	inline bool ReadKernelSource(std::string& out)
	{
		const char* paths[] =
		{
			"src/LatticeReverb.cl",
			"LatticeOpt/src/LatticeReverb.cl",
			"../src/LatticeReverb.cl",
			"../../src/LatticeReverb.cl",
			"../../../src/LatticeReverb.cl"
		};

		for (const char* path : paths)
		{
			if (ReadTextFile(path, out))
				return true;
		}

		std::printf("OpenCL test: failed to find LatticeReverb.cl\n");
		return false;
	}

	inline const char* DeviceTypeName(cl_device_type type)
	{
		if (type & CL_DEVICE_TYPE_GPU)
			return "GPU";
		if (type & CL_DEVICE_TYPE_CPU)
			return "CPU";
		if (type & CL_DEVICE_TYPE_ACCELERATOR)
			return "ACCELERATOR";
		return "OTHER";
	}

	class LossBatchEvaluator
	{
	private:
		cl_platform_id platform = nullptr;
		cl_device_id device = nullptr;
		cl_context context = nullptr;
		cl_command_queue queue = nullptr;
		cl_program program = nullptr;
		cl_kernel kernel = nullptr;
		cl_kernel renderKernel = nullptr;
		bool initialized = false;

	private:
		void Release()
		{
			if (renderKernel) clReleaseKernel(renderKernel);
			if (kernel) clReleaseKernel(kernel);
			if (program) clReleaseProgram(program);
			if (queue) clReleaseCommandQueue(queue);
			if (context) clReleaseContext(context);

			renderKernel = nullptr;
			kernel = nullptr;
			program = nullptr;
			queue = nullptr;
			context = nullptr;
			device = nullptr;
			platform = nullptr;
			initialized = false;
		}

		bool PickDevice()
		{
			cl_uint numPlatforms = 0;
			cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
			if (err != CL_SUCCESS || numPlatforms == 0)
			{
				std::printf("OpenCL test: no platform found (err=%d)\n", err);
				return false;
			}

			std::vector<cl_platform_id> platforms(numPlatforms);
			err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
			if (err != CL_SUCCESS)
			{
				std::printf("OpenCL test: clGetPlatformIDs failed (err=%d)\n", err);
				return false;
			}

			const cl_device_type preferredTypes[] = { CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU };
			for (cl_device_type preferredType : preferredTypes)
			{
				for (cl_platform_id p : platforms)
				{
					cl_uint numDevices = 0;
					err = clGetDeviceIDs(p, preferredType, 0, nullptr, &numDevices);
					if (err != CL_SUCCESS || numDevices == 0)
						continue;

					std::vector<cl_device_id> devices(numDevices);
					err = clGetDeviceIDs(p, preferredType, numDevices, devices.data(), nullptr);
					if (err != CL_SUCCESS || devices.empty())
						continue;

					platform = p;
					device = devices[0];
					return true;
				}
			}

			std::printf("OpenCL test: no GPU/CPU device found\n");
			return false;
		}

		void PrintDeviceInfo()
		{
			char name[256] = {};
			char platformName[256] = {};
			cl_device_type type = 0;
			clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
			clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
			clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
			std::printf("OpenCL test: using %s device: %s / %s\n", DeviceTypeName(type), platformName, name);
		}

		bool BuildProgram()
		{
			std::string source;
			if (!ReadKernelSource(source))
				return false;

			const char* src = source.c_str();
			size_t len = source.size();
			cl_int err = CL_SUCCESS;
			program = clCreateProgramWithSource(context, 1, &src, &len, &err);
			if (err != CL_SUCCESS || !program)
			{
				std::printf("OpenCL test: clCreateProgramWithSource failed (err=%d)\n", err);
				return false;
			}

			const char* options = "-cl-std=CL1.2";
			err = clBuildProgram(program, 1, &device, options, nullptr, nullptr);
			if (err != CL_SUCCESS)
			{
				size_t logSize = 0;
				clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
				std::vector<char> log(logSize + 1, '\0');
				if (logSize > 0)
					clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
				std::printf("OpenCL test: clBuildProgram failed (err=%d)\n%s\n", err, log.data());
				return false;
			}

			kernel = clCreateKernel(program, "EvalLatticeLossBatch", &err);
			if (err != CL_SUCCESS || !kernel)
			{
				std::printf("OpenCL test: clCreateKernel failed (err=%d)\n", err);
				return false;
			}

			renderKernel = clCreateKernel(program, "RenderLatticeIRBatch", &err);
			if (err != CL_SUCCESS || !renderKernel)
			{
				std::printf("OpenCL test: clCreateKernel(RenderLatticeIRBatch) failed (err=%d)\n", err);
				return false;
			}

			return true;
		}

	public:
		~LossBatchEvaluator()
		{
			Release();
		}

		bool Initialize()
		{
			if (initialized)
				return true;

			if (!PickDevice())
				return false;

			PrintDeviceInfo();

			cl_int err = CL_SUCCESS;
			context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
			if (err != CL_SUCCESS || !context)
			{
				std::printf("OpenCL test: clCreateContext failed (err=%d)\n", err);
				Release();
				return false;
			}

			queue = clCreateCommandQueue(context, device, 0, &err);
			if (err != CL_SUCCESS || !queue)
			{
				std::printf("OpenCL test: clCreateCommandQueue failed (err=%d)\n", err);
				Release();
				return false;
			}

			if (!BuildProgram())
			{
				Release();
				return false;
			}

			initialized = true;
			return true;
		}

		bool Evaluate(const float* params, int numTasks, std::vector<float>& losses)
		{
			if (!Initialize())
				return false;

			if (!params || numTasks <= 0)
				return false;

			losses.assign(numTasks, 0.0f);

			const size_t paramsBytes = (size_t)numTasks * NumParams * sizeof(float);
			const size_t lossesBytes = (size_t)numTasks * sizeof(float);
			const size_t scratchBytes = (size_t)numTasks * ScratchFloatsPerTask * sizeof(float);

			cl_int err = CL_SUCCESS;
			cl_mem paramsBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, paramsBytes, (void*)params, &err);
			if (err != CL_SUCCESS || !paramsBuf)
			{
				std::printf("OpenCL test: params buffer allocation failed (err=%d)\n", err);
				return false;
			}

			cl_mem lossesBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, lossesBytes, nullptr, &err);
			if (err != CL_SUCCESS || !lossesBuf)
			{
				std::printf("OpenCL test: losses buffer allocation failed (err=%d)\n", err);
				clReleaseMemObject(paramsBuf);
				return false;
			}

			cl_mem scratchBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, scratchBytes, nullptr, &err);
			if (err != CL_SUCCESS || !scratchBuf)
			{
				std::printf("OpenCL test: scratch buffer allocation failed (err=%d, bytes=%zu)\n", err, scratchBytes);
				clReleaseMemObject(lossesBuf);
				clReleaseMemObject(paramsBuf);
				return false;
			}

			err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &paramsBuf);
			err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &lossesBuf);
			err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &scratchBuf);
			err |= clSetKernelArg(kernel, 3, sizeof(int), &numTasks);
			if (err != CL_SUCCESS)
			{
				std::printf("OpenCL test: clSetKernelArg failed (err=%d)\n", err);
				clReleaseMemObject(scratchBuf);
				clReleaseMemObject(lossesBuf);
				clReleaseMemObject(paramsBuf);
				return false;
			}

			const size_t globalSize = (size_t)numTasks;
			err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
			if (err == CL_SUCCESS)
				err = clFinish(queue);
			if (err != CL_SUCCESS)
			{
				std::printf("OpenCL test: kernel launch failed (err=%d)\n", err);
				clReleaseMemObject(scratchBuf);
				clReleaseMemObject(lossesBuf);
				clReleaseMemObject(paramsBuf);
				return false;
			}

			err = clEnqueueReadBuffer(queue, lossesBuf, CL_TRUE, 0, lossesBytes, losses.data(), 0, nullptr, nullptr);
			if (err != CL_SUCCESS)
			{
				std::printf("OpenCL test: readback failed (err=%d)\n", err);
				clReleaseMemObject(scratchBuf);
				clReleaseMemObject(lossesBuf);
				clReleaseMemObject(paramsBuf);
				return false;
			}

			clReleaseMemObject(scratchBuf);
			clReleaseMemObject(lossesBuf);
			clReleaseMemObject(paramsBuf);
			return true;
		}

		bool RenderIR(const float* params, int numTasks, int numSamples, std::vector<float>& irL, std::vector<float>& irR)
		{
			if (!Initialize())
				return false;

			if (!params || numTasks <= 0 || numSamples <= 0)
				return false;

			irL.assign((size_t)numTasks * numSamples, 0.0f);
			irR.assign((size_t)numTasks * numSamples, 0.0f);

			const size_t paramsBytes = (size_t)numTasks * NumParams * sizeof(float);
			const size_t irBytes = (size_t)numTasks * numSamples * sizeof(float);
			const size_t scratchBytes = (size_t)numTasks * ScratchFloatsPerTask * sizeof(float);

			cl_int err = CL_SUCCESS;
			cl_mem paramsBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, paramsBytes, (void*)params, &err);
			if (err != CL_SUCCESS || !paramsBuf)
			{
				std::printf("OpenCL test: render params buffer allocation failed (err=%d)\n", err);
				return false;
			}

			cl_mem outLBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, irBytes, nullptr, &err);
			if (err != CL_SUCCESS || !outLBuf)
			{
				std::printf("OpenCL test: render left buffer allocation failed (err=%d)\n", err);
				clReleaseMemObject(paramsBuf);
				return false;
			}

			cl_mem outRBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, irBytes, nullptr, &err);
			if (err != CL_SUCCESS || !outRBuf)
			{
				std::printf("OpenCL test: render right buffer allocation failed (err=%d)\n", err);
				clReleaseMemObject(outLBuf);
				clReleaseMemObject(paramsBuf);
				return false;
			}

			cl_mem scratchBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, scratchBytes, nullptr, &err);
			if (err != CL_SUCCESS || !scratchBuf)
			{
				std::printf("OpenCL test: render scratch buffer allocation failed (err=%d, bytes=%zu)\n", err, scratchBytes);
				clReleaseMemObject(outRBuf);
				clReleaseMemObject(outLBuf);
				clReleaseMemObject(paramsBuf);
				return false;
			}

			err = clSetKernelArg(renderKernel, 0, sizeof(cl_mem), &paramsBuf);
			err |= clSetKernelArg(renderKernel, 1, sizeof(cl_mem), &outLBuf);
			err |= clSetKernelArg(renderKernel, 2, sizeof(cl_mem), &outRBuf);
			err |= clSetKernelArg(renderKernel, 3, sizeof(cl_mem), &scratchBuf);
			err |= clSetKernelArg(renderKernel, 4, sizeof(int), &numTasks);
			err |= clSetKernelArg(renderKernel, 5, sizeof(int), &numSamples);
			if (err != CL_SUCCESS)
			{
				std::printf("OpenCL test: render clSetKernelArg failed (err=%d)\n", err);
				clReleaseMemObject(scratchBuf);
				clReleaseMemObject(outRBuf);
				clReleaseMemObject(outLBuf);
				clReleaseMemObject(paramsBuf);
				return false;
			}

			const size_t globalSize = (size_t)numTasks;
			err = clEnqueueNDRangeKernel(queue, renderKernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
			if (err == CL_SUCCESS)
				err = clFinish(queue);
			if (err != CL_SUCCESS)
			{
				std::printf("OpenCL test: render kernel launch failed (err=%d)\n", err);
				clReleaseMemObject(scratchBuf);
				clReleaseMemObject(outRBuf);
				clReleaseMemObject(outLBuf);
				clReleaseMemObject(paramsBuf);
				return false;
			}

			err = clEnqueueReadBuffer(queue, outLBuf, CL_TRUE, 0, irBytes, irL.data(), 0, nullptr, nullptr);
			if (err == CL_SUCCESS)
				err = clEnqueueReadBuffer(queue, outRBuf, CL_TRUE, 0, irBytes, irR.data(), 0, nullptr, nullptr);
			if (err != CL_SUCCESS)
			{
				std::printf("OpenCL test: render readback failed (err=%d)\n", err);
				clReleaseMemObject(scratchBuf);
				clReleaseMemObject(outRBuf);
				clReleaseMemObject(outLBuf);
				clReleaseMemObject(paramsBuf);
				return false;
			}

			clReleaseMemObject(scratchBuf);
			clReleaseMemObject(outRBuf);
			clReleaseMemObject(outLBuf);
			clReleaseMemObject(paramsBuf);
			return true;
		}
	};

	inline void MakeRandomParams(std::vector<float>& params, int numTasks)
	{
		params.resize((size_t)numTasks * NumParams);
		std::mt19937 rng(12345u);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		for (float& v : params)
			v = dist(rng);
	}

	inline bool EvaluateLossBatch(const std::vector<float>& params, std::vector<float>& losses, int numTasks = NumTasks)
	{
		if ((int)params.size() < numTasks * NumParams)
		{
			std::printf("OpenCL test: params vector is too small\n");
			return false;
		}

		LossBatchEvaluator evaluator;
		return evaluator.Evaluate(params.data(), numTasks, losses);
	}

	template<typename CpuRenderFunc>
	bool RunRandomIRSelfTest(CpuRenderFunc cpuRenderFunc, int numTasks = NumTasks, int numSamples = 512)
	{
		constexpr int NumProbeTasks = 16;
		constexpr float AbsTolerance = 1e-3f;
		constexpr float RelTolerance = 1e-4f;

		std::vector<float> params;
		std::vector<float> gpuL;
		std::vector<float> gpuR;
		MakeRandomParams(params, numTasks);

		LossBatchEvaluator evaluator;
		if (!evaluator.RenderIR(params.data(), numTasks, numSamples, gpuL, gpuR))
			return false;

		float globalMaxAbs = 0.0f;
		float globalMaxRel = 0.0f;
		float globalMeanAbs = 0.0f;
		int globalCount = 0;
		int worstTask = 0;
		int worstSample = 0;
		char worstChannel = 'L';
		float worstCpu = 0.0f;
		float worstGpu = 0.0f;

		const int numProbes = (std::min)(NumProbeTasks, numTasks);
		for (int task = 0; task < numProbes; ++task)
		{
			std::vector<float> oneParams(NumParams);
			for (int i = 0; i < NumParams; ++i)
				oneParams[i] = params[(size_t)task * NumParams + i];

			std::vector<float> cpuL;
			std::vector<float> cpuR;
			cpuRenderFunc(oneParams, cpuL, cpuR, numSamples);

			float taskMaxAbs = 0.0f;
			float taskMeanAbs = 0.0f;
			float taskMaxRel = 0.0f;

			for (int i = 0; i < numSamples; ++i)
			{
				const float cpuVals[2] = { cpuL[i], cpuR[i] };
				const float gpuVals[2] =
				{
					gpuL[(size_t)task * numSamples + i],
					gpuR[(size_t)task * numSamples + i]
				};

				for (int ch = 0; ch < 2; ++ch)
				{
					const float absDiff = std::fabs(cpuVals[ch] - gpuVals[ch]);
					const float relDiff = absDiff / (std::max)(1e-6f, std::fabs(cpuVals[ch]));

					taskMaxAbs = (std::max)(taskMaxAbs, absDiff);
					taskMeanAbs += absDiff;
					taskMaxRel = (std::max)(taskMaxRel, relDiff);
					globalMeanAbs += absDiff;
					++globalCount;

					if (absDiff > globalMaxAbs)
					{
						globalMaxAbs = absDiff;
						globalMaxRel = relDiff;
						worstTask = task;
						worstSample = i;
						worstChannel = ch == 0 ? 'L' : 'R';
						worstCpu = cpuVals[ch];
						worstGpu = gpuVals[ch];
					}
				}
			}

			taskMeanAbs /= (float)(std::max)(1, numSamples * 2);
			std::printf("OpenCL IR test: task=%d maxAbs=%.9g meanAbs=%.9g maxRel=%.9g\n",
				task, taskMaxAbs, taskMeanAbs, taskMaxRel);
		}

		globalMeanAbs /= (float)(std::max)(1, globalCount);
		const bool closeEnough = globalMaxAbs <= AbsTolerance || globalMaxRel <= RelTolerance;

		std::printf("OpenCL IR test: probed=%d samples=%d globalMaxAbs=%.9g globalMeanAbs=%.9g worstRel=%.9g tolerance(abs=%.9g rel=%.9g)\n",
			numProbes, numSamples, globalMaxAbs, globalMeanAbs, globalMaxRel, AbsTolerance, RelTolerance);
		std::printf("OpenCL IR test: worst task=%d sample=%d ch=%c CPU=%.9g GPU=%.9g diff=%.9g\n",
			worstTask, worstSample, worstChannel, worstCpu, worstGpu, globalMaxAbs);

		if (!closeEnough)
		{
			std::printf("OpenCL IR test: CPU/GPU IR mismatch is larger than tolerance\n");
			return false;
		}

		return true;
	}

	template<typename CpuLossFunc>
	bool RunRandomBatchSelfTest(CpuLossFunc cpuLossFunc, int numTasks = NumTasks)
	{
		constexpr int NumProbeTasks = 16;
		constexpr float AbsTolerance = 2e-2f;
		constexpr float RelTolerance = 2e-4f;

		std::vector<float> params;
		std::vector<float> losses;
		MakeRandomParams(params, numTasks);

		LossBatchEvaluator evaluator;
		if (!evaluator.Evaluate(params.data(), numTasks, losses))
			return false;

		float maxAbsDiff = 0.0f;
		float maxRelDiff = 0.0f;
		float sumAbsDiff = 0.0f;
		float firstCpuLoss = 0.0f;
		float firstGpuLoss = 0.0f;
		float firstAbsDiff = 0.0f;
		float firstRelDiff = 0.0f;
		bool bitExact = true;
		bool finite = true;

		const int numProbes = (std::min)(NumProbeTasks, numTasks);
		for (int task = 0; task < numProbes; ++task)
		{
			std::vector<float> oneParams(NumParams);
			for (int i = 0; i < NumParams; ++i)
				oneParams[i] = params[(size_t)task * NumParams + i];

			const float cpuLoss = cpuLossFunc(oneParams);
			const float gpuLoss = losses[task];
			const float absDiff = std::fabs(cpuLoss - gpuLoss);
			const float relDiff = absDiff / (std::max)(1.0f, std::fabs(cpuLoss));

			if (task == 0)
			{
				firstCpuLoss = cpuLoss;
				firstGpuLoss = gpuLoss;
				firstAbsDiff = absDiff;
				firstRelDiff = relDiff;
			}

			if (cpuLoss != gpuLoss)
				bitExact = false;
			if (!std::isfinite(cpuLoss) || !std::isfinite(gpuLoss))
				finite = false;

			maxAbsDiff = (std::max)(maxAbsDiff, absDiff);
			maxRelDiff = (std::max)(maxRelDiff, relDiff);
			sumAbsDiff += absDiff;
		}

		const float meanAbsDiff = sumAbsDiff / (float)(std::max)(1, numProbes);
		const bool closeEnough = finite && (maxAbsDiff <= AbsTolerance || maxRelDiff <= RelTolerance);

		std::printf("OpenCL test: numTasks=%d\n", numTasks);
		std::printf("OpenCL test: CPU loss[0]=%.9g GPU loss[0]=%.9g absDiff=%.9g relDiff=%.9g %s\n",
			firstCpuLoss, firstGpuLoss, firstAbsDiff, firstRelDiff, bitExact ? "(bit-exact)" : "");
		std::printf("OpenCL test: probed=%d maxAbsDiff=%.9g meanAbsDiff=%.9g maxRelDiff=%.9g tolerance(abs=%.9g rel=%.9g)\n",
			numProbes, maxAbsDiff, meanAbsDiff, maxRelDiff, AbsTolerance, RelTolerance);

		if (!closeEnough)
		{
			std::printf("OpenCL test: CPU/GPU loss mismatch is larger than tolerance\n");
			return false;
		}

		return true;
	}
}
