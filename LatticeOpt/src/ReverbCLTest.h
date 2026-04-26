#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#include "CL/cl.h"
#include "Eigen/Dense"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>


namespace ReverbCLTest
{
	constexpr int NumLayers = 6;
	constexpr int NumParams = 8 * NumLayers + 4;
	constexpr int NumTasks = 500;
	constexpr int TestBlockSize = 65536 * 4;
	constexpr int MaxDelay = 8192;
	constexpr int DelayLines = 14;
	constexpr int DelayScratchFloatsPerTask = DelayLines * MaxDelay;
	constexpr int LossScratchFloatsPerTask = 14 * TestBlockSize;
	constexpr int ScratchFloatsPerTask = DelayScratchFloatsPerTask + LossScratchFloatsPerTask;

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

	inline bool ParseKernelIntDefine(const std::string& source, const char* name, int& value)
	{
		const std::string needle = std::string("#define ") + name;
		size_t pos = source.find(needle);
		if (pos == std::string::npos)
			return false;

		pos += needle.size();
		while (pos < source.size() && (source[pos] == ' ' || source[pos] == '\t'))
			++pos;

		char* endPtr = nullptr;
		const long parsed = std::strtol(source.c_str() + pos, &endPtr, 10);
		if (endPtr == source.c_str() + pos || parsed <= 0)
			return false;

		value = (int)parsed;
		return true;
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
		size_t scratchFloatsPerTask = ScratchFloatsPerTask;
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

			int kernelTestBlockSize = TestBlockSize;
			int kernelMaxDelay = MaxDelay;
			ParseKernelIntDefine(source, "LR_TEST_BLOCK_SIZE", kernelTestBlockSize);
			ParseKernelIntDefine(source, "LR_MAX_DELAY", kernelMaxDelay);
			scratchFloatsPerTask = (size_t)DelayLines * (size_t)kernelMaxDelay + 14u * (size_t)kernelTestBlockSize;
			std::printf("OpenCL test: scratchPerTask=%zu floats (testBlock=%d maxDelay=%d)\n",
				scratchFloatsPerTask, kernelTestBlockSize, kernelMaxDelay);

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
			const size_t scratchBytes = (size_t)numTasks * scratchFloatsPerTask * sizeof(float);

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
			const size_t scratchBytes = (size_t)numTasks * scratchFloatsPerTask * sizeof(float);

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

	namespace ReverbCLOptimizer
	{
		constexpr int SampleRate = 48000;
		constexpr int CheckpointSeconds = 5;
		constexpr int CheckpointSamples = SampleRate * CheckpointSeconds;
		using CheckpointWriter = bool (*)(const std::vector<float>& normalizedRoomParams);

		struct RandomSearchConfig
		{
			int numTasks = NumTasks;
			int eliteCount = 64;
			unsigned int seed = 20260426u;
			float initialRadius = 2.5f;
			float minRadius = 0.01f;
			float radiusShrink = 0.985f;
			float eliteStdScale = 1.5f;
			float centerBlend = 0.72f;
			float rawClamp = 12.0f;
			int checkpointSamples = CheckpointSamples;
			CheckpointWriter checkpointWriter = nullptr;
		};

		inline float Soft01(float x, float minv)
		{
			float v = 1.0f - std::exp(-std::fabs(x));
			return v * (1.0f - minv) + minv;
		}

		inline float TanhNo0(float x, float minv)
		{
			float v = std::tanh(x);
			if (v > 0.0f)
				v = v * (1.0f - minv) + minv;
			else
				v = v * (1.0f - minv) - minv;
			return v;
		}

		inline void NormalizeParams(const std::vector<float>& raw, std::vector<float>& normalized)
		{
			normalized = raw;
			normalized.resize(NumParams, 0.0f);

			for (int i = 0; i < NumLayers; ++i)
			{
				normalized[i + 0 * NumLayers] = Soft01(normalized[i + 0 * NumLayers], 0.01f);
				normalized[i + 1 * NumLayers] = Soft01(normalized[i + 1 * NumLayers], 0.01f);
				normalized[i + 2 * NumLayers] = TanhNo0(normalized[i + 2 * NumLayers], 0.05f) * 0.9f;
				normalized[i + 3 * NumLayers] = TanhNo0(normalized[i + 3 * NumLayers], 0.05f) * 0.9f;
				normalized[i + 4 * NumLayers] = Soft01(normalized[i + 4 * NumLayers], 0.9995f) * 0.99999999f;
				normalized[i + 5 * NumLayers] = Soft01(normalized[i + 5 * NumLayers], 0.9995f) * 0.99999999f;
				normalized[i + 6 * NumLayers] = Soft01(normalized[i + 6 * NumLayers], 0.01f);
				normalized[i + 7 * NumLayers] = Soft01(normalized[i + 7 * NumLayers], 0.01f);
			}

			normalized[8 * NumLayers + 2] = Soft01(normalized[8 * NumLayers + 2], 0.01f);
			normalized[8 * NumLayers + 3] = Soft01(normalized[8 * NumLayers + 3], 0.01f);
			normalized[8 * NumLayers + 0] = -Soft01(normalized[8 * NumLayers + 0], 0.65f) * 0.99999999f;
			normalized[8 * NumLayers + 1] = -Soft01(normalized[8 * NumLayers + 1], 0.65f) * 0.99999999f;
		}

		inline short FloatToPcm16(float x)
		{
			if (x > 1.0f) x = 1.0f;
			if (x < -1.0f) x = -1.0f;
			int v = (int)std::lrint(x * 32767.0f);
			if (v > 32767) v = 32767;
			if (v < -32768) v = -32768;
			return (short)v;
		}

		inline void WriteLE16(FILE* fp, unsigned short v)
		{
			unsigned char b[2];
			b[0] = (unsigned char)(v & 0xFF);
			b[1] = (unsigned char)((v >> 8) & 0xFF);
			fwrite(b, 1, 2, fp);
		}

		inline void WriteLE32(FILE* fp, unsigned int v)
		{
			unsigned char b[4];
			b[0] = (unsigned char)(v & 0xFF);
			b[1] = (unsigned char)((v >> 8) & 0xFF);
			b[2] = (unsigned char)((v >> 16) & 0xFF);
			b[3] = (unsigned char)((v >> 24) & 0xFF);
			fwrite(b, 1, 4, fp);
		}

		inline bool SaveWav(const char* filename, const std::vector<float>& irL, const std::vector<float>& irR, int numSamples)
		{
			if ((int)irL.size() < numSamples || (int)irR.size() < numSamples)
				return false;

			FILE* fp = nullptr;
#if defined(_MSC_VER)
			if (fopen_s(&fp, filename, "wb") != 0) fp = nullptr;
#else
			fp = fopen(filename, "wb");
#endif
			if (!fp) return false;

			const unsigned short numChannels = 2;
			const unsigned short bitsPerSample = 16;
			const unsigned short blockAlign = (unsigned short)(numChannels * bitsPerSample / 8);
			const unsigned int byteRate = SampleRate * blockAlign;
			const unsigned int dataSize = (unsigned int)numSamples * blockAlign;
			const unsigned int riffSize = 36 + dataSize;

			fwrite("RIFF", 1, 4, fp);
			WriteLE32(fp, riffSize);
			fwrite("WAVE", 1, 4, fp);

			fwrite("fmt ", 1, 4, fp);
			WriteLE32(fp, 16);
			WriteLE16(fp, 1);
			WriteLE16(fp, numChannels);
			WriteLE32(fp, SampleRate);
			WriteLE32(fp, byteRate);
			WriteLE16(fp, blockAlign);
			WriteLE16(fp, bitsPerSample);

			fwrite("data", 1, 4, fp);
			WriteLE32(fp, dataSize);

			for (int i = 0; i < numSamples; ++i)
			{
				WriteLE16(fp, (unsigned short)FloatToPcm16(irL[i]));
				WriteLE16(fp, (unsigned short)FloatToPcm16(irR[i]));
			}

			fclose(fp);
			return true;
		}

		inline bool SaveParamsTxt(const char* filename, const std::vector<float>& params)
		{
			FILE* fp = nullptr;
#if defined(_MSC_VER)
			if (fopen_s(&fp, filename, "wb") != 0) fp = nullptr;
#else
			fp = fopen(filename, "wb");
#endif
			if (!fp) return false;

			fprintf(fp, "NumParams=%zu\n", params.size());
			for (float v : params)
				fprintf(fp, "%.9g\n", v);

			fclose(fp);
			return true;
		}

		inline bool SaveCheckpoint(LossBatchEvaluator& evaluator, const std::vector<float>& bestRaw, int numSamples)
		{
			std::vector<float> normalized;
			std::vector<float> irL;
			std::vector<float> irR;

			NormalizeParams(bestRaw, normalized);

			if (!evaluator.RenderIR(bestRaw.data(), 1, numSamples, irL, irR))
				return false;

			if (!SaveParamsTxt("checkpoint.txt", normalized))
				return false;

			if (!SaveWav("checkpoint.wav", irL, irR, numSamples))
				return false;

			return true;
		}

		inline float AverageRadius(const std::vector<float>& radius)
		{
			float sum = 0.0f;
			for (float v : radius)
				sum += v;
			return sum / (float)radius.size();
		}

		inline void PrintIterationStats(
			int iter,
			const std::vector<float>& losses,
			const std::vector<int>& order,
			int eliteCount,
			float bestEver,
			const std::vector<float>& radius,
			bool improved)
		{
			double mean = 0.0;
			float minLoss = losses[order.front()];
			float maxLoss = losses[order.back()];
			for (float v : losses)
				mean += (double)v;
			mean /= (double)losses.size();

			float median = losses[order[losses.size() / 2]];
			float p10 = losses[order[losses.size() / 10]];
			float p90 = losses[order[(losses.size() * 9) / 10]];

			double eliteMean = 0.0;
			for (int i = 0; i < eliteCount; ++i)
				eliteMean += (double)losses[order[i]];
			eliteMean /= (double)eliteCount;

			double variance = 0.0;
			for (float v : losses)
			{
				double d = (double)v - mean;
				variance += d * d;
			}
			variance /= (double)losses.size();

			const auto minmaxRadius = std::minmax_element(radius.begin(), radius.end());
			std::printf(
				"search %05d loss[min=%.6f p10=%.6f median=%.6f p90=%.6f max=%.6f mean=%.6f std=%.6f eliteMean=%.6f best=%.6f] radius[avg=%.6f min=%.6f max=%.6f]%s\n",
				iter,
				minLoss,
				p10,
				median,
				p90,
				maxLoss,
				(float)mean,
				(float)std::sqrt(variance),
				(float)eliteMean,
				bestEver,
				AverageRadius(radius),
				*minmaxRadius.first,
				*minmaxRadius.second,
				improved ? " checkpoint" : "");
		}

		class RandomSearchOptimizer
		{
		private:
			RandomSearchConfig cfg;
			LossBatchEvaluator evaluator;
			std::mt19937 rng;
			std::normal_distribution<float> normal;

			std::vector<float> center;
			std::vector<float> radius;
			std::vector<float> params;
			std::vector<float> losses;
			std::vector<float> bestRaw;
			float bestLoss = std::numeric_limits<float>::infinity();
			int iteration = 0;

		private:
			float ClampRaw(float v) const
			{
				if (v > cfg.rawClamp) return cfg.rawClamp;
				if (v < -cfg.rawClamp) return -cfg.rawClamp;
				return v;
			}

			void InitializeSearch()
			{
				center.assign(NumParams, 0.0f);
				radius.assign(NumParams, cfg.initialRadius);
				params.assign((size_t)cfg.numTasks * NumParams, 0.0f);
				losses.assign(cfg.numTasks, 0.0f);
				bestRaw.assign(NumParams, 0.0f);

				for (int i = 0; i < NumLayers; ++i)
				{
					center[i + 4 * NumLayers] = 2.0f;
					center[i + 5 * NumLayers] = 2.0f;
				}
				center[8 * NumLayers + 0] = 0.5f;
				center[8 * NumLayers + 1] = 0.5f;
			}

			void BuildCandidates()
			{
				for (int j = 0; j < NumParams; ++j)
				{
					params[j] = center[j];
					if (bestLoss < std::numeric_limits<float>::infinity())
						params[NumParams + j] = bestRaw[j];
				}

				const int randomStart = bestLoss < std::numeric_limits<float>::infinity() ? 2 : 1;
				for (int task = randomStart; task < cfg.numTasks; ++task)
				{
					float* dst = params.data() + (size_t)task * NumParams;
					for (int j = 0; j < NumParams; ++j)
						dst[j] = ClampRaw(center[j] + normal(rng) * radius[j]);
				}
			}

			void UpdateSearchDistribution(const std::vector<int>& order)
			{
				std::vector<float> eliteMean(NumParams, 0.0f);
				std::vector<float> eliteStd(NumParams, 0.0f);

				for (int e = 0; e < cfg.eliteCount; ++e)
				{
					const float* src = params.data() + (size_t)order[e] * NumParams;
					for (int j = 0; j < NumParams; ++j)
						eliteMean[j] += src[j];
				}
				for (float& v : eliteMean)
					v /= (float)cfg.eliteCount;

				for (int e = 0; e < cfg.eliteCount; ++e)
				{
					const float* src = params.data() + (size_t)order[e] * NumParams;
					for (int j = 0; j < NumParams; ++j)
					{
						const float d = src[j] - eliteMean[j];
						eliteStd[j] += d * d;
					}
				}

				for (int j = 0; j < NumParams; ++j)
				{
					eliteStd[j] = std::sqrt(eliteStd[j] / (float)(std::max)(1, cfg.eliteCount - 1));
					center[j] = ClampRaw(center[j] * (1.0f - cfg.centerBlend) + eliteMean[j] * cfg.centerBlend);

					const float targetRadius = (std::max)(cfg.minRadius, eliteStd[j] * cfg.eliteStdScale);
					const float shrinkRadius = (std::max)(cfg.minRadius, radius[j] * cfg.radiusShrink);
					radius[j] = (std::max)(cfg.minRadius, (std::min)(shrinkRadius, targetRadius));
				}
			}

			bool SaveBestCheckpoint()
			{
				if (cfg.checkpointWriter)
				{
					std::vector<float> normalized;
					NormalizeParams(bestRaw, normalized);
					if (!cfg.checkpointWriter(normalized))
					{
						std::printf("ReverbCLOptimizer: failed to write checkpoint.txt/checkpoint.wav\n");
						return false;
					}

					std::printf("ReverbCLOptimizer: saved checkpoint loss=%.9g\n", bestLoss);
					return true;
				}

				if (!SaveCheckpoint(evaluator, bestRaw, cfg.checkpointSamples))
				{
					std::printf("ReverbCLOptimizer: failed to write checkpoint.txt/checkpoint.wav\n");
					return false;
				}

				std::printf("ReverbCLOptimizer: saved checkpoint loss=%.9g\n", bestLoss);
				return true;
			}

		public:
			explicit RandomSearchOptimizer(RandomSearchConfig config = RandomSearchConfig())
				: cfg(config), rng(config.seed), normal(0.0f, 1.0f)
			{
				if (cfg.numTasks < 2)
					cfg.numTasks = 2;
				if (cfg.eliteCount < 2)
					cfg.eliteCount = 2;
				if (cfg.eliteCount > cfg.numTasks)
					cfg.eliteCount = cfg.numTasks;
				InitializeSearch();
			}

			bool Step()
			{
				BuildCandidates();

				if (!evaluator.Evaluate(params.data(), cfg.numTasks, losses))
					return false;

				for (float& loss : losses)
				{
					if (!std::isfinite(loss))
						loss = std::numeric_limits<float>::infinity();
				}

				std::vector<int> order(cfg.numTasks);
				for (int i = 0; i < cfg.numTasks; ++i)
					order[i] = i;
				std::sort(order.begin(), order.end(),
					[this](int a, int b)
					{
						return losses[a] < losses[b];
					});

				const int bestIndex = order.front();
				const float iterBest = losses[bestIndex];
				bool improved = false;

				if (std::isfinite(iterBest) && iterBest < bestLoss)
				{
					bestLoss = iterBest;
					bestRaw.assign(
						params.begin() + (size_t)bestIndex * NumParams,
						params.begin() + (size_t)(bestIndex + 1) * NumParams);
					center = bestRaw;
					improved = true;
					if (!SaveBestCheckpoint())
						return false;
				}

				UpdateSearchDistribution(order);
				PrintIterationStats(iteration, losses, order, cfg.eliteCount, bestLoss, radius, improved);
				++iteration;
				return true;
			}

			bool RunForever()
			{
				while (Step())
				{
				}
				return false;
			}
		};

		inline bool RunRandomSearchForever(RandomSearchConfig config = RandomSearchConfig())
		{
			RandomSearchOptimizer optimizer(config);
			return optimizer.RunForever();
		}

		struct CMAConfig
		{
			int numTasks = NumTasks;
			int eliteCount = 0;
			unsigned int seed = 20260426u;
			float initialSigma = 2.0f;
			float minSigma = 0.001f;
			float maxSigma = 12.0f;
			float rawClamp = 12.0f;
			int checkpointSamples = CheckpointSamples;
			int eigenUpdateEvery = 1;
			CheckpointWriter checkpointWriter = nullptr;
		};

		class CMAOptimizer
		{
		private:
			using Vec = Eigen::VectorXf;
			using Mat = Eigen::MatrixXf;

			CMAConfig cfg;
			LossBatchEvaluator evaluator;
			std::mt19937 rng;
			std::normal_distribution<float> normal;

			Vec mean;
			Mat cov;
			Mat eigenBasis;
			Vec axisScale;
			Mat invSqrtCov;
			Vec evolutionCov;
			Vec evolutionSigma;
			Vec weights;

			std::vector<float> params;
			std::vector<float> losses;
			std::vector<float> bestRaw;
			std::vector<float> radius;

			float sigma = 1.0f;
			float mueff = 1.0f;
			float cc = 0.0f;
			float cs = 0.0f;
			float c1 = 0.0f;
			float cmu = 0.0f;
			float damps = 0.0f;
			float chiN = 1.0f;
			float bestLoss = std::numeric_limits<float>::infinity();
			int iteration = 0;

		private:
			float ClampRaw(float v) const
			{
				if (v > cfg.rawClamp) return cfg.rawClamp;
				if (v < -cfg.rawClamp) return -cfg.rawClamp;
				return v;
			}

			void InitializeMean()
			{
				mean = Vec::Zero(NumParams);
				for (int i = 0; i < NumLayers; ++i)
				{
					mean[i + 4 * NumLayers] = 2.0f;
					mean[i + 5 * NumLayers] = 2.0f;
				}
				mean[8 * NumLayers + 0] = 0.5f;
				mean[8 * NumLayers + 1] = 0.5f;
			}

			void InitializeWeightsAndRates()
			{
				const int n = NumParams;
				if (cfg.eliteCount <= 0)
					cfg.eliteCount = cfg.numTasks / 2;
				if (cfg.eliteCount < 2)
					cfg.eliteCount = 2;
				if (cfg.eliteCount > cfg.numTasks)
					cfg.eliteCount = cfg.numTasks;

				weights = Vec::Zero(cfg.eliteCount);
				float sumWeights = 0.0f;
				for (int i = 0; i < cfg.eliteCount; ++i)
				{
					weights[i] = std::log((float)cfg.eliteCount + 0.5f) - std::log((float)i + 1.0f);
					sumWeights += weights[i];
				}
				weights /= sumWeights;
				mueff = 1.0f / weights.squaredNorm();

				const float nf = (float)n;
				cc = (4.0f + mueff / nf) / (nf + 4.0f + 2.0f * mueff / nf);
				cs = (mueff + 2.0f) / (nf + mueff + 5.0f);
				c1 = 2.0f / (((nf + 1.3f) * (nf + 1.3f)) + mueff);
				cmu = (std::min)(1.0f - c1, 2.0f * (mueff - 2.0f + 1.0f / mueff) / (((nf + 2.0f) * (nf + 2.0f)) + mueff));
				if (cmu < 0.0f)
					cmu = 0.0f;
				damps = 1.0f + 2.0f * (std::max)(0.0f, std::sqrt((mueff - 1.0f) / (nf + 1.0f)) - 1.0f) + cs;
				chiN = std::sqrt(nf) * (1.0f - 1.0f / (4.0f * nf) + 1.0f / (21.0f * nf * nf));
			}

			void InitializeSearch()
			{
				InitializeWeightsAndRates();
				InitializeMean();

				sigma = cfg.initialSigma;
				cov = Mat::Identity(NumParams, NumParams);
				eigenBasis = Mat::Identity(NumParams, NumParams);
				axisScale = Vec::Ones(NumParams);
				invSqrtCov = Mat::Identity(NumParams, NumParams);
				evolutionCov = Vec::Zero(NumParams);
				evolutionSigma = Vec::Zero(NumParams);

				params.assign((size_t)cfg.numTasks * NumParams, 0.0f);
				losses.assign(cfg.numTasks, 0.0f);
				bestRaw.assign(NumParams, 0.0f);
				radius.assign(NumParams, sigma);
			}

			void CopyVectorToCandidate(int task, const Vec& src)
			{
				float* dst = params.data() + (size_t)task * NumParams;
				for (int j = 0; j < NumParams; ++j)
					dst[j] = ClampRaw(src[j]);
			}

			void BuildCandidates()
			{
				CopyVectorToCandidate(0, mean);

				int randomStart = 1;
				if (bestLoss < std::numeric_limits<float>::infinity())
				{
					for (int j = 0; j < NumParams; ++j)
						params[NumParams + j] = bestRaw[j];
					randomStart = 2;
				}

				for (int task = randomStart; task < cfg.numTasks; ++task)
				{
					Vec z(NumParams);
					for (int j = 0; j < NumParams; ++j)
						z[j] = normal(rng);

					Vec y = eigenBasis * axisScale.asDiagonal() * z;
					Vec x = mean + sigma * y;
					CopyVectorToCandidate(task, x);
				}
			}

			Vec CandidateVector(int task) const
			{
				Vec v(NumParams);
				const float* src = params.data() + (size_t)task * NumParams;
				for (int j = 0; j < NumParams; ++j)
					v[j] = src[j];
				return v;
			}

			void UpdateAxisRadius()
			{
				for (int j = 0; j < NumParams; ++j)
					radius[j] = sigma * std::sqrt((std::max)(0.0f, cov(j, j)));
			}

			void UpdateEigensystem()
			{
				cov = 0.5f * (cov + cov.transpose()).eval();

				Eigen::SelfAdjointEigenSolver<Mat> solver(cov);
				if (solver.info() != Eigen::Success)
				{
					cov += Mat::Identity(NumParams, NumParams) * 1.0e-6f;
					solver.compute(cov);
				}

				Vec eigenValues = solver.eigenvalues();
				for (int i = 0; i < NumParams; ++i)
				{
					if (!std::isfinite(eigenValues[i]) || eigenValues[i] < 1.0e-12f)
						eigenValues[i] = 1.0e-12f;
				}

				eigenBasis = solver.eigenvectors();
				axisScale = eigenValues.cwiseSqrt();
				Vec invAxis = axisScale.cwiseInverse();
				invSqrtCov = eigenBasis * invAxis.asDiagonal() * eigenBasis.transpose();
				cov = eigenBasis * eigenValues.asDiagonal() * eigenBasis.transpose();
				cov = 0.5f * (cov + cov.transpose()).eval();
				UpdateAxisRadius();
			}

			void UpdateDistribution(const std::vector<int>& order)
			{
				const Vec oldMean = mean;
				mean.setZero();
				for (int e = 0; e < cfg.eliteCount; ++e)
					mean += weights[e] * CandidateVector(order[e]);
				for (int j = 0; j < NumParams; ++j)
					mean[j] = ClampRaw(mean[j]);

				const Vec yWeighted = (mean - oldMean) / sigma;
				const float csFactor = std::sqrt(cs * (2.0f - cs) * mueff);
				evolutionSigma = (1.0f - cs) * evolutionSigma + csFactor * (invSqrtCov * yWeighted);

				const float normSigma = evolutionSigma.norm();
				const float correction = std::sqrt(1.0f - std::pow(1.0f - cs, 2.0f * (float)(iteration + 1)));
				const float hsigLimit = (1.4f + 2.0f / ((float)NumParams + 1.0f)) * chiN;
				const bool hsig = correction > 0.0f && normSigma / correction < hsigLimit;

				const float ccFactor = std::sqrt(cc * (2.0f - cc) * mueff);
				evolutionCov = (1.0f - cc) * evolutionCov + (hsig ? ccFactor : 0.0f) * yWeighted;

				Mat rankMu = Mat::Zero(NumParams, NumParams);
				for (int e = 0; e < cfg.eliteCount; ++e)
				{
					const Vec y = (CandidateVector(order[e]) - oldMean) / sigma;
					rankMu += weights[e] * (y * y.transpose());
				}

				const float oldCovScale = 1.0f - c1 - cmu + (hsig ? 0.0f : c1 * cc * (2.0f - cc));
				cov = oldCovScale * cov + c1 * (evolutionCov * evolutionCov.transpose()) + cmu * rankMu;
				cov = 0.5f * (cov + cov.transpose()).eval();

				sigma *= std::exp((cs / damps) * (normSigma / chiN - 1.0f));
				if (!std::isfinite(sigma))
					sigma = cfg.initialSigma;
				if (sigma < cfg.minSigma)
					sigma = cfg.minSigma;
				if (sigma > cfg.maxSigma)
					sigma = cfg.maxSigma;

				if (cfg.eigenUpdateEvery < 1)
					cfg.eigenUpdateEvery = 1;
				if ((iteration % cfg.eigenUpdateEvery) == 0)
					UpdateEigensystem();
				else
					UpdateAxisRadius();
			}

			void PrintCMAState() const
			{
				const float minAxis = axisScale.minCoeff();
				const float maxAxis = axisScale.maxCoeff();
				const float cond = minAxis > 0.0f ? maxAxis / minAxis : std::numeric_limits<float>::infinity();
				std::printf(
					"cma    %05d sigma=%.6f axis[min=%.6f max=%.6f cond=%.6f] ps=%.6f\n",
					iteration,
					sigma,
					minAxis,
					maxAxis,
					cond,
					evolutionSigma.norm());
			}

			bool SaveBestCheckpoint()
			{
				if (cfg.checkpointWriter)
				{
					std::vector<float> normalized;
					NormalizeParams(bestRaw, normalized);
					if (!cfg.checkpointWriter(normalized))
					{
						std::printf("CMAOptimizer: failed to write checkpoint.txt/checkpoint.wav\n");
						return false;
					}

					std::printf("CMAOptimizer: saved checkpoint loss=%.9g\n", bestLoss);
					return true;
				}

				if (!SaveCheckpoint(evaluator, bestRaw, cfg.checkpointSamples))
				{
					std::printf("CMAOptimizer: failed to write checkpoint.txt/checkpoint.wav\n");
					return false;
				}

				std::printf("CMAOptimizer: saved checkpoint loss=%.9g\n", bestLoss);
				return true;
			}

		public:
			explicit CMAOptimizer(CMAConfig config = CMAConfig())
				: cfg(config), rng(config.seed), normal(0.0f, 1.0f)
			{
				if (cfg.numTasks < 4)
					cfg.numTasks = 4;
				if (cfg.initialSigma <= 0.0f)
					cfg.initialSigma = 1.0f;
				if (cfg.minSigma <= 0.0f)
					cfg.minSigma = 0.001f;
				if (cfg.maxSigma < cfg.minSigma)
					cfg.maxSigma = cfg.minSigma;
				InitializeSearch();
			}

			bool Step()
			{
				BuildCandidates();

				if (!evaluator.Evaluate(params.data(), cfg.numTasks, losses))
					return false;

				for (float& loss : losses)
				{
					if (!std::isfinite(loss))
						loss = std::numeric_limits<float>::infinity();
				}

				std::vector<int> order(cfg.numTasks);
				for (int i = 0; i < cfg.numTasks; ++i)
					order[i] = i;
				std::sort(order.begin(), order.end(),
					[this](int a, int b)
					{
						return losses[a] < losses[b];
					});

				const int bestIndex = order.front();
				const float iterBest = losses[bestIndex];
				bool improved = false;

				if (std::isfinite(iterBest) && iterBest < bestLoss)
				{
					bestLoss = iterBest;
					bestRaw.assign(
						params.begin() + (size_t)bestIndex * NumParams,
						params.begin() + (size_t)(bestIndex + 1) * NumParams);
					improved = true;
					if (!SaveBestCheckpoint())
						return false;
				}

				UpdateDistribution(order);
				PrintIterationStats(iteration, losses, order, cfg.eliteCount, bestLoss, radius, improved);
				PrintCMAState();
				++iteration;
				return true;
			}

			bool RunForever()
			{
				while (Step())
				{
				}
				return false;
			}
		};

		inline bool RunCMAForever(CMAConfig config = CMAConfig())
		{
			CMAOptimizer optimizer(config);
			return optimizer.RunForever();
		}
	}
}
