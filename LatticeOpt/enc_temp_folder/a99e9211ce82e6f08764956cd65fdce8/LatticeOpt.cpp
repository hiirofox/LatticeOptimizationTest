#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

#include "optimizer.h"

#include <windows.h>

void fft(float re[], float im[], int N, int inv)
{
	int i, j, k, m;
	int len = N;
	j = 0;
	for (i = 1; i < len; i++) {
		int bit = len >> 1;
		while (j & bit) {
			j ^= bit;
			bit >>= 1;
		}
		j ^= bit;
		if (i < j) {
			float tmp;
			tmp = re[i]; re[i] = re[j]; re[j] = tmp;
			tmp = im[i]; im[i] = im[j]; im[j] = tmp;
		}
	}
	for (m = 2; m <= len; m <<= 1) {
		float angle = -inv * 2.0f * 3.1415926535897932384626f / m;
		float wm_re = cosf(angle);
		float wm_im = sinf(angle);
		for (k = 0; k < len; k += m) {
			float w_re = 1.0f;
			float w_im = 0.0f;

			for (j = 0; j < m / 2; j++) {
				int t = k + j;
				int u = t + m / 2;
				float tr = w_re * re[u] - w_im * im[u];
				float ti = w_re * im[u] + w_im * re[u];
				float ur = re[t];
				float ui = im[t];
				re[t] = ur + tr;
				im[t] = ui + ti;
				re[u] = ur - tr;
				im[u] = ui - ti;
				float next_w_re = w_re * wm_re - w_im * wm_im;
				float next_w_im = w_re * wm_im + w_im * wm_re;
				w_re = next_w_re;
				w_im = next_w_im;
			}
		}
	}
	if (inv == -1) {
		for (i = 0; i < len; i++) {
			re[i] /= len;
			im[i] /= len;
		}
	}
}

namespace LatticeReverb3
{
	template<int MaxDelayLen>
	class DelayLine
	{
	private:
		float dat[MaxDelayLen] = { 0 };
		float out = 0;

		// 平滑处理
		float currentDelay = 0;
		float targetDelay = 2;
		float delayVelocity = 0; // 用于线性平滑延迟时间本身

		int pos = 0;

		int updateTime = 0;

		inline float ReadSampleHermite(float delay)
		{
			float readPos = (float)pos - delay;
			while (readPos < 0) readPos += MaxDelayLen;
			while (readPos >= MaxDelayLen) readPos -= MaxDelayLen;

			int i1 = (int)readPos;
			float f = readPos - i1;

			int i0 = (i1 - 1 + MaxDelayLen) % MaxDelayLen;
			int i2 = (i1 + 1) % MaxDelayLen;
			int i3 = (i1 + 2) % MaxDelayLen;

			float y0 = dat[i0];
			float y1 = dat[i1];
			float y2 = dat[i2];
			float y3 = dat[i3];

			float c0 = y1;
			float c1 = 0.5f * (y2 - y0);
			float c2 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
			float c3 = 0.5f * (y3 - y0) + 1.5f * (y1 - y2);

			return ((c3 * f + c2) * f + c1) * f + c0;
		}
		inline float ReadSampleLinear(float delay)
		{
			float readPos = (float)pos - delay;
			while (readPos < 0) readPos += MaxDelayLen;
			while (readPos >= MaxDelayLen) readPos -= MaxDelayLen;
			int i1 = (int)readPos;
			int i2 = (i1 + 1) % MaxDelayLen;
			float f = readPos - i1;
			return dat[i1] * (1.0f - f) + dat[i2] * f;
		}
	public:
		constexpr static int GradientSamples = 10000;

		DelayLine()
		{
			std::fill(dat, dat + MaxDelayLen, 0.0f);
		}

		inline void SetDelayTime(float t)
		{
			if (t < 2)t = 2;
			if (t > MaxDelayLen - 4)t = MaxDelayLen - 4;
			targetDelay = t;
		}

		inline float ReadSample()
		{
			return out;
		}

		inline void WriteSample(float val)
		{
			dat[pos] = val;

			updateTime--;
			if (updateTime <= 0)
			{
				delayVelocity = (targetDelay - currentDelay) / (float)GradientSamples;
				updateTime = 512;
			}

			if (fabsf(targetDelay - currentDelay) > 0.0001f)
			{
				currentDelay += delayVelocity;
			}
			else
			{
				currentDelay = targetDelay;
			}

			out = ReadSampleHermite(currentDelay);

			if (++pos >= MaxDelayLen) pos = 0;
		}
		void Reset()
		{
			for (auto& v : dat)v = 0;
			out = 0;
			pos = 0;
			currentDelay = targetDelay;
		}
	};

	class LatticeCascade
	{
	public:
		constexpr static int NumLayers = 6;
		constexpr static int MaxDelayLength = 48000;

	private:
		float roomSize = 512;

		DelayLine<MaxDelayLength> delays[NumLayers];
		float ks[NumLayers];
		float ds[NumLayers];
		float outks[NumLayers];

		float tapmix = 0;
		template<int inLayer>//NumLayers-1
		inline float HProcSamp(float x)
		{
			if constexpr (inLayer >= 0)
			{
				if constexpr (inLayer == NumLayers - 1) tapmix = 0;
				float y = HProcSamp<inLayer - 1>(delays[inLayer].ReadSample());
				float a = (x + ks[inLayer] * y) * ds[inLayer];
				delays[inLayer].WriteSample(a);
				float out = y - a * ks[inLayer];
				tapmix += out * outks[inLayer];
				return out;
			}
			else
			{
				return x;
			}
		}

	public:
		inline float ProcessSample(float x)// direct
		{
			return HProcSamp<NumLayers - 1>(x);
		}
		inline float GetTapMix()
		{
			return tapmix;
		}

		void Reset()
		{
			for (auto& it : delays)it.Reset();
		}
		void SetKs(float* ks)
		{
			for (int i = 0; i < NumLayers; ++i)this->ks[i] = ks[i];
		}
		void SetDs(float* ds)
		{
			for (int i = 0; i < NumLayers; ++i)this->ds[i] = ds[i];
		}
		void SetDelaysLength(float* delaysLength)//归一化最大长度
		{
			float maxv = 0;
			for (int i = 0; i < NumLayers; ++i)
			{
				if (delaysLength[i] > maxv)maxv = delaysLength[i];
			}
			for (int i = 0; i < NumLayers; ++i)
			{
				float delayt = delaysLength[i] / maxv * roomSize;
				delays[i].SetDelayTime(delayt);
			}
		}
		void SetOutKs(float* outKs)//归一化能量
		{
			float energy = 0;
			for (int i = 0; i < NumLayers; ++i)
			{
				energy += outKs[i] * outKs[i];
			}
			energy = sqrtf(energy) * sqrtf(NumLayers);
			for (int i = 0; i < NumLayers; ++i)
			{
				float outk = outKs[i] / energy;
				outks[i] = outk;
			}
		}
		void SetRoomSize(float roomSize)
		{
			this->roomSize = roomSize;
		}
	};

	class LatticeReverb3
	{
	public:
		constexpr static int NumLayers = LatticeCascade::NumLayers;
		constexpr static int NumRoomParams = 8 * LatticeCascade::NumLayers + 4;
	private:
		LatticeCascade latl, latr;
		DelayLine<LatticeCascade::MaxDelayLength> crossDelayL, crossDelayR;
		struct RoomParams
		{
			float tsl[LatticeCascade::NumLayers];//延迟线长度比例
			float tsr[LatticeCascade::NumLayers];
			float fbtl;
			float fbtr;
			//反射系数受diffusion约束
			float ksl[LatticeCascade::NumLayers];//各节反射系数
			float ksr[LatticeCascade::NumLayers];
			//衰减系数受decayTime约束
			float dsl[LatticeCascade::NumLayers];//各节衰减系数
			float dsr[LatticeCascade::NumLayers];
			float fbdl;//莫比乌斯环结构的反馈衰减系数
			float fbdr;

			float outksl[LatticeCascade::NumLayers];//各节tap输出混合比例
			float outksr[LatticeCascade::NumLayers];//各节tap输出混合比例
		};
		struct ReverbParams
		{
			float roomSize = 2400;//roomsize在这里改！！！
			float decayTime = 0.99;//这个需要算RT60补偿
			float diffusion = 1.0;
			float mixTapDirect = 0.5;
		};
		RoomParams roomParams, applyRoomParams;
		ReverbParams reverbParams;
	public:
		void Reset()
		{
			latl.Reset();
			latr.Reset();
			crossDelayL.Reset();
			crossDelayR.Reset();
		}
		void ProcessBlock(const float* inl, const float* inr, float* outl, float* outr, int numSamples)
		{
			for (int i = 0; i < numSamples; ++i)
			{
				float lastoutl = crossDelayL.ReadSample();
				float lastoutr = crossDelayR.ReadSample();
				float xl = inl[i] + lastoutr * applyRoomParams.fbdl;
				float xr = inr[i] + lastoutl * applyRoomParams.fbdr;
				float outvl = latl.ProcessSample(xl);
				float outvr = latr.ProcessSample(xr);
				crossDelayL.WriteSample(outvr);//cross
				crossDelayR.WriteSample(outvl);
				float tapMixOutl = latl.GetTapMix();
				float tapMixOutr = latr.GetTapMix();

				outl[i] = outvl * (1.0f - reverbParams.mixTapDirect) + tapMixOutl * reverbParams.mixTapDirect;
				outr[i] = outvr * (1.0f - reverbParams.mixTapDirect) + tapMixOutr * reverbParams.mixTapDirect;
			}
		}
		void SetRoomSize(float roomSize)
		{
			reverbParams.roomSize = roomSize;
			applyRoomParams.fbtl = roomParams.fbtl * roomSize;
			applyRoomParams.fbtr = roomParams.fbtr * roomSize;
			latl.SetRoomSize(roomSize);
			latr.SetRoomSize(roomSize);
			for (int i = 0; i < LatticeCascade::NumLayers; ++i)
			{
				applyRoomParams.tsl[i] = roomParams.tsl[i];
				applyRoomParams.tsr[i] = roomParams.tsr[i];
			}
			latl.SetDelaysLength(applyRoomParams.tsl);
			latr.SetDelaysLength(applyRoomParams.tsr);
			crossDelayL.SetDelayTime(applyRoomParams.fbtl);
			crossDelayR.SetDelayTime(applyRoomParams.fbtr);
		}
		void SetDecayTime(float decayTime)//n代表n秒之后到达-60dB
		{
			reverbParams.decayTime = decayTime;
			for (int i = 0; i < LatticeCascade::NumLayers; ++i)
			{
				applyRoomParams.dsl[i] = roomParams.dsl[i] * reverbParams.decayTime;
				applyRoomParams.dsr[i] = roomParams.dsr[i] * reverbParams.decayTime;
			}
			applyRoomParams.fbdl = roomParams.fbdl * reverbParams.decayTime;
			applyRoomParams.fbdr = roomParams.fbdr * reverbParams.decayTime;
			latl.SetDs(applyRoomParams.dsl);
			latr.SetDs(applyRoomParams.dsr);
		}
		void SetDiffusion(float diffusion)//0-1
		{
			reverbParams.diffusion = diffusion;
			for (int i = 0; i < LatticeCascade::NumLayers; ++i)
			{
				applyRoomParams.ksl[i] = roomParams.ksl[i] * diffusion;
				applyRoomParams.ksr[i] = roomParams.ksr[i] * diffusion;
			}
			latl.SetKs(applyRoomParams.ksl);
			latr.SetKs(applyRoomParams.ksr);
		}
		void SetMixTapDirect(float mix)//0-1
		{
			reverbParams.mixTapDirect = mix;
		}
		void SetupRoomCharacteristics(std::vector<float>& roomParamsPack)
		{
			for (int i = 0; i < LatticeCascade::NumLayers; ++i)
			{
				roomParams.tsl[i] = roomParamsPack[i + 0 * LatticeCascade::NumLayers];
				roomParams.tsr[i] = roomParamsPack[i + 1 * LatticeCascade::NumLayers];
				roomParams.ksl[i] = roomParamsPack[i + 2 * LatticeCascade::NumLayers];
				roomParams.ksr[i] = roomParamsPack[i + 3 * LatticeCascade::NumLayers];
				roomParams.dsl[i] = roomParamsPack[i + 4 * LatticeCascade::NumLayers];
				roomParams.dsr[i] = roomParamsPack[i + 5 * LatticeCascade::NumLayers];
				roomParams.outksl[i] = roomParamsPack[i + 6 * LatticeCascade::NumLayers];
				roomParams.outksr[i] = roomParamsPack[i + 7 * LatticeCascade::NumLayers];
			}
			roomParams.fbdl = roomParamsPack[8 * LatticeCascade::NumLayers + 0];
			roomParams.fbdr = roomParamsPack[8 * LatticeCascade::NumLayers + 1];
			roomParams.fbtl = roomParamsPack[8 * LatticeCascade::NumLayers + 2];
			roomParams.fbtr = roomParamsPack[8 * LatticeCascade::NumLayers + 3];

			SetRoomSize(reverbParams.roomSize);
			SetDecayTime(reverbParams.decayTime);
			SetDiffusion(reverbParams.diffusion);
			SetMixTapDirect(reverbParams.mixTapDirect);
			latl.SetOutKs(roomParams.outksl);
			latr.SetOutKs(roomParams.outksr);

			Reset();
		}
		void InitRoomParams(std::vector<float>& roomParamsPack)
		{
			for (int i = 0; i < LatticeCascade::NumLayers; ++i)
			{
				roomParamsPack[i + 0 * LatticeCascade::NumLayers] = (float)(rand() % 1000) / 1000.0f;//tsl
				roomParamsPack[i + 1 * LatticeCascade::NumLayers] = (float)(rand() % 1000) / 1000.0f;//tsr
				roomParamsPack[i + 2 * LatticeCascade::NumLayers] = (float)(rand() % 1000) / 1000.0f;//ksl
				roomParamsPack[i + 3 * LatticeCascade::NumLayers] = (float)(rand() % 1000) / 1000.0f;//ksr
				roomParamsPack[i + 4 * LatticeCascade::NumLayers] = 1.0;//dsl
				roomParamsPack[i + 5 * LatticeCascade::NumLayers] = 1.0;//dsr
				roomParamsPack[i + 6 * LatticeCascade::NumLayers] = (float)(rand() % 1000) / 1000.0f;//outksl
				roomParamsPack[i + 7 * LatticeCascade::NumLayers] = (float)(rand() % 1000) / 1000.0f;//outksr
			}
			roomParamsPack[8 * LatticeCascade::NumLayers + 0] = -0.5;//fbdl
			roomParamsPack[8 * LatticeCascade::NumLayers + 1] = 0.5;//fbdr
			roomParamsPack[8 * LatticeCascade::NumLayers + 2] = (float)(rand() % 1000) / 1000.0f;//fbtl
			roomParamsPack[8 * LatticeCascade::NumLayers + 3] = (float)(rand() % 1000) / 1000.0f;//fbtr
		}
	};
}

/*
优化方向：
频谱方差小（无染色且噪）
接近RT60（方便校准）
左右声道相位差 方差大（声场宽）
*/

class LatticeOptimizer
{
public:
	constexpr static int NumRoomParams = LatticeReverb3::LatticeReverb3::NumRoomParams;
	constexpr static int NumLayers = LatticeReverb3::LatticeReverb3::NumLayers;
private:
	LatticeReverb3::LatticeReverb3 instance;
	AdamOptimizer optimizer;

	std::vector<float> roomParams;//未正则化
	std::vector<float> applyRoomParams;//正则化后实际应用的参数
	inline float soft01(float x, float minv = 0.0)
	{
		float v = 1.0 - expf(-fabsf(x));
		return v * (1.0 - minv) + minv;
	}
	void Regularization(std::vector<float>& roomParams)
	{
		for (int i = 0; i < NumLayers; ++i)
		{
			auto& tsl = roomParams[i + 0 * NumLayers];
			auto& tsr = roomParams[i + 1 * NumLayers];
			auto& ksl = roomParams[i + 2 * NumLayers];
			auto& ksr = roomParams[i + 3 * NumLayers];
			auto& dsl = roomParams[i + 4 * NumLayers];
			auto& dsr = roomParams[i + 5 * NumLayers];
			auto& outksl = roomParams[i + 6 * NumLayers];
			auto& outksr = roomParams[i + 7 * NumLayers];

			tsl = soft01(tsl);
			tsr = soft01(tsr);
			ksl = tanhf(ksl);
			ksr = tanhf(ksr);
			dsl = soft01(dsl, 0.98) * 0.99999999;
			dsr = soft01(dsr, 0.98) * 0.99999999;
			outksl = soft01(outksl);
			outksr = soft01(outksr);
		}

		auto& fbdl = roomParams[8 * NumLayers + 0];
		auto& fbdr = roomParams[8 * NumLayers + 1];
		auto& fbtl = roomParams[8 * NumLayers + 2];
		auto& fbtr = roomParams[8 * NumLayers + 3];

		fbtl = soft01(fbtl);
		fbtr = soft01(fbtr);
		fbdl = -soft01(fbdl, 0.5) * 0.99999999;
		fbdr = -soft01(fbdr, 0.5) * 0.99999999;
	}

	constexpr static int testBlockSize = 2048;
	float zeroBuf[testBlockSize] = { 0 };
	float bufrel[testBlockSize];
	float bufiml[testBlockSize];
	float bufrer[testBlockSize];
	float bufimr[testBlockSize];
	float magl[testBlockSize / 2];
	float magr[testBlockSize / 2];
	float Error(std::vector<float>& roomParamsPack)
	{
		std::copy(roomParamsPack.begin(), roomParamsPack.end(), applyRoomParams.begin());
		Regularization(applyRoomParams);
		instance.SetupRoomCharacteristics(applyRoomParams);
		instance.Reset();
		float tmpl = 1, tmpr = 1;
		instance.ProcessBlock(&tmpl, &tmpr, &tmpl, &tmpr, 1);//impulse response
		//instance.ProcessBlock(zeroBuf, zeroBuf, bufrel, bufrer, testBlockSize);
		//instance.ProcessBlock(zeroBuf, zeroBuf, bufrel, bufrer, testBlockSize);
		instance.ProcessBlock(zeroBuf, zeroBuf, bufrel, bufrer, testBlockSize);
		for (int i = 0; i < testBlockSize; ++i)
		{
			float window = 0.5f * (1.0f - cosf(2.0f * 3.1415926535897932384626f * i / testBlockSize));
			bufrel[i] *= window;
			bufrer[i] *= window;
			bufiml[i] = 0;
			bufimr[i] = 0;
		}
		fft(bufrel, bufiml, testBlockSize, 1);
		fft(bufrer, bufimr, testBlockSize, 1);

		float avg = 0;
		float avgl = 0, avgr = 0;
		float s2 = 0;
		const int numBins = testBlockSize / 2 * 0.5;//只关心nyquist*0.5以内的频响

		for (int i = 0; i < numBins; ++i)
		{
			float maglv = sqrtf(bufrel[i] * bufrel[i] + bufiml[i] * bufiml[i]);
			float magrv = sqrtf(bufrer[i] * bufrer[i] + bufimr[i] * bufimr[i]);
			magl[i] = maglv;
			magr[i] = magrv;
			avg += (maglv + magrv) * 0.5;
			avgl += maglv;
			avgr += magrv;
		}
		avg /= numBins;
		avgl /= numBins;
		avgr /= numBins;
		for (int i = 0; i < numBins; ++i)
		{
			float magv = (magl[i] + magr[i]) * 0.5 / avg;
			s2 += (magv - 1.0) * (magv - 1.0);
		}

		float specflatloss = s2 * 100.0;
		float diffloss = avgl - avgr;
		diffloss = diffloss * diffloss * 100.0;
		return specflatloss + diffloss;
		//return specflatloss;
	}
public:
	LatticeOptimizer()
	{
		roomParams.resize(NumRoomParams);
		applyRoomParams.resize(NumRoomParams);
		instance.InitRoomParams(roomParams);
		instance.InitRoomParams(applyRoomParams);
		optimizer.SetupOptimizer(LatticeReverb3::LatticeReverb3::NumRoomParams,
			roomParams, 0.005f);
		optimizer.SetErrorFunc([this](std::vector<float>& params) {return Error(params); });
	}
	void RunOptimizer(int numCycle)
	{
		optimizer.RunOptimizer(numCycle);
	}
	float GetNowLoss()
	{
		return optimizer.GetNowError();
	}
	void GetNowRoomParams(std::vector<float>& outParams)
	{
		optimizer.GetNowVec(roomParams);
		Regularization(roomParams);
		std::copy(roomParams.begin(), roomParams.end(), outParams.begin());
	}
};


///////////////////////////////////////////////////////////////////////
namespace IRPlot2D
{
	constexpr int SamplesPerCharX = 256;   // 横向 1 字符 = 64 samples 平均能量
	constexpr float DbPerCharY = 3.0f;    // 纵向 1 字符 = 3 dB

	// 控制上限，全部静态分配
	constexpr int MaxConsoleW = 256;
	constexpr int MaxConsoleH = 96;
	constexpr int MaxPlotW = MaxConsoleW - 8;
	constexpr int MaxPlotH = MaxConsoleH - 4;
	constexpr int MaxIRSamples = MaxPlotW * SamplesPerCharX;

	// 全部静态，禁止函数内大对象/大数组
	static LatticeReverb3::LatticeReverb3 s_reverb;

	static float s_inL[1] = { 1.0f };
	static float s_inR[1] = { 1.0f };
	static float s_out0L[1] = { 0.0f };
	static float s_out0R[1] = { 0.0f };

	static float s_zeroL[MaxIRSamples] = { 0.0f };
	static float s_zeroR[MaxIRSamples] = { 0.0f };
	static float s_irL[MaxIRSamples] = { 0.0f };
	static float s_irR[MaxIRSamples] = { 0.0f };

	// 每列一个能量值 / dB 值 / 行坐标
	static float s_colEnergy[MaxPlotW] = { 0.0f };
	static float s_colDb[MaxPlotW] = { 0.0f };
	static int   s_colRow[MaxPlotW] = { 0 };

	// 终端画布
	static char s_canvas[MaxPlotH][MaxPlotW + 1];

	static int GetConsoleWidth()
	{
#ifdef _WIN32
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
		{
			int w = (int)(csbi.srWindow.Right - csbi.srWindow.Left + 1);
			if (w > 0) return w;
		}
#else
		struct winsize ws;
		if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0)
		{
			if (ws.ws_col > 0) return (int)ws.ws_col;
		}
#endif
		return 120;
	}

	static int GetConsoleHeight()
	{
#ifdef _WIN32
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
		{
			int h = (int)(csbi.srWindow.Bottom - csbi.srWindow.Top + 1);
			if (h > 0) return h;
		}
#else
		struct winsize ws;
		if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0)
		{
			if (ws.ws_row > 0) return (int)ws.ws_row;
		}
#endif
		return 40;
	}

	static void ClearCanvas(int w, int h)
	{
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x) s_canvas[y][x] = ' ';
			s_canvas[y][w] = '\0';
		}
	}

	static void BuildIR(std::vector<float>& roomParams, int irSamples)
	{
		if (irSamples < 1) return;
		if (irSamples > MaxIRSamples) irSamples = MaxIRSamples;

		s_reverb.SetupRoomCharacteristics(roomParams);
		s_reverb.Reset();

		// 先打一发冲激
		s_inL[0] = 1.0f;
		s_inR[0] = 1.0f;
		s_reverb.ProcessBlock(s_inL, s_inR, s_out0L, s_out0R, 1);
		s_irL[0] = s_out0L[0];
		s_irR[0] = s_out0R[0];

		// 后续零输入
		if (irSamples > 1)
		{
			s_reverb.ProcessBlock(s_zeroL, s_zeroR, s_irL + 1, s_irR + 1, irSamples - 1);
		}
	}

	static void BuildColumnEnergyMono(int plotW)
	{
		float maxE = 0.0f;

		for (int x = 0; x < plotW; ++x)
		{
			const int begin = x * SamplesPerCharX;
			const int end = begin + SamplesPerCharX;

			float e = 0.0f;
			for (int i = begin; i < end; ++i)
			{
				const float l = s_irL[i];
				const float r = s_irR[i];
				e += 0.5f * (l * l + r * r);
			}
			e /= (float)SamplesPerCharX;
			s_colEnergy[x] = e;
			if (e > maxE) maxE = e;
		}

		if (maxE < 1e-30f) maxE = 1e-30f;

		for (int x = 0; x < plotW; ++x)
		{
			const float norm = s_colEnergy[x] / maxE;
			s_colDb[x] = 10.0f * log10f(norm + 1e-30f); // <= 0 dB
		}
	}

	static void BuildRows(int plotW, int plotH)
	{
		const float bottomDb = -(float)(plotH - 1) * DbPerCharY;

		for (int x = 0; x < plotW; ++x)
		{
			float db = s_colDb[x];
			if (db > 0.0f) db = 0.0f;
			if (db < bottomDb) db = bottomDb;

			// row=0 顶部(0dB), row=plotH-1 底部(最小dB)
			int row = (int)floorf((-db) / DbPerCharY + 0.5f);
			if (row < 0) row = 0;
			if (row >= plotH) row = plotH - 1;
			s_colRow[x] = row;
		}
	}

	static void DrawAxesAndBars(int plotW, int plotH)
	{
		// 纵轴刻度线：每 3 dB 一行本来就是一格
		// 横轴：底边
		for (int x = 0; x < plotW; ++x)
		{
			s_canvas[plotH - 1][x] = '_';
		}

		// 左轴
		for (int y = 0; y < plotH; ++y)
		{
			s_canvas[y][0] = '|';
		}
		s_canvas[plotH - 1][0] = '+';

		// 画柱
		for (int x = 0; x < plotW; ++x)
		{
			int row = s_colRow[x];
			if (x == 0) continue; // 留给纵轴

			for (int y = plotH - 2; y >= row; --y)
			{
				s_canvas[y][x] = '#';
			}
		}
	}

	static void DrawPolyline(int plotW, int plotH)
	{
		for (int x = 1; x < plotW; ++x)
		{
			const int y0 = s_colRow[x - 1];
			const int y1 = s_colRow[x];

			if (y0 == y1)
			{
				if (x - 1 > 0) s_canvas[y0][x - 1] = '*';
				if (x > 0)     s_canvas[y1][x] = '*';
			}
			else
			{
				int ya = y0 < y1 ? y0 : y1;
				int yb = y0 < y1 ? y1 : y0;
				for (int y = ya; y <= yb; ++y)
				{
					if (x > 0) s_canvas[y][x] = '*';
				}
			}
		}
	}

	static void PrintCanvas(int plotW, int plotH)
	{
		for (int y = 0; y < plotH; ++y)
		{
			printf("%s\n", s_canvas[y]);
		}
	}

	// 直接拿 GetNowRoomParams 输出后的系数向量来画
	// 要求：roomParams 已经是可直接 SetupRoomCharacteristics 的参数
	static void Draw(std::vector<float>& roomParams)
	{
		int cw = GetConsoleWidth();
		int ch = GetConsoleHeight();

		int plotW = cw;
		int plotH = ch - 2;

		if (plotW > MaxPlotW) plotW = MaxPlotW;
		if (plotH > MaxPlotH) plotH = MaxPlotH;

		if (plotW < 8) plotW = 8;
		if (plotH < 4) plotH = 4;

		const int irSamples = plotW * SamplesPerCharX;

		BuildIR(roomParams, irSamples);
		BuildColumnEnergyMono(plotW);
		BuildRows(plotW, plotH);
		ClearCanvas(plotW, plotH);
		DrawAxesAndBars(plotW, plotH);
		DrawPolyline(plotW, plotH);
		PrintCanvas(plotW, plotH);
	}
}
namespace IRCheckpoint
{
	constexpr int SampleRate = 48000;
	constexpr int IRSeconds = 5;
	constexpr int IRNumSamples = SampleRate * IRSeconds;

	// 全部静态，禁止函数内大对象/大数组
	static LatticeReverb3::LatticeReverb3 s_reverb;

	static float s_inL[1] = { 1.0f };
	static float s_inR[1] = { 1.0f };
	static float s_out0L[1] = { 0.0f };
	static float s_out0R[1] = { 0.0f };

	static float s_zeroL[IRNumSamples - 1] = { 0.0f };
	static float s_zeroR[IRNumSamples - 1] = { 0.0f };
	static float s_irL[IRNumSamples] = { 0.0f };
	static float s_irR[IRNumSamples] = { 0.0f };

	static void BuildIR(std::vector<float>& roomParams)
	{
		s_reverb.SetupRoomCharacteristics(roomParams);
		s_reverb.Reset();

		s_inL[0] = 1.0f;
		s_inR[0] = 1.0f;
		s_reverb.ProcessBlock(s_inL, s_inR, s_out0L, s_out0R, 1);
		s_irL[0] = s_out0L[0];
		s_irR[0] = s_out0R[0];

		s_reverb.ProcessBlock(s_zeroL, s_zeroR, s_irL + 1, s_irR + 1, IRNumSamples - 1);
	}

	static short FloatToPcm16(float x)
	{
		if (x > 1.0f) x = 1.0f;
		if (x < -1.0f) x = -1.0f;
		int v = (int)lrintf(x * 32767.0f);
		if (v > 32767) v = 32767;
		if (v < -32768) v = -32768;
		return (short)v;
	}

	static void WriteLE16(FILE* fp, unsigned short v)
	{
		unsigned char b[2];
		b[0] = (unsigned char)(v & 0xFF);
		b[1] = (unsigned char)((v >> 8) & 0xFF);
		fwrite(b, 1, 2, fp);
	}

	static void WriteLE32(FILE* fp, unsigned int v)
	{
		unsigned char b[4];
		b[0] = (unsigned char)(v & 0xFF);
		b[1] = (unsigned char)((v >> 8) & 0xFF);
		b[2] = (unsigned char)((v >> 16) & 0xFF);
		b[3] = (unsigned char)((v >> 24) & 0xFF);
		fwrite(b, 1, 4, fp);
	}

	static bool SaveWav(const char* filename)
	{
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
		const unsigned int dataSize = IRNumSamples * blockAlign;
		const unsigned int riffSize = 36 + dataSize;

		fwrite("RIFF", 1, 4, fp);
		WriteLE32(fp, riffSize);
		fwrite("WAVE", 1, 4, fp);

		fwrite("fmt ", 1, 4, fp);
		WriteLE32(fp, 16);
		WriteLE16(fp, 1); // PCM
		WriteLE16(fp, numChannels);
		WriteLE32(fp, SampleRate);
		WriteLE32(fp, byteRate);
		WriteLE16(fp, blockAlign);
		WriteLE16(fp, bitsPerSample);

		fwrite("data", 1, 4, fp);
		WriteLE32(fp, dataSize);

		for (int i = 0; i < IRNumSamples; ++i)
		{
			short sl = FloatToPcm16(s_irL[i]);
			short sr = FloatToPcm16(s_irR[i]);
			WriteLE16(fp, (unsigned short)sl);
			WriteLE16(fp, (unsigned short)sr);
		}

		fclose(fp);
		return true;
	}

	static bool SaveParamsTxt(std::vector<float>& roomParams, const char* filename)
	{
		FILE* fp = nullptr;
#if defined(_MSC_VER)
		if (fopen_s(&fp, filename, "wb") != 0) fp = nullptr;
#else
		fp = fopen(filename, "wb");
#endif
		if (!fp) return false;

		fprintf(fp, "NumParams=%zu\n", roomParams.size());
		for (size_t i = 0; i < roomParams.size(); ++i)
		{
			fprintf(fp, "%.9g\n", roomParams[i]);
		}

		fclose(fp);
		return true;
	}

	// roomParams 必须是可直接传给 SetupRoomCharacteristics 的参数向量
	static bool Save(std::vector<float>& roomParams)
	{
		if ((int)roomParams.size() != LatticeReverb3::LatticeReverb3::NumRoomParams)
			return false;

		BuildIR(roomParams);

		if (!SaveWav("checkpoint.wav"))
			return false;

		if (!SaveParamsTxt(roomParams, "checkpoint.txt"))
			return false;

		return true;
	}
}
///////////////////////////////////////////////////////////////////////
LatticeOptimizer lattopt;
std::vector<float> roomParams(LatticeOptimizer::NumRoomParams);
float minLoss = 1e30f;
int main()
{
	for (;;)
	{
		lattopt.RunOptimizer(20);
		float nowloss = lattopt.GetNowLoss();
		printf("loss:%.5f %s\n", nowloss, nowloss < minLoss ? "(new minloss)" : "");
		lattopt.GetNowRoomParams(roomParams);
		IRPlot2D::Draw(roomParams);

		if (nowloss < minLoss)
		{
			minLoss = nowloss;
			IRCheckpoint::Save(roomParams);
		}
	}
}