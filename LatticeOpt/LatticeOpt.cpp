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
			float roomSize = 1200;//roomsize在这里改！！！
			float decayTime = 0.9999;//这个需要算RT60补偿
			float diffusion = 1.0;
			float mixTapDirect = 0.2;
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
		float randf()
		{
			return (float)(rand() % 1000) / 1000.0f * (rand() % 2 ? 1 : -1);
		}
		void InitRoomParams(std::vector<float>& roomParamsPack)
		{
			for (int i = 0; i < LatticeCascade::NumLayers; ++i)
			{
				float tsl = fabsf(randf()) * 0.8 + 0.2;//tsl
				float tsr = fabsf(randf()) * 0.8 + 0.2;//tsr
				float ksl = randf();//ksl
				float ksr = randf();//ksr
				float dsl = randf() * 0.1 + 0.90;//dsl
				float dsr = randf() * 0.1 + 0.90;//dsr
				float outksl = randf();//outksl
				float outksr = randf();//outksr

				roomParamsPack[i + 0 * LatticeCascade::NumLayers] = tsl;//tsl
				roomParamsPack[i + 1 * LatticeCascade::NumLayers] = tsr;//tsr
				roomParamsPack[i + 2 * LatticeCascade::NumLayers] = ksl;//ksl
				roomParamsPack[i + 3 * LatticeCascade::NumLayers] = ksr;//ksr
				roomParamsPack[i + 4 * LatticeCascade::NumLayers] = dsl;//dsl
				roomParamsPack[i + 5 * LatticeCascade::NumLayers] = dsr;//dsr
				roomParamsPack[i + 6 * LatticeCascade::NumLayers] = outksl;//outksl
				roomParamsPack[i + 7 * LatticeCascade::NumLayers] = outksr;//outksr
			}
			roomParamsPack[8 * LatticeCascade::NumLayers + 0] = -0.75;//fbdl
			roomParamsPack[8 * LatticeCascade::NumLayers + 1] = -0.75;//fbdr
			roomParamsPack[8 * LatticeCascade::NumLayers + 2] = fabsf(randf());//fbtl
			roomParamsPack[8 * LatticeCascade::NumLayers + 3] = fabsf(randf());//fbtr
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
	constexpr static int testBlockSize = 512;
	//这个设的反而越小越好
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
	inline float tanhfno0(float x, float minv = 0.0)
	{
		float v = tanhf(x);
		if (v > 0)v = v * (1.0f - minv) + minv;
		else v = v * (1.0f - minv) - minv;
		return v;
	}

	float zeroBuf[testBlockSize] = { 0 };
	float bufrel[testBlockSize];
	float bufiml[testBlockSize];
	float bufrer[testBlockSize];
	float bufimr[testBlockSize];

	float autocorrl[testBlockSize * 2];
	float autocorrr[testBlockSize * 2];
	float autocorriml[testBlockSize * 2];
	float autocorrimr[testBlockSize * 2];

	float magl[testBlockSize / 2];
	float magr[testBlockSize / 2];
	float gdl[testBlockSize / 2];
	float gdr[testBlockSize / 2];
	float Error(std::vector<float>& roomParamsPack)
	{
		std::copy(roomParamsPack.begin(), roomParamsPack.end(), applyRoomParams.begin());
		Regularization(applyRoomParams);
		instance.SetupRoomCharacteristics(applyRoomParams);
		instance.Reset();
		float tmpl = 1, tmpr = 1;
		instance.ProcessBlock(&tmpl, &tmpr, &tmpl, &tmpr, 1);//impulse response
		instance.ProcessBlock(zeroBuf, zeroBuf, bufrel, bufrer, testBlockSize);
		//instance.ProcessBlock(zeroBuf, zeroBuf, bufrel, bufrer, testBlockSize);
		for (int i = 0; i < testBlockSize; ++i)
		{
			//float window = 0.5f * (1.0f - cosf(2.0f * 3.1415926535897932384626f * i / testBlockSize));
			float window = 1.0;
			autocorrl[i] = bufrel[i] * window;
			autocorrr[i] = bufrer[i] * window;
			autocorrl[testBlockSize + i] = 0;
			autocorrr[testBlockSize + i] = 0;
		}
		for (int i = 0; i < testBlockSize * 2; ++i)
		{
			autocorriml[i] = 0;
			autocorrimr[i] = 0;
		}
		fft(autocorrl, autocorriml, testBlockSize * 2, 1);
		fft(autocorrr, autocorrimr, testBlockSize * 2, 1);
		for (int i = 0; i < testBlockSize * 2; ++i)
		{
			autocorrl[i] = autocorrl[i] * autocorrl[i] + autocorriml[i] * autocorriml[i];
			autocorrr[i] = autocorrr[i] * autocorrr[i] + autocorrimr[i] * autocorrimr[i];
			autocorriml[i] = 0;
			autocorrimr[i] = 0;
		}
		fft(autocorrl, autocorriml, testBlockSize * 2, -1);
		fft(autocorrr, autocorrimr, testBlockSize * 2, -1);
		float maxcorrl = 1e-30f;
		float maxcorrr = 1e-30f;
		for (int i = 1; i < testBlockSize * 2 - 1; ++i)
		{
			if (maxcorrl < autocorrl[i])maxcorrl = autocorrl[i];
			if (maxcorrr < autocorrr[i])maxcorrr = autocorrr[i];
		}

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
		float s2l = 0, s2r = 0;
		const int numBins = testBlockSize / 2 * 0.7;//只关心nyquist*0.5以内的频响
		float specmax = 1e-30, specmin = 1e30;
		float lastphasel = 0, lastphaser = 0;
		float gdavgl = 0, gdavgr = 0;
		float gds2l = 0, gds2r = 0;

		int startbin = 20;
		for (int i = startbin; i < numBins; ++i)
		{
			float maglv = sqrtf(bufrel[i] * bufrel[i] + bufiml[i] * bufiml[i]);
			float magrv = sqrtf(bufrer[i] * bufrer[i] + bufimr[i] * bufimr[i]);
			magl[i] = maglv;
			magr[i] = magrv;
			avg += (maglv + magrv) * 0.5;
			avgl += maglv;
			avgr += magrv;
			if (specmax < maglv)specmax = maglv;
			if (specmax < magrv)specmax = magrv;
			if (specmin > maglv)specmin = maglv;
			if (specmin > magrv)specmin = magrv;

			const float PI = 3.14159265358979323846f;
			const float deltaOmega = 2.0f * PI / testBlockSize;
			float phasel = atan2f(bufiml[i], bufrel[i]);
			float phaser = atan2f(bufimr[i], bufrer[i]);
			float dphil = phasel - lastphasel;
			float dphir = phaser - lastphaser;
			while (dphil > PI) dphil -= 2.0f * PI;
			while (dphil < -PI) dphil += 2.0f * PI;
			while (dphir > PI) dphir -= 2.0f * PI;
			while (dphir < -PI) dphir += 2.0f * PI;
			float groupdelayl = -dphil / deltaOmega;
			float groupdelayr = -dphir / deltaOmega;
			lastphasel = phasel;
			lastphaser = phaser;
			gdavgl += groupdelayl;
			gdavgr += groupdelayr;
			gdl[i] = groupdelayl;
			gdr[i] = groupdelayr;
		}
		avg /= numBins - startbin;
		avgl /= numBins - startbin;
		avgr /= numBins - startbin;
		gdavgl /= numBins - startbin;
		gdavgr /= numBins - startbin;
		specmax /= avg;
		specmin /= avg;
		for (int i = startbin; i < numBins; ++i)
		{
			float magv = (magl[i] + magr[i]) * 0.5 / avg;
			float magvl = magl[i] / avgl;
			float magvr = magr[i] / avgr;
			s2 += (magv - 1.0) * (magv - 1.0);
			s2l += (magvl - 1.0) * (magvl - 1.0);
			s2r += (magvr - 1.0) * (magvr - 1.0);
			float groupdelayl = gdl[i] / gdavgl;
			float groupdelayr = gdr[i] / gdavgr;
			gds2l += (groupdelayl - 1.0) * (groupdelayl - 1.0);
			gds2r += (groupdelayr - 1.0) * (groupdelayr - 1.0);
		}

		float specflatloss = (s2 * 0.25 + max(s2l, s2r) * 0.75) * 0.1;//能量方差平坦（这个很一般）
		float diffloss = avgl - avgr;//左右声道能量差
		float maxminloss = specmax - specmin;//频谱极大极小平坦（这个一般）
		float gdloss = 1.0 / (min(gds2l, gds2r) + 0.1);//群延迟最不平坦化（这个还可以吧）
		float maxcorrloss = logf(maxcorrl + maxcorrr + max(maxcorrl, maxcorrr) * 4.0) + 80.0;//自相关峰值越小越好（这个效果不错）
		diffloss = diffloss * diffloss * 100000.0;
		maxminloss = maxminloss * maxminloss * 1.0;
		gdloss = 1000.0 / (gdloss + 1e-20f);
		return  maxcorrloss;
	}
	/////////////////////////////////////////////////////////////

	float targetReL[testBlockSize];
	float targetImL[testBlockSize];
	float targetReR[testBlockSize];
	float targetImR[testBlockSize];

	float magCurL[testBlockSize / 2];
	float magCurR[testBlockSize / 2];
	float magTarL[testBlockSize / 2];
	float magTarR[testBlockSize / 2];

	float phaseCurL[testBlockSize / 2];
	float phaseCurR[testBlockSize / 2];
	float phaseTarL[testBlockSize / 2];
	float phaseTarR[testBlockSize / 2];

	float gdCurL[testBlockSize / 2];
	float gdCurR[testBlockSize / 2];
	float gdTarL[testBlockSize / 2];
	float gdTarR[testBlockSize / 2];

	float ErrorGlobalIR(std::vector<float>& roomParamsPack)
	{
		std::copy(roomParamsPack.begin(), roomParamsPack.end(), applyRoomParams.begin());
		Regularization(applyRoomParams);
		instance.SetupRoomCharacteristics(applyRoomParams);
		instance.Reset();

		// 生成当前参数对应的 IR
		float tmpl = 1.0f, tmpr = 1.0f;
		instance.ProcessBlock(&tmpl, &tmpr, &tmpl, &tmpr, 1);
		instance.ProcessBlock(zeroBuf, zeroBuf, bufrel, bufrer, testBlockSize);

		// 拷贝目标 IR 到频域缓冲
		for (int i = 0; i < testBlockSize; ++i)
		{
			// 当前 IR
			bufiml[i] = 0.0f;
			bufimr[i] = 0.0f;

			// 目标 IR
			targetReL[i] = global_ir_l[i];
			targetImL[i] = 0.0f;
			targetReR[i] = global_ir_r[i];
			targetImR[i] = 0.0f;
		}

		// 频域变换
		fft(bufrel, bufiml, testBlockSize, 1);
		fft(bufrer, bufimr, testBlockSize, 1);
		fft(targetReL, targetImL, testBlockSize, 1);
		fft(targetReR, targetImR, testBlockSize, 1);

		const int numBins = testBlockSize / 2 - 2;
		const float eps = 1e-20f;
		const float dOmega = 2.0f * 3.1415926535897932384626f / (float)testBlockSize;

		// 计算幅度和相位
		for (int i = 0; i < numBins; ++i)
		{
			float crl = bufrel[i];
			float cil = bufiml[i];
			float crr = bufrer[i];
			float cir = bufimr[i];

			float trl = targetReL[i];
			float til = targetImL[i];
			float trr = targetReR[i];
			float tir = targetImR[i];

			magCurL[i] = sqrtf(crl * crl + cil * cil) + eps;
			magCurR[i] = sqrtf(crr * crr + cir * cir) + eps;
			magTarL[i] = sqrtf(trl * trl + til * til) + eps;
			magTarR[i] = sqrtf(trr * trr + tir * tir) + eps;

			phaseCurL[i] = atan2f(cil, crl);
			phaseCurR[i] = atan2f(cir, crr);
			phaseTarL[i] = atan2f(til, trl);
			phaseTarR[i] = atan2f(tir, trr);
		}

		// unwrap phase
		auto unwrapPhase = [&](float* ph)
			{
				for (int i = 1; i < numBins; ++i)
				{
					float d = ph[i] - ph[i - 1];
					while (d > 3.1415926535897932384626f)
					{
						ph[i] -= 2.0f * 3.1415926535897932384626f;
						d = ph[i] - ph[i - 1];
					}
					while (d < -3.1415926535897932384626f)
					{
						ph[i] += 2.0f * 3.1415926535897932384626f;
						d = ph[i] - ph[i - 1];
					}
				}
			};

		unwrapPhase(phaseCurL);
		unwrapPhase(phaseCurR);
		unwrapPhase(phaseTarL);
		unwrapPhase(phaseTarR);

		// 群延迟（物理定义：-dphi/domega）
		// 用中心差分，边界用单边差分
		gdCurL[0] = -(phaseCurL[1] - phaseCurL[0]) / dOmega;
		gdCurR[0] = -(phaseCurR[1] - phaseCurR[0]) / dOmega;
		gdTarL[0] = -(phaseTarL[1] - phaseTarL[0]) / dOmega;
		gdTarR[0] = -(phaseTarR[1] - phaseTarR[0]) / dOmega;

		for (int i = 1; i < numBins - 1; ++i)
		{
			gdCurL[i] = -(phaseCurL[i + 1] - phaseCurL[i - 1]) / (2.0f * dOmega);
			gdCurR[i] = -(phaseCurR[i + 1] - phaseCurR[i - 1]) / (2.0f * dOmega);
			gdTarL[i] = -(phaseTarL[i + 1] - phaseTarL[i - 1]) / (2.0f * dOmega);
			gdTarR[i] = -(phaseTarR[i + 1] - phaseTarR[i - 1]) / (2.0f * dOmega);
		}

		gdCurL[numBins - 1] = -(phaseCurL[numBins - 1] - phaseCurL[numBins - 2]) / dOmega;
		gdCurR[numBins - 1] = -(phaseCurR[numBins - 1] - phaseCurR[numBins - 2]) / dOmega;
		gdTarL[numBins - 1] = -(phaseTarL[numBins - 1] - phaseTarL[numBins - 2]) / dOmega;
		gdTarR[numBins - 1] = -(phaseTarR[numBins - 1] - phaseTarR[numBins - 2]) / dOmega;

		// 误差计算
		float magLossL = 0.0f;
		float magLossR = 0.0f;
		float gdLossL = 0.0f;
		float gdLossR = 0.0f;

		// 只看较低到中频，减少高频相位噪声影响
		const int beginBin = 20;
		const int endBin = (int)(numBins * 0.45f);

		for (int i = beginBin; i < endBin; ++i)
		{
			// 对数幅度差
			float dMagL = logf(magCurL[i]) - logf(magTarL[i]);
			float dMagR = logf(magCurR[i]) - logf(magTarR[i]);
			magLossL += dMagL * dMagL;
			magLossR += dMagR * dMagR;

			// 群延迟差
			float dGdL = gdCurL[i] - gdTarL[i];
			float dGdR = gdCurR[i] - gdTarR[i];
			gdLossL += dGdL * dGdL;
			gdLossR += dGdR * dGdR;
		}

		float invN = 1.0f / (float)(endBin - beginBin);
		magLossL *= invN;
		magLossR *= invN;
		gdLossL *= invN;
		gdLossR *= invN;

		// 适当压一下群延迟量纲，避免它直接主导
		float magLoss = 0.5f * (magLossL + magLossR);
		float gdLoss = 0.5f * (gdLossL + gdLossR);

		// 你可以后面自己调权重
		float loss = magLoss * 10.0f;//+ gdLoss * 0.001f;
		return loss;
	}
public:
	float global_ir_l[testBlockSize] = { 0 };
	float global_ir_r[testBlockSize] = { 0 };

	LatticeOptimizer()
	{
		roomParams.resize(NumRoomParams);
		applyRoomParams.resize(NumRoomParams);
		instance.InitRoomParams(roomParams);
		instance.InitRoomParams(applyRoomParams);
		optimizer.SetupOptimizer(LatticeReverb3::LatticeReverb3::NumRoomParams,
			roomParams, 0.05f);
		optimizer.SetErrorFunc([this](std::vector<float>& params) {return Error(params); });
		//optimizer.SetErrorFunc([this](std::vector<float>& params) {return ErrorGlobalIR(params); });
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

			tsl = soft01(tsl, 0.01);
			tsr = soft01(tsr, 0.01);
			ksl = tanhfno0(ksl, 0.05) * 0.9;
			ksr = tanhfno0(ksr, 0.05) * 0.9;
			dsl = soft01(dsl, 0.9995) * 0.99999999;
			dsr = soft01(dsr, 0.9995) * 0.99999999;
			outksl = soft01(outksl, 0.01);
			outksr = soft01(outksr, 0.01);
		}

		auto& fbdl = roomParams[8 * NumLayers + 0];
		auto& fbdr = roomParams[8 * NumLayers + 1];
		auto& fbtl = roomParams[8 * NumLayers + 2];
		auto& fbtr = roomParams[8 * NumLayers + 3];

		fbtl = soft01(fbtl, 0.01);
		fbtr = soft01(fbtr, 0.01);
		fbdl = -soft01(fbdl, 0.65) * 0.99999999;
		fbdr = -soft01(fbdr, 0.65) * 0.99999999;
	}
	void Reset()
	{
		instance.Reset();
		instance.InitRoomParams(roomParams);
		instance.InitRoomParams(applyRoomParams);
		optimizer.SetBasin(roomParams);
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
		s_reverb.SetRoomSize(2400);///////////////////////////////!
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
namespace IRTrainDataGen
{
	constexpr int SampleRate = 48000;
	constexpr int IRSeconds = 5;
	constexpr int IRNumSamples = SampleRate * IRSeconds;

	static LatticeReverb3::LatticeReverb3 s_reverb;

	static float s_inL[1] = { 1.0f };
	static float s_inR[1] = { 1.0f };
	static float s_out0L[1] = { 0.0f };
	static float s_out0R[1] = { 0.0f };

	static float s_zeroL[IRNumSamples - 1] = { 0.0f };
	static float s_zeroR[IRNumSamples - 1] = { 0.0f };
	static float s_irL[IRNumSamples] = { 0.0f };
	static float s_irR[IRNumSamples] = { 0.0f };

	static int g_index = 0;

	// ================= IO =================
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
		b[0] = v & 0xFF;
		b[1] = (v >> 8) & 0xFF;
		fwrite(b, 1, 2, fp);
	}

	static void WriteLE32(FILE* fp, unsigned int v)
	{
		unsigned char b[4];
		b[0] = v & 0xFF;
		b[1] = (v >> 8) & 0xFF;
		b[2] = (v >> 16) & 0xFF;
		b[3] = (v >> 24) & 0xFF;
		fwrite(b, 1, 4, fp);
	}

	static void EnsureFolder()
	{
#ifdef _WIN32
		CreateDirectoryA("traindata", NULL);
#else
		mkdir("traindata", 0777);
#endif
	}

	// ================= IR =================
	static void BuildIR(std::vector<float>& params)
	{
		s_reverb.SetupRoomCharacteristics(params);
		s_reverb.Reset();

		s_reverb.ProcessBlock(s_inL, s_inR, s_out0L, s_out0R, 1);
		s_irL[0] = s_out0L[0];
		s_irR[0] = s_out0R[0];

		s_reverb.ProcessBlock(s_zeroL, s_zeroR, s_irL + 1, s_irR + 1, IRNumSamples - 1);
	}

	static void SaveWav(const char* filename)
	{
		FILE* fp = nullptr;
#if defined(_MSC_VER)
		fopen_s(&fp, filename, "wb");
#else
		fp = fopen(filename, "wb");
#endif
		if (!fp) return;

		const int dataSize = IRNumSamples * 4;

		fwrite("RIFF", 1, 4, fp);
		WriteLE32(fp, 36 + dataSize);
		fwrite("WAVE", 1, 4, fp);

		fwrite("fmt ", 1, 4, fp);
		WriteLE32(fp, 16);
		WriteLE16(fp, 1);
		WriteLE16(fp, 2);
		WriteLE32(fp, SampleRate);
		WriteLE32(fp, SampleRate * 4);
		WriteLE16(fp, 4);
		WriteLE16(fp, 16);

		fwrite("data", 1, 4, fp);
		WriteLE32(fp, dataSize);

		for (int i = 0; i < IRNumSamples; ++i)
		{
			WriteLE16(fp, FloatToPcm16(s_irL[i]));
			WriteLE16(fp, FloatToPcm16(s_irR[i]));
		}

		fclose(fp);
	}

	static void SaveParams(const char* filename, std::vector<float>& params)
	{
		FILE* fp = nullptr;
#if defined(_MSC_VER)
		fopen_s(&fp, filename, "wb");
#else
		fp = fopen(filename, "wb");
#endif
		if (!fp) return;

		for (auto& v : params)
			fprintf(fp, "%.9g\n", v);

		fclose(fp);
	}

	static void SaveSample(std::vector<float>& params)
	{
		char wavname[256];
		char txtname[256];

		sprintf_s(wavname, "traindata/ir_%06d.wav", g_index);
		sprintf_s(txtname, "traindata/param_%06d.txt", g_index);

		BuildIR(params);
		SaveWav(wavname);
		SaveParams(txtname, params);

		g_index++;
	}

	// ================= 主函数 =================
	std::vector<float> params(LatticeOptimizer::NumRoomParams);
	LatticeOptimizer opt;
	static void GenerateDataset(int numRandom)
	{
		EnsureFolder();
		for (int n = 0; n < numRandom; ++n)
		{
			// step0: 随机
			opt.Reset();
			opt.GetNowRoomParams(params);
			opt.Regularization(params);
			SaveSample(params);

			// step1~3: 优化路径
			for (int k = 0; k < 2; ++k)
			{
				opt.RunOptimizer(1);
				opt.GetNowRoomParams(params);
				opt.Regularization(params);
				SaveSample(params);
			}

			printf("seed %d done (total %d samples)\n", n + 1, g_index);
		}
	}
}
///////////////////////////////////////////////////////////////////////
namespace IROutputFromTxt
{
	constexpr int SampleRate = 48000;
	constexpr int IRSeconds = 5;
	constexpr int IRNumSamples = SampleRate * IRSeconds;
	constexpr int NumParams = LatticeReverb3::LatticeReverb3::NumRoomParams;

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

	static bool LoadParamsTxt(const char* filename, std::vector<float>& roomParams)
	{
		roomParams.clear();
		roomParams.resize(NumParams);

		FILE* fp = nullptr;
#if defined(_MSC_VER)
		if (fopen_s(&fp, filename, "rb") != 0) fp = nullptr;
#else
		fp = fopen(filename, "rb");
#endif
		if (!fp) return false;

		for (int i = 0; i < NumParams; ++i)
		{
			if (fscanf_s(fp, "%f", &roomParams[i]) != 1)
			{
				fclose(fp);
				return false;
			}
		}

		fclose(fp);
		return true;
	}

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

	// 主入口：
	// 读取 output.txt -> 渲染 IR -> 输出 output.wav
	static bool Render(const char* txtFilename = "output.txt", const char* wavFilename = "output.wav")
	{
		std::vector<float> roomParams;
		if (!LoadParamsTxt(txtFilename, roomParams))
		{
			printf("Failed to load params from %s\n", txtFilename);
			return false;
		}

		if ((int)roomParams.size() != NumParams)
		{
			printf("Param count mismatch: got %d, expected %d\n", (int)roomParams.size(), NumParams);
			return false;
		}

		BuildIR(roomParams);

		if (!SaveWav(wavFilename))
		{
			printf("Failed to save wav to %s\n", wavFilename);
			return false;
		}

		printf("Rendered IR wav: %s\n", wavFilename);
		return true;
	}
}
///////////////////////////////////////////////////////////////////////
namespace IRLoadFromWav
{
	static float ReadFloatSample16(short v)
	{
		return (float)v / 32768.0f;
	}

	static float ReadFloatSample32(float v)
	{
		return v;
	}

	static bool Load(const char* filename, LatticeOptimizer& opt)
	{
		FILE* fp = nullptr;
#if defined(_MSC_VER)
		if (fopen_s(&fp, filename, "rb") != 0) fp = nullptr;
#else
		fp = fopen(filename, "rb");
#endif
		if (!fp)
		{
			printf("Failed to open %s\n", filename);
			return false;
		}

		// ===== 读 WAV header =====
		char chunkId[4];
		unsigned int chunkSize;
		char format[4];

		fread(chunkId, 1, 4, fp); // RIFF
		fread(&chunkSize, 4, 1, fp);
		fread(format, 1, 4, fp);  // WAVE

		unsigned short audioFormat = 1;
		unsigned short numChannels = 0;
		unsigned int sampleRate = 0;
		unsigned short bitsPerSample = 0;

		// 找 fmt 和 data
		while (!feof(fp))
		{
			char subchunkId[4];
			unsigned int subchunkSize;

			if (fread(subchunkId, 1, 4, fp) != 4) break;
			fread(&subchunkSize, 4, 1, fp);

			if (memcmp(subchunkId, "fmt ", 4) == 0)
			{
				fread(&audioFormat, 2, 1, fp);
				fread(&numChannels, 2, 1, fp);
				fread(&sampleRate, 4, 1, fp);

				fseek(fp, 6, SEEK_CUR); // byteRate + blockAlign
				fread(&bitsPerSample, 2, 1, fp);

				fseek(fp, subchunkSize - 16, SEEK_CUR);
			}
			else if (memcmp(subchunkId, "data", 4) == 0)
			{
				int bytesPerSample = bitsPerSample / 8;
				int totalSamples = subchunkSize / bytesPerSample / numChannels;

				const int N = LatticeOptimizer::testBlockSize;

				for (int i = 0; i < N; ++i)
				{
					float l = 0.0f;
					float r = 0.0f;

					if (i < totalSamples)
					{
						if (audioFormat == 1 && bitsPerSample == 16)
						{
							short s[2] = { 0,0 };
							fread(s, sizeof(short), numChannels, fp);

							if (numChannels == 1)
							{
								l = r = ReadFloatSample16(s[0]);
							}
							else
							{
								l = ReadFloatSample16(s[0]);
								r = ReadFloatSample16(s[1]);
							}
						}
						else if (audioFormat == 3 && bitsPerSample == 32)
						{
							float s[2] = { 0,0 };
							fread(s, sizeof(float), numChannels, fp);

							if (numChannels == 1)
							{
								l = r = s[0];
							}
							else
							{
								l = s[0];
								r = s[1];
							}
						}
						else
						{
							printf("Unsupported wav format\n");
							fclose(fp);
							return false;
						}
					}

					opt.global_ir_l[i] = l;
					opt.global_ir_r[i] = r;
				}

				fclose(fp);

				// ===== RMS 归一化（强烈建议）=====
				float energy = 0.0f;
				for (int i = 0; i < N; ++i)
				{
					energy += opt.global_ir_l[i] * opt.global_ir_l[i];
					energy += opt.global_ir_r[i] * opt.global_ir_r[i];
				}

				float rms = sqrtf(energy / (float)(2 * N) + 1e-20f);
				if (rms > 1e-6f)
				{
					float inv = 1.0f / rms;
					for (int i = 0; i < N; ++i)
					{
						opt.global_ir_l[i] *= inv;
						opt.global_ir_r[i] *= inv;
					}
				}

				printf("Loaded IR from %s (sr=%d, ch=%d)\n", filename, sampleRate, numChannels);
				return true;
			}
			else
			{
				fseek(fp, subchunkSize, SEEK_CUR);
			}
		}

		fclose(fp);
		printf("Invalid WAV file\n");
		return false;
	}
}
///////////////////////////////////////////////////////////////////////
LatticeOptimizer lattopt;
std::vector<float> roomParams(LatticeOptimizer::NumRoomParams);
float minLoss = 1e30f;
int main()
{
	//IRTrainDataGen::GenerateDataset(25);
	//IROutputFromTxt::Render();
	//IRLoadFromWav::Load("target_ir.wav", lattopt);
	for (;;)
	{
		lattopt.RunOptimizer(1);
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