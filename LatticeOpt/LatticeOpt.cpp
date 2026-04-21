#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

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
		float roomSize = 4800;

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
				if (inLayer == NumLayers - 1) tapmix = 0;
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
			float roomSize = 4800;
			float decayTime = 0.99;//这个需要算RT60补偿
			float diffusion = 1.0;
			float mixTapDirect = 0.0;
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
		}
	};
}

/*
优化方向：
频谱方差小（无染色且噪）
接近RT60（方便校准）
左右声道相位差 方差大（声场宽）
*/
int main()
{

}