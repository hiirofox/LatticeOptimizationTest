#include <math.h>
#include <stdio.h>
#include <algorithm>

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
		constexpr static int NumLayers = 8;
		constexpr static int MaxDelayLength = 48000;

	private:
		DelayLine<MaxDelayLength> delays[NumLayers];
		float ks[NumLayers];
		float ds[NumLayers];
		float roomSize = 4800;

		template<int inLayer>//NumLayers-1
		inline float HProcSamp(float x)
		{
			if constexpr (inLayer >= 0)
			{
				float y = HProcSamp<inLayer - 1>(delays[inLayer].ReadSample());
				float a = (x + ks[inLayer] * y) * ds[inLayer];
				delays[inLayer].WriteSample(a);
				return y - a * ks[inLayer];
			}
			else
			{
				return x;
			}
		}

	public:
		float ProcessSample(float x)
		{
			return HProcSamp<NumLayers - 1>(x);
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
		void SetDelaysLength(float* delaysLength)
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
		void SetRoomSize(float roomSize)
		{
			this->roomSize = roomSize;
		}
	};
}
int main()
{

}