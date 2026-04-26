#pragma OPENCL EXTENSION cl_khr_fp64 : disable
#pragma OPENCL FP_CONTRACT OFF

#define LR_NUM_LAYERS 6
#define LR_NUM_PARAMS 52
#define LR_TEST_BLOCK_SIZE (65536*8)
#define LR_AUTO_FFT_SIZE (LR_TEST_BLOCK_SIZE*2)
#define LR_MAX_DELAY 8192
#define LR_DELAY_LINES 14
#define LR_DELAY_SCRATCH_FLOATS (LR_DELAY_LINES * LR_MAX_DELAY)
#define LR_LOSS_SCRATCH_FLOATS (LR_TEST_BLOCK_SIZE * 14)

#define LR_LINE_LATL0 0
#define LR_LINE_LATR0 6
#define LR_LINE_CROSS_L 12
#define LR_LINE_CROSS_R 13

#define LR_SCRATCH_BUFREL 0
#define LR_SCRATCH_BUFRER (LR_SCRATCH_BUFREL + LR_TEST_BLOCK_SIZE)
#define LR_SCRATCH_BUFIML (LR_SCRATCH_BUFRER + LR_TEST_BLOCK_SIZE)
#define LR_SCRATCH_BUFIMR (LR_SCRATCH_BUFIML + LR_TEST_BLOCK_SIZE)
#define LR_SCRATCH_AUTOCORRL (LR_SCRATCH_BUFIMR + LR_TEST_BLOCK_SIZE)
#define LR_SCRATCH_AUTOCORRR (LR_SCRATCH_AUTOCORRL + LR_AUTO_FFT_SIZE)
#define LR_SCRATCH_AUTOCORRIML (LR_SCRATCH_AUTOCORRR + LR_AUTO_FFT_SIZE)
#define LR_SCRATCH_AUTOCORRIMR (LR_SCRATCH_AUTOCORRIML + LR_AUTO_FFT_SIZE)
#define LR_SCRATCH_MAGL (LR_SCRATCH_AUTOCORRIMR + LR_AUTO_FFT_SIZE)
#define LR_SCRATCH_MAGR (LR_SCRATCH_MAGL + LR_TEST_BLOCK_SIZE / 2)
#define LR_SCRATCH_GDL (LR_SCRATCH_MAGR + LR_TEST_BLOCK_SIZE / 2)
#define LR_SCRATCH_GDR (LR_SCRATCH_GDL + LR_TEST_BLOCK_SIZE / 2)

#define LR_PI 3.1415926535897932384626f

typedef struct DelayLineCL
{
	__global float* dat;
	float out;
	float currentDelay;
	float targetDelay;
	float delayVelocity;
	int pos;
	int updateTime;
} DelayLineCL;

typedef struct LatticeCascadeCL
{
	DelayLineCL delays[LR_NUM_LAYERS];
	float ks[LR_NUM_LAYERS];
	float ds[LR_NUM_LAYERS];
	float outks[LR_NUM_LAYERS];
	float roomSize;
	float tapmix;
} LatticeCascadeCL;

typedef struct LatticeReverbCL
{
	LatticeCascadeCL latl;
	LatticeCascadeCL latr;
	DelayLineCL crossDelayL;
	DelayLineCL crossDelayR;
	float fbdl;
	float fbdr;
	float mixTapDirect;
} LatticeReverbCL;

static float lr_soft01(float x, float minv)
{
	float v = 1.0f - exp(-fabs(x));
	return v * (1.0f - minv) + minv;
}

static float lr_tanhfno0(float x, float minv)
{
	float v = tanh(x);
	if (v > 0.0f)
		v = v * (1.0f - minv) + minv;
	else
		v = v * (1.0f - minv) - minv;
	return v;
}

static void lr_regularize(const __global float* raw, float rp[LR_NUM_PARAMS])
{
	for (int i = 0; i < LR_NUM_PARAMS; ++i)
		rp[i] = raw[i];

	for (int i = 0; i < LR_NUM_LAYERS; ++i)
	{
		rp[i + 0 * LR_NUM_LAYERS] = lr_soft01(rp[i + 0 * LR_NUM_LAYERS], 0.01f);
		rp[i + 1 * LR_NUM_LAYERS] = lr_soft01(rp[i + 1 * LR_NUM_LAYERS], 0.01f);
		rp[i + 2 * LR_NUM_LAYERS] = lr_tanhfno0(rp[i + 2 * LR_NUM_LAYERS], 0.05f) * 0.9f;
		rp[i + 3 * LR_NUM_LAYERS] = lr_tanhfno0(rp[i + 3 * LR_NUM_LAYERS], 0.05f) * 0.9f;
		rp[i + 4 * LR_NUM_LAYERS] = lr_soft01(rp[i + 4 * LR_NUM_LAYERS], 0.9995f) * 0.99999999f;
		rp[i + 5 * LR_NUM_LAYERS] = lr_soft01(rp[i + 5 * LR_NUM_LAYERS], 0.9995f) * 0.99999999f;
		rp[i + 6 * LR_NUM_LAYERS] = lr_soft01(rp[i + 6 * LR_NUM_LAYERS], 0.01f);
		rp[i + 7 * LR_NUM_LAYERS] = lr_soft01(rp[i + 7 * LR_NUM_LAYERS], 0.01f);
	}

	rp[8 * LR_NUM_LAYERS + 2] = lr_soft01(rp[8 * LR_NUM_LAYERS + 2], 0.01f);
	rp[8 * LR_NUM_LAYERS + 3] = lr_soft01(rp[8 * LR_NUM_LAYERS + 3], 0.01f);
	rp[8 * LR_NUM_LAYERS + 0] = -lr_soft01(rp[8 * LR_NUM_LAYERS + 0], 0.65f) * 0.99999999f;
	rp[8 * LR_NUM_LAYERS + 1] = -lr_soft01(rp[8 * LR_NUM_LAYERS + 1], 0.65f) * 0.99999999f;
}

static void lr_delay_bind(DelayLineCL* d, __global float* base)
{
	d->dat = base;
	d->out = 0.0f;
	d->currentDelay = 0.0f;
	d->targetDelay = 2.0f;
	d->delayVelocity = 0.0f;
	d->pos = 0;
	d->updateTime = 0;
}

static void lr_delay_zero_data(DelayLineCL* d)
{
	for (int i = 0; i < LR_MAX_DELAY; ++i)
		d->dat[i] = 0.0f;
}

static void lr_delay_set_delay_time(DelayLineCL* d, float t)
{
	if (t < 2.0f)
		t = 2.0f;
	if (t > (float)(LR_MAX_DELAY - 4))
		t = (float)(LR_MAX_DELAY - 4);
	d->targetDelay = t;
}

static void lr_delay_reset(DelayLineCL* d)
{
	lr_delay_zero_data(d);
	d->out = 0.0f;
	d->pos = 0;
	d->currentDelay = d->targetDelay;
}

static float lr_delay_read_hermite(DelayLineCL* d, float delay)
{
	float readPos = (float)d->pos - delay;
	while (readPos < 0.0f)
		readPos += (float)LR_MAX_DELAY;
	while (readPos >= (float)LR_MAX_DELAY)
		readPos -= (float)LR_MAX_DELAY;

	int i1 = (int)readPos;
	float f = readPos - (float)i1;

	int i0 = (i1 - 1 + LR_MAX_DELAY) % LR_MAX_DELAY;
	int i2 = (i1 + 1) % LR_MAX_DELAY;
	int i3 = (i1 + 2) % LR_MAX_DELAY;

	float y0 = d->dat[i0];
	float y1 = d->dat[i1];
	float y2 = d->dat[i2];
	float y3 = d->dat[i3];

	float c0 = y1;
	float c1 = 0.5f * (y2 - y0);
	float c2 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
	float c3 = 0.5f * (y3 - y0) + 1.5f * (y1 - y2);

	return ((c3 * f + c2) * f + c1) * f + c0;
}

static void lr_delay_write(DelayLineCL* d, float val)
{
	d->dat[d->pos] = val;

	d->updateTime--;
	if (d->updateTime <= 0)
	{
		d->delayVelocity = (d->targetDelay - d->currentDelay) / 10000.0f;
		d->updateTime = 512;
	}

	if (fabs(d->targetDelay - d->currentDelay) > 0.0001f)
		d->currentDelay += d->delayVelocity;
	else
		d->currentDelay = d->targetDelay;

	d->out = lr_delay_read_hermite(d, d->currentDelay);

	if (++d->pos >= LR_MAX_DELAY)
		d->pos = 0;
}

static void lr_cascade_bind(LatticeCascadeCL* c, __global float* taskScratch, int lineBase)
{
	c->roomSize = 512.0f;
	c->tapmix = 0.0f;
	for (int i = 0; i < LR_NUM_LAYERS; ++i)
	{
		lr_delay_bind(&c->delays[i], taskScratch + (lineBase + i) * LR_MAX_DELAY);
		c->ks[i] = 0.0f;
		c->ds[i] = 0.0f;
		c->outks[i] = 0.0f;
	}
}

static void lr_cascade_reset(LatticeCascadeCL* c)
{
	for (int i = 0; i < LR_NUM_LAYERS; ++i)
		lr_delay_reset(&c->delays[i]);
}

static void lr_cascade_set_delays_length(LatticeCascadeCL* c, float delaysLength[LR_NUM_LAYERS])
{
	float maxv = 0.0f;
	for (int i = 0; i < LR_NUM_LAYERS; ++i)
	{
		if (delaysLength[i] > maxv)
			maxv = delaysLength[i];
	}
	for (int i = 0; i < LR_NUM_LAYERS; ++i)
	{
		float delayt = delaysLength[i] / maxv * c->roomSize;
		lr_delay_set_delay_time(&c->delays[i], delayt);
	}
}

static void lr_cascade_set_outks(LatticeCascadeCL* c, float outKs[LR_NUM_LAYERS])
{
	float energy = 0.0f;
	for (int i = 0; i < LR_NUM_LAYERS; ++i)
		energy += outKs[i] * outKs[i];
	energy = sqrt(energy) * sqrt((float)LR_NUM_LAYERS);
	for (int i = 0; i < LR_NUM_LAYERS; ++i)
		c->outks[i] = outKs[i] / energy;
}

static float lr_cascade_process_sample(LatticeCascadeCL* c, float x)
{
	float xs[LR_NUM_LAYERS];
	float cur = x;
	c->tapmix = 0.0f;

	for (int i = LR_NUM_LAYERS - 1; i >= 0; --i)
	{
		xs[i] = cur;
		cur = c->delays[i].out;
	}

	float y = cur;
	for (int i = 0; i < LR_NUM_LAYERS; ++i)
	{
		float a = (xs[i] + c->ks[i] * y) * c->ds[i];
		lr_delay_write(&c->delays[i], a);
		float out = y - a * c->ks[i];
		c->tapmix += out * c->outks[i];
		y = out;
	}

	return y;
}

static void lr_reverb_bind(LatticeReverbCL* r, __global float* taskScratch)
{
	lr_cascade_bind(&r->latl, taskScratch, LR_LINE_LATL0);
	lr_cascade_bind(&r->latr, taskScratch, LR_LINE_LATR0);
	lr_delay_bind(&r->crossDelayL, taskScratch + LR_LINE_CROSS_L * LR_MAX_DELAY);
	lr_delay_bind(&r->crossDelayR, taskScratch + LR_LINE_CROSS_R * LR_MAX_DELAY);
	r->fbdl = 0.0f;
	r->fbdr = 0.0f;
	r->mixTapDirect = 0.2f;
}

static void lr_reverb_reset(LatticeReverbCL* r)
{
	lr_cascade_reset(&r->latl);
	lr_cascade_reset(&r->latr);
	lr_delay_reset(&r->crossDelayL);
	lr_delay_reset(&r->crossDelayR);
}

static void lr_reverb_setup(LatticeReverbCL* r, float rp[LR_NUM_PARAMS])
{
	float tsl[LR_NUM_LAYERS];
	float tsr[LR_NUM_LAYERS];
	float ksl[LR_NUM_LAYERS];
	float ksr[LR_NUM_LAYERS];
	float dsl[LR_NUM_LAYERS];
	float dsr[LR_NUM_LAYERS];
	float outksl[LR_NUM_LAYERS];
	float outksr[LR_NUM_LAYERS];

	for (int i = 0; i < LR_NUM_LAYERS; ++i)
	{
		tsl[i] = rp[i + 0 * LR_NUM_LAYERS];
		tsr[i] = rp[i + 1 * LR_NUM_LAYERS];
		ksl[i] = rp[i + 2 * LR_NUM_LAYERS];
		ksr[i] = rp[i + 3 * LR_NUM_LAYERS];
		dsl[i] = rp[i + 4 * LR_NUM_LAYERS] * 0.9999f;
		dsr[i] = rp[i + 5 * LR_NUM_LAYERS] * 0.9999f;
		outksl[i] = rp[i + 6 * LR_NUM_LAYERS];
		outksr[i] = rp[i + 7 * LR_NUM_LAYERS];
	}

	r->latl.roomSize = 1200.0f;
	r->latr.roomSize = 1200.0f;
	lr_cascade_set_delays_length(&r->latl, tsl);
	lr_cascade_set_delays_length(&r->latr, tsr);

	for (int i = 0; i < LR_NUM_LAYERS; ++i)
	{
		r->latl.ks[i] = ksl[i];
		r->latr.ks[i] = ksr[i];
		r->latl.ds[i] = dsl[i];
		r->latr.ds[i] = dsr[i];
	}

	r->fbdl = rp[8 * LR_NUM_LAYERS + 0] * 0.9999f;
	r->fbdr = rp[8 * LR_NUM_LAYERS + 1] * 0.9999f;
	lr_delay_set_delay_time(&r->crossDelayL, rp[8 * LR_NUM_LAYERS + 2] * 1200.0f);
	lr_delay_set_delay_time(&r->crossDelayR, rp[8 * LR_NUM_LAYERS + 3] * 1200.0f);

	lr_cascade_set_outks(&r->latl, outksl);
	lr_cascade_set_outks(&r->latr, outksr);

	lr_reverb_reset(r);
}

static void lr_reverb_process_sample(LatticeReverbCL* r, float inl, float inr, float* outl, float* outr)
{
	float lastoutl = r->crossDelayL.out;
	float lastoutr = r->crossDelayR.out;
	float xl = inl + lastoutr * r->fbdl;
	float xr = inr + lastoutl * r->fbdr;
	float outvl = lr_cascade_process_sample(&r->latl, xl);
	float outvr = lr_cascade_process_sample(&r->latr, xr);
	lr_delay_write(&r->crossDelayL, outvr);
	lr_delay_write(&r->crossDelayR, outvl);

	float tapMixOutl = r->latl.tapmix;
	float tapMixOutr = r->latr.tapmix;

	*outl = outvl * (1.0f - r->mixTapDirect) + tapMixOutl * r->mixTapDirect;
	*outr = outvr * (1.0f - r->mixTapDirect) + tapMixOutr * r->mixTapDirect;
}

static void lr_fft(__global float* re, __global float* im, int n, int inv)
{
	int j = 0;
	for (int i = 1; i < n; ++i)
	{
		int bit = n >> 1;
		while (j & bit)
		{
			j ^= bit;
			bit >>= 1;
		}
		j ^= bit;
		if (i < j)
		{
			float tmp = re[i];
			re[i] = re[j];
			re[j] = tmp;
			tmp = im[i];
			im[i] = im[j];
			im[j] = tmp;
		}
	}

	for (int m = 2; m <= n; m <<= 1)
	{
		float angle = -(float)inv * 2.0f * LR_PI / (float)m;
		float wm_re = cos(angle);
		float wm_im = sin(angle);
		for (int k = 0; k < n; k += m)
		{
			float w_re = 1.0f;
			float w_im = 0.0f;
			for (int jj = 0; jj < m / 2; ++jj)
			{
				int t = k + jj;
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

	if (inv == -1)
	{
		for (int i = 0; i < n; ++i)
		{
			re[i] /= (float)n;
			im[i] /= (float)n;
		}
	}
}

static float lr_eval_loss_one(const __global float* rawParams, __global float* taskScratch)
{
	float rp[LR_NUM_PARAMS];
	__global float* delayScratch = taskScratch;
	__global float* lossScratch = taskScratch + LR_DELAY_SCRATCH_FLOATS;
	__global float* bufrel = lossScratch + LR_SCRATCH_BUFREL;
	__global float* bufrer = lossScratch + LR_SCRATCH_BUFRER;
	__global float* bufiml = lossScratch + LR_SCRATCH_BUFIML;
	__global float* bufimr = lossScratch + LR_SCRATCH_BUFIMR;
	__global float* autocorrl = lossScratch + LR_SCRATCH_AUTOCORRL;
	__global float* autocorrr = lossScratch + LR_SCRATCH_AUTOCORRR;
	__global float* autocorriml = lossScratch + LR_SCRATCH_AUTOCORRIML;
	__global float* autocorrimr = lossScratch + LR_SCRATCH_AUTOCORRIMR;
	__global float* magl = lossScratch + LR_SCRATCH_MAGL;
	__global float* magr = lossScratch + LR_SCRATCH_MAGR;
	__global float* gdl = lossScratch + LR_SCRATCH_GDL;
	__global float* gdr = lossScratch + LR_SCRATCH_GDR;
	LatticeReverbCL reverb;

	lr_regularize(rawParams, rp);
	lr_reverb_bind(&reverb, delayScratch);
	lr_reverb_setup(&reverb, rp);

	float tmpl = 1.0f;
	float tmpr = 1.0f;
	lr_reverb_process_sample(&reverb, tmpl, tmpr, &tmpl, &tmpr);

	for (int i = 0; i < LR_TEST_BLOCK_SIZE; ++i)
	{
		float outl;
		float outr;
		lr_reverb_process_sample(&reverb, 0.0f, 0.0f, &outl, &outr);
		bufrel[i] = outl;
		bufrer[i] = outr;
	}

	for (int i = 0; i < LR_TEST_BLOCK_SIZE; ++i)
	{
		float window = 1.0f;
		autocorrl[i] = bufrel[i] * window;
		autocorrr[i] = bufrer[i] * window;
		autocorrl[LR_TEST_BLOCK_SIZE + i] = 0.0f;
		autocorrr[LR_TEST_BLOCK_SIZE + i] = 0.0f;
	}
	for (int i = 0; i < LR_AUTO_FFT_SIZE; ++i)
	{
		autocorriml[i] = 0.0f;
		autocorrimr[i] = 0.0f;
	}

	lr_fft(autocorrl, autocorriml, LR_AUTO_FFT_SIZE, 1);
	lr_fft(autocorrr, autocorrimr, LR_AUTO_FFT_SIZE, 1);

	for (int i = 0; i < LR_AUTO_FFT_SIZE; ++i)
	{
		autocorrl[i] = autocorrl[i] * autocorrl[i] + autocorriml[i] * autocorriml[i];
		autocorrr[i] = autocorrr[i] * autocorrr[i] + autocorrimr[i] * autocorrimr[i];
		autocorriml[i] = 0.0f;
		autocorrimr[i] = 0.0f;
	}

	lr_fft(autocorrl, autocorriml, LR_AUTO_FFT_SIZE, -1);
	lr_fft(autocorrr, autocorrimr, LR_AUTO_FFT_SIZE, -1);

	float maxcorrl = 1e-30f;
	float maxcorrr = 1e-30f;
	for (int i = 1; i < LR_AUTO_FFT_SIZE - 1; ++i)
	{
		if (maxcorrl < autocorrl[i])
			maxcorrl = autocorrl[i];
		if (maxcorrr < autocorrr[i])
			maxcorrr = autocorrr[i];
	}

	for (int i = 0; i < LR_TEST_BLOCK_SIZE; ++i)
	{
		float window = 0.5f * (1.0f - cos(2.0f * LR_PI * (float)i / (float)LR_TEST_BLOCK_SIZE));
		bufrel[i] *= window;
		bufrer[i] *= window;
		bufiml[i] = 0.0f;
		bufimr[i] = 0.0f;
	}

	lr_fft(bufrel, bufiml, LR_TEST_BLOCK_SIZE, 1);
	lr_fft(bufrer, bufimr, LR_TEST_BLOCK_SIZE, 1);

	float avg = 0.0f;
	float avgl = 0.0f;
	float avgr = 0.0f;
	float s2 = 0.0f;
	float s2l = 0.0f;
	float s2r = 0.0f;
	int numBins = (int)((float)(LR_TEST_BLOCK_SIZE / 2) * 0.7f);
	float specmax = 1e-30f;
	float specmin = 1e30f;
	float lastphasel = 0.0f;
	float lastphaser = 0.0f;
	float gdavgl = 0.0f;
	float gdavgr = 0.0f;
	float gds2l = 0.0f;
	float gds2r = 0.0f;
	int startbin = 20;

	for (int i = startbin; i < numBins; ++i)
	{
		float maglv = sqrt(bufrel[i] * bufrel[i] + bufiml[i] * bufiml[i]);
		float magrv = sqrt(bufrer[i] * bufrer[i] + bufimr[i] * bufimr[i]);
		magl[i] = maglv;
		magr[i] = magrv;
		avg += (maglv + magrv) * 0.5f;
		avgl += maglv;
		avgr += magrv;
		if (specmax < maglv)
			specmax = maglv;
		if (specmax < magrv)
			specmax = magrv;
		if (specmin > maglv)
			specmin = maglv;
		if (specmin > magrv)
			specmin = magrv;

		float deltaOmega = 2.0f * LR_PI / (float)LR_TEST_BLOCK_SIZE;
		float phasel = atan2(bufiml[i], bufrel[i]);
		float phaser = atan2(bufimr[i], bufrer[i]);
		float dphil = phasel - lastphasel;
		float dphir = phaser - lastphaser;
		while (dphil > LR_PI) dphil -= 2.0f * LR_PI;
		while (dphil < -LR_PI) dphil += 2.0f * LR_PI;
		while (dphir > LR_PI) dphir -= 2.0f * LR_PI;
		while (dphir < -LR_PI) dphir += 2.0f * LR_PI;
		float groupdelayl = -dphil / deltaOmega;
		float groupdelayr = -dphir / deltaOmega;
		lastphasel = phasel;
		lastphaser = phaser;
		gdavgl += groupdelayl;
		gdavgr += groupdelayr;
		gdl[i] = groupdelayl;
		gdr[i] = groupdelayr;
	}

	avg /= (float)(numBins - startbin);
	avgl /= (float)(numBins - startbin);
	avgr /= (float)(numBins - startbin);
	gdavgl /= (float)(numBins - startbin);
	gdavgr /= (float)(numBins - startbin);
	specmax /= avg;
	specmin /= avg;

	for (int i = startbin; i < numBins; ++i)
	{
		float magv = (magl[i] + magr[i]) * 0.5f / avg;
		float magvl = magl[i] / avgl;
		float magvr = magr[i] / avgr;
		s2 += (magv - 1.0f) * (magv - 1.0f);
		s2l += (magvl - 1.0f) * (magvl - 1.0f);
		s2r += (magvr - 1.0f) * (magvr - 1.0f);
		float groupdelayl = gdl[i] / gdavgl;
		float groupdelayr = gdr[i] / gdavgr;
		gds2l += (groupdelayl - 1.0f) * (groupdelayl - 1.0f);
		gds2r += (groupdelayr - 1.0f) * (groupdelayr - 1.0f);
	}

	float specflatloss = (s2 * 0.25f + fmax(s2l, s2r) * 0.75f) * 0.1f;
	float diffloss = avgl - avgr;
	float maxminloss = specmax - specmin;
	float gdloss = 1.0f / (fmin(gds2l, gds2r) + 0.1f);
	float maxcorrloss = log(maxcorrl + maxcorrr + fmax(maxcorrl, maxcorrr) * 4.0f) + 80.0f;
	diffloss = diffloss * diffloss * 100000.0f;
	maxminloss = maxminloss * maxminloss * 1.0f;
	gdloss = 1000.0f / (gdloss + 1e-20f);
	(void)specflatloss;
	(void)diffloss;
	(void)maxminloss;
	(void)gdloss;

	return maxcorrloss;
}

__kernel void EvalLatticeLossBatch(
	__global const float* params,
	__global float* losses,
	__global float* delayScratch,
	const int numTasks)
{
	int taskId = get_global_id(0);
	if (taskId >= numTasks)
		return;

	__global const float* taskParams = params + taskId * LR_NUM_PARAMS;
	__global float* taskScratch = delayScratch + taskId * (LR_DELAY_SCRATCH_FLOATS + LR_LOSS_SCRATCH_FLOATS);
	losses[taskId] = lr_eval_loss_one(taskParams, taskScratch);
}

__kernel void RenderLatticeIRBatch(
	__global const float* params,
	__global float* outL,
	__global float* outR,
	__global float* delayScratch,
	const int numTasks,
	const int numSamples)
{
	int taskId = get_global_id(0);
	if (taskId >= numTasks)
		return;

	__global const float* taskParams = params + taskId * LR_NUM_PARAMS;
	__global float* taskScratch = delayScratch + taskId * (LR_DELAY_SCRATCH_FLOATS + LR_LOSS_SCRATCH_FLOATS);
	__global float* taskOutL = outL + taskId * numSamples;
	__global float* taskOutR = outR + taskId * numSamples;

	float rp[LR_NUM_PARAMS];
	LatticeReverbCL reverb;

	lr_regularize(taskParams, rp);
	lr_reverb_bind(&reverb, taskScratch);
	lr_reverb_setup(&reverb, rp);

	float tmpl = 1.0f;
	float tmpr = 1.0f;
	lr_reverb_process_sample(&reverb, tmpl, tmpr, &tmpl, &tmpr);

	for (int i = 0; i < numSamples; ++i)
	{
		float l;
		float r;
		lr_reverb_process_sample(&reverb, 0.0f, 0.0f, &l, &r);
		taskOutL[i] = l;
		taskOutR[i] = r;
	}
}
