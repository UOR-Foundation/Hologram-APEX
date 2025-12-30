/**
 * Official ONNX Runtime Web - Stable Diffusion Pipeline
 *
 * Uses the official onnxruntime-web library for comparison with hologram-onnx.
 * Implements full SD Turbo diffusion pipeline.
 */

import * as ort from 'onnxruntime-web';

export interface OnnxTensor {
  data: Float32Array;
  shape: number[];
}

/**
 * Euler Discrete Scheduler for SD Turbo
 */
class EulerScheduler {
  private numTrainTimesteps: number = 1000;
  private betaStart: number = 0.00085;
  private betaEnd: number = 0.012;
  private timesteps: number[] = [];
  private sigmas: Float32Array = new Float32Array();

  constructor() {}

  setTimesteps(numInferenceSteps: number): void {
    const step = Math.floor(this.numTrainTimesteps / numInferenceSteps);
    this.timesteps = [];
    for (let i = 0; i < numInferenceSteps; i++) {
      this.timesteps.push(this.numTrainTimesteps - 1 - i * step);
    }
    this.timesteps.reverse();

    const betas = new Float32Array(this.numTrainTimesteps);
    for (let i = 0; i < this.numTrainTimesteps; i++) {
      betas[i] =
        this.betaStart +
        ((this.betaEnd - this.betaStart) * i) / (this.numTrainTimesteps - 1);
    }

    const alphas = new Float32Array(this.numTrainTimesteps);
    let alphasProd = 1.0;
    for (let i = 0; i < this.numTrainTimesteps; i++) {
      alphasProd *= 1.0 - betas[i];
      alphas[i] = alphasProd;
    }

    this.sigmas = new Float32Array(numInferenceSteps + 1);
    for (let i = 0; i < numInferenceSteps; i++) {
      const t = this.timesteps[i];
      const alphaProd = alphas[t];
      this.sigmas[i] = Math.sqrt((1 - alphaProd) / alphaProd);
    }
    this.sigmas[numInferenceSteps] = 0;
  }

  step(latents: Float32Array, modelOutput: Float32Array, timestepIdx: number): Float32Array {
    const sigma = this.sigmas[timestepIdx];
    const sigmaNext = this.sigmas[timestepIdx + 1];
    const dt = sigmaNext - sigma;

    const result = new Float32Array(latents.length);
    for (let i = 0; i < latents.length; i++) {
      const pred = latents[i] - sigma * modelOutput[i];
      result[i] = latents[i] + ((pred - latents[i]) * dt) / sigma;
    }

    return result;
  }

  getTimesteps(): number[] {
    return this.timesteps;
  }
}

/**
 * ONNX Runtime Web Diffusion Pipeline
 */
export class OnnxRuntimeWebDiffusion {
  private textEncoder: ort.InferenceSession | null = null;
  private unet: ort.InferenceSession | null = null;
  private vaeDecoder: ort.InferenceSession | null = null;
  private scheduler: EulerScheduler;

  private readonly latentChannels = 4;
  private readonly latentSize = 64;
  private readonly modelPath = '/models/onnx/sd-turbo-external';

  constructor() {
    this.scheduler = new EulerScheduler();
  }

  async initialize(): Promise<void> {
    if (this.textEncoder && this.unet && this.vaeDecoder) {
      console.log('[ONNX Runtime Web] Already initialized');
      return;
    }

    console.log('[ONNX Runtime Web] Initializing...');

    // Configure ONNX Runtime to use WebGPU
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    try {
      // Load text encoder
      console.log('[ONNX Runtime Web] Loading text encoder...');
      this.textEncoder = await ort.InferenceSession.create(
        `${this.modelPath}/text_encoder/model.onnx`,
        { executionProviders: ['webgpu'] }
      );
      console.log('[ONNX Runtime Web] ✓ Text encoder loaded');

      // Load U-Net
      console.log('[ONNX Runtime Web] Loading U-Net (1.7GB)...');
      this.unet = await ort.InferenceSession.create(
        `${this.modelPath}/unet/model.onnx`,
        { executionProviders: ['webgpu'] }
      );
      console.log('[ONNX Runtime Web] ✓ U-Net loaded');

      // Load VAE decoder
      console.log('[ONNX Runtime Web] Loading VAE decoder...');
      this.vaeDecoder = await ort.InferenceSession.create(
        `${this.modelPath}/vae_decoder/model.onnx`,
        { executionProviders: ['webgpu'] }
      );
      console.log('[ONNX Runtime Web] ✓ VAE decoder loaded');

      console.log('[ONNX Runtime Web] ✅ Ready!');
    } catch (error) {
      console.error('[ONNX Runtime Web] Initialization failed:', error);
      throw error;
    }
  }

  isReady(): boolean {
    return this.textEncoder !== null && this.unet !== null && this.vaeDecoder !== null;
  }

  async generateImage(
    tokenIds: number[],
    numInferenceSteps: number = 1,
    seed: number = 42
  ): Promise<Float32Array> {
    if (!this.isReady()) {
      throw new Error('Pipeline not initialized');
    }

    console.log(`[ONNX Runtime Web] Generating with ${numInferenceSteps} steps...`);

    // 1. Text encoding
    console.log('[ONNX Runtime Web] 1/4 Encoding text...');
    const textEmbeddings = await this.encodePrompt(tokenIds);
    console.log('[ONNX Runtime Web] ✓ Text encoded');

    // 2. Initialize latents
    console.log('[ONNX Runtime Web] 2/4 Initializing latents...');
    const latents = this.prepareLatents(seed);
    console.log('[ONNX Runtime Web] ✓ Latents initialized');

    // 3. Denoising loop
    console.log('[ONNX Runtime Web] 3/4 Running diffusion...');
    this.scheduler.setTimesteps(numInferenceSteps);
    const denoisedLatents = await this.denoisingLoop(latents, textEmbeddings, numInferenceSteps);
    console.log('[ONNX Runtime Web] ✓ Denoising complete');

    // 4. VAE decode
    console.log('[ONNX Runtime Web] 4/4 Decoding to image...');
    const image = await this.decodeLatents(denoisedLatents);
    console.log('[ONNX Runtime Web] ✓ Image decoded');

    return image;
  }

  private async encodePrompt(tokenIds: number[]): Promise<ort.Tensor> {
    if (!this.textEncoder) throw new Error('Text encoder not loaded');

    const maxLength = 77;
    const inputIds = new Float32Array(maxLength);
    for (let i = 0; i < Math.min(tokenIds.length, maxLength); i++) {
      inputIds[i] = tokenIds[i];
    }
    for (let i = tokenIds.length; i < maxLength; i++) {
      inputIds[i] = 49407; // CLIP padding token
    }

    const inputTensor = new ort.Tensor('float32', inputIds, [1, maxLength]);
    const outputs = await this.textEncoder.run({ input_ids: inputTensor });

    return outputs[this.textEncoder.outputNames[0]];
  }

  private prepareLatents(seed: number): Float32Array {
    const batchSize = 1;
    const numChannels = this.latentChannels;
    const height = this.latentSize;
    const width = this.latentSize;

    const latents = new Float32Array(batchSize * numChannels * height * width);

    let random = seed;
    const lcg = () => {
      random = (1103515245 * random + 12345) & 0x7fffffff;
      return random / 0x7fffffff;
    };

    for (let i = 0; i < latents.length; i += 2) {
      const u1 = lcg();
      const u2 = lcg();
      const r = Math.sqrt(-2.0 * Math.log(u1));
      const theta = 2.0 * Math.PI * u2;
      latents[i] = r * Math.cos(theta);
      if (i + 1 < latents.length) {
        latents[i + 1] = r * Math.sin(theta);
      }
    }

    return latents;
  }

  private async denoisingLoop(
    initialLatents: Float32Array,
    textEmbeddings: ort.Tensor,
    numInferenceSteps: number
  ): Promise<Float32Array> {
    if (!this.unet) throw new Error('U-Net not loaded');

    let latents: Float32Array = new Float32Array(initialLatents);
    const timesteps = this.scheduler.getTimesteps();

    for (let i = 0; i < numInferenceSteps; i++) {
      const t = timesteps[i];
      console.log(`[ONNX Runtime Web]   Step ${i + 1}/${numInferenceSteps}, t=${t}`);

      const timestepTensor = new ort.Tensor('float32', new Float32Array([t]), [1]);
      const latentTensor = new ort.Tensor('float32', latents, [
        1,
        this.latentChannels,
        this.latentSize,
        this.latentSize,
      ]);

      const outputs = await this.unet.run({
        sample: latentTensor,
        timestep: timestepTensor,
        encoder_hidden_states: textEmbeddings,
      });

      const noisePred = outputs[this.unet.outputNames[0]].data as Float32Array;
      latents = this.scheduler.step(latents, noisePred as Float32Array, i) as Float32Array;
    }

    return latents;
  }

  private async decodeLatents(latents: Float32Array): Promise<Float32Array> {
    if (!this.vaeDecoder) throw new Error('VAE decoder not loaded');

    const scaledLatents = new Float32Array(latents.length);
    const scaleFactor = 1.0 / 0.18215;
    for (let i = 0; i < latents.length; i++) {
      scaledLatents[i] = latents[i] * scaleFactor;
    }

    const latentTensor = new ort.Tensor('float32', scaledLatents, [
      1,
      this.latentChannels,
      this.latentSize,
      this.latentSize,
    ]);

    const outputs = await this.vaeDecoder.run({ latent_sample: latentTensor });
    const decodedImage = outputs[this.vaeDecoder.outputNames[0]].data as Float32Array;

    const height = 512;
    const width = 512;
    const channels = 3;
    const image = new Float32Array(height * width * channels);

    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        for (let c = 0; c < channels; c++) {
          const srcIdx = c * height * width + h * width + w;
          const dstIdx = (h * width + w) * channels + c;
          image[dstIdx] = (decodedImage[srcIdx] + 1.0) / 2.0;
          image[dstIdx] = Math.max(0, Math.min(1, image[dstIdx]));
        }
      }
    }

    return image;
  }
}
