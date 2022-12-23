import '@webmachinelearning/webnn-polyfill';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';

import {expose, proxy} from 'comlink';
import {setWasmPath, loadTFLiteModel, TFLiteWebModelRunnerOptions} from '@tensorflow/tfjs-tflite';


export type Predictor = {predict: (data: Float32Array) => Promise<Float32Array>};

const api = {
  async setWebNNPolyfillBackend(deviceType: string): Promise<void> {
    // Initiate webnn-polyfill
    console.warn(`WebNN ${deviceType} backend is not supported in this browser, \
fall back to webnn-polyfill.`);
    const context = navigator.ml.createContextSync();
    const tf = context.tf;
    const backend = deviceType == 'gpu' ? 'webgl' : 'wasm';
    await tf.setBackend(backend);
    await tf.ready();
    proxy(this.setWebNNPolyfillBackend);
  },
  setWasmPath: proxy(setWasmPath),
  async loadTFLiteModel(modelPath: string, options: TFLiteWebModelRunnerOptions): Promise<Predictor> {

    const model = await loadTFLiteModel(modelPath, options)

    const wrapped: Predictor = {
      async predict(data) {
        // Hacky hard-coded tensor shape :(
        const prediction = model.predict(tf.tensor(data, [1, 224, 224, 3]));
        return (prediction as tf.Tensor).dataSync() as Float32Array;
      }
    }

    return proxy(wrapped);
  }
}

export type Api = typeof api;

expose(api);
