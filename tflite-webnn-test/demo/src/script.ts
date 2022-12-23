/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-cpu';

import * as tf from '@tensorflow/tfjs-core';
//import * as tflite from '@tensorflow/tfjs-tflite';
import * as Comlink from 'comlink';
import type { Api, Predictor } from './worker';
import { IMAGENET_CLASSES } from './imagenet_classes';


const MODEL_LINK =
    'https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/default/1';

const worker = new Worker('/src/worker.js');

let tfliteModel: Predictor;
let stream: MediaStream;
let stopAnimation: boolean;
const imgEle = ele('img') as HTMLImageElement;
const camEle = ele('video') as HTMLVideoElement;
const backendElem = ele('#backend') as HTMLSelectElement;

const backendInfo: { [key: string]: string } = {
  'wasm': "XNNPACK delegate CPU",
  'cpu': 'WebNN delegate CPU',
  'gpu': 'WebNN delegate GPU',
};

// Handle backends' switching
backendElem.addEventListener('change', async () => {
  // Clear messages
  ele('.imgs-container').classList.add('hide');
  ele('#loading-msg').innerHTML = "";
  ele('#running-msg').innerHTML = "";
  document.querySelectorAll('.stats').forEach(elem => elem.innerHTML = "");

  stopAnimation = true;
  // Leave more time for stop last animation frame
  await new Promise(resolve => {
    setTimeout(async () => {
      await start(backendElem.value);
      resolve;
    }, 1000);
  });
});

async function loadModel(backend: string) {
  const {loadTFLiteModel, setWasmPath, setWebNNPolyfillBackend} =
      Comlink.wrap<Api>(worker);
  
  await setWasmPath('/node_modules/@tensorflow/tfjs-tflite/dist/');

  if (backend !== 'wasm') {
    // Use WebNN external delegate
    if (typeof MLGraphBuilder !== 'undefined') {
      // WebNN is supported in this browser
      try {
        await navigator.ml.createContext({deviceType: backend});
        console.log(`WebNN ${backend} backend is supported in this browser!`);
      } catch(e) {
        await setWebNNPolyfillBackend(backend);
      }
    } else {
      await setWebNNPolyfillBackend(backend);
    }
  }

  // Load model runner with the MobileNet V2 tflite model.
  const start = Date.now();
  tfliteModel = await loadTFLiteModel(MODEL_LINK, {
      // delegatePath: '/dummy_external_delegate_wasm.wasm',
      delegatePath: '/webnn_external_delegate_wasm.wasm',
    }
  );

  ele('#loading-msg').innerHTML = `Loaded <a href='${
    MODEL_LINK}' target='blank'>TFLite model</a> with <span class='backend'>${
    backendInfo[backend]}</span> backend in ${Date.now() - start} ms`;
}

async function setupCam() {
  const constraints = {
    video: {
      width: {
        min: 224,
        ideal: 224,
        max: 224,
      },
      height: {
        min: 224,
        ideal: 224,
        max: 224,
      },
    }
  };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  camEle.srcObject = stream;
  await new Promise(resolve => camEle.onplaying = resolve);
}

async function start(backend: string) {
  ele('#loading-msg').innerHTML = "Loading...";

  // Load TFLite model
  await loadModel(backend);
  ele('.imgs-container').classList.remove('hide');

  // Setup cam.
  if (!stream) await setupCam();

  ele('#running-msg').innerHTML = "Processing...";

  // Predict image element.
  await run(imgEle);

  // Predict camera element.
  stopAnimation = false;
  await run(camEle);

  ele('#running-msg').innerHTML += "Done!";
}

async function run(srcMedia: HTMLImageElement|HTMLVideoElement) {
  const stats = srcMedia.closest('.img-container')!.querySelector('.stats')!;
  // Run inference and render the result.
  const inferenceStart = Date.now();
  const output = await classifier(tfliteModel, srcMedia);
  const latency = Date.now() - inferenceStart;
  
  const topClass = tf.tidy(() => {
    // Post processing
    const slice = tf.slice(tf.tensor(output, [1, 1001]), [0, 1], [-1, 1000]);
    const argmax = tf.argMax(slice, 1);
    const labelIndex = argmax.dataSync()[0];
    const label = IMAGENET_CLASSES[labelIndex];
    const score = slice.dataSync()[labelIndex];
    return {label, score};
  });

  // Show output stat.
  stats.classList.add('show');
  stats.innerHTML = `
    <p><b>Label</b>: ${topClass.label}</p>
    <p><b>Score</b>: ${(topClass.score*100).toFixed(2)}%</p>
    <p><b>Latency</b>: ${latency.toFixed(1)} ms</p>
  `;

  if (srcMedia instanceof HTMLVideoElement && !stopAnimation) {
    requestAnimationFrame(() => {run(camEle)});
  }
}

// Predict and classify image/video element
async function classifier(
  tfliteModel: Predictor,
  ele: HTMLImageElement|HTMLVideoElement): Promise<Float32Array> {
  const input = tf.tidy(() => {
    // Get pixels data.
    const img = tf.browser.fromPixels(ele);
    // Normalize.
    //
    // Since the images are already 224*224 that matches the model's input size,
    // we don't resize them here.
    const input = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);
    return input;
  });

  // Run the inference.
  const output = await tfliteModel.predict(input.dataSync() as Float32Array);
  return output;
}

function ele(selector: string) {
  return document.querySelector(selector)!;
}

start('cpu');
