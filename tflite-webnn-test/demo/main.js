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
import './webnn-polyfill/dist/webnn-polyfill.js';


(async () => {
  const context = await navigator.ml.createContext();
  const tf = context.tf;
  await tf.setBackend('webgl');
  await tf.ready();
  // The following code multiplies matrix a [3, 4] with matrix b [4, 3]
// into matrix c [3, 3].
const builder = new MLGraphBuilder(context);
const descA = {type: 'float32', dimensions: [3, 4]};
const a = builder.input('a', descA);
const descB = {type: 'float32', dimensions: [4, 3]};
const bufferB = new Float32Array(12).fill(0.5);
const b = builder.constant(descB, bufferB);
const c = builder.matmul(a, b);

const graph = await builder.build({c});
const bufferA = new Float32Array(12).fill(0.5);
const bufferC = new Float32Array(9);
const inputs = {'a': bufferA};
const outputs = {'c': bufferC};
await context.compute(graph, inputs, outputs);
console.log(`in node values: ${bufferC}`);
})();


// console.log(polyfill);
// const worker = new Worker('worker.js');

// Send init message to web worker. 
// worker.postMessage([
//   'create'
// ]);

