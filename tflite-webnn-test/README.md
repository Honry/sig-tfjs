# WebNN external delegate support for Tensorflow.js TFLite

_WORK IN PROGRESS_

This is a new TensorFlow Lite external [Delegate][Delegate] based on
[Web Neural Network API][WebNN] (WebNN), a W3C specification for constructing
and executing computational graphs of neural networks. Under the hood, the WebNN
external delegate C++ implementation is packaged as a standalone WASM module
(Emscripten side module), [dynamically linked][dynamic linking] by the
[`tfjs-tflite`][tfjs-tflite] package, that would enable hardware acceleration of
TensorFlow Lite WebAssembly runtime by leveraging on-device accelerators for
web browsers.

Check out this [demo][demo] where we use this delegate to run a
[MobileNetV2][model] TFLite model on the web.

# Usage

Since this is just a delegate of [`tfjs-tflite`][tfjs-tflite] package, please
refer to the [full usage][tfjs-tflite usage] of [`tfjs-tflite`][tfjs-tflite]
package firstly.

## Load a TFLite model with WebNN external delegate
```js
const tfliteModel = await tflite.loadTFLiteModel(
    'url/to/your/model.tflite',
    {delegatePath: 'url/to/webnn-external-delegate.wasm'});
```


[Delegate]: https://www.tensorflow.org/lite/performance/delegates
[WebNN]: https://www.w3.org/TR/webnn/
[dynamic linking]: https://emscripten.org/docs/compiling/Dynamic-Linking.html
[model]: https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/default/1
[tfjs-tflite]: https://www.npmjs.com/package/@tensorflow/tfjs-tflite
[tfjs-tflite usage]: https://github.com/tensorflow/tfjs/tree/master/tfjs-tflite#usage