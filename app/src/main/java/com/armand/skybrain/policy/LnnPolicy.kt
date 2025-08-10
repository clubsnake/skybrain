package com.armand.skybrain.policy

import org.tensorflow.lite.Interpreter

class LnnPolicy(private val tflite: Interpreter?) {
    data class Inputs(val features: FloatArray)

    data class Output(
        val vCapFwd: Float,
        val vCapLat: Float,
        val vCapVert: Float,
        val yawRateCap: Float,
        val scanProb: Float,
        val rampGain: Float,
        val confidence: Float,
    )

    fun infer(inp: Inputs): Output {
        // TODO: run TFLite and map outputs
        return Output(4f, 2f, 1.5f, 25f, 0.1f, 0.5f, 0.9f)
    }
}
