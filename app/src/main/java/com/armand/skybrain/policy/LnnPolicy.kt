package com.armand.skybrain.policy

import org.tensorflow.lite.Interpreter

private const val DEFAULT_VCAP_FWD = 1f
private const val DEFAULT_VCAP_LAT = 1f
private const val DEFAULT_VCAP_VERT = 0.5f
private const val DEFAULT_YAW_CAP = 30f
private const val DEFAULT_SCAN_PROB = 0f
private const val DEFAULT_RAMP_GAIN = 0.5f
private const val OUTPUT_SIZE = 7

private const val IDX_VCAP_FWD = 0
private const val IDX_VCAP_LAT = 1
private const val IDX_VCAP_VERT = 2
private const val IDX_YAW_CAP = 3
private const val IDX_SCAN_PROB = 4
private const val IDX_RAMP_GAIN = 5
private const val IDX_CONFIDENCE = 6

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
        // If no interpreter is provided, return conservative defaults
        val interpreter =
            tflite ?: return Output(
                vCapFwd = DEFAULT_VCAP_FWD,
                vCapLat = DEFAULT_VCAP_LAT,
                vCapVert = DEFAULT_VCAP_VERT,
                yawRateCap = DEFAULT_YAW_CAP,
                scanProb = DEFAULT_SCAN_PROB,
                rampGain = DEFAULT_RAMP_GAIN,
                confidence = 0f,
            )
        // Allocate an output array for model outputs
        val out = Array(1) { FloatArray(OUTPUT_SIZE) }
        // Run inference.  The interpreter expects a 2D input [batch, features]
        interpreter.run(arrayOf(inp.features), out)
        val arr = out[0]
        // Map outputs to typed values; clamp to reasonable ranges
        return Output(
            vCapFwd = (arr.getOrNull(IDX_VCAP_FWD) ?: DEFAULT_VCAP_FWD),
            vCapLat = (arr.getOrNull(IDX_VCAP_LAT) ?: DEFAULT_VCAP_LAT),
            vCapVert = (arr.getOrNull(IDX_VCAP_VERT) ?: DEFAULT_VCAP_VERT),
            yawRateCap = (arr.getOrNull(IDX_YAW_CAP) ?: DEFAULT_YAW_CAP),
            scanProb = arr.getOrNull(IDX_SCAN_PROB)?.coerceIn(0f, 1f) ?: DEFAULT_SCAN_PROB,
            rampGain = arr.getOrNull(IDX_RAMP_GAIN)?.coerceAtLeast(0f) ?: DEFAULT_RAMP_GAIN,
            confidence = arr.getOrNull(IDX_CONFIDENCE)?.coerceIn(0f, 1f) ?: 0f,
        )
    }
}
