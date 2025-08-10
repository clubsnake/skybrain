package com.armand.skybrain.perception

import kotlin.math.max

class MapGrid(
    val sizeXM: Int = 120,
    val sizeYM: Int = 120,
    val sizeZM: Int = 60,
    val resM: Int = 1,
) {
    data class Cell(var pOcc: Float, var sigma: Float, var ageS: Float, var heightHintM: Float?, var conf: Float)

    private val cells = HashMap<Long, Cell>()

    private fun key(
        x: Int,
        y: Int,
        z: Int,
    ) = (x.toLong() shl 42) or (y.toLong() shl 21) or (z.toLong() and 0x1FFFFF)

    fun integrate() { /* TODO */ }

    fun decay(dtS: Float) {
        cells.values.forEach {
            it.ageS += dtS
            it.pOcc = max(0f, it.pOcc - dtS * 0.01f)
        }
    }

    fun costAhead(): Float = 0f
}
