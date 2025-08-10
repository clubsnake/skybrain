package com.armand.skybrain.policy

object OaClamp {
    data class Cmd(val vx: Float, val vy: Float, val vz: Float, val yawRate: Float)

    fun apply(
        cmd: Cmd,
        oaDistM: Float,
    ): Cmd {
        // TODO: distance/TTC curve → slow down forward speed smoothly
        return cmd
    }
}
