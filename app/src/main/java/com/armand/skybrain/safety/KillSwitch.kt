package com.armand.skybrain.safety

import com.armand.skybrain.sdk.VirtualStickLoop

class KillSwitch(private val vs: VirtualStickLoop, private val disableVs: (Boolean) -> Unit) {
    @Volatile private var engaged = false

    fun engage() {
        engaged = true
        vs.stop()
        disableVs(false)
    }

    fun isEngaged() = engaged
}
