package com.armand.skybrain.sdk

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class VirtualStickLoop(
    private val dji: DjiBridge,
    private val commandProvider: () -> VsCommand,
) {
    data class VsCommand(val roll: Float, val pitch: Float, val yawRate: Float, val throttle: Float)

    private var job: Job? = null

    fun start(scope: CoroutineScope = CoroutineScope(Dispatchers.Default)) {
        stop()
        job =
            scope.launch {
                val periodMs = 50L // 20 Hz
                while (isActive) {
                    val cmd = commandProvider.invoke()
                    dji.sendVirtualStick(cmd.roll, cmd.pitch, cmd.yawRate, cmd.throttle)
                    delay(periodMs)
                }
            }
    }

    fun stop() {
        job?.cancel()
        job = null
    }
}
