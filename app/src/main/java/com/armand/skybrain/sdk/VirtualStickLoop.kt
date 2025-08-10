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
    private val onWatchdogTrip: (String) -> Unit = {},
) {
    /** Simple data class representing body-frame velocity commands. */
    data class VsCommand(val roll: Float, val pitch: Float, val yawRate: Float, val throttle: Float)

    private var job: Job? = null

    /**
     * Starts the virtual stick sender loop. This loop runs at a fixed period
     * (default 25 Hz) and will stop automatically if a send call takes too long
     * (>200 ms) or throws an exception. When that happens, [onWatchdogTrip]
     * is invoked on a background thread and the VS mode is disabled via
     * [DjiBridge.enableVirtualStick].
     *
     * @param scope Coroutine scope for launching the sender loop
     */
    private companion object {
        private const val PERIOD_MS: Long = 40L // 25 Hz
        private const val MAX_SEND_MS: Long = 200L
        private const val NS_PER_MS: Long = 1_000_000L
    }

    fun start(scope: CoroutineScope = CoroutineScope(Dispatchers.Default)) {
        stop()
        job =
            scope.launch {
                while (isActive) {
                    val t0 = System.nanoTime()
                    var shouldStop = false
                    try {
                        val cmd = commandProvider.invoke()
                        dji.sendVirtualStick(cmd.roll, cmd.pitch, cmd.yawRate, cmd.throttle)
                    } catch (e: IllegalStateException) {
                        onWatchdogTrip("sendVirtualStick exception: ${e.localizedMessage}")
                        dji.enableVirtualStick(false)
                        shouldStop = true
                    }
                    val elapsedMs = (System.nanoTime() - t0) / NS_PER_MS
                    if (elapsedMs > MAX_SEND_MS) {
                        onWatchdogTrip("Virtual stick send took ${elapsedMs}ms (>${MAX_SEND_MS}ms)")
                        dji.enableVirtualStick(false)
                        shouldStop = true
                    }
                    if (shouldStop) break
                    delay(PERIOD_MS)
                }
            }
    }

    /**
     * Stops the sender loop, if running. Safe to call repeatedly. Virtual stick
     * mode itself is not toggled here; callers should disable virtual stick
     * via [DjiBridge.enableVirtualStick] when appropriate.
     */
    fun stop() {
        job?.cancel()
        job = null
    }
}
