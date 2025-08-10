package com.armand.skybrain.ui

import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.armand.skybrain.policy.CommandMixer
import com.armand.skybrain.policy.LnnPolicy
import com.armand.skybrain.policy.LookThenGo
import com.armand.skybrain.policy.buildNavFeatures
import com.armand.skybrain.sdk.DjiBridge
import com.armand.skybrain.sdk.VirtualStickLoop
import kotlinx.coroutines.flow.collectLatest
import org.tensorflow.lite.Interpreter

class MainActivity : ComponentActivity() {
    private val djiBridge = DjiBridge()
    private lateinit var vsLoop: VirtualStickLoop

    // LNN policy; interpreter is lazy-loaded (null for now)
    private val navLnnPolicy: LnnPolicy by lazy {
        // Attempt to load TFLite model from assets.  If it fails, pass null.
        val assetManager = assets
        val modelName = "nav_lnn.tflite"
        val interpreter: Interpreter? =
            try {
                val fd = assetManager.openFd(modelName)
                val input = fd.createInputStream()
                val modelBytes = ByteArray(fd.length.toInt())
                input.read(modelBytes)
                input.close()
                Interpreter(modelBytes)
            } catch (e: java.io.IOException) {
                Log.e("MainActivity", "Failed to load $modelName: ${e.message}")
                null
            } catch (e: IllegalArgumentException) {
                Log.e("MainActivity", "Invalid TFLite model $modelName: ${e.message}")
                null
            }
        LnnPolicy(interpreter)
    }

    // Current desired command from UI sliders (body-frame velocities).  We store
    // as a LookThenGo.Desired so that FeatureBuilder can use it directly.
    private var currentDesired by mutableStateOf(LookThenGo.Desired(0f, 0f, 0f, 0f))

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Initialize VS loop with command provider reading from currentVsCommand
        vsLoop =
            VirtualStickLoop(djiBridge, {
                // Build final VS command by mixing policy, look-then-go, and OA clamp
                val state = djiBridge.flightStateStore.state.value
                // Build feature vector for NavLNN
                val feats = buildNavFeatures(state, currentDesired)
                val navOut = navLnnPolicy.infer(LnnPolicy.Inputs(feats))
                // Mix commands
                CommandMixer.mix(state, currentDesired, navOut)
            }) { msg ->
                // Called when watchdog trips; show toast on UI thread
                runOnUiThread {
                    Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
                }
                // Disable VS on watchdog
                djiBridge.enableVirtualStick(false)
            }
        setContent {
            appRoot(
                djiBridge = djiBridge,
                vsLoop = vsLoop,
                onDesiredChange = { des -> currentDesired = des },
            )
        }
    }
}

@Suppress("LongMethod")
@Composable
fun appRoot(
    djiBridge: DjiBridge,
    vsLoop: VirtualStickLoop,
    onDesiredChange: (LookThenGo.Desired) -> Unit,
) {
    val ctx = LocalContext.current
    // Local state for VS enabled flag
    var vsEnabled by remember { mutableStateOf(false) }
    // Local state for whether simulator is running
    var simRunning by remember { mutableStateOf(false) }
    // Desired body-frame velocities (vx, vy, vz, yawRate)
    var desired by remember { mutableStateOf(LookThenGo.Desired(0f, 0f, 0f, 0f)) }
    // Observe connection status from DjiBridge
    val statusFlow = djiBridge.status

    // Launch effect to collect status changes and react
    LaunchedEffect(Unit) {
        statusFlow.collectLatest { status ->
            // Show toasts on major transitions
            when (status) {
                is DjiBridge.Status.Connected -> {
                    Toast.makeText(ctx, "Connected to ${status.model}", Toast.LENGTH_SHORT).show()
                }
                is DjiBridge.Status.Disconnected -> {
                    Toast.makeText(ctx, "Disconnected", Toast.LENGTH_SHORT).show()
                    vsEnabled = false
                    simRunning = false
                }
                is DjiBridge.Status.Connecting -> {
                    Toast.makeText(ctx, "Connecting…", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    MaterialTheme {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("SkyBrain — Pilot", style = MaterialTheme.typography.titleLarge)

            // Buttons row
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {
                    // Register DJI; uses manifest App Key
                    djiBridge.registerAndConnect(ctx)
                }) { Text("Init DJI") }
                Button(onClick = {
                    // Enable VS and start loop
                    djiBridge.enableVirtualStick(true)
                    vsEnabled = true
                    vsLoop.start()
                }, enabled = (statusFlow.value is DjiBridge.Status.Connected) && !vsEnabled) { Text("Enable VS") }
                Button(onClick = {
                    // Disable VS and stop loop
                    vsLoop.stop()
                    djiBridge.enableVirtualStick(false)
                    vsEnabled = false
                }, enabled = vsEnabled) { Text("Disable VS") }
            }

            // Simulator controls row
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {
                    djiBridge.startSimulator()
                    simRunning = true
                }, enabled = (statusFlow.value is DjiBridge.Status.Connected) && !simRunning) { Text("Start Sim") }
                Button(onClick = {
                    djiBridge.stopSimulator()
                    simRunning = false
                }, enabled = simRunning) { Text("Stop Sim") }
            }

            // Display status
            val statusText =
                when (val s = statusFlow.value) {
                    is DjiBridge.Status.Connected -> "Connected (${s.model})"
                    is DjiBridge.Status.Connecting -> "Connecting…"
                    else -> "Disconnected"
                }
            Text("Status: $statusText")
            Text("VS: ${if (vsEnabled) "ON" else "OFF"} | Sim: ${if (simRunning) "RUN" else "OFF"}")

            // Sliders for manual body-frame commands
            if (vsEnabled) {
                Spacer(Modifier.height(8.dp))
                Text("Forward/back (vx) m/s: %.2f".format(desired.vx))
                Slider(value = desired.vx, onValueChange = {
                    desired = desired.copy(vx = it)
                    onDesiredChange(desired)
                }, valueRange = -1f..1f)
                Text("Left/right (vy) m/s: %.2f".format(desired.vy))
                Slider(value = desired.vy, onValueChange = {
                    desired = desired.copy(vy = it)
                    onDesiredChange(desired)
                }, valueRange = -1f..1f)
                Text("Yaw rate (deg/s): %.1f".format(desired.yawRate))
                Slider(value = desired.yawRate, onValueChange = {
                    desired = desired.copy(yawRate = it)
                    onDesiredChange(desired)
                }, valueRange = -90f..90f)
                Text("Vertical (vz) m/s: %.2f".format(desired.vz))
                Slider(value = desired.vz, onValueChange = {
                    desired = desired.copy(vz = it)
                    onDesiredChange(desired)
                }, valueRange = -1f..1f)
            }
        }
    }
}
