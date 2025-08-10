package com.armand.skybrain.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { appRoot() }
    }
}

@Composable
fun appRoot() {
    MaterialTheme {
        Column(Modifier.padding(16.dp)) {
            Text("SkyBrain — Dashboard", style = MaterialTheme.typography.titleLarge)
            Spacer(Modifier.height(12.dp))
            Row {
                Button(onClick = { /* TODO: register DJI + enable VS */ }) { Text("Init DJI") }
                Spacer(Modifier.width(8.dp))
                Button(onClick = { /* TODO: start sim */ }) { Text("Start Sim") }
                Spacer(Modifier.width(8.dp))
                Button(onClick = { /* TODO: stop sim */ }) { Text("Stop Sim") }
            }
            Spacer(Modifier.height(12.dp))
            Text("Status: ...  Battery: ...  GPS: ...")
        }
    }
}
