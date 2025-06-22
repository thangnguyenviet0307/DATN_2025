#ifndef MQTT_CONNECTION_H
#define MQTT_CONNECTION_H

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include "mqtt_config.h"

WiFiClientSecure espClient;
PubSubClient mqttClient(espClient);

// MQTT connection state
volatile bool mqttConnected = false;
unsigned long lastMqttReconnectAttempt = 0;

void mqttSetup(MQTT_CALLBACK_SIGNATURE) {
  // Nếu bạn có CA certificate, hãy dùng espClient.setCACert(...);
  // Nếu không, có thể bỏ verify để dễ thử nghiệm:
  espClient.setInsecure();
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(callback);
  mqttClient.setKeepAlive(MQTT_KEEP_ALIVE);
  mqttClient.setSocketTimeout(MQTT_SOCKET_TIMEOUT);
}

bool mqttReconnect() {
  if (mqttClient.connected()) {
    return true;
  }

  unsigned long now = millis();
  if (now - lastMqttReconnectAttempt < MQTT_RECONNECT_INTERVAL) {
    return false;
  }
  lastMqttReconnectAttempt = now;

  Serial.println("Attempting MQTT connection...");
  
  if (mqttClient.connect(MQTT_CLIENT_ID, MQTT_USERNAME, MQTT_PASSWORD)) {
    Serial.println("MQTT connected");
    mqttClient.subscribe(MQTT_TOPIC_CONTROL, MQTT_QOS);
    mqttClient.subscribe(MQTT_TOPIC_COMMANDS, MQTT_QOS);
    mqttConnected = true;
    return true;
  } else {
    Serial.print("MQTT connection failed, rc=");
    Serial.print(mqttClient.state());
    Serial.println(" retrying in 5s");
    Serial.print("WiFi status: ");
    Serial.println(WiFi.status() == WL_CONNECTED ? "Connected" : "Disconnected");
    Serial.print("Free heap: ");
    Serial.println(ESP.getFreeHeap());
    mqttConnected = false;
    return false;
  }
}

void mqttLoop() {
  if (!mqttClient.connected()) {
    mqttConnected = false;
    mqttReconnect();
  } else {
    mqttClient.loop();
  }
}

#endif // MQTT_CONNECTION_H