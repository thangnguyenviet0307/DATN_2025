#ifndef MQTT_CONFIG_H
#define MQTT_CONFIG_H

// MQTT Broker settings
#define MQTT_BROKER "ced301392ebf462bb446dcb1f658181b.s1.eu.hivemq.cloud"
#define MQTT_PORT 8883
#define MQTT_USERNAME "raspberrypi"
#define MQTT_PASSWORD "Pi123456"

// MQTT Topics
#define MQTT_TOPIC_CONTROL "device/control"    // Topic nhận lệnh điều khiển
#define MQTT_TOPIC_STATUS "device/status"      // Topic gửi trạng thái
#define MQTT_TOPIC_COMMANDS "device/commands"  // Topic nhận danh sách lệnh

// MQTT Client settings
#define MQTT_CLIENT_ID "ESP32_IR"
#define MQTT_KEEP_ALIVE 60
#define MQTT_CLEAN_SESSION true
#define MQTT_SOCKET_TIMEOUT 10
#define MQTT_BUFFER_SIZE 512
#define MQTT_MAX_PACKET_SIZE 512

// MQTT QoS levels
#define MQTT_QOS 1

// MQTT Message settings
#define MQTT_RETAIN false
#define MQTT_DUP false

// MQTT Reconnection settings
#define MQTT_RECONNECT_INTERVAL 5000  // 5 seconds
#define MQTT_MAX_RECONNECT_ATTEMPTS 5

#endif // MQTT_CONFIG_H
