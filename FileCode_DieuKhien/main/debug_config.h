#ifndef DEBUG_CONFIG_H
#define DEBUG_CONFIG_H

#include <Arduino.h>

// Debug logging macro
#define ENABLE_DEBUG 1
#if ENABLE_DEBUG
#define DEBUG_LOG(fmt, ...) Serial.printf("[%.3f] " fmt, millis() / 1000.0, ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

#endif // DEBUG_CONFIG_H