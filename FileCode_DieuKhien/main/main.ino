#include <Arduino.h>
#include <IRremote.hpp>
#include <EEPROM.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ArduinoJson.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <freertos/queue.h>
#include "debug_config.h"
#include "wifi_setting.h"
#include "mqtt_config.h"
#include "mqtt_connection.h"

// Hardware pin definitions
#define IR_RECEIVE_PIN 15
#define IR_SEND_PIN 33
#define BUTTON_LEARN_PIN 27
#define BUTTON_RESET_PIN 25
#define BUTTON_DELETE_PIN 26

// EEPROM addresses
#define IR_BASE_ADDR 230 // Sau vùng WiFi (0-228)
#define SIGNAL_COUNT_ADDR (IR_BASE_ADDR + MAX_SIGNALS * SIGNAL_SIZE_MAX) // 3290
#define COMMAND_BASE_ADDR (SIGNAL_COUNT_ADDR + 1) // 3291
#define COMMAND_COUNT_ADDR (COMMAND_BASE_ADDR + MAX_SIGNALS * COMMAND_SIZE) // 3451
#define RESET_COUNT_ADDR (COMMAND_COUNT_ADDR + 1) // 3452

// System limits
#define MAX_SIGNALS 10
#define MAX_RAW_LEN 150
#define SIGNAL_SIZE_MAX 306 // 4 (hexData) + 2 (rawLen) + (150 * 2)
#define COMMAND_SIZE 16
#define JSON_BUFFER_SIZE 512
#define INPUT_BUFFER_SIZE 64
#define BUTTON_DEBOUNCE_DELAY 50
#define MAX_RESET_COUNT 5
#define STATUS_DISPLAY_TIME 3000 // 3 giây cho thông báo trạng thái

// Global variables
bool lastLearnButtonState = HIGH;
bool lastResetButtonState = HIGH;
bool lastDeleteButtonState = HIGH;
unsigned long lastDebounceTime = 0;
uint8_t resetCount = 0;
unsigned long lastInvalidIRLog = 0;
const unsigned long invalidIRLogInterval = 1000; // Log lỗi IR mỗi 1 giây

// Thêm biến toàn cục để kiểm soát trạng thái IR receiver
bool irEnabled = false;

struct IRSignal {
  uint32_t hexData;
  uint16_t rawData[MAX_RAW_LEN];
  uint16_t rawLen;
};

struct Command {
  String name;
  uint8_t index;
};

IRSignal signals[MAX_SIGNALS];
Command COMMANDS[MAX_SIGNALS];
uint8_t signalCount = 0, COMMAND_COUNT = 0;
enum Mode { IDLE, LEARN };
Mode currentMode = IDLE;

// LCD configuration
#define LCD_ADDRESS 0x27
#define LCD_COLUMNS 20
#define LCD_ROWS 4
LiquidCrystal_I2C lcd(LCD_ADDRESS, LCD_COLUMNS, LCD_ROWS);

// Task priorities and stack sizes
#define LCD_TASK_PRIORITY 3
#define IR_TASK_PRIORITY 4
#define MQTT_TASK_PRIORITY 3
#define SERIAL_TASK_PRIORITY 2
#define BUTTON_TASK_PRIORITY 3

#define LCD_TASK_STACK_SIZE 4096
#define IR_TASK_STACK_SIZE 4096
#define MQTT_TASK_STACK_SIZE 8192
#define SERIAL_TASK_STACK_SIZE 4096
#define BUTTON_TASK_STACK_SIZE 4096

// Task handles
TaskHandle_t lcdTaskHandle = NULL;
TaskHandle_t irTaskHandle = NULL;
TaskHandle_t mqttTaskHandle = NULL;
TaskHandle_t serialTaskHandle = NULL;
TaskHandle_t buttonTaskHandle = NULL;

// Semaphores for thread safety
SemaphoreHandle_t lcdMutex = NULL;
SemaphoreHandle_t irMutex = NULL;

// Queue for button events
QueueHandle_t buttonQueue = NULL;

// LCD display variables
String currentModeStr = "IDLE";
String irStatus = "";
String wifiIP = "";
unsigned long lastIrStatusTime = 0;

// Variables to control log spam
unsigned long lastWifiDisconnectLog = 0;
const unsigned long logInterval = 10000; // 10 giây

// Helper functions
int8_t findCommandIndex(const String& command) {
  for (uint8_t i = 0; i < COMMAND_COUNT; i++)
    if (command == COMMANDS[i].name) return COMMANDS[i].index;
  return -1;
}

void sendStatusResponse(const String& command, bool success, const String& message = "") {
  String response = "{\"command\":\"" + command + "\",\"status\":\"" + (success ? "success" : "error") + "\"";
  if (message.length() > 0) response += ",\"message\":\"" + message + "\"";
  response += "}";
  if (WiFi.status() == WL_CONNECTED) {
    mqttClient.publish(MQTT_TOPIC_STATUS, (const uint8_t*)response.c_str(), response.length(), MQTT_RETAIN);
  } else {
    unsigned long now = millis();
    if (now - lastWifiDisconnectLog >= logInterval) {
      DEBUG_LOG("Cannot publish MQTT status, WiFi disconnected\n");
      lastWifiDisconnectLog = now;
    }
  }
}

// EEPROM functions
void saveSignalToEEPROM(uint8_t index, const IRSignal& signal) {
  int addr = IR_BASE_ADDR + index * SIGNAL_SIZE_MAX;
  EEPROM.put(addr, signal.hexData);
  addr += 4;
  EEPROM.put(addr, signal.rawLen);
  addr += 2;
  for (uint16_t i = 0; i < signal.rawLen; i++) {
    EEPROM.put(addr, signal.rawData[i]);
    addr += 2;
  }
  EEPROM.commit();
}

void loadSignalFromEEPROM(uint8_t index, IRSignal& signal) {
  int addr = IR_BASE_ADDR + index * SIGNAL_SIZE_MAX;
  EEPROM.get(addr, signal.hexData);
  addr += 4;
  EEPROM.get(addr, signal.rawLen);
  addr += 2;
  for (uint16_t i = 0; i < signal.rawLen; i++) {
    EEPROM.get(addr, signal.rawData[i]);
    addr += 2;
  }
}

void saveAllSignalsToEEPROM() {
  for (uint8_t i = 0; i < signalCount; i++) {
    saveSignalToEEPROM(i, signals[i]);
  }
  EEPROM.write(SIGNAL_COUNT_ADDR, signalCount);
  EEPROM.commit();
}

void loadAllSignalsFromEEPROM() {
  signalCount = EEPROM.read(SIGNAL_COUNT_ADDR);
  if (signalCount > MAX_SIGNALS) {
    DEBUG_LOG("Invalid signal count in EEPROM: %d, resetting to 0\n", signalCount);
    signalCount = 0;
  } else {
    DEBUG_LOG("Loaded signal count from EEPROM: %d\n", signalCount);
  }
  for (uint8_t i = 0; i < signalCount; i++) {
    loadSignalFromEEPROM(i, signals[i]);
  }
}

void saveResetCountToEEPROM() {
  EEPROM.write(RESET_COUNT_ADDR, resetCount);
  EEPROM.commit();
  uint8_t verify = EEPROM.read(RESET_COUNT_ADDR);
  if (verify != resetCount) {
    DEBUG_LOG("EEPROM write verification failed at RESET_COUNT_ADDR: expected %d, read %d\n", resetCount, verify);
  } else {
    DEBUG_LOG("Saved reset count to EEPROM: %d\n", resetCount);
  }
}

void loadResetCountFromEEPROM() {
  resetCount = EEPROM.read(RESET_COUNT_ADDR);
  if (resetCount == 0xFF || resetCount > MAX_RESET_COUNT) {
    DEBUG_LOG("Invalid reset count in EEPROM: %d, resetting to 0\n", resetCount);
    resetCount = 0;
    saveResetCountToEEPROM(); // Lưu giá trị mặc định ngay lập tức
  } else {
    DEBUG_LOG("Loaded reset count from EEPROM: %d\n", resetCount);
  }
}

void clearAllEEPROM() {
  for (int i = 0; i < 4096; i++) {
    EEPROM.write(i, 0);
  }
  EEPROM.commit();
  DEBUG_LOG("Cleared all EEPROM data\n");
}

// Helper functions
String formatCommandName(const String& name) {
  if (name.length() <= 5) return name;
  return name.substring(0, 4) + "..";
}

String formatHexCode(uint32_t hexData) {
  char hexStr[9];
  sprintf(hexStr, "%08X", hexData);
  return String(hexStr);
}

// LCD Task
void lcdTask(void *pvParameters) {
  while (1) {
    if (xSemaphoreTake(lcdMutex, portMAX_DELAY) == pdTRUE) {
      lcd.clear();
      
      // Line 1: Current Mode
      lcd.setCursor(0, 0);
      lcd.print("Mode: ");
      lcd.print(currentModeStr);
      
      // Line 2: Saved Codes Count
      lcd.setCursor(0, 1);
      lcd.print("Codes: ");
      lcd.print(signalCount);
      lcd.print("/");
      lcd.print(MAX_SIGNALS);
      
      // Line 3: IR Status with auto-clear
      lcd.setCursor(0, 2);
      lcd.print("IR: ");
      if (irStatus.length() > 0) {
        lcd.print(irStatus);
        if (millis() - lastIrStatusTime > STATUS_DISPLAY_TIME) {
          irStatus = "";
        }
      }
      
      // Line 4: WiFi IP
      lcd.setCursor(0, 3);
      lcd.print("IP: ");
      if (WiFi.status() == WL_CONNECTED) {
        wifiIP = WiFi.localIP().toString();
      }
      lcd.print(wifiIP);
      
      xSemaphoreGive(lcdMutex);
    }
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}

// Helper functions
bool isDuplicateCode(uint32_t hexData) {
  for (uint8_t i = 0; i < signalCount; i++) {
    if (signals[i].hexData == hexData) {
      return true;
    }
  }
  return false;
}

int8_t findDuplicateCodeIndex(uint32_t hexData) {
  for (uint8_t i = 0; i < signalCount; i++) {
    if (signals[i].hexData == hexData) {
      return i;
    }
  }
  return -1;
}

// IR Task
void irTask(void *pvParameters) {
  while (1) {
    if (currentMode == LEARN && IrReceiver.decode()) {
      if (xSemaphoreTake(irMutex, portMAX_DELAY) == pdTRUE) {
        // Kiểm tra bộ nhớ đầy trước khi truy cập mảng signals
        if (signalCount >= MAX_SIGNALS) {
          currentMode = IDLE;
          currentModeStr = "IDLE";
          if (irEnabled) {
            IrReceiver.disableIRIn();
            irEnabled = false;
            DEBUG_LOG("IR receiver disabled due to memory full\n");
          }
          irStatus = "MEMORY FULL";
          lastIrStatusTime = millis();
          DEBUG_LOG("Memory full! Cannot learn more codes.\n");
          xSemaphoreGive(irMutex);
          IrReceiver.resume();
          continue; // Bỏ qua xử lý tín hiệu
        }

        // Nếu bộ nhớ chưa đầy, an toàn để truy cập mảng signals
        IRSignal& sig = signals[signalCount];
        sig.hexData = IrReceiver.decodedIRData.decodedRawData;
        sig.rawLen = 0;

        // Lấy dữ liệu raw
        if (IrReceiver.decodedIRData.rawDataPtr != NULL) {
          for (uint16_t i = 0; i < IrReceiver.decodedIRData.rawDataPtr->rawlen && sig.rawLen < MAX_RAW_LEN; i++) {
            uint16_t value = IrReceiver.decodedIRData.rawDataPtr->rawbuf[i] * MICROS_PER_TICK;
            if (value > 0) sig.rawData[sig.rawLen++] = value;
          }
        }

        // Kiểm tra tín hiệu hợp lệ
        if (sig.hexData == 0 || sig.rawLen < 2) {
          unsigned long now = millis();
          if (now - lastInvalidIRLog >= invalidIRLogInterval) {
            DEBUG_LOG("Invalid IR signal: HEX=0x%08X, Raw Length=%d\n", sig.hexData, sig.rawLen);
            irStatus = "INVALID IR";
            lastIrStatusTime = millis();
            lastInvalidIRLog = now;
          }
        } else {
          String hexStr = formatHexCode(sig.hexData);
          DEBUG_LOG("Received IR code: HEX=0x%s, Raw Length=%d\n", hexStr.c_str(), sig.rawLen);

          int8_t duplicateIndex = findDuplicateCodeIndex(sig.hexData);
          if (duplicateIndex >= 0) {
            String commandName = "";
            for (uint8_t i = 0; i < COMMAND_COUNT; i++) {
              if (COMMANDS[i].index == duplicateIndex) {
                commandName = formatCommandName(COMMANDS[i].name);
                break;
              }
            }

            if (commandName.length() > 0) {
              irStatus = "DUP #" + String(duplicateIndex) + " " + commandName;
            } else {
              irStatus = "DUP #" + String(duplicateIndex);
            }
            lastIrStatusTime = millis();

            DEBUG_LOG("WARNING: Duplicate code detected!\n");
            DEBUG_LOG("Code exists at #%d", duplicateIndex);
            if (commandName.length() > 0) {
              DEBUG_LOG(" (%s)", commandName.c_str());
            }
            DEBUG_LOG(" [0x%s]\n", hexStr.c_str());
          } else {
            signalCount++;
            saveAllSignalsToEEPROM();
            irStatus = "NEW #" + String(signalCount - 1) + " 0x" + hexStr;
            lastIrStatusTime = millis();
            DEBUG_LOG("New code saved at #%d [0x%s]\n", signalCount - 1, hexStr.c_str());

            // Kiểm tra lại sau khi lưu mã mới
            if (signalCount >= MAX_SIGNALS) {
              currentMode = IDLE;
              currentModeStr = "IDLE";
              if (irEnabled) {
                IrReceiver.disableIRIn();
                irEnabled = false;
                DEBUG_LOG("IR receiver disabled due to memory full after saving\n");
              }
              irStatus = "MEMORY FULL";
              lastIrStatusTime = millis();
              DEBUG_LOG("Memory full after saving code #%d! Exiting LEARN mode.\n", signalCount - 1);
            }
          }
        }
        xSemaphoreGive(irMutex);
      }
      IrReceiver.resume();
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

// MQTT Task
void mqttTask(void *pvParameters) {
  while (1) {
    mqttLoop();
    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

// Modify sendIR function to update LCD with HEX code
void sendIR(uint8_t index) {
  if (index >= signalCount) {
    DEBUG_LOG("Invalid signal index: %d\n", index);
    return;
  }
  IRSignal& signal = signals[index];
  DEBUG_LOG("Sending signal at index %d: HEX=0x%08X\n", index, signal.hexData);
  if (signal.rawLen == 0) {
    DEBUG_LOG("No raw data to send!\n");
    return;
  }
  IrSender.sendRaw(signal.rawData, signal.rawLen, 38);
  irStatus = "Tx 0x" + String(signal.hexData, HEX);
  lastIrStatusTime = millis();
  DEBUG_LOG("IR signal sent\n");
}

void listSavedCodes() {
  if (signalCount == 0) {
    DEBUG_LOG("No codes saved!\n");
    return;
  }
  DEBUG_LOG("\nSaved codes:\n");
  DEBUG_LOG("----------------------\n");
  for (uint8_t i = 0; i < signalCount; i++) {
    DEBUG_LOG("Index %d: HEX=0x%08X", i, signals[i].hexData);
    
    bool found = false;
    for (uint8_t j = 0; j < COMMAND_COUNT; j++) {
      if (COMMANDS[j].index == i) {
        DEBUG_LOG(" - Command: %s", COMMANDS[j].name.c_str());
        found = true;
        break;
      }
    }
    
    if (!found) {
      DEBUG_LOG(" - Command: (Chưa gán tên)");
    }
    DEBUG_LOG("\n");
  }
  DEBUG_LOG("----------------------\n");
  DEBUG_LOG("Tổng số mã: %d\n", signalCount);
  DEBUG_LOG("Tổng số lệnh: %d\n", COMMAND_COUNT);
}

// Command management
void saveCommandsToEEPROM() {
  EEPROM.write(COMMAND_COUNT_ADDR, COMMAND_COUNT);
  for (uint8_t i = 0; i < COMMAND_COUNT; i++) {
    int addr = COMMAND_BASE_ADDR + i * COMMAND_SIZE;
    for (uint8_t j = 0; j < COMMANDS[i].name.length() && j < COMMAND_SIZE - 1; j++) {
      EEPROM.write(addr + j, COMMANDS[i].name[j]);
    }
    EEPROM.write(addr + COMMANDS[i].name.length(), 0);
    EEPROM.write(addr + COMMAND_SIZE - 1, COMMANDS[i].index);
  }
  EEPROM.commit();
}

void loadCommandsFromEEPROM() {
  COMMAND_COUNT = EEPROM.read(COMMAND_COUNT_ADDR);
  if (COMMAND_COUNT > MAX_SIGNALS) {
    DEBUG_LOG("Invalid command count in EEPROM: %d, resetting to 0\n", COMMAND_COUNT);
    COMMAND_COUNT = 0;
  } else {
    DEBUG_LOG("Loaded command count from EEPROM: %d\n", COMMAND_COUNT);
  }
  for (uint8_t i = 0; i < COMMAND_COUNT; i++) {
    int addr = COMMAND_BASE_ADDR + i * COMMAND_SIZE;
    String name = "";
    for (uint8_t j = 0; j < COMMAND_SIZE - 1; j++) {
      char c = EEPROM.read(addr + j);
      if (c == 0) break;
      name += c;
    }
    COMMANDS[i].name = name;
    COMMANDS[i].index = EEPROM.read(addr + COMMAND_SIZE - 1);
  }
}

void publishCommandToCommandsTopic(const String& name, uint8_t index) {
  String json = "[{\"name\":\"" + name + "\",\"index\":" + String(index) + "}]";
  if (WiFi.status() == WL_CONNECTED) {
    mqttClient.publish(MQTT_TOPIC_COMMANDS, (const uint8_t*)json.c_str(), json.length(), MQTT_RETAIN);
  } else {
    unsigned long now = millis();
    if (now - lastWifiDisconnectLog >= logInterval) {
      DEBUG_LOG("Cannot publish MQTT, WiFi disconnected\n");
      lastWifiDisconnectLog = now;
    }
  }
}

String getCommandNameAt(uint8_t index) {
  for (uint8_t i = 0; i < COMMAND_COUNT; i++) {
    if (COMMANDS[i].index == index) {
      return formatCommandName(COMMANDS[i].name);
    }
  }
  return "";
}

void deleteCodeAtIndex(uint8_t index) {
  if (index >= signalCount) {
    DEBUG_LOG("Invalid index! Must be between 0 and %d\n", signalCount - 1);
    return;
  }

  String commandName = getCommandNameAt(index);
  uint32_t hexCode = signals[index].hexData;
  String hexStr = formatHexCode(hexCode);

  for (uint8_t i = index; i < signalCount - 1; i++) {
    signals[i] = signals[i + 1];
  }
  signalCount--;
  saveAllSignalsToEEPROM();

  for (uint8_t i = 0; i < COMMAND_COUNT; i++) {
    if (COMMANDS[i].index == index) {
      for (uint8_t j = i; j < COMMAND_COUNT - 1; j++) {
        COMMANDS[j] = COMMANDS[j + 1];
      }
      COMMAND_COUNT--;
      break;
    } else if (COMMANDS[i].index > index) {
      COMMANDS[i].index--;
    }
  }
  saveCommandsToEEPROM();

  if (commandName.length() > 0) {
    irStatus = "DEL #" + String(index) + " " + commandName;
    DEBUG_LOG("Deleted code at #%d (%s) [0x%s]\n", index, commandName.c_str(), hexStr.c_str());
  } else {
    irStatus = "DEL #" + String(index);
    DEBUG_LOG("Deleted code at #%d [0x%s]\n", index, hexStr.c_str());
  }
  lastIrStatusTime = millis();
}

int8_t findFirstEmptyIndex() {
  bool used[MAX_SIGNALS] = { false };
  for (uint8_t i = 0; i < COMMAND_COUNT; i++) {
    used[COMMANDS[i].index] = true;
  }
  for (uint8_t i = 0; i < MAX_SIGNALS; i++) {
    if (!used[i]) return i;
  }
  return -1;
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  if (length > JSON_BUFFER_SIZE - 1) {
    DEBUG_LOG("MQTT Task: Message too long\n");
    return;
  }
  String t = String(topic);
  char msg[JSON_BUFFER_SIZE];
  for (unsigned int i = 0; i < length && i < JSON_BUFFER_SIZE - 1; i++) {
    msg[i] = (char)payload[i];
  }
  msg[length] = '\0';
  String message = String(msg);
  message.trim();
  DEBUG_LOG("MQTT: Topic=%s, Msg=%s\n", t.c_str(), message.c_str());

  if (t == MQTT_TOPIC_CONTROL) {
    int8_t idx = findCommandIndex(message);
    if (idx >= 0) {
      if (idx < signalCount) {
        sendIR(idx);
        sendStatusResponse(message, true);
      } else {
        sendStatusResponse(message, false, "No IR signal");
      }
    } else if (COMMAND_COUNT < MAX_SIGNALS) {
      int8_t emptyIndex = findFirstEmptyIndex();
      if (emptyIndex >= 0) {
        COMMANDS[COMMAND_COUNT].name = message;
        COMMANDS[COMMAND_COUNT].index = emptyIndex;
        COMMAND_COUNT++;
        saveCommandsToEEPROM();
        publishCommandToCommandsTopic(message, emptyIndex);
        sendStatusResponse(message, true, "Added command at index " + String(emptyIndex));
        // Thêm thông báo trên LCD
        irStatus = "Added at #" + String(emptyIndex);
        lastIrStatusTime = millis();
        DEBUG_LOG("Added new command '%s' at index %d\n", message.c_str(), emptyIndex);
      } else {
        sendStatusResponse(message, false, "No empty index");
      }
    } else {
      sendStatusResponse(message, false, "Command list full");
    }
  } else if (t == MQTT_TOPIC_COMMANDS) {
    StaticJsonDocument<JSON_BUFFER_SIZE> doc;
    DeserializationError error = deserializeJson(doc, message);
    if (!error && doc.is<JsonArray>()) {
      JsonArray commands = doc.as<JsonArray>();
      for (JsonObject cmd : commands) {
        if (!cmd.containsKey("name") || !cmd.containsKey("index")) continue;
        String name = cmd["name"].as<String>();
        int index = cmd["index"].as<int>();
        if (index < 0 || index >= MAX_SIGNALS || name.length() > COMMAND_SIZE - 1) continue;
        bool found = false;
        for (uint8_t i = 0; i < COMMAND_COUNT; i++) {
          if (COMMANDS[i].index == index) {
            COMMANDS[i].name = name;
            found = true;
            break;
          }
        }
        if (!found && COMMAND_COUNT < MAX_SIGNALS) {
          COMMANDS[COMMAND_COUNT].name = name;
          COMMANDS[COMMAND_COUNT].index = index;
          COMMAND_COUNT++;
        }
      }
      if (COMMAND_COUNT > 0) {
        saveCommandsToEEPROM();
        DEBUG_LOG("MQTT Task: Updated commands\n");
      }
    } else {
      DEBUG_LOG("MQTT Task: Invalid JSON\n");
    }
  }
}

// Serial Task
void serialTask(void *pvParameters) {
  while (1) {
    if (Serial.available()) {
      String input = Serial.readStringUntil('\n');
      input.trim();
      DEBUG_LOG("Received: [%s]\n", input.c_str());

      if (input.equalsIgnoreCase("out")) {
        if (currentMode == LEARN) {
          currentMode = IDLE;
          currentModeStr = "IDLE";
          if (irEnabled) {
            IrReceiver.disableIRIn();
            irEnabled = false;
            DEBUG_LOG("IR receiver disabled\n");
          }
          irStatus = "IDLE Mode";
          lastIrStatusTime = millis();
          DEBUG_LOG("Exited learning mode!\n");
        }
      } else if (input.equalsIgnoreCase("list")) {
        listSavedCodes();
      } else if (input.startsWith("learn")) {
        if (signalCount >= MAX_SIGNALS) {
          irStatus = "MEMORY FULL";
          lastIrStatusTime = millis();
          DEBUG_LOG("Memory full!\n");
        } else {
          currentMode = LEARN;
          currentModeStr = "LEARN";
          if (!irEnabled) {
            delay(50);
            IrReceiver.enableIRIn();
            irEnabled = true;
            DEBUG_LOG("IR receiver enabled\n");
          }
          irStatus = "Waiting IR";
          lastIrStatusTime = millis();
          DEBUG_LOG("Waiting for IR code...\n");
        }
      } else if (input.startsWith("deleteall")) {
        signalCount = 0;
        COMMAND_COUNT = 0;
        saveAllSignalsToEEPROM();
        saveCommandsToEEPROM();
        irStatus = "ALL DELETED";
        lastIrStatusTime = millis();
        DEBUG_LOG("All codes deleted!\n");
      } else if (input.startsWith("delete ")) {
        int index = input.substring(7).toInt();
        if (index >= 0) {
          deleteCodeAtIndex(index);
        } else {
          DEBUG_LOG("Invalid index!\n");
        }
      } else if (input.startsWith("send ")) {
        int index = input.substring(5).toInt();
        sendIR(index);
      } else {
        DEBUG_LOG("Unknown command!\n");
      }
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

// Button Task
enum ButtonType { BUTTON_LEARN, BUTTON_RESET, BUTTON_DELETE };

void IRAM_ATTR handleLearnButtonISR() {
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  ButtonType button = BUTTON_LEARN;
  xQueueSendFromISR(buttonQueue, &button, &xHigherPriorityTaskWoken);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

void IRAM_ATTR handleResetButtonISR() {
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  ButtonType button = BUTTON_RESET;
  xQueueSendFromISR(buttonQueue, &button, &xHigherPriorityTaskWoken);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

void IRAM_ATTR handleDeleteButtonISR() {
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  ButtonType button = BUTTON_DELETE;
  xQueueSendFromISR(buttonQueue, &button, &xHigherPriorityTaskWoken);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

void buttonTask(void *pvParameters) {
  ButtonType button;
  while (1) {
    if (xQueueReceive(buttonQueue, &button, portMAX_DELAY) == pdTRUE) {
      unsigned long currentTime = millis();
      if (currentTime - lastDebounceTime < BUTTON_DEBOUNCE_DELAY) {
        continue;
      }
      lastDebounceTime = currentTime;

      if (button == BUTTON_LEARN) {
        bool currentState = digitalRead(BUTTON_LEARN_PIN);
        if (currentState == LOW && lastLearnButtonState == HIGH) {
          DEBUG_LOG("Learn button pressed\n");
          if (currentMode == IDLE) {
            if (signalCount >= MAX_SIGNALS) {
              irStatus = "MEMORY FULL";
              lastIrStatusTime = millis();
              DEBUG_LOG("Memory full, cannot enter LEARN mode\n");
            } else {
              currentMode = LEARN;
              currentModeStr = "LEARN";
              if (!irEnabled) {
                delay(50);
                IrReceiver.enableIRIn();
                irEnabled = true;
                DEBUG_LOG("IR receiver enabled\n");
              }
              irStatus = "Waiting IR";
              lastIrStatusTime = millis();
              DEBUG_LOG("Entered LEARN mode\n");
            }
          } else {
            currentMode = IDLE;
            currentModeStr = "IDLE";
            if (irEnabled) {
              IrReceiver.disableIRIn();
              irEnabled = false;
              DEBUG_LOG("IR receiver disabled\n");
            }
            irStatus = "IDLE Mode";
            lastIrStatusTime = millis();
            DEBUG_LOG("Exited LEARN mode\n");
          }
        }
        lastLearnButtonState = currentState;
      } else if (button == BUTTON_RESET) {
        bool currentState = digitalRead(BUTTON_RESET_PIN);
        if (currentState == LOW && lastResetButtonState == HIGH) {
          DEBUG_LOG("Reset button pressed\n");
          resetCount++;
          saveResetCountToEEPROM();
          irStatus = "Reset #" + String(resetCount);
          lastIrStatusTime = millis();
          currentMode = IDLE;
          currentModeStr = "IDLE";
          DEBUG_LOG("Reset count: %d\n", resetCount);
          if (resetCount >= MAX_RESET_COUNT) {
            DEBUG_LOG("Reset count reached %d, clearing EEPROM and restarting\n", MAX_RESET_COUNT);
            irStatus = "Clearing EEPROM";
            lastIrStatusTime = millis();
            clearAllEEPROM();
            delay(1000);
            ESP.restart();
          } else {
            DEBUG_LOG("Restarting ESP32\n");
            delay(1000);
            ESP.restart();
          }
        }
        lastResetButtonState = currentState;
      } else if (button == BUTTON_DELETE) {
        bool currentState = digitalRead(BUTTON_DELETE_PIN);
        if (currentState == LOW && lastDeleteButtonState == HIGH) {
          DEBUG_LOG("Delete button pressed\n");
          signalCount = 0;
          COMMAND_COUNT = 0;
          saveAllSignalsToEEPROM();
          saveCommandsToEEPROM();
          irStatus = "ALL DELETED";
          lastIrStatusTime = millis();
          DEBUG_LOG("All IR codes and commands deleted\n");
        }
        lastDeleteButtonState = currentState;
      }
    }
  }
}

// LCD function implementations
void updateLCDStatus(const String& line1, const String& line2, const String& line3, const String& line4) {
  if (xSemaphoreTake(lcdMutex, portMAX_DELAY) == pdTRUE) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(line1);
    if (line2.length() > 0) {
      lcd.setCursor(0, 1);
      lcd.print(line2);
    }
    if (line3.length() > 0) {
      lcd.setCursor(0, 2);
      lcd.print(line3);
    }
    if (line4.length() > 0) {
      lcd.setCursor(0, 3);
      lcd.print(line4);
    }
    xSemaphoreGive(lcdMutex);
  }
}

void clearLCD() {
  if (xSemaphoreTake(lcdMutex, portMAX_DELAY) == pdTRUE) {
    lcd.clear();
    xSemaphoreGive(lcdMutex);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  // Initialize I2C
  Wire.begin();
  
  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.print("ESP32 START!");
  delay(1000);
  
  // Create mutexes
  lcdMutex = xSemaphoreCreateMutex();
  irMutex = xSemaphoreCreateMutex();

  // Create button queue
  buttonQueue = xQueueCreate(10, sizeof(ButtonType));
  
  // Initialize EEPROM
  EEPROM.begin(4096);
  DEBUG_LOG("ESP32 START!\n");

  // Load reset count from EEPROM
  loadResetCountFromEEPROM();

  // Initialize IR receiver and sender
  IrReceiver.begin(IR_RECEIVE_PIN, ENABLE_LED_FEEDBACK);
  IrSender.begin(IR_SEND_PIN);
  delay(200);
  IrReceiver.disableIRIn();
  irEnabled = false;

  // Initialize button pins
  pinMode(BUTTON_LEARN_PIN, INPUT_PULLUP);
  pinMode(BUTTON_RESET_PIN, INPUT_PULLUP);
  pinMode(BUTTON_DELETE_PIN, INPUT_PULLUP);

  // Attach interrupts for buttons
  attachInterrupt(digitalPinToInterrupt(BUTTON_LEARN_PIN), handleLearnButtonISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(BUTTON_RESET_PIN), handleResetButtonISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(BUTTON_DELETE_PIN), handleDeleteButtonISR, CHANGE);

  // Start WiFi AP mode for configuration
  wifiSettingPortal();

  // Update WiFi IP for LCD display
  if (WiFi.status() == WL_CONNECTED) {
    wifiIP = WiFi.localIP().toString();
    DEBUG_LOG("WiFi IP updated: %s\n", wifiIP.c_str());
  }

  // Setup MQTT
  mqttSetup(mqttCallback);

  // Load initial data from EEPROM
  loadAllSignalsFromEEPROM();
  loadCommandsFromEEPROM();

  // Create tasks
  xTaskCreatePinnedToCore(lcdTask, "LCD Task", LCD_TASK_STACK_SIZE, NULL, LCD_TASK_PRIORITY, &lcdTaskHandle, 0);
  xTaskCreatePinnedToCore(irTask, "IR Task", IR_TASK_STACK_SIZE, NULL, IR_TASK_PRIORITY, &irTaskHandle, 1);
  xTaskCreatePinnedToCore(mqttTask, "MQTT Task", MQTT_TASK_STACK_SIZE, NULL, MQTT_TASK_PRIORITY, &mqttTaskHandle, 1);
  xTaskCreatePinnedToCore(serialTask, "Serial Task", SERIAL_TASK_STACK_SIZE, NULL, SERIAL_TASK_PRIORITY, &serialTaskHandle, 0);
  xTaskCreatePinnedToCore(buttonTask, "Button Task", BUTTON_TASK_STACK_SIZE, NULL, BUTTON_TASK_PRIORITY, &buttonTaskHandle, 0);
  
  DEBUG_LOG("Ready to receive/send IR\n");
  DEBUG_LOG("MQTT topic: device/control\n");
  DEBUG_LOG("Supported commands:\n");
  DEBUG_LOG("- learn: Learn new code\n");
  DEBUG_LOG("- list: View saved codes\n");
  DEBUG_LOG("- send <index>: Send code at index\n");
  DEBUG_LOG("- deleteall: Delete all codes\n");
  DEBUG_LOG("- out: Exit learning mode\n");
}

void loop() {
  vTaskDelay(portMAX_DELAY);
}