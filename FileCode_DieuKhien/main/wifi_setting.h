#ifndef WIFI_SETTING_H
#define WIFI_SETTING_H

#include <WiFi.h>
#include <EEPROM.h>
#include <WebServer.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "debug_config.h"

// Định nghĩa các hằng số
#define WIFI_SSID_ADDR 0
#define WIFI_PASS_ADDR 64
#define WIFI_SUCCESS_FLAG_ADDR (WIFI_PASS_ADDR + WIFI_MAX_LEN * WIFI_MAX_SAVED) // 224
#define WIFI_MAX_LEN 32
#define WIFI_MAX_SAVED 5

#define WIFI_CONNECT_TIMEOUT 10000  // 15 giây timeout kết nối
#define MAX_WIFI_NETWORKS 10        // Giới hạn số lượng mạng WiFi trả về khi quét

// Biến toàn cục
String currentSSID = "";
String currentPass = "";
WebServer server(80);
extern String wifiIP; // Biến toàn cục để lưu IP WiFi
bool connected = false; // Biến để theo dõi trạng thái kết nối

// Forward declaration of LCD functions
void updateLCDStatus(const String& line1, const String& line2 = "", const String& line3 = "", const String& line4 = "");

// Lưu SSID và mật khẩu vào EEPROM
void saveWiFiCredentials(const String& ssid, const String& pass) {
  for (int i = 0; i < WIFI_MAX_SAVED; i++) {
    int base = WIFI_SSID_ADDR + i * WIFI_MAX_LEN;
    char first = EEPROM.read(base);
    if (first == 0) {
      for (int j = 0; j < WIFI_MAX_LEN; j++) {
        EEPROM.write(WIFI_SSID_ADDR + i * WIFI_MAX_LEN + j, j < ssid.length() ? ssid[j] : 0);
        EEPROM.write(WIFI_PASS_ADDR + i * WIFI_MAX_LEN + j, j < pass.length() ? pass[j] : 0);
      }
      EEPROM.write(WIFI_SUCCESS_FLAG_ADDR + i, 1); // Đánh dấu đã kết nối thành công
      EEPROM.commit();
      DEBUG_LOG("Saved WiFi credentials to slot %d: SSID=%s\n", i, ssid.c_str());
      return;
    }
  }
  DEBUG_LOG("No available slots to save WiFi credentials\n");
}

// Kiểm tra xem SSID đã từng kết nối thành công chưa
bool hasWiFiConnectedSuccessfully(int slot) {
  return EEPROM.read(WIFI_SUCCESS_FLAG_ADDR + slot) == 1;
}

// Xóa thông tin WiFi trong EEPROM tại slot cụ thể
void clearWiFiCredentialsAtSlot(int slot) {
  for (int j = 0; j < WIFI_MAX_LEN; j++) {
    EEPROM.write(WIFI_SSID_ADDR + slot * WIFI_MAX_LEN + j, 0);
    EEPROM.write(WIFI_PASS_ADDR + slot * WIFI_MAX_LEN + j, 0);
  }
  EEPROM.write(WIFI_SUCCESS_FLAG_ADDR + slot, 0);
  EEPROM.commit();
  DEBUG_LOG("Cleared WiFi credentials at slot %d\n", slot);
}

// Xóa toàn bộ thông tin WiFi trong EEPROM
void clearWiFiCredentials() {
  for (int i = 0; i < WIFI_MAX_SAVED; i++) {
    for (int j = 0; j < WIFI_MAX_LEN; j++) {
      EEPROM.write(WIFI_SSID_ADDR + i * WIFI_MAX_LEN + j, 0);
      EEPROM.write(WIFI_PASS_ADDR + i * WIFI_MAX_LEN + j, 0);
    }
    EEPROM.write(WIFI_SUCCESS_FLAG_ADDR + i, 0);
  }
  EEPROM.commit();
  DEBUG_LOG("Cleared WiFi credentials from EEPROM\n");
}

// Đọc SSID và mật khẩu từ slot cụ thể trong EEPROM
bool readWiFiCredentialsAtSlot(int slot, String& ssid, String& pass) {
  char s[WIFI_MAX_LEN + 1], p[WIFI_MAX_LEN + 1];
  bool validSlot = true;
  for (int i = 0; i < WIFI_MAX_LEN; i++) {
    s[i] = EEPROM.read(WIFI_SSID_ADDR + slot * WIFI_MAX_LEN + i);
    p[i] = EEPROM.read(WIFI_PASS_ADDR + slot * WIFI_MAX_LEN + i);
    if (s[i] == 0xFF || p[i] == 0xFF) validSlot = false; // EEPROM uninitialized
  }
  s[WIFI_MAX_LEN] = '\0';
  p[WIFI_MAX_LEN] = '\0';
  ssid = String(s);
  pass = String(p);
  if (validSlot && ssid.length() > 0 && pass.length() > 0) {
    DEBUG_LOG("Read WiFi credentials from slot %d: SSID=%s\n", slot, ssid.c_str());
    return true;
  }
  return false;
}

// Đọc SSID và mật khẩu từ slot đầu tiên có dữ liệu hợp lệ (giữ tương thích)
void readWiFiCredentials(String& ssid, String& pass, int& slot) {
  for (slot = 0; slot < WIFI_MAX_SAVED; slot++) {
    if (readWiFiCredentialsAtSlot(slot, ssid, pass)) {
      return;
    }
  }
  ssid = "";
  pass = "";
  slot = -1;
  DEBUG_LOG("No valid WiFi credentials found in EEPROM\n");
}

// Kết nối đến WiFi
bool connectToWiFi(const String& ssid, const String& pass) {
  WiFi.mode(WIFI_OFF);
  delay(500);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  DEBUG_LOG("Connecting to WiFi: SSID=%s\n", ssid.c_str());
  updateLCDStatus("Connecting WiFi", ssid, "Please wait...");

  WiFi.begin(ssid.c_str(), pass.c_str());
  unsigned long startTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startTime < WIFI_CONNECT_TIMEOUT) {
    delay(500);
  }

  connected = (WiFi.status() == WL_CONNECTED);
  if (connected) {
    String ip = WiFi.localIP().toString();
    wifiIP = ip; // Cập nhật wifiIP ngay sau khi kết nối
    DEBUG_LOG("WiFi connected, IP: %s\n", ip.c_str());
    updateLCDStatus("WiFi Connected!", "IP: " + ip);
    currentSSID = ssid;
    currentPass = pass;
  } else {
    DEBUG_LOG("Failed to connect to WiFi: SSID=%s\n", ssid.c_str());
    updateLCDStatus("Wifi connecting failed");
    delay(2000); // Hiển thị thông báo trong 2 giây
    // Tắt chế độ STA để chuẩn bị quay lại AP mode hoặc thử WiFi khác
    WiFi.mode(WIFI_OFF);
    delay(1000);
  }
  return connected;
}

// Xử lý yêu cầu quét WiFi
void handleScanAjax() {
  // Kiểm tra bộ nhớ heap trước khi quét
  size_t freeHeap = ESP.getFreeHeap();
  if (freeHeap < 10000) {
    server.send(500, "text/plain", "Low memory");
    DEBUG_LOG("Low memory during scan, free heap: %d bytes\n", freeHeap);
    return;
  }

  DEBUG_LOG("Starting WiFi scan...\n");
  WiFi.scanDelete(); // Xóa kết quả quét cũ
  int n = WiFi.scanNetworks(false, true); // Non-blocking scan
  unsigned long scanStart = millis();
  while (n == WIFI_SCAN_RUNNING && (millis() - scanStart) < 5000) {
    delay(100);
    n = WiFi.scanComplete();
  }

  if (n < 0) {
    server.send(500, "text/plain", "Scan failed");
    DEBUG_LOG("WiFi scan failed or timed out, error code: %d\n", n);
    return;
  }

  String json = "[";
  int count = min(n, MAX_WIFI_NETWORKS); // Giới hạn số lượng mạng trả về
  for (int i = 0; i < count; ++i) {
    if (i > 0) json += ",";
    json += "{\"ssid\":\"" + WiFi.SSID(i) + "\",\"rssi\":" + String(WiFi.RSSI(i)) + "}";
  }
  json += "]";
  server.send(200, "application/json", json);
  DEBUG_LOG("Scanned %d WiFi networks, sent %d\n", n, count);
  WiFi.scanDelete(); // Xóa kết quả quét để giải phóng bộ nhớ
}

// Xử lý yêu cầu xóa WiFi
void handleClear() {
  clearWiFiCredentials();
  server.send(200, "text/plain", "WiFi cleared");
  DEBUG_LOG("Cleared WiFi credentials\n");
}

// Xử lý yêu cầu hiển thị danh sách WiFi đã lưu
void handleListSaved() {
  String json = "[";
  bool first = true;
  for (int slot = 0; slot < WIFI_MAX_SAVED; slot++) {
    char s[WIFI_MAX_LEN + 1];
    for (int i = 0; i < WIFI_MAX_LEN; i++) {
      s[i] = EEPROM.read(WIFI_SSID_ADDR + slot * WIFI_MAX_LEN + i);
    }
    s[WIFI_MAX_LEN] = '\0';
    String ssid = String(s);
    if (ssid.length() > 0) {
      if (!first) json += ",";
      json += "\"" + ssid + "\"";
      first = false;
    }
  }
  json += "]";
  server.send(200, "application/json", json);
  DEBUG_LOG("Sent list of saved WiFi networks\n");
}

// Tạo trang HTML (động, tùy thuộc vào trạng thái kết nối)
String htmlPage(bool connected = false) {
  String html = "<!DOCTYPE html><html><head><meta charset=\"UTF-8\">";
  html += "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,maximum-scale=2,shrink-to-fit=no\">";
  html += "<title>Cai dat WiFi</title>";
  html += "<style>body{font-family:sans-serif}h2{text-align:center}label{font-size:18px}input{font-size:18px}input,button{width:100%;padding:10px;margin:6px 0}button{background:#2196F3;color:#fff;border:0;border-radius:4px;font-size:18px}#scanBtn{margin-top:12px}#wifiList{margin-top:16px}.wifi-item{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}.bar{height:12px;width:100px;background:#eee;border-radius:6px;overflow:hidden;margin-left:8px}.bar-inner{height:100%;background:#4caf50}#savedList{margin-top:16px}</style>";
  html += "<script>let showingSaved=false;function scanWifi(){showingSaved=false;document.getElementById(\"wifiList\").innerHTML=\"Dang quet...\";fetch(\"/scan\").then(r=>r.json()).then(list=>{let html=\"\";list.forEach(function(w){let percent=Math.min(100,Math.max(0,2*(w.rssi+100)));html+=\"<div class=\\\"wifi-item\\\"><span onclick=\\\"document.getElementById('ssid').value='\"+w.ssid+\"'\\\" style=\\\"cursor:pointer\\\">\"+w.ssid+\" (\"+w.rssi+\" dBm)</span><div class=\\\"bar\\\"><div class=\\\"bar-inner\\\" style=\\\"width:\"+percent+\"%\\\"></div></div></div>\";});if(html===\"\")html=\"Khong tim thay WiFi nao!\";document.getElementById(\"wifiList\").innerHTML=html;}).catch(e=>console.error(\"Scan failed:\",e))}function clearWifi(){fetch(\"/clear\",{method:\"POST\"}).then(r=>r.text()).then(txt=>{alert(\"Da xoa WiFi da luu!\");document.getElementById(\"wifiList\").innerHTML=\"\";}).catch(e=>console.error(\"Clear failed:\",e))}function listSaved(){showingSaved=true;fetch(\"/list\").then(r=>r.json()).then(list=>{let html=\"\";list.forEach(w=>{html+=\"<li>\"+w+\"</li>\";});if(html===\"\")html=\"<i>Khong co WiFi nao da luu</i>\";else html=\"<ul>\"+html+\"</ul>\";document.getElementById(\"wifiList\").innerHTML=html;}).catch(e=>console.error(\"List failed:\",e))}</script></head><body>";
  html += "<h2 style=\"color:#2196F3\">Cai dat WiFi</h2>";
  if (connected) {
    html += "<div style=\"color:green;font-weight:bold\">Da ket noi WiFi thanh cong!</div>";
  }
  html += "<form action=\"/save\" method=\"POST\">";
  html += "<label>Ten WiFi:</label><input type=\"text\" id=\"ssid\" name=\"ssid\" placeholder=\"Nhap ten WiFi\" maxlength=\"20\"><br>";
  html += "<label>Mat khau:</label><input type=\"password\" name=\"password\" maxlength=\"15\"><br>";
  html += "<button type=\"submit\">Ket noi</button>";
  html += "</form>";
  html += "<button id=\"scanBtn\" onclick=\"scanWifi();return false\">Quet WiFi</button>";
  html += "<button onclick=\"clearWifi();return false\" style=\"background:#f44336\">Xoa tat ca mat khau da luu</button>";
  html += "<button onclick=\"listSaved();return false\" style=\"background:#4CAF50\">Xem danh sach WiFi da luu</button>";
  html += "<div id=\"wifiList\"></div>";
  html += "</body></html>";
  return html;
}

// Xử lý yêu cầu hiển thị trang chính
void handleRoot() {
  size_t freeHeap = ESP.getFreeHeap();
  if (freeHeap < 10000) {
    server.send(500, "text/plain", "Low memory");
    DEBUG_LOG("Low memory in handleRoot, free heap: %d bytes\n", freeHeap);
    return;
  }
  DEBUG_LOG("Serving root page, free heap: %d bytes\n", freeHeap);
  server.send(200, "text/html", htmlPage());
}

// Xử lý yêu cầu lưu WiFi
void handleSave() {
  if (!server.hasArg("ssid") || !server.hasArg("password")) {
    server.send(400, "text/plain", "Missing SSID or password");
    DEBUG_LOG("Missing SSID or password in handleSave\n");
    return;
  }

  currentSSID = server.arg("ssid");
  currentPass = server.arg("password");
  if (currentSSID.length() == 0 || currentPass.length() == 0) {
    server.send(400, "text/plain", "Invalid SSID or password");
    DEBUG_LOG("Invalid SSID or password in handleSave\n");
    return;
  }

  // Kiểm tra xem SSID này đã từng kết nối thành công chưa
  String storedSsid, storedPass;
  int slot;
  bool hasConnectedBefore = false;
  for (int i = 0; i < WIFI_MAX_SAVED; i++) {
    if (readWiFiCredentialsAtSlot(i, storedSsid, storedPass) && storedSsid == currentSSID && hasWiFiConnectedSuccessfully(i)) {
      hasConnectedBefore = true;
      slot = i;
      break;
    }
  }

  DEBUG_LOG("Attempting to connect to WiFi: SSID=%s\n", currentSSID.c_str());
  bool connectedResult = connectToWiFi(currentSSID, currentPass);
  DEBUG_LOG("Connection result: %d, global connected: %d\n", connectedResult, connected);

  if (connected) {
    // Nếu kết nối thành công, lưu thông tin WiFi vào EEPROM (nếu chưa có hoặc cần cập nhật mật khẩu)
    if (!hasConnectedBefore || (hasConnectedBefore && storedPass != currentPass)) {
      if (hasConnectedBefore) {
        // Nếu SSID đã tồn tại nhưng mật khẩu khác, xóa thông tin cũ trước khi lưu mới
        clearWiFiCredentialsAtSlot(slot);
      }
      saveWiFiCredentials(currentSSID, currentPass);
    }
    // Trả về JavaScript để hiển thị thông báo thành công và khởi động lại
    String js = "<script>setTimeout(function(){window.location.href='/';}, 2000);</script>";
    server.send(200, "text/html", js);
    DEBUG_LOG("WiFi saved in handleSave, connected=%d\n", connected);
    delay(2000);
    ESP.restart();
  } else {
    // Nếu kết nối thất bại, không hiển thị thông báo lỗi, chỉ log và tiếp tục ở AP mode
    DEBUG_LOG("WiFi connection failed, staying in AP mode\n");
  }
}

// Chế độ AP để cấu hình WiFi
void wifiSettingPortal() {
  // Thử kết nối với tất cả WiFi đã lưu trong EEPROM
  for (int slot = 0; slot < WIFI_MAX_SAVED; slot++) {
    String ssid, pass;
    if (readWiFiCredentialsAtSlot(slot, ssid, pass)) {
      bool connected = connectToWiFi(ssid, pass);
      if (connected) {
        return; // Nếu kết nối thành công, thoát khỏi hàm
      } else if (!hasWiFiConnectedSuccessfully(slot)) {
        // Nếu chưa từng kết nối thành công với SSID này, xóa thông tin WiFi
        clearWiFiCredentialsAtSlot(slot);
      }
      // Nếu đã từng kết nối thành công, không xóa thông tin, thử slot tiếp theo
    }
  }

  // Nếu không kết nối được với bất kỳ WiFi nào, vào chế độ AP
  DEBUG_LOG("Starting AP mode: ESP32_Setup\n");
  Serial.println("ESP32 dang phat WiFi AP: ESP32_Setup, hay ket noi va truy cap 192.168.4.1 de cau hinh!");
  WiFi.mode(WIFI_AP);
  if (!WiFi.softAP("ESP32_Setup")) {
    DEBUG_LOG("Failed to start AP mode!\n");
    updateLCDStatus("AP Mode Failed", "Please restart...");
    return;
  }
  delay(100);

  IPAddress apIP = WiFi.softAPIP();
  String ipStr = apIP.toString();
  DEBUG_LOG("AP started, access at http://%s\n", ipStr.c_str());

  server.on("/", HTTP_GET, handleRoot);
  server.on("/scan", HTTP_GET, handleScanAjax);
  server.on("/save", HTTP_POST, handleSave);
  server.on("/clear", HTTP_POST, handleClear);
  server.on("/list", HTTP_GET, handleListSaved);
  server.onNotFound([]() {
    server.send(404, "text/plain", "Not found");
  });

  server.begin();
  DEBUG_LOG("Web server started in AP mode\n");

  unsigned long lastLogTime = millis();
  unsigned long lastLcdUpdate = millis();
  while (true) { // Duy trì AP mode vô hạn cho đến khi kết nối thành công
    // Kiểm tra bộ nhớ heap trước khi xử lý yêu cầu
    size_t freeHeap = ESP.getFreeHeap();
    if (freeHeap < 10000) {
      DEBUG_LOG("Low memory in wifiSettingPortal, free heap: %d bytes\n", freeHeap);
      break;
    }

    server.handleClient();

    // Log định kỳ để debug
    if (millis() - lastLogTime > 5000) {
      DEBUG_LOG("AP mode active, free heap: %d bytes\n", ESP.getFreeHeap());
      lastLogTime = millis();
    }

    // Cập nhật LCD mỗi 1 giây để giảm nhấp nháy
    if (millis() - lastLcdUpdate >= 1000) {
      updateLCDStatus("AP Mode Active", "SSID: ESP32_Setup", "IP: " + ipStr);
      lastLcdUpdate = millis();
    }

    // Kiểm tra trạng thái AP mode và khởi động lại nếu cần
    if (WiFi.softAPgetStationNum() == 0 && WiFi.getMode() != WIFI_AP) {
      DEBUG_LOG("AP mode not active, restarting AP...\n");
      WiFi.mode(WIFI_AP);
      if (!WiFi.softAP("ESP32_Setup")) {
        DEBUG_LOG("Failed to restart AP mode!\n");
        updateLCDStatus("AP Mode Failed", "Please restart...");
        return;
      }
      DEBUG_LOG("AP restarted, access at http://%s\n", ipStr.c_str());
    }

    delay(50); // Giảm delay để tăng khả năng xử lý yêu cầu từ client
  }
}

#endif // WIFI_SETTING_H