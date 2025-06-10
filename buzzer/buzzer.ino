#define BUZZER_PIN 8
unsigned long lastCommandTime = 0;
unsigned long timeout = 3000;  // 3 giây

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(9600);
  digitalWrite(BUZZER_PIN, LOW);  // Đảm bảo tắt buzzer khi khởi động
}

void loop() {
  if (Serial.available()) {
    char command = Serial.read();
    if (command == '1') {
      digitalWrite(BUZZER_PIN, HIGH);
      Serial.println("Buzzer ON");
      lastCommandTime = millis();  // Cập nhật thời điểm nhận lệnh
    } else if (command == '0') {
      digitalWrite(BUZZER_PIN, LOW);
      Serial.println("Buzzer OFF");
      lastCommandTime = millis();  // Cập nhật luôn
    }
  }

  // Nếu quá 3 giây không có lệnh → tắt buzzer
  if (millis() - lastCommandTime > timeout) {
    digitalWrite(BUZZER_PIN, LOW);
  }
}
