#include <LiquidCrystal.h>

// Pre definitions  
#define BUZZER_PIN 8
#define BLUE_LED_PIN 13
#define WHITE_LED_PIN 12

unsigned long lastCommandTime = 0;
unsigned long timeout = 3000; 

LiquidCrystal lcd(3, 2, 4, 5, 6, 7); // LiquidCrystal(rs, e, d4, d5, d6, d7)

// Functions
void setBuzzerOn() {
  digitalWrite(BUZZER_PIN, LOW);
  tone(BUZZER_PIN, 100);
}

void setBuzzerOff() {
  noTone(BUZZER_PIN);
  digitalWrite(BUZZER_PIN, HIGH);
}

void turnBlueLedOn() {
  digitalWrite(WHITE_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, HIGH);
}

void turnWhiteLedOn() {
  digitalWrite(BLUE_LED_PIN, LOW);
  digitalWrite(WHITE_LED_PIN, HIGH);
}

void showLCDMessage(String msg) {
  lcd.clear();
  lcd.print(msg);
}

void setup() {
  // Begin Serial
  Serial.begin(9600);

  // Begin Buzzer
  pinMode(BUZZER_PIN, OUTPUT);
  setBuzzerOff();

  // Buzzer Led
  turnBlueLedOn();

  // Begin LCD
  lcd.begin(16, 2);
  showLCDMessage("Hello! Starto");
}

void loop() {
  if (Serial.available()) {
    char command = Serial.read();

    if (command == '1') {
      setBuzzerOn();
      turnWhiteLedOn();
      showLCDMessage("Eyes Closed");
    } else if (command == '0') {
      setBuzzerOff();
      turnBlueLedOn();
      showLCDMessage("Eyes Opened");
    }
  }
}
