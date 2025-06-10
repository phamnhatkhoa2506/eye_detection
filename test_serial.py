import serial
import time
import sys


def connect_to_arduino(port='COM3', baudrate=9600, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Attempting to connect to {port} (attempt {attempt + 1}/{max_retries})...")
            arduino = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            print(f"Successfully connected to {port}")
            return arduino
        except serial.SerialException as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("Failed to connect after multiple attempts.")
                print("Please check if:")
                print("1. The correct COM port is selected")
                print("2. No other program is using the port")
                print("3. You have administrator privileges")
                sys.exit(1)


def buzzer_on(arduino: serial.Serial):
    arduino.write(b'1')
    print("Sent: 1 (Buzzer ON)")
    print("Arduino:", arduino.readline().decode().strip())


def buzzer_off(arduino: serial.Serial):
    arduino.write(b'0')
    print("Sent: 0 (Buzzer OFF)")
    print("Arduino:", arduino.readline().decode().strip())
