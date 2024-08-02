#include "mpu9250.h"
#include <i2c_driver_wire.h>
// #define PERIOD 800 // ensure EMG sampling rate to be 1250 Hz

bfs::Mpu9250 imu;

int a[8];
unsigned long last_us = 0L;
int timer = 0;
bool curr_record_status = false;
bool prev_record_status = false;


void setup() {

  Serial.begin(1000000);
  analogReadRes(12);
  analog_init();

  pinMode(6, INPUT_PULLUP);


  while(!Serial) {}
  Wire.begin();
  Wire.setClock(400000);
  imu.Config(&Wire, bfs::Mpu9250::I2C_ADDR_PRIM);
  if (!imu.Begin()) {
    Serial.println("Error initializing communication with IMU 1");
    while(1) {}
  }
  if (!imu.ConfigSrd(4)) {
    // 100 Hz IMU & Mag -> 1000 Hz
    Serial.println("Error configured SRD");
    while(1) {}
  }
  Serial.println("Initialization Completed");
}


void loop() {

//     int curr_us = micros();
//     if (curr_us - last_us > PERIOD){
        last_us = micros();
        sample();
        // Serial.send_now();
//     }

}

inline void sample () {
    if (digitalRead(6) == HIGH) {
        delayMicroseconds(10);
        a[0] = analogRead(A0);

        delayMicroseconds(10);
        a[1] = analogRead(A1);

        delayMicroseconds(10);
        a[2] = analogRead(A2);

        delayMicroseconds(10);
        a[3] = analogRead(A3);

        delayMicroseconds(10);
        a[4] = analogRead(A6);

        delayMicroseconds(10);
        a[5] = analogRead(A7);

        delayMicroseconds(10);
        a[6] = analogRead(A8);

        delayMicroseconds(10);
        a[7] = analogRead(A9);


        Serial.print(last_us);

        Serial.print(";EMG1:");
        for (int i = 0; i < 8; i++) {
            Serial.print(a[i]);
            Serial.print(',');
        }
        Serial.println();

        if (timer > 9){
            while(!imu.Read()){}
            Serial.print(last_us);
            Serial.print(";IMU1:");
            Serial.print(imu.accel_x_mps2());
            Serial.print(",");
            Serial.print(imu.accel_y_mps2());
            Serial.print(",");
            Serial.print(imu.accel_z_mps2());
            Serial.print(",");
            Serial.print(imu.gyro_x_radps());
            Serial.print(",");
            Serial.print(imu.gyro_y_radps());
            Serial.print(",");
            Serial.print(imu.gyro_z_radps());
            Serial.print(",");
            Serial.print(imu.mag_x_ut());
            Serial.print(",");
            Serial.print(imu.mag_y_ut());
            Serial.print(",");
            Serial.print(imu.mag_z_ut());
            Serial.print(",");
            Serial.println();
            timer = 0;
        }

        timer +=1;
    }
}
