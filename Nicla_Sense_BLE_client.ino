/* An example sketch created for BLE project demonstrating both transmission and
reception of BLE messages.
* This sketch will transmit selected sensor measurements via BLE to the
  example Flutter app (tested on Android),  'Nicla App'.
* This sketch accepts control commands from the BLE App which switches the
  onboard LED.

 << Pavan - 12/01/2023 >>
 */

#include "Arduino.h"
#include "Arduino_BHY2.h"

#include <ArduinoBLE.h>
#include <Nicla_System.h>
#define bitRead(value, bit) (((value) >> (bit)) & 0x01)

// Comment out the following line if you want the acceleration values
// including the gravity component
#define COMPENSATE_GRAVITY

// List of sensors to read from Nicla Sense
Sensor temp(SENSOR_ID_TEMP);                   // Temperature sensor
Sensor humidity(SENSOR_ID_HUM);                // Humidity sensor
SensorBSEC BSEC_data(SENSOR_ID_BSEC);          // Air quality sensor (to read IAQ and CO2 levels)

#if defined (COMPENSATE_GRAVITY)
SensorXYZ acc(SENSOR_ID_LACC);                  // Acceleration
#else
SensorXYZ acc(SENSOR_ID_ACC);                  // Acceleration
#endif

//  Accepted values: 2 (+/-2g), 4 (+/-4g), 8 (+/-8g), 16 (+/-16g)
#define G_SCALE 8              // Set the acceleromter range setting value here
#define BLE_UPDATE_INTERVAL 10 // Set the sensor data send interval in ms

// Create BLE Service
BLEService niclaControlService("a2e1f6de-5f14-4b7f-b8f3-4c34a2c6d4c9");

// Create BLE Characteristics
BLEByteCharacteristic toggleCharacteristic("12d6b1df-87a9-4e0b-af2b-7b4d01e55644", BLERead | BLEWrite);
BLEStringCharacteristic niclaDataCharacteristic("58471a42-c8d0-4046-8f7a-cb230ae286f8", BLERead | BLENotify, 32);

float max_gForce = 0; // Holds the max_gForce reported in the measurement cycle

// Convert the acceleration scale to G's
float get_G_value(int acc)
{
  return ((float)acc) / (32768.0 / G_SCALE);
}

// Serial print and send a measurement via BLE
void send_niclaData(String str)
{
  Serial.println(str);
  niclaDataCharacteristic.writeValue(str);
}

void setup()
{
  // Begin BLE
  BLE.begin();
  Serial.begin(9600);

  // print BLE MAC address
  Serial.print("BLE Address: ");
  Serial.println(BLE.address());

  // Set pin modes
  pinMode(LED_BUILTIN, OUTPUT);

  // Set up BLE Service and Characteristics
  BLE.setLocalName("Nicla Sense for Numorpho");
  BLE.setAdvertisedService(niclaControlService);

  niclaControlService.addCharacteristic(toggleCharacteristic);
  niclaControlService.addCharacteristic(niclaDataCharacteristic);

  BLE.addService(niclaControlService);

  // Set initial values
  toggleCharacteristic.writeValue(0);
  niclaDataCharacteristic.writeValue("null,null");

  // Start advertising
  BLE.advertise();

  // Configure the Nicla Sense device to work in standalone mode
  BHY2.begin(NICLA_STANDALONE);

  // Start sensors
  temp.begin();
  humidity.begin();
  BSEC_data.begin();
  acc.begin();

  // Set G_Scale for acceleration readings
  // This changes the range for all acceleration related measurements above
  // Data takes the full range of int16_t
  //  2 (+/-2g), 4 (+/-4g), 8 (+/-8g), 16 (+/-16g)
  acc.setRange(G_SCALE);
}

void loop()
{

  BLEDevice central = BLE.central();
  static auto bleDataUpdateTime = millis(); // timer used for measurement sending
  static auto printTime = millis();         // timer used for serial printing

  if (central)
  {
    while (central.connected())
    {

      // Update function should be continuously polled
      BHY2.update();

      // Record the max_gForce in the current measurement cycle
      float acc_x = get_G_value(acc.x());
      float acc_y = get_G_value(acc.y());
      float acc_z = get_G_value(acc.z());

      max_gForce = max(max_gForce, (acc_x * acc_x + acc_y * acc_y + acc_z * acc_z));

      if (toggleCharacteristic.written())
      {
        byte toggleValue = toggleCharacteristic.value();
        Serial.print("received bytes ");
        Serial.println(toggleValue);
        handleToggles(toggleValue);
      }

      // Update measurements every BLE_UPDATE_INTERVAL (in ms)
      if (millis() - bleDataUpdateTime > BLE_UPDATE_INTERVAL)
      {
        bleDataUpdateTime = millis();

        // Send temperature
        String data_str = "";
        data_str = "temp," + String(temp.value(), 1);
        send_niclaData(data_str);

        // Send humidity
        data_str = "hum," + String(humidity.value(), 1);
        send_niclaData(data_str);

        // Send IAQ
        data_str = "IAQ," + String(BSEC_data.iaq_s(), 6);
        send_niclaData(data_str);

        // Send CO2
        data_str = "co2," + String(BSEC_data.co2_eq(), 6);
        send_niclaData(data_str);

        // Send x acceleration
        data_str = "x_accel," + String(abs(get_G_value(acc.x())), 2);
        send_niclaData(data_str);

        // Send y acceleration
        data_str = "y_accel," + String(abs(get_G_value(acc.y())), 2);
        send_niclaData(data_str);

        // Send z acceleration
        data_str = "z_accel," + String(abs(get_G_value(acc.z())), 2);
        send_niclaData(data_str);

        // Send max_gForce
        data_str = "gForce," + String(max_gForce, 2);
        send_niclaData(data_str);
        max_gForce = 0;
        
      }

      if (millis() - printTime >= 1000)
      {
        printTime = millis();

        Serial.println(String("Temperature: ") + String(temp.value(), 1));
        Serial.println(String("Humidity: ") + String(humidity.value(), 1));
        Serial.println(String("IAQ: ") + String(BSEC_data.iaq_s(), 6));
        Serial.println(String("CO2: ") + String(BSEC_data.co2_eq(), 6));
        Serial.println(String("Acceleration x: ") + get_G_value(acc.x()));
        Serial.println(String("Acceleration y: ") + get_G_value(acc.y()));
        Serial.println(String("Acceleration z: ") + get_G_value(acc.z()));
      }
    }
  }
}

void handleToggles(byte value)
{
  // digitalWrite(LED_BUILTIN, HIGH);
  if (value == 0x01)
    digitalWrite(LED_BUILTIN, HIGH);
  else
    digitalWrite(LED_BUILTIN, LOW);
}
