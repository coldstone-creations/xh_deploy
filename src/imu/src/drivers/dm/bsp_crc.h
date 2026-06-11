#pragma once

#include <stdint.h>

namespace dmbot_imu
{

uint8_t Get_CRC8(uint8_t init_value, uint8_t *ptr, uint8_t len);
uint16_t Get_CRC16(uint8_t *ptr, uint16_t len);

}  // namespace dmbot_imu
