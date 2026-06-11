#pragma once

#include <memory>
#include <string>
#include <mutex>
#include <vector>

#include "imu_driver.hpp"
#include "protocol/serial/serial_port.hpp"
#include "bsp_crc.h"

#define DM_IMU_FRAME_SIZE  57
#define DM_IMU_DEG_TO_RAD  0.017453292519943295

namespace dmbot_imu
{

#pragma pack(1)
typedef struct
{
    uint8_t  FrameHeader1;
    uint8_t  flag1;
    uint8_t  slave_id1;
    uint8_t  reg_acc;
    uint32_t accx_u32;
    uint32_t accy_u32;
    uint32_t accz_u32;
    uint16_t crc1;
    uint8_t  FrameEnd1;

    uint8_t  FrameHeader2;
    uint8_t  flag2;
    uint8_t  slave_id2;
    uint8_t  reg_gyro;
    uint32_t gyrox_u32;
    uint32_t gyroy_u32;
    uint32_t gyroz_u32;
    uint16_t crc2;
    uint8_t  FrameEnd2;

    uint8_t  FrameHeader3;
    uint8_t  flag3;
    uint8_t  slave_id3;
    uint8_t  reg_euler;   // r-p-y
    uint32_t roll_u32;
    uint32_t pitch_u32;
    uint32_t yaw_u32;
    uint16_t crc3;
    uint8_t  FrameEnd3;
} IMU_Receive_Frame;
#pragma pack()

typedef struct
{
    float accx;
    float accy;
    float accz;
    float gyrox;
    float gyroy;
    float gyroz;
    float roll;
    float pitch;
    float yaw;
} IMU_Data;

class DmIMUDriver : public IMUDriver
{
public:
    DmIMUDriver(uint16_t imu_id, const std::string& interface_type, const std::string& interface, int baudrate);
    ~DmIMUDriver();

    void serial_rx_cbk(const uint8_t* data, size_t length);

    std::vector<float> get_ang_vel() override;
    std::vector<float> get_quat() override;
    std::vector<float> get_lin_acc() override;
    std::vector<float> get_euler();

private:
    void send_imu_command(const uint8_t* txbuf, size_t len, int repeat = 5);
    void enter_setting_mode();
    void turn_on_accel();
    void turn_on_gyro();
    void turn_on_euler();
    void turn_off_quat();
    void set_output_1000HZ();
    void save_imu_para();
    void exit_setting_mode();

    bool parse_byte(uint8_t byte);

    std::string interface_;
    int baudrate_;
    int error_count_{0};
    mutable std::mutex imu_mutex_;

    std::shared_ptr<IMUSerialPort> serial_;

    IMU_Receive_Frame receive_data_{};
    IMU_Data data_{};

    // Parsing state machine
    uint8_t rx_buf_[DM_IMU_FRAME_SIZE];
    uint8_t rx_index_{0};
    bool frame_started_{false};
};

}  // namespace dmbot_imu
