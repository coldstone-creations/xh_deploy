#include "dm_imu_driver.hpp"

#include <cmath>
#include <cstring>
#include <thread>
#include <chrono>

namespace dmbot_imu
{

DmIMUDriver::DmIMUDriver(uint16_t imu_id, const std::string& interface_type, const std::string& interface, int baudrate)
    : IMUDriver()
    , interface_(interface)
    , baudrate_(baudrate)
{
    imu_id_ = imu_id;
    memset(&receive_data_, 0, sizeof(receive_data_));
    memset(&data_, 0, sizeof(data_));
    memset(rx_buf_, 0, sizeof(rx_buf_));

    if (interface_type != "serial")
    {
        throw std::runtime_error("DM IMU driver only supports SERIAL interface");
    }

    // Open serial port
    serial_ = IMUSerialPort::open(interface_, baudrate_);

    // Set up callback
    IMUSerialPort::SerialCbkFunc serial_callback =
        std::bind(&DmIMUDriver::serial_rx_cbk, this, std::placeholders::_1, std::placeholders::_2);
    serial_->set_serial_callback(serial_callback);

    // Configure IMU
    logger_->info("DM IMU: starting configuration sequence...");

    enter_setting_mode();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    turn_on_accel();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    turn_on_gyro();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    turn_on_euler();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    turn_off_quat();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    set_output_1000HZ();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    save_imu_para();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    exit_setting_mode();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    logger_->info("DM IMU: init complete");
}

DmIMUDriver::~DmIMUDriver()
{
    logger_->info("DM IMU: shutting down");
    if (serial_)
    {
        serial_->close();
    }
}

void DmIMUDriver::send_imu_command(const uint8_t* txbuf, size_t len, int repeat)
{
    if (!serial_)
    {
        logger_->error("DM IMU: serial port not available");
        return;
    }
    for (int i = 0; i < repeat; i++)
    {
        serial_->write(txbuf, len);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void DmIMUDriver::enter_setting_mode()
{
    uint8_t txbuf[4] = {0xAA, 0x06, 0x01, 0x0D};
    send_imu_command(txbuf, sizeof(txbuf));
}

void DmIMUDriver::turn_on_accel()
{
    uint8_t txbuf[4] = {0xAA, 0x01, 0x14, 0x0D};
    send_imu_command(txbuf, sizeof(txbuf));
}

void DmIMUDriver::turn_on_gyro()
{
    uint8_t txbuf[4] = {0xAA, 0x01, 0x15, 0x0D};
    send_imu_command(txbuf, sizeof(txbuf));
}

void DmIMUDriver::turn_on_euler()
{
    uint8_t txbuf[4] = {0xAA, 0x01, 0x16, 0x0D};
    send_imu_command(txbuf, sizeof(txbuf));
}

void DmIMUDriver::turn_off_quat()
{
    uint8_t txbuf[4] = {0xAA, 0x01, 0x07, 0x0D};
    send_imu_command(txbuf, sizeof(txbuf));
}

void DmIMUDriver::set_output_1000HZ()
{
    uint8_t txbuf[5] = {0xAA, 0x02, 0x01, 0x00, 0x0D};
    send_imu_command(txbuf, sizeof(txbuf));
}

void DmIMUDriver::save_imu_para()
{
    uint8_t txbuf[4] = {0xAA, 0x03, 0x01, 0x0D};
    send_imu_command(txbuf, sizeof(txbuf));
}

void DmIMUDriver::exit_setting_mode()
{
    uint8_t txbuf[4] = {0xAA, 0x06, 0x00, 0x0D};
    send_imu_command(txbuf, sizeof(txbuf));
}

bool DmIMUDriver::parse_byte(uint8_t byte)
{
    if (!frame_started_)
    {
        // Look for frame header 0x55
        if (byte == 0x55)
        {
            frame_started_ = true;
            rx_index_ = 0;
            rx_buf_[rx_index_++] = byte;
        }
        return false;
    }

    // Accumulate bytes
    rx_buf_[rx_index_++] = byte;

    // Need at least 4 bytes to verify header
    if (rx_index_ == 4)
    {
        uint8_t* p = rx_buf_;
        if (p[0] != 0x55 || p[1] != 0xAA || p[2] != 0x01 || p[3] != 0x01)
        {
            // Bad header, reset search
            frame_started_ = false;
            rx_index_ = 0;
            return false;
        }
    }

    // Full frame received
    if (rx_index_ >= DM_IMU_FRAME_SIZE)
    {
        frame_started_ = false;
        rx_index_ = 0;

        // Copy into receive_data struct
        memcpy(&receive_data_, rx_buf_, DM_IMU_FRAME_SIZE);

        // Validate CRC16 for each sub-frame (each sub-frame is 16 bytes excluding crc and end byte)
        bool valid = true;

        if (Get_CRC16((uint8_t*)(&receive_data_.FrameHeader1), 16) != receive_data_.crc1)
        {
            valid = false;
        }
        if (Get_CRC16((uint8_t*)(&receive_data_.FrameHeader2), 16) != receive_data_.crc2)
        {
            valid = false;
        }
        if (Get_CRC16((uint8_t*)(&receive_data_.FrameHeader3), 16) != receive_data_.crc3)
        {
            valid = false;
        }

        if (valid)
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);

            // Extract float data from uint32_t fields
            data_.accx  = *((float*)(&receive_data_.accx_u32));
            data_.accy  = *((float*)(&receive_data_.accy_u32));
            data_.accz  = *((float*)(&receive_data_.accz_u32));
            data_.gyrox = *((float*)(&receive_data_.gyrox_u32));
            data_.gyroy = *((float*)(&receive_data_.gyroy_u32));
            data_.gyroz = *((float*)(&receive_data_.gyroz_u32));
            data_.roll  = *((float*)(&receive_data_.roll_u32));
            data_.pitch = *((float*)(&receive_data_.pitch_u32));
            data_.yaw   = *((float*)(&receive_data_.yaw_u32));

            // Update base class data
            lin_acc_[0] = data_.accx;
            lin_acc_[1] = data_.accy;
            lin_acc_[2] = data_.accz;
            ang_vel_[0] = data_.gyrox;
            ang_vel_[1] = data_.gyroy;
            ang_vel_[2] = data_.gyroz;

            // Convert Euler angles (degrees) to quaternion
            double roll_rad  = data_.roll  * DM_IMU_DEG_TO_RAD;
            double pitch_rad = data_.pitch * DM_IMU_DEG_TO_RAD;
            double yaw_rad   = data_.yaw   * DM_IMU_DEG_TO_RAD;

            double cr = std::cos(roll_rad * 0.5);
            double sr = std::sin(roll_rad * 0.5);
            double cp = std::cos(pitch_rad * 0.5);
            double sp = std::sin(pitch_rad * 0.5);
            double cy = std::cos(yaw_rad * 0.5);
            double sy = std::sin(yaw_rad * 0.5);

            quat_[0] = static_cast<float>(cr * cp * cy + sr * sp * sy);  // w
            quat_[1] = static_cast<float>(sr * cp * cy - cr * sp * sy);  // x
            quat_[2] = static_cast<float>(cr * sp * cy + sr * cp * sy);  // y
            quat_[3] = static_cast<float>(cr * cp * sy - sr * sp * cy);  // z

            return true;
        }
        else
        {
            error_count_++;
            if (error_count_ > 1200)
            {
                logger_->warn("DM IMU: fail to get correct data, CRC mismatch");
                error_count_ = 0;
            }
        }
    }

    return false;
}

void DmIMUDriver::serial_rx_cbk(const uint8_t* data, size_t length)
{
    for (size_t i = 0; i < length; i++)
    {
        parse_byte(data[i]);
    }
}

std::vector<float> DmIMUDriver::get_ang_vel()
{
    std::lock_guard<std::mutex> lock(imu_mutex_);
    return ang_vel_;
}

std::vector<float> DmIMUDriver::get_quat()
{
    std::lock_guard<std::mutex> lock(imu_mutex_);
    return quat_;
}

std::vector<float> DmIMUDriver::get_lin_acc()
{
    std::lock_guard<std::mutex> lock(imu_mutex_);
    return lin_acc_;
}

std::vector<float> DmIMUDriver::get_euler()
{
    std::lock_guard<std::mutex> lock(imu_mutex_);
    return {data_.roll, data_.pitch, data_.yaw};
}

}  // namespace dmbot_imu
