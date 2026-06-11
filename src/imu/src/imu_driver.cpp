#include "imu_driver.hpp"
#include "drivers/dm/dm_imu_driver.hpp"

std::shared_ptr<IMUDriver> IMUDriver::create_imu(uint16_t imu_id, const std::string& interface_type, const std::string& interface,
                                                const std::string& imu_type, const int baudrate) {
    if (imu_type == "DM") {
        return std::make_shared<dmbot_imu::DmIMUDriver>(imu_id, interface_type, interface, baudrate);
    } else {
        throw std::runtime_error("IMU type not supported: " + imu_type);
    }
}
