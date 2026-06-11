AUX_SOURCE_DIRECTORY(. SOURCE_LIST_DM_IMU)
add_library(dm_imu
        STATIC
        ${SOURCE_LIST_DM_IMU}
)
target_include_directories(dm_imu PUBLIC ./ ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(dm_imu PUBLIC ${PUBLIC_DEPENDENCIES} imu_protocol)
