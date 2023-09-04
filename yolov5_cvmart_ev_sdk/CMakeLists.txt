cmake_minimum_required(VERSION 3.5)
project(ji)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX "/usr/local/ev_sdk")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_BUILD_TYPE "Release")


#配置点线框解析相关头文件路径和源文件
include_directories(${CMAKE_SOURCE_DIR}/3rd/wkt_parser/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd/wkt_parser)

include_directories(/usr/local/evdeploy/include)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(EVDEPLOY_LIB_PATH "/usr/local/evdeploy/x86_64/lib")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
	set(EVDEPLOY_LIB_PATH "/usr/local/evdeploy/aarch64/lib")
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:/usr/local/ev_sdk/lib")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${EVDEPLOY_LIB_PATH}")
# 编译libji.so
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRCS)
list(APPEND libji_SOURCES  ${SRCS})

add_library (${PROJECT_NAME} SHARED ${libji_SOURCES})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        glog
        wktparser
	-L${EVDEPLOY_LIB_PATH} -levdeploy
        )

# Install libji.so
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
# ---  END EV_SDK DEPENDENCIES  ---
