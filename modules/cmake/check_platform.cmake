

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86|AMD64")
    message(STATUS "Target platform is x86")
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        set(INFER_TENSORRT ON)
        if(MSVC)
			# MSVC禁用已弃用API警告（对应C4996警告）
			add_compile_options(/wd4996)
		else()
			# GCC/Clang禁用弃用声明警告
			add_compile_options(-Wno-deprecated-declarations)
		endif()
        message(STATUS "INFER WITH TENSORRT")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    message(STATUS "Target platform is ARM")
    #RKNN
    execute_process(
    COMMAND bash -c "cat /proc/cpuinfo | grep 'Rockchip RK3588' | uniq | cut -d ':' -f 2"
    OUTPUT_VARIABLE CPU_MODEL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(CPU_MODEL STREQUAL " Rockchip RK3588")
        message(STATUS "Detected RK3588 chip")
        add_definitions(-DRK3588)
        set(INFER_RKNN ON)
        message(STATUS "INFER WITH RKNN")
        if (CMAKE_SYSTEM_NAME STREQUAL "Android")
            set (TARGET_LIB_ARCH ${CMAKE_ANDROID_ARCH_ABI})
        else()
            if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                set (TARGET_LIB_ARCH aarch64)
            else()
                set (TARGET_LIB_ARCH armhf)
            endif()
            if (CMAKE_C_COMPILER MATCHES "uclibc")
                set (TARGET_LIB_ARCH ${TARGET_LIB_ARCH}_uclibc)
            endif()
        endif()
    endif()
    #RDK X5
    execute_process(
    COMMAND bash -c "cat /proc/device-tree/model | grep -a 'D-Robotics RDK X5' | uniq | cut -d 'V' -f 1"
    OUTPUT_VARIABLE BOARD_PLATFORM
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (BOARD_PLATFORM STREQUAL "D-Robotics RDK X5")
        option(INFER_HBDNN "USE HBDNN" ON)
    endif()

    # Jetson Orin Nano
    execute_process(
    COMMAND bash -c "cat /proc/device-tree/model | grep -a 'NVIDIA Jetson Orin Nano' | uniq | cut -d ' ' -f 1-4"
    OUTPUT_VARIABLE BOARD_PLATFORM
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (BOARD_PLATFORM STREQUAL "NVIDIA Jetson Orin Nano")
        option(INFER_TENSORRT "USE TENSORRT" ON)
    endif()
else()
    message(STATUS "Unknown platform: ${CMAKE_SYSTEM_PROCESSOR}")
endif()