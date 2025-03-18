#========================
# libs
#========================
set(CUR_LIB yolov8)

find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)

LIST(APPEND CUR_INCLUDES include)
set(CUR_SRCS
    src/common/common.cpp
    src/common/input_msg.cpp
    src/common/msg.cpp
    src/common/out_msg.cpp
    src/yolov8/config.cpp
    src/yolov8/preprocess.cpp
    src/yolov8/postprocess.cpp
    src/interface.cpp)
if(INFER_TENSORRT)
    list(APPEND CUR_SRCS src/yolov8/yolov8_det_nn_trt.cpp)
elseif(INFER_RKNN)
    list(APPEND CUR_SRCS src/yolov8/yolov8_det_nn_rk.cpp)
endif()

if(INFER_TENSORRT)
	find_package(CUDA REQUIRED)
	# TensorRT配置
	if(WIN32)
		find_path(TENSORRT_INCLUDE_DIR NvInfer.h
		PATHS c:/TensorRT/include)
		find_library(TENSORRT_LIBRARY_INFER nvinfer 
		PATHS c:/TensorRT/lib)
	else()
		find_path(TENSORRT_INCLUDE_DIR NvInfer.h
		PATHS /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include /usr/local/TensorRT-8.5.2.2/include)
		find_library(TENSORRT_LIBRARY_INFER nvinfer
		PATHS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib /usr/local/TensorRT-8.5.2.2/lib)
	endif()
	include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
	list(APPEND INFER_LIBS ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY_INFER})
elseif(INFER_RKNN)
    set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknpu2)
    set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librknnrt.so)
    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)
    include_directories(${LIBRKNNRT_INCLUDES})
    list(APPEND INFER_LIBS ${LIBRKNNRT})
endif()

if(WIN32)
    add_library(${CUR_LIB} STATIC ${CUR_SRCS})
else()
    add_library(${CUR_LIB} SHARED ${CUR_SRCS})
endif()

target_include_directories(${CUR_LIB}
        PUBLIC
        ${CUR_INCLUDES}
        ${OpenCV_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(${CUR_LIB}
        ${OpenCV_LIBS}
        ${YAML_CPP_LIBRARIES}
        ${INFER_LIBS}
        )
if(INFER_TENSORRT)
    message("ENABLE_TENSORRT")
    target_compile_definitions(${CUR_LIB} PUBLIC ENABLE_TENSORRT)
elseif(INFER_RKNN)
    message("ENABLE_RKNN")
    target_compile_definitions(${CUR_LIB} PUBLIC ENABLE_RKNN)
endif()
ament_target_dependencies(${CUR_LIB}
    PCL
    pcl_conversions
    cv_bridge
)
#=============================
# install
#=============================
install(TARGETS ${CUR_LIB}
  DESTINATION lib/${PROJECT_NAME})
