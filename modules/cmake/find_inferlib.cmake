set(INFER_LIBS "" PARENT_SCOPE)

if(INFER_TENSORRT)
	find_package(CUDA REQUIRED)
	# TensorRT配置
	if(WIN32)
		find_path(TENSORRT_INCLUDE_DIR NvInfer.h
		PATHS c:/TensorRT/include)
		find_library(TENSORRT_LIBRARY_INFER nvinfer 
		PATHS c:/TensorRT/lib)
	elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
		find_path(TENSORRT_INCLUDE_DIR NvInfer.h
		PATHS /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include /usr/local/TensorRT-8.5.2.2/include)
		find_library(TENSORRT_LIBRARY_INFER nvinfer
		PATHS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib /usr/local/TensorRT-8.5.2.2/lib)
	elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
		find_path(TENSORRT_INCLUDE_DIR NvInfer.h
					PATHS /usr/include/aarch64-linux-gnu)
		find_library(TENSORRT_LIBRARY_INFER nvinfer
						PATHS /usr/lib/aarch64-linux-gnu)
	endif()
	list(APPEND INFER_LIBS ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY_INFER})
	list(APPEND INFER_INCLUDES ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
elseif(INFER_RKNN)
    set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknpu2)
    set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librknnrt.so)
    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)
    include_directories(${LIBRKNNRT_INCLUDES})
    list(APPEND INFER_LIBS ${LIBRKNNRT})
elseif(INFER_HBDNN)
    set(HBDNN_PATH /usr)
    set(LIBHBDNN ${HBDNN_PATH}/lib/libdnn.so)
    set(INFER_INCLUDES ${HBDNN_PATH}/include/dnn)
    list(APPEND INFER_LIBS ${LIBHBDNN})
endif()

