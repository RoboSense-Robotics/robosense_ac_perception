#========================
# libs
#========================
set(CUR_LIB ppseg)

find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)

LIST(APPEND CUR_INCLUDES include)
set(CUR_SRCS
    src/common/common.cpp
    src/common/input_msg.cpp
    src/common/msg.cpp
    src/common/out_msg.cpp
    src/ppseg/config.cpp
    src/ppseg/preprocess.cpp
    src/ppseg/postprocess.cpp
    src/interface.cpp)

if(INFER_TENSORRT)
    list(APPEND CUR_SRCS src/ppseg/ppseg_nn_trt.cpp)
elseif(INFER_RKNN)
    list(APPEND CUR_SRCS src/ppseg/ppseg_nn_rk.cpp)
elseif(INFER_HBDNN)
    list(APPEND CUR_SRCS src/ppseg/ppseg_nn_hbdnn.cpp)
endif()


if(WIN32)
    add_library(${CUR_LIB} STATIC ${CUR_SRCS})
else()
    add_library(${CUR_LIB} SHARED ${CUR_SRCS})
endif()

target_include_directories(${CUR_LIB}
        PUBLIC
        ${CUR_INCLUDES}
        ${INFER_INCLUDES}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${pcl_conversions_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(${CUR_LIB}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${pcl_conversions_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        ${INFER_LIBS}
)


if(INFER_TENSORRT)
    message("ENABLE_TENSORRT IN PPSEG")
    target_compile_definitions(${CUR_LIB} PUBLIC ENABLE_TENSORRT)
elseif(INFER_RKNN)
    message("ENABLE_RKNN IN PPSEG")
    target_compile_definitions(${CUR_LIB} PUBLIC ENABLE_RKNN)
elseif(INFER_HBDNN)
    message("ENABLE_HBDNN IN PPSEG")
    target_compile_definitions(${CUR_LIB} PUBLIC ENABLE_HBDNN)
endif()

#=============================
# install
#=============================
install(TARGETS ${CUR_LIB}
  DESTINATION lib/${PROJECT_NAME})
