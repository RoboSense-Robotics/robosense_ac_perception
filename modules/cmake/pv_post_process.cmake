#========================
# libs
#========================
set(CUR_LIB pv_post_process)
set(CUR_SRCS "")
LIST(APPEND CUR_INCLUDES include)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${CUR_LIB} STATIC
        src/pv_post_process/pv_post_process.cpp
)

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
        )
ament_target_dependencies(${CUR_LIB}
        PCL
        pcl_conversions
      )
#=============================
# install
#=============================
install(TARGETS ${CUR_LIB}
  DESTINATION lib/${PROJECT_NAME})
