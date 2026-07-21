# Build separate libraries the define custom classes/operators used from our Python tests.
# These are intended to be used with torch.ops.load_library() in our Python test suite.
add_library(aoti_custom_ops SHARED custom_ops.cpp)

if(USE_MACA)
  target_compile_definitions(aoti_custom_ops PRIVATE USE_MACA)
  target_compile_definitions(aoti_custom_ops PRIVATE NV_ARCH_A100)
endif()

target_link_libraries(aoti_custom_ops torch)

if(INSTALL_TEST)
  install(TARGETS aoti_custom_ops DESTINATION lib)
endif()
