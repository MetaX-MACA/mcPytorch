include_directories(${PROJECT_SOURCE_DIR}/test_api)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(TORCH_API_TEST_DIR "${PROJECT_SOURCE_DIR}/test_api/test/cpp/api")
set(TORCH_API_TEST_SOURCES
  ${TORCH_API_TEST_DIR}/../common/main.cpp
  ${TORCH_API_TEST_DIR}/autograd.cpp
  ${TORCH_API_TEST_DIR}/any.cpp
  ${TORCH_API_TEST_DIR}/dataloader.cpp
  ${TORCH_API_TEST_DIR}/enum.cpp
  ${TORCH_API_TEST_DIR}/expanding-array.cpp
  ${TORCH_API_TEST_DIR}/fft.cpp
  ${TORCH_API_TEST_DIR}/functional.cpp
  ${TORCH_API_TEST_DIR}/integration.cpp
  ${TORCH_API_TEST_DIR}/init.cpp
  ${TORCH_API_TEST_DIR}/jit.cpp
  ${TORCH_API_TEST_DIR}/memory.cpp
  ${TORCH_API_TEST_DIR}/meta_tensor.cpp
  ${TORCH_API_TEST_DIR}/misc.cpp
  ${TORCH_API_TEST_DIR}/module.cpp
  ${TORCH_API_TEST_DIR}/moduledict.cpp
  ${TORCH_API_TEST_DIR}/modulelist.cpp
  ${TORCH_API_TEST_DIR}/modules.cpp
  ${TORCH_API_TEST_DIR}/parameterdict.cpp
  ${TORCH_API_TEST_DIR}/parameterlist.cpp
  ${TORCH_API_TEST_DIR}/namespace.cpp
  ${TORCH_API_TEST_DIR}/nn_utils.cpp
  ${TORCH_API_TEST_DIR}/optim.cpp
  ${TORCH_API_TEST_DIR}/ordered_dict.cpp
  ${TORCH_API_TEST_DIR}/rnn.cpp
  ${TORCH_API_TEST_DIR}/sequential.cpp
  ${TORCH_API_TEST_DIR}/transformer.cpp
  ${TORCH_API_TEST_DIR}/serialize.cpp
  ${TORCH_API_TEST_DIR}/special.cpp
  ${TORCH_API_TEST_DIR}/static.cpp
  ${TORCH_API_TEST_DIR}/support.cpp
  ${TORCH_API_TEST_DIR}/tensor_cuda.cpp
  ${TORCH_API_TEST_DIR}/tensor_indexing.cpp
  ${TORCH_API_TEST_DIR}/tensor_options_cuda.cpp
  ${TORCH_API_TEST_DIR}/tensor_options.cpp
  ${TORCH_API_TEST_DIR}/tensor.cpp
  ${TORCH_API_TEST_DIR}/torch_include.cpp
  ${TORCH_API_TEST_DIR}/inference_mode.cpp
  ${TORCH_API_TEST_DIR}/grad_mode.cpp
  ${TORCH_API_TEST_DIR}/operations.cpp
)

add_executable(test_api ${TORCH_API_TEST_SOURCES})
target_link_libraries(test_api "${TORCH_LIBRARIES}")




