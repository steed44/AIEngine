file(GLOB TEST_SOURCES *.cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(aicore_tests ${TEST_SOURCES})

target_include_directories(aicore_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(aicore_tests PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_include_directories(aicore_tests PRIVATE ${OpenCV_INCLUDE_DIRS})

target_sources(aicore_tests PRIVATE
    # Phase 1 — Core
    ${CMAKE_SOURCE_DIR}/src/backend/backend_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/config/config_parser.cpp
    ${CMAKE_SOURCE_DIR}/src/config/pipeline_builder.cpp
    ${CMAKE_SOURCE_DIR}/src/engine/thread_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/engine/engine_pool.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/pipeline_impl.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/model_node.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/composite_node.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/merge_node.cpp
    ${CMAKE_SOURCE_DIR}/src/preprocess/resize_node.cpp
    ${CMAKE_SOURCE_DIR}/src/preprocess/normalize_node.cpp
    ${CMAKE_SOURCE_DIR}/src/postprocess/nms_node.cpp
    # Phase 2 — Optimizer
    ${CMAKE_SOURCE_DIR}/src/optimizer/python_embedding.cpp
    ${CMAKE_SOURCE_DIR}/src/optimizer/onnx_exporter.cpp
    ${CMAKE_SOURCE_DIR}/src/optimizer/tensorrt_builder.cpp
    ${CMAKE_SOURCE_DIR}/src/optimizer/int8_calibrator.cpp
    ${CMAKE_SOURCE_DIR}/src/optimizer/model_optimizer.cpp
    ${CMAKE_SOURCE_DIR}/src/optimizer/optimizer_api.cpp
    # Phase 3 — Trainer
    ${CMAKE_SOURCE_DIR}/src/trainer/data/dataset.cpp
    ${CMAKE_SOURCE_DIR}/src/trainer/data/data_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/trainer/data/augmentation.cpp
    ${CMAKE_SOURCE_DIR}/src/trainer/model/model_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/trainer/model/python_trainer.cpp
    ${CMAKE_SOURCE_DIR}/src/trainer/training/training_loop.cpp
    ${CMAKE_SOURCE_DIR}/src/trainer/training/training_scheduler.cpp
    ${CMAKE_SOURCE_DIR}/src/trainer/training/checkpoint.cpp
    ${CMAKE_SOURCE_DIR}/src/trainer/training/early_stopping.cpp
    ${CMAKE_SOURCE_DIR}/src/trainer/validation/validator.cpp
    ${CMAKE_SOURCE_DIR}/src/trainer/export/model_exporter.cpp
    ${CMAKE_SOURCE_DIR}/src/trainer/trainer_api.cpp
    # PatchCore
    ${CMAKE_SOURCE_DIR}/src/patchcore/memory_bank.cpp
    ${CMAKE_SOURCE_DIR}/src/patchcore/coreset_sampler.cpp
    ${CMAKE_SOURCE_DIR}/src/patchcore/patchcore_node.cpp
    ${CMAKE_SOURCE_DIR}/src/patchcore/patchcore_trainer.cpp
    ${CMAKE_SOURCE_DIR}/src/patchcore/folder_dataset.cpp
)

target_compile_definitions(aicore_tests PRIVATE
    AICORE_EXPORTS
    AICORE_OPTIMIZER_EXPORTS
    AICORE_TRAINER_EXPORTS
)

target_link_libraries(aicore_tests PRIVATE GTest::GTest GTest::Main ${OpenCV_LIBS})
add_test(NAME aicore_tests COMMAND aicore_tests)
