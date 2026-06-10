cmake_minimum_required(VERSION 3.20)
project(AICore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.7.0 REQUIRED)
find_package(GTest REQUIRED)

# LibTorch (PatchCore backbone, 可选)
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "Found LibTorch: ${TORCH_INSTALL_PREFIX}")
    add_compile_definitions(AICORE_HAS_LIBTORCH)
endif()

# ──────────────────────────────────────────────
# nlohmann-json (单头文件)
# ──────────────────────────────────────────────
set(NLOHMANN_JSON_DIR ${CMAKE_SOURCE_DIR}/third_party)

# ──────────────────────────────────────────────
# 主推理引擎 DLL (aicore.dll)
# ──────────────────────────────────────────────
set(AICORE_SOURCES
    src/api/aicore_api.cpp
    src/config/config_parser.cpp
    src/config/pipeline_builder.cpp
    src/engine/thread_pool.cpp
    src/engine/engine_pool.cpp
    src/pipeline/pipeline_impl.cpp
    src/pipeline/model_node.cpp
    src/pipeline/composite_node.cpp
    src/pipeline/merge_node.cpp
    src/backend/backend_factory.cpp
    src/backend/tensorrt_backend.cpp
    src/backend/onnxruntime_backend.cpp
    src/backend/libtorch_backend.cpp
    src/preprocess/resize_node.cpp
    src/preprocess/normalize_node.cpp
    src/postprocess/nms_node.cpp
    src/patchcore/memory_bank.cpp
    src/patchcore/coreset_sampler.cpp
    src/patchcore/patchcore_node.cpp
    src/patchcore/patchcore_trainer.cpp
    src/patchcore/folder_dataset.cpp
)

add_library(aicore SHARED ${AICORE_SOURCES})
target_include_directories(aicore
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${NLOHMANN_JSON_DIR}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(aicore PRIVATE ${OpenCV_LIBS})
if(Torch_FOUND)
    target_link_libraries(aicore PRIVATE ${TORCH_LIBRARIES})
endif()
target_compile_definitions(aicore PRIVATE AICORE_EXPORTS)

# ──────────────────────────────────────────────
# 模型优化工具 DLL (aicore_optimizer.dll)
# ──────────────────────────────────────────────
set(AICORE_OPTIMIZER_SOURCES
    src/optimizer/python_embedding.cpp
    src/optimizer/onnx_exporter.cpp
    src/optimizer/tensorrt_builder.cpp
    src/optimizer/int8_calibrator.cpp
    src/optimizer/model_optimizer.cpp
    src/optimizer/optimizer_api.cpp
)

add_library(aicore_optimizer SHARED ${AICORE_OPTIMIZER_SOURCES})
target_include_directories(aicore_optimizer
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${NLOHMANN_JSON_DIR}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(aicore_optimizer PRIVATE ${OpenCV_LIBS})
target_compile_definitions(aicore_optimizer PRIVATE AICORE_OPTIMIZER_EXPORTS)

# ──────────────────────────────────────────────
# 训练模块 DLL (aicore_trainer.dll)
# ──────────────────────────────────────────────
set(AICORE_TRAINER_SOURCES
    src/trainer/data/dataset.cpp
    src/trainer/data/data_loader.cpp
    src/trainer/data/augmentation.cpp
    src/trainer/model/model_factory.cpp
    src/trainer/model/python_trainer.cpp
    src/trainer/training/training_loop.cpp
    src/trainer/training/training_scheduler.cpp
    src/trainer/training/checkpoint.cpp
    src/trainer/training/early_stopping.cpp
    src/trainer/validation/validator.cpp
    src/trainer/export/model_exporter.cpp
    src/trainer/trainer_api.cpp
)

add_library(aicore_trainer SHARED ${AICORE_TRAINER_SOURCES})
target_include_directories(aicore_trainer
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${NLOHMANN_JSON_DIR}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(aicore_trainer PRIVATE aicore_optimizer ${OpenCV_LIBS})
target_compile_definitions(aicore_trainer PRIVATE AICORE_TRAINER_EXPORTS)

# ──────────────────────────────────────────────
# CLI 可执行文件
# ──────────────────────────────────────────────
add_executable(ModelOptimizer cli/model_optimizer_main.cpp)
target_include_directories(ModelOptimizer PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(ModelOptimizer PRIVATE ${NLOHMANN_JSON_DIR})
target_include_directories(ModelOptimizer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ModelOptimizer PRIVATE aicore_optimizer ${OpenCV_LIBS})

add_executable(AICoreTrainer cli/trainer_main.cpp)
target_include_directories(AICoreTrainer PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(AICoreTrainer PRIVATE ${NLOHMANN_JSON_DIR})
target_include_directories(AICoreTrainer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(AICoreTrainer PRIVATE aicore_trainer ${OpenCV_LIBS})

# ──────────────────────────────────────────────
# PatchCore 训练 CLI (PatchCoreTrain.exe)
# ──────────────────────────────────────────────
add_executable(PatchCoreTrain cli/patchcore_train_main.cpp)
target_include_directories(PatchCoreTrain PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(PatchCoreTrain PRIVATE ${NLOHMANN_JSON_DIR})
target_include_directories(PatchCoreTrain PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(PatchCoreTrain PRIVATE aicore ${OpenCV_LIBS})

# ──────────────────────────────────────────────
# 测试
# ──────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)

# ──────────────────────────────────────────────
# Qt5 上位机 (AICoreUI.exe)
# ──────────────────────────────────────────────
set(CMAKE_AUTOMOC ON)
find_package(Qt5 REQUIRED COMPONENTS Widgets Concurrent)

add_executable(AICoreUI
    gui/main.cpp
    gui/main_window.cpp
)
target_include_directories(AICoreUI PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(AICoreUI PRIVATE Qt5::Widgets Qt5::Concurrent aicore)
