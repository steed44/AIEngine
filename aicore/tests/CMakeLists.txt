# ============================================================
# 文件: tests/CMakeLists.txt
# 用途: AICore 单元测试构建配置
#   编译所有测试源文件 + 被测源文件 → 单个测试可执行文件
#   测试阶段按模块分组: Core → Optimizer → Trainer → PatchCore
# ============================================================

# GTest via FetchContent — vcpkg 无需安装
include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0
)
FetchContent_MakeAvailable(googletest)

# 自动搜集 tests/*.cpp 测试用例文件
# 排除 YOLO 测试（需要 LibTorch）
file(GLOB TEST_SOURCES *.cpp)
# 排除需要 LibTorch 的 YOLO 训练测试（test_yolo_decode_node.cpp 不需要 LibTorch）
list(FILTER TEST_SOURCES EXCLUDE REGEX "test_yolo_data\\.cpp$|test_yolo_loss\\.cpp$|test_yolo_model\\.cpp$|test_yolo_trainer\\.cpp$")

set(CMAKE_CXX_STANDARD 17)

# 测试可执行文件
add_executable(aicore_tests ${TEST_SOURCES})

target_include_directories(aicore_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(aicore_tests PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_include_directories(aicore_tests PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(aicore_tests PRIVATE ${Python3_INCLUDE_DIRS})

# 将各模块的被测源文件编译进测试程序
target_sources(aicore_tests PRIVATE
    # Phase 1 — Core: 引擎核心模块
    ${PROJECT_SOURCE_DIR}/src/backend/backend_factory.cpp
    ${PROJECT_SOURCE_DIR}/src/config/config_parser.cpp
    ${PROJECT_SOURCE_DIR}/src/config/pipeline_builder.cpp
    ${PROJECT_SOURCE_DIR}/src/engine/thread_pool.cpp
    ${PROJECT_SOURCE_DIR}/src/engine/engine_pool.cpp
    ${PROJECT_SOURCE_DIR}/src/engine/ai_engine.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/pipeline_impl.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/model_node.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/composite_node.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/merge_node.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline/fusion_node.cpp
    ${PROJECT_SOURCE_DIR}/src/preprocess/resize_node.cpp
    ${PROJECT_SOURCE_DIR}/src/preprocess/normalize_node.cpp
    ${PROJECT_SOURCE_DIR}/src/preprocess/letterbox_node.cpp
    ${PROJECT_SOURCE_DIR}/src/postprocess/nms_node.cpp
    ${PROJECT_SOURCE_DIR}/src/postprocess/nms_common.cpp
    ${PROJECT_SOURCE_DIR}/src/utils/draw_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/postprocess/yolo_decode_node.cpp
    ${PROJECT_SOURCE_DIR}/src/api/aicore_api.cpp
    # Phase 2 — Optimizer: 模型优化模块
    ${PROJECT_SOURCE_DIR}/src/optimizer/python_embedding.cpp
    ${PROJECT_SOURCE_DIR}/src/optimizer/onnx_exporter.cpp
    ${PROJECT_SOURCE_DIR}/src/optimizer/tensorrt_builder.cpp
    ${PROJECT_SOURCE_DIR}/src/optimizer/int8_calibrator.cpp
    ${PROJECT_SOURCE_DIR}/src/optimizer/model_optimizer.cpp
    ${PROJECT_SOURCE_DIR}/src/optimizer/optimizer_api.cpp
    # Phase 3 — Trainer: 训练模块
    ${PROJECT_SOURCE_DIR}/src/trainer/data/dataset.cpp
    ${PROJECT_SOURCE_DIR}/src/trainer/data/data_loader.cpp
    ${PROJECT_SOURCE_DIR}/src/trainer/data/augmentation.cpp
    ${PROJECT_SOURCE_DIR}/src/trainer/model/model_factory.cpp
    ${PROJECT_SOURCE_DIR}/src/trainer/model/python_trainer.cpp
    ${PROJECT_SOURCE_DIR}/src/trainer/training/training_loop.cpp
    ${PROJECT_SOURCE_DIR}/src/trainer/training/training_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/trainer/training/checkpoint.cpp
    ${PROJECT_SOURCE_DIR}/src/trainer/training/early_stopping.cpp
    ${PROJECT_SOURCE_DIR}/src/trainer/validation/validator.cpp
    ${PROJECT_SOURCE_DIR}/src/trainer/export/model_exporter.cpp
    ${PROJECT_SOURCE_DIR}/src/trainer/trainer_api.cpp
    # PatchCore: 异常检测模块
    ${PROJECT_SOURCE_DIR}/src/patchcore/memory_bank.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/coreset_sampler.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/backbone.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/backbone_opencv.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/backbone_model.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/patchcore_node.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/patchcore_trainer.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/folder_dataset.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/tiered_memory_bank.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/memory_manager.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/anomaly_evaluator.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/patchcore_visualize.cpp
    # Multi-ROI: 多区域多模型支持
    ${PROJECT_SOURCE_DIR}/src/patchcore/roi_def.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/roi_trainer.cpp
    ${PROJECT_SOURCE_DIR}/src/patchcore/multi_roi_node.cpp
    # Scheduler API: GPU 调度器 C API
    ${PROJECT_SOURCE_DIR}/src/api/scheduler_api.cpp
    # Server: 推理服务器
    ${PROJECT_SOURCE_DIR}/src/server/model_registry.cpp
    ${PROJECT_SOURCE_DIR}/src/server/inference_server.cpp
    ${PROJECT_SOURCE_DIR}/src/server/server_api.cpp
)

# 可选：LibTorch 源文件
if(Torch_FOUND)
    target_sources(aicore_tests PRIVATE
        ${PROJECT_SOURCE_DIR}/src/patchcore/backbone_libtorch.cpp
        ${PROJECT_SOURCE_DIR}/src/trainer/model/yolo_model.cpp
        ${PROJECT_SOURCE_DIR}/src/trainer/model/yolo_loss.cpp
        ${PROJECT_SOURCE_DIR}/src/trainer/data/yolo_data.cpp
        ${PROJECT_SOURCE_DIR}/src/trainer/training/yolo_trainer.cpp
        # YOLO 测试用例
        test_yolo_data.cpp
        test_yolo_loss.cpp
        test_yolo_model.cpp
        test_yolo_trainer.cpp
    )
endif()

# 编译宏定义：允许测试访问内部符号
target_compile_definitions(aicore_tests PRIVATE
    AICORE_EXPORTS
    AICORE_OPTIMIZER_EXPORTS
    AICORE_TRAINER_EXPORTS
)
if(Torch_FOUND)
    target_compile_definitions(aicore_tests PRIVATE AICORE_HAS_LIBTORCH)
endif()

# 链接 GTest (FetchContent) 和 OpenCV
target_link_libraries(aicore_tests PRIVATE gtest gtest_main ${OpenCV_LIBS} Python3::Python CUDA::cudart)
get_filename_component(PYTHON3_HOME "${Python3_EXECUTABLE}" DIRECTORY)
target_compile_definitions(aicore_tests PRIVATE PYTHON3_HOME="${PYTHON3_HOME}")

# ──────────────────────────────────────────────
# 复制运行时依赖 DLL 到测试输出目录
# 测试二进制直接链接 OpenCV / LibTorch / CUDA / Python，
# 运行时需要这些 DLL 在同一目录或 PATH 中
# ──────────────────────────────────────────────

# Python DLL：需要完整 python3XY.dll（如 python314.dll），
# 而非仅 python3.dll（稳定 ABI 存根）。在 Python 二进制目录找。
if(WIN32)
    # Python3 运行时 DLL
    get_filename_component(PYTHON3_BIN_DIR "${Python3_EXECUTABLE}" DIRECTORY)
    find_file(PYTHON3_RUNTIME_DLL
        NAMES "python3${Python3_VERSION_MINOR}.dll" "python3.dll"
        PATHS "${PYTHON3_BIN_DIR}"
              "${PYTHON3_LIBRARY_DIRS}"
              "${Python3_RUNTIME_LIBRARY_DIRS}"
        NO_DEFAULT_PATH
    )
    if(PYTHON3_RUNTIME_DLL)
        add_custom_command(TARGET aicore_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${PYTHON3_RUNTIME_DLL}" "$<TARGET_FILE_DIR:aicore_tests>")
        message(STATUS "Copying Python DLL: ${PYTHON3_RUNTIME_DLL}")
    else()
        # 兜底：复制目录下所有 python*.dll
        file(GLOB PYTHON_DLLS "${PYTHON3_BIN_DIR}/python*.dll")
        if(PYTHON_DLLS)
            add_custom_command(TARGET aicore_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${PYTHON_DLLS} "$<TARGET_FILE_DIR:aicore_tests>")
        else()
            message(WARNING "Python DLL not found — tests may fail at runtime")
        endif()
    endif()

    # OpenCV DLL
    find_file(OPENCV_DLL opencv_world4100.dll
        PATHS "${OpenCV_DIR}/../../bin"
              "${OpenCV_DIR}/x64/vc16/bin"
              "${OpenCV_DIR}/bin"
              "${OpenCV_INCLUDE_DIRS}/../bin"
              "${OpenCV_INCLUDE_DIRS}/../../x64/vc16/bin"
        NO_DEFAULT_PATH
    )
    if(OPENCV_DLL)
        add_custom_command(TARGET aicore_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENCV_DLL}" "$<TARGET_FILE_DIR:aicore_tests>")
    else()
        message(WARNING "OpenCV DLL (opencv_world4100.dll) not found — tests may fail at runtime")
    endif()

    # CUDA runtime DLL（cudart）
    find_file(CUDART_DLL cudart64_110.dll
        PATHS "${CUDAToolkit_BIN_DIR}"
              "${CUDAToolkit_ROOT_DIR}/bin"
              "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin"
              "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin"
              "$ENV{CUDA_PATH}/bin"
        NO_DEFAULT_PATH
    )
    if(CUDART_DLL)
        add_custom_command(TARGET aicore_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CUDART_DLL}" "$<TARGET_FILE_DIR:aicore_tests>")
    endif()
endif()

# LibTorch DLLs：复制 lib/ 下全部 DLL（含 torch_cpu/torch_cuda/c10/fbgemm 等）
if(Torch_FOUND AND WIN32)
    set(TORCH_LIB_DIR "${TORCH_INSTALL_PREFIX}/lib")
    file(GLOB TORCH_DLLS "${TORCH_LIB_DIR}/*.dll")
    if(TORCH_DLLS)
        add_custom_command(TARGET aicore_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS} "$<TARGET_FILE_DIR:aicore_tests>")
        message(STATUS "Copying ${TORCH_LIB_DIR}/*.dll to test output dir")
    else()
        message(WARNING "No LibTorch DLLs found in ${TORCH_LIB_DIR} — tests may fail at runtime")
    endif()
endif()
if(Torch_FOUND)
    target_include_directories(aicore_tests PRIVATE "${TORCH_INSTALL_PREFIX}/include")
    target_include_directories(aicore_tests PRIVATE "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include")
    target_link_libraries(aicore_tests PRIVATE ${TORCH_LIBRARIES})
endif()

# 注册 CTest 测试
add_test(NAME aicore_tests COMMAND aicore_tests)
