#include "optimizer/python_embedding.h"
#include <Python.h>
#include <filesystem>
#include <fstream>

namespace aicore {

namespace fs = std::filesystem;

PythonEmbedding::PythonEmbedding() {}

PythonEmbedding::~PythonEmbedding() { Finalize(); }

Status PythonEmbedding::ensureInitialized() {
    if (initialized_) return Status{};
    return Initialize();
}

Status PythonEmbedding::Initialize() {
    if (initialized_) return Status{};

    // 设置 PYTHONHOME 确保嵌入式 Python 能找到标准库
    // 默认使用 CMake 注入的 PYTHON3_HOME，可通过环境变量 AICORE_PYTHON_HOME 覆盖
    const char* home = getenv("AICORE_PYTHON_HOME");
    if (!home) home = PYTHON3_HOME;

    PyStatus status;
    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);

    if (home) {
        status = PyConfig_SetBytesString(&config, &config.home, home);
        if (PyStatus_Exception(status)) {
            PyConfig_Clear(&config);
            return Status{StatusCode::ErrorInternal, "Failed to set Python home"};
        }
    }

    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        return Status{StatusCode::ErrorInternal, "Py_InitializeFromConfig failed"};
    }

    PyRun_SimpleString("import sys; sys.path.insert(0, '.')");
    initialized_ = true;
    return Status{};
}

Status PythonEmbedding::RunScript(const std::string& script,
                                   const std::string& configJson,
                                   std::string& output) {
    auto s = ensureInitialized();
    if (!s) return s;

    PyGILState_STATE gstate = PyGILState_Ensure();

    // 1. Resolve script path and add dir to sys.path
    fs::path scriptPath = fs::absolute(script);
    std::string scriptDir = scriptPath.parent_path().string();
    std::string moduleName = scriptPath.stem().string();

    std::string pathCmd = "import sys; sys.path.insert(0, '" + scriptDir + "')";
    PyRun_SimpleString(pathCmd.c_str());

    // 2. Inject progress_hook into __main__ (if callback registered)
    if (progressCb_) {
        // Build temp file path for progress communication
        std::string progressFile = (fs::temp_directory_path() / "aicore_progress.jsonl").string();
        std::string injectCode = R"(
import json, os
_progress_file = ')" + progressFile + R"('
def progress_hook(data):
    with open(_progress_file, 'a') as f:
        f.write(json.dumps(data) + '\n')
)";
        PyRun_SimpleString(injectCode.c_str());
        // Store progress file path for polling
        (void)progressFile;  // TODO: store and expose for GUI polling
    }

    // 3. Import module
    PyObject* pModuleName = PyUnicode_DecodeFSDefault(moduleName.c_str());
    PyObject* pModule = PyImport_Import(pModuleName);
    Py_DECREF(pModuleName);

    if (!pModule) {
        PyErr_Print();
        PyGILState_Release(gstate);
        return Status{StatusCode::ErrorInternal,
                      "Failed to import module: " + moduleName};
    }

    // 4. Get train() function
    PyObject* pFunc = PyObject_GetAttrString(pModule, "train");
    if (!pFunc || !PyCallable_Check(pFunc)) {
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
        PyGILState_Release(gstate);
        return Status{StatusCode::ErrorInternal,
                      "Function 'train' not found in " + moduleName};
    }

    // 5. Call train(configJson)
    PyObject* pArgs = Py_BuildValue("(s)", configJson.c_str());
    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
    Py_DECREF(pArgs);

    Status ret;
    if (!pResult) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        if (pvalue) {
            PyObject* pStr = PyObject_Str(pvalue);
            if (pStr) {
                const char* errStr = PyUnicode_AsUTF8(pStr);
                if (errStr) ret = Status{StatusCode::ErrorInternal, errStr};
                Py_DECREF(pStr);
            }
        }
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
    } else {
        const char* resultStr = PyUnicode_AsUTF8(pResult);
        if (resultStr) output = resultStr;
        else ret = Status{StatusCode::ErrorInternal, "train() returned non-string"};
    }

    Py_XDECREF(pResult);
    Py_DECREF(pFunc);
    Py_DECREF(pModule);
    PyGILState_Release(gstate);
    return ret;
}

void PythonEmbedding::Finalize() {
    if (!initialized_) return;
    PyGILState_STATE gstate = PyGILState_Ensure();
    Py_FinalizeEx();
    initialized_ = false;
}

} // namespace aicore
