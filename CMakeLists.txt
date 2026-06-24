cmake_minimum_required(VERSION 3.20)
project(AICore)

# VS2022 打开此文件夹即可识别整个项目
add_subdirectory(aicore)

# 构建时将 .sln 复制到根目录（双击即开）
# 路径从 aicore\ -> out\build\aicore\ 以匹配实际 .vcxproj 位置
set(_sln_src "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.sln")
set(_sln_dst "${PROJECT_SOURCE_DIR}/${PROJECT_NAME}.sln")
add_custom_target(copy_sln ALL
  COMMAND powershell -ExecutionPolicy Bypass -File
    "${PROJECT_SOURCE_DIR}/scripts/copy-sln.ps1"
    -SourceSln "${_sln_src}"
    -DestSln "${_sln_dst}"
  COMMENT "Syncing ${PROJECT_NAME}.sln to source root"
)
unset(_sln_src)
unset(_sln_dst)


