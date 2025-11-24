# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name.
project("glasspro")

# --- OpenCV Configuration (Corrected) ---
# 1. Set the path to your OpenCV jni folder (using your provided path)
set(OpenCV_DIR "D:/CourseResources/2025autumn/ResearchClass/Project/Research_classroom_2025-main/opencv-4.11.0-android-sdk/OpenCV-android-sdk/sdk/native/jni")

# 2. Find the OpenCV package
find_package(OpenCV REQUIRED)

# 3. Include the OpenCV header files
include_directories(${OpenCV_INCLUDE_DIRS})
# --- End of OpenCV Configuration ---


# ===================================================================
# 1. 定义你所有的功能模块库
# ===================================================================

add_library( # Sets the name of the library.
        dehaze
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        dehaze.cpp
        dehaze.h)

add_library( # Sets the name of the library.
        enhance
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        enhance.cpp
        enhance.h)

add_library( # Sets the name of the library.
        threshold
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        threshold.cpp
        threshold.h)
add_library( # Sets the name of the library.
        stab
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        videoStab.cpp
        videoStab.h)
add_library( # Sets the name of the library.
        motion
        SHARED
        motion.cpp
        motion.h)

# --- [!! 新增库：DNN 目标检测 !!] ---
add_library( # Sets the name of the library.
        vision_processor
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        vision_processor.cpp
        vision_processor.h)
# --- [!! 新增结束 !!] ---


# 这是你的主 JNI 库 (native-lib.cpp 所在的位置)
add_library(
        native-lib
        SHARED
        native-lib.cpp
)

# ===================================================================
# 2. 查找系统库
# ===================================================================

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# ===================================================================
# 3. 链接所有库
# ===================================================================

# --- 链接所有功能模块 ---

target_link_libraries( # Specifies the target library.
        dehaze
        enhance
        ${OpenCV_LIBS} # <-- MODIFIED
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})

target_link_libraries( # Specifies the target library.
        enhance
        ${OpenCV_LIBS} # <-- MODIFIED
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})

target_link_libraries( # Specifies the target library.
        threshold
        ${OpenCV_LIBS} # <-- MODIFIED
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})
target_link_libraries( # Specifies the target library.
        stab
        ${OpenCV_LIBS} # <-- MODIFIED
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})

target_link_libraries( # Specifies the target library.
        motion
        ${OpenCV_LIBS} # 链接 OpenCV
        stab           # 链接 stab
        ${log-lib}     # 链接 log
)

# --- [!! 新增库的链接 !!] ---
target_link_libraries( # Specifies the target library.
        vision_processor
        ${OpenCV_LIBS} # 链接 OpenCV (DNN 模块)
        ${log-lib}     # 链接 log
)
# --- [!! 新增结束 !!] ---


# --- 链接你的主库 "native-lib" ---
# 你的 MainActivity 只加载 "native-lib"，
# 所以 "native-lib" 必须链接 *所有* 其他模块。
target_link_libraries(
        native-lib
        dehaze
        enhance
        threshold
        stab
        motion
        vision_processor  # <-- [!! 核心修改：添加新库 !!]
        ${OpenCV_LIBS}
        ${log-lib}
)