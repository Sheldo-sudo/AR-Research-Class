# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name.
project("glasspro")

# --- OpenCV Configuration (Corrected) ---
# 1. Set the path to your OpenCV jni folder (using your provided path)
set(OpenCV_DIR "D:/Erudite3/Research Seminar/opencv-4.11.0-android-sdk/OpenCV-android-sdk/sdk/native/jni")

# 2. Find the OpenCV package
find_package(OpenCV REQUIRED)

# 3. Include the OpenCV header files
include_directories(${OpenCV_INCLUDE_DIRS})
# --- End of OpenCV Configuration ---


# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

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
# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.
add_library(
        native-lib
        SHARED
        native-lib.cpp
)

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

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

# --- [!! 关键修复 !!] ---
# 你缺少了这个代码块。
# 它告诉 motion 库去链接 OpenCV 和 stab 库。
target_link_libraries( # Specifies the target library.
        motion
        ${OpenCV_LIBS} # 链接 OpenCV
        stab           # 链接 stab
        ${log-lib}     # 链接 log
)
# --- [!! 修复结束 !!] ---

target_link_libraries(
        native-lib
        dehaze
        enhance
        threshold
        stab
        motion         # <-- [!! 关键修复 !!] 你也缺少了这一行
        ${OpenCV_LIBS} # <-- MODIFIED
        ${log-lib}
)