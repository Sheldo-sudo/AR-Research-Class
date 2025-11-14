
#ifndef VISION_PROCESSOR_H
#define VISION_PROCESSOR_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     com_example_glasspro_NativeProcessor
 * Method:    loadObjectDetector
 * Signature: (Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL
Java_com_example_glasspro_NativeProcessor_loadObjectDetector(JNIEnv *, jclass, jstring, jstring);

/*
 * Class:     com_example_glasspro_NativeProcessor
 * Method:    releaseObjectDetector
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_com_example_glasspro_NativeProcessor_releaseObjectDetector(JNIEnv *, jclass, jlong);

/*
 * Class:     com_example_glasspro_NativeProcessor
 * Method:    detectObjectsNN
 * Signature: (JJJJf)V
 */
JNIEXPORT void JNICALL
Java_com_example_glasspro_NativeProcessor_detectObjectsNN(JNIEnv *, jclass, jlong, jlong,
                                                          jlong, jlong, jfloat);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VISION_PROCESSOR_H