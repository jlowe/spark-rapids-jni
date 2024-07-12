/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cudf_jni_apis.hpp"

namespace {

constexpr char const* SPLIT_RESULT_CLASS =
  "com/nvidia/spark/rapids/jni/ShuffleSplitAssemble$SplitResult";

bool is_initialized = false;
jclass Split_result_jclass;

void initialize(JNIEnv* env)
{
  if (is_initialized) {
    return;
  }
  jclass cls = env->FindClass(SPLIT_RESULT_CLASS);
  if (cls == nullptr) {
    throw std::runtime_error(std::string("Unable to load ") + SPLIT_RESULT_CLASS);
  }
  // Convert local reference to global so it cannot be garbage collected.
  Split_result_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (Split_result_jclass == nullptr) {
    throw std::runtime_error("Unable to convert local ref to global ref");
  }
  is_initialized = true;
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_ShuffleSplitAssemble_split(
  JNIEnv* env, jclass, jlong metadata_addr, jlong metadata_size, jlong jtable,
  jintArray jsplit_indices)
{
  JNI_NULL_CHECK(metadata_addr, "metadata is null", nullptr);
  JNI_NULL_CHECK(env, jtable, "input table is null", nullptr);
  try {
    cudf::jni::auto_set_device(env);
    auto const table = reinterpret_cast<cudf::table_view const*>(jtable);
    cudf::jni::native_jintArray indices(env, jsplit_indices);
    auto split_result = spark_rapids_jni::zipsplit(table, indices.to_vector(),
                                                   cudf::get_default_stream(),
                                                   rmm::mr::get_current_device_resource());
    cudf::jni::jintArray offsets(env, split_result.offsets);
    return env->NewObject(Split_result_jclass,
                          cudf::jni::ptr_as_jlong(split_result.partitions->data()),
                          split_result.partitions->size(),
                          cudf::jni::release_as_jlong(split_result.partitions),
                          offsets.get_jArray());
  }
  CATCH_STD(env, nullptr);
}

}  // extern "C"
