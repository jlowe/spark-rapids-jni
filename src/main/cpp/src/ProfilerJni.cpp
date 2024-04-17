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

#include <cupti.h>
#include <jni.h>

#include <iostream>
#include <stdlib.h>
#include <string.h>

// CUPTI headers do not declare their NVTX initialization, so we need to declare it per
// https://docs.nvidia.com/cupti/main/main.html#nvidia-tools-extension-callbacks
extern "C" CUptiResult CUPTIAPI cuptiNvtxInitialize2(void* pfnGetExportTable);

// HACK - FIXME
#define JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val)                                                 \
  {                                                                                                \
    if (env->ExceptionOccurred()) {                                                                \
      return ret_val;                                                                              \
    }                                                                                              \
  }

#define JNI_THROW_NEW(env, class_name, message, ret_val)                                           \
  {                                                                                                \
    jclass ex_class = env->FindClass(class_name);                                                  \
    if (ex_class == NULL) {                                                                        \
      return ret_val;                                                                              \
    }                                                                                              \
    env->ThrowNew(ex_class, message);                                                              \
    return ret_val;                                                                                \
  }

#define CATCH_STD_CLASS(env, class_name, ret_val)                                                  \
  catch (const std::exception &e) {                                                                \
    JNI_THROW_NEW(env, class_name, e.what(), ret_val)                                              \
  }

#define CATCH_STD(env, ret_val) CATCH_STD_CLASS(env, "java/lang/RuntimeException", ret_val)

// END HACK - FIXME


namespace {

constexpr size_t ALIGN_BYTES = 8;
constexpr size_t BUFF_SIZE = 8 * 1024 * 1024;

struct subscriber_state {
  subscriber_state(JNIEnv* env) : jni(env), callback_errored(false) {}
  JNIEnv* jni;
  CUpti_SubscriberHandle subscriber_handle;
  bool callback_errored;
};

subscriber_state* State = nullptr;

char const* get_cupti_error(CUptiResult rc)
{
  char const* err;
  if (cuptiGetResultString(rc, &err) != CUPTI_SUCCESS) {
    err = "UNKNOWN";
  }
  return err;
}

void check_cupti(CUptiResult rc, std::string msg)
{
  if (rc != CUPTI_SUCCESS) {
    throw std::runtime_error(msg + ": " + get_cupti_error(rc));
  }
}

//void domain_state_callback(CUpti_CallbackId callback_id, CUpti_StateData const* data_ptr)
//{
//  switch (callback_id) {
//    case CUPTI_CBID_STATE_FATAL_ERROR:
//    {
//      auto error = get_cupti_error(data_ptr->notification.result);
//      std::cerr << "CUPTI reported a fatal error: " << error << std::endl;
//      if (data_ptr->notification.message != nullptr) {
//        std::cerr << "CUPTI: " << data_ptr->notification.message << std::endl;
//      }
//    }
//    case CUPTI_CBID_STATE_ERROR:
//    {
//      auto error = get_cupti_error(data_ptr->notification.result);
//      std::cerr << "CUPTI reported an error: " << error << std::endl;
//      if (data_ptr->notification.message != nullptr) {
//        std::cerr << "CUPTI: " << data_ptr->notification.message << std::endl;
//      }
//    }
//    case CUPTI_CBID_STATE_WARNING:
//    {
//      auto error = get_cupti_error(data_ptr->notification.result);
//      std::cerr << "CUPTI reported a warning: " << error << std::endl;
//      if (data_ptr->notification.message != nullptr) {
//        std::cerr << "CUPTI: " << data_ptr->notification.message << std::endl;
//      }
//    }
//    default:
//      std::cer << "Ignoring CUPTI domain state callback for " << callback_id << std::endl;
//      break;
//  }
//}

void domain_runtime_callback(CUpti_CallbackId callback_id, CUpti_CallbackData const* data_ptr)
{
  switch (callback_id) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020:
      if (data_ptr->callbackSite == CUPTI_API_ENTER) {
        auto rc = cuptiActivityFlushAll(0);
        if (rc != CUPTI_SUCCESS) {
          std::cerr << "Error flushing CUPTI activity on device reset: " << get_cupti_error(rc) << std::endl;
        }
      }
      break;
    default:
      break;
  }
}

void CUPTIAPI callback_handler(void*, CUpti_CallbackDomain domain,
    CUpti_CallbackId callback_id, const void* callback_data_ptr)
{
  auto rc = cuptiGetLastError();
  if (rc != CUPTI_SUCCESS && !State->callback_errored) {
    //State->callback_errored = true;
    std::cerr << "ERROR HANDLING CALLBACK: " << get_cupti_error(rc) << std::endl;
    return;
  }

  switch (domain) {
//    case CUPTI_CB_DOMAIN_STATE:
//    {
//      auto domain_data = static_cast<CUpti_StateData const *>(callback_data_ptr);
//      domain_state_callback(callback_id, domain_data);
//      break;
//    }
    case CUPTI_CB_DOMAIN_RUNTIME_API:
    {
      auto runtime_data = static_cast<CUpti_CallbackData const *>(callback_data_ptr);
      domain_runtime_callback(callback_id, runtime_data);
      break;
    }
    default:
      break;
  }
}

std::string activity_kind_to_string(CUpti_ActivityKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: return "CUPTI_ACTIVITY_KIND_MEMCPY";
    case CUPTI_ACTIVITY_KIND_MEMSET: return "CUPTI_ACTIVITY_KIND_MEMSET";
    case CUPTI_ACTIVITY_KIND_KERNEL: return "CUPTI_ACTIVITY_KIND_KERNEL";
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: return "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL";
    case CUPTI_ACTIVITY_KIND_DRIVER: return "CPUTI_ACTIVITY_KIND_DRIVER";
    case CUPTI_ACTIVITY_KIND_RUNTIME: return "CUPTI_ACTIVITY_KIND_RUNTIME";
    case CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API: return "CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API";
    case CUPTI_ACTIVITY_KIND_EVENT: return "CUPTI_ACTIVITY_KIND_EVENT";
    case CUPTI_ACTIVITY_KIND_METRIC: return "CUPTI_ACTIVITY_KIND_METRIC";
    case CUPTI_ACTIVITY_KIND_DEVICE: return "CUPTI_ACTIVITY_KIND_DEVICE";
    case CUPTI_ACTIVITY_KIND_CONTEXT: return "CUPTI_ACTIVITY_KIND_CONTEXT";
    case CUPTI_ACTIVITY_KIND_NAME: return "CUPTI_ACTIVITY_KIND_NAME";
    case CUPTI_ACTIVITY_KIND_MARKER: return "CUPTI_ACTIVITY_KIND_MARKER";
    case CUPTI_ACTIVITY_KIND_MARKER_DATA: return "CUPTI_ACTIVITY_KIND_MARKER_DATA";
    case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR: return "CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR";
    case CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS: return "CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS";
    case CUPTI_ACTIVITY_KIND_BRANCH: return "CUPTI_ACTIVITY_KIND_BRANCH";
    case CUPTI_ACTIVITY_KIND_OVERHEAD: return "CUPTI_ACTIVITY_KIND_OVERHEAD";
    case CUPTI_ACTIVITY_KIND_CDP_KERNEL: return "CUPTI_ACTIVITY_KIND_CDP_KERNEL";
    case CUPTI_ACTIVITY_KIND_PREEMPTION: return "CUPTI_ACTIVITY_KIND_PREEMPTION";
    case CUPTI_ACTIVITY_KIND_ENVIRONMENT: return "CUPTI_ACTIVITY_KIND_ENVIRONMENT";
    case CUPTI_ACTIVITY_KIND_EVENT_INSTANCE: return "CUPTI_ACTIVITY_KIND_EVENT_INSTANCE";
    case CUPTI_ACTIVITY_KIND_MEMCPY2: return "CUPTI_ACTIVITY_KIND_MEMCPY2";
    case CUPTI_ACTIVITY_KIND_METRIC_INSTANCE: return "CUPTI_ACTIVITY_KIND_METRIC_INSTANCE";
    case CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION: return "CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION";
    case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER: return "CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER";
    case CUPTI_ACTIVITY_KIND_FUNCTION: return "CUPTI_ACTIVITY_KIND_FUNCTION";
    case CUPTI_ACTIVITY_KIND_MODULE: return "CUPTI_ACTIVITY_KIND_MODULE";
    case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE: return "CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE";
    case CUPTI_ACTIVITY_KIND_SHARED_ACCESS: return "CUPTI_ACTIVITY_KIND_SHARED_ACCESS";
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING: return "CUPTI_ACTIVITY_KIND_PC_SAMPLING";
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO: return "CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO";
    case CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION: return "CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION";
    case CUPTI_ACTIVITY_KIND_OPENACC_DATA: return "CUPTI_ACTIVITY_KIND_OPENACC_DATA";
    case CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH: return "CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH";
    case CUPTI_ACTIVITY_KIND_OPENACC_OTHER: return "CUPTI_ACTIVITY_KIND_OPENACC_OTHER";
    case CUPTI_ACTIVITY_KIND_CUDA_EVENT: return "CUPTI_ACTIVITY_KIND_CUDA_EVENT";
    case CUPTI_ACTIVITY_KIND_STREAM: return "CUPTI_ACTIVITY_KIND_STREAM";
    case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: return "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION";
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: return "CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION";
    case CUPTI_ACTIVITY_KIND_NVLINK: return "CUPTI_ACTIVITY_KIND_NVLINK";
    case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT: return "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT";
    case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE: return "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE";
    case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC: return "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC";
    case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE: return "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE";
    case CUPTI_ACTIVITY_KIND_MEMORY: return "CUPTI_ACTIVITY_KIND_MEMORY";
    case CUPTI_ACTIVITY_KIND_PCIE: return "CUPTI_ACTIVITY_KIND_PCIE";
    case CUPTI_ACTIVITY_KIND_OPENMP: return "CUPTI_ACTIVITY_KIND_OPENMP";
    case CUPTI_ACTIVITY_KIND_MEMORY2: return "CUPTI_ACTIVITY_KIND_MEMORY2";
    case CUPTI_ACTIVITY_KIND_MEMORY_POOL: return "CUPTI_ACTIVITY_KIND_MEMORY_POOL";
    case CUPTI_ACTIVITY_KIND_GRAPH_TRACE: return "CUPTI_ACTIVITY_KIND_GRAPH_TRACE";
    case CUPTI_ACTIVITY_KIND_JIT: return "CUPTI_ACTIVITY_KIND_JIT";
    default: return "UNKNOWN";
  }
}

void CUPTIAPI buffer_requested_callback(uint8_t** buffer_ptr_ptr, size_t* size_ptr,
    size_t* max_num_records_ptr)
{
  // TODO: Reuse completed buffers
  *max_num_records_ptr = 0;
  *size_ptr = BUFF_SIZE;
  auto rc = posix_memalign(reinterpret_cast<void**>(buffer_ptr_ptr), ALIGN_BYTES, *size_ptr);
  if (rc != 0) {
    std::cerr << "FAILED TO ALLOCATE CUPTI BUFFER: " << strerror(rc) << std::endl;
    *buffer_ptr_ptr = 0;
    *size_ptr = 0;
  }
}

void CUPTIAPI buffer_completed_callback(CUcontext, uint32_t,
    uint8_t* buffer, size_t buffer_size, size_t valid_size)
{
  if (valid_size > 0) {
    // TODO: Process buffer on a background thread
    std::cerr << "GOT A BUFFER OF DATA FROM CUPTI, SIZE: " << valid_size << std::endl;
    CUpti_Activity* record_ptr = nullptr;
    auto rc = cuptiActivityGetNextRecord(buffer, valid_size, &record_ptr);
    while (rc == CUPTI_SUCCESS) {
      std::cerr << "RECORD: " << activity_kind_to_string(record_ptr->kind) << std::endl;
      switch (record_ptr->kind) {
        case CUPTI_ACTIVITY_KIND_DRIVER:
        {
          auto api_record = reinterpret_cast<CUpti_ActivityAPI const*>(record_ptr);
          char const* name = nullptr;
          cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, api_record->cbid, &name);
          name = name ? name : "NULL";
          std::cerr << "  NAME: " << name << " THREAD: " << api_record->threadId << std::endl;
          break;
        }
        case CUPTI_ACTIVITY_KIND_DEVICE:
        {
          auto device_record = reinterpret_cast<CUpti_ActivityDevice4 const*>(record_ptr);
          char const* name = device_record->name != nullptr ? device_record->name : "NULL";
          std::cerr << "  " << activity_kind_to_string(device_record->kind) << " " << name << std::endl;
          break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        {
          auto api_record = reinterpret_cast<CUpti_ActivityAPI const*>(record_ptr);
          char const* name = nullptr;
          cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api_record->cbid, &name);
          name = name ? name : "NULL";
          std::cerr << "  NAME: " << name << " THREAD: " << api_record->threadId << std::endl;
          break;
        }
        default:
          break;
      }
      rc = cuptiActivityGetNextRecord(buffer, valid_size, &record_ptr);
    }
  }
  free(buffer);
}

}

extern "C" {

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_Profiler_nativeInit(JNIEnv* env, jclass)
{
  try {
    State = new subscriber_state(env);
    auto rc = cuptiSubscribe(&State->subscriber_handle, callback_handler, nullptr);
    check_cupti(rc, "Error initializing CUPTI");
    rc = cuptiEnableCallback(1, State->subscriber_handle, CUPTI_CB_DOMAIN_RUNTIME_API,
        CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020);
    check_cupti(rc, "Error enabling device reset callback");
//    rc = cuptiEnableCallback(1, State->subscriber_handle, CUPTI_CB_DOMAIN_STATE,
//        CUPTI_CBID_STATE_FATAL_ERROR);
//    check_cupti(rc, "Error enabling fatal error callback");
//    rc = cuptiEnableCallback(1, State->subscriber_handle, CUPTI_CB_DOMAIN_STATE,
//        CUPTI_CBID_STATE_ERROR);
//    check_cupti(rc, "Error enabling error callback");
//    rc = cuptiEnableCallback(1, State->subscriber_handle, CUPTI_CB_DOMAIN_STATE,
//        CUPTI_CBID_STATE_WARNING);
//    check_cupti(rc, "Error enabling warning callback");
    rc = cuptiActivityRegisterCallbacks(buffer_requested_callback, buffer_completed_callback);
    check_cupti(rc, "Error registering activity buffer callbacks");

    check_cupti(cuptiEnableDomain(1, State->subscriber_handle, CUPTI_CB_DOMAIN_NVTX), "Error enabling NVTX domain");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE), "Error enabling device activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT), "Error enabling context activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER), "Error enabling driver activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME), "Error enabling runtime activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY), "Error enabling memcpy activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET), "Error enabling memset activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME), "Error enabling name activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER), "Error enabling marker activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "Error enabling concurrent kernel activity");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD), "Error enabling overhead activity");
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_Profiler_nativeShutdown(JNIEnv* env, jclass)
{
  try {
    if (State != nullptr) {
      auto unsub_rc = cuptiUnsubscribe(State->subscriber_handle);
      auto flush_rc = cuptiActivityFlushAll(1);
      delete State;
      State = nullptr;
      check_cupti(unsub_rc, "Error unsubscribing from CUPTI");
      check_cupti(flush_rc, "Error flushing CUPTI records");
    }
  }
  CATCH_STD(env, );
}

int InitializeInjectionNvtx2_fnptr(void* p)
{
  std::cerr << "INITIALIZING NVTX" << std::endl;
  CUptiResult rc = cuptiNvtxInitialize2(p);
  return (rc == CUPTI_SUCCESS) ? 1 : 0;
}

}
