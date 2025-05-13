#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <torch/csrc/autograd/accuracy_tool.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/dump/ivalue_dump.h>
#include <torch/csrc/autograd/profiler_legacy.h>

namespace torch {
namespace autograd {

static thread_local std::vector<std::string> call_stack;
static thread_local bool skip_cpu = false;
static thread_local at::CallbackHandle handle_tls = 0;
// TODO: ops in black list are which arguments are not supported to be dumped.
// more detail please refer to ivalue_dump.cpp:DIValue::setDIValue.
static std::vector<std::string> global_ops_black_lst = {"empty", "empty_like", "empty_strided", "as_strided", "randn_like",
                                                        "new_empty_strided", "ones", "ones_like", "zeros", "zeros_like"};

// RAII guard for cpu callfack and compare.
struct ToCPUGuard {
  ToCPUGuard() {
    skip_cpu = true;
  }

  ~ToCPUGuard() {
    skip_cpu = false;
  }
};

std::string getDir(){
  char *dir;
  TORCH_CHECK((dir = getcwd(nullptr,0)) != nullptr, "Dump dir not existed!");
  std::string dir_s(dir);
  free(dir);
  for (auto str:call_stack) {
    dir_s+= "/";
    dir_s+= str;
  }
  dir_s += "/";
  return dir_s;
}

inline bool checkNameInBlackLst(const std::string& str) {
  auto it = std::find(global_ops_black_lst.begin(), global_ops_black_lst.end(), str);

  if (it != global_ops_black_lst.end()) {
      return true;
  } else {
      return false;
  }
}

torch::jit::Stack call_cpu_fallback(const c10::OperatorHandle& op,const torch::jit::Stack* stack) {
  const auto& schema_returns = op.schema().returns();
  const auto& num_returns = schema_returns.size();
  const auto& args_size = stack->size();
  torch::jit::Stack stack_(*stack);
  at::native::cpu_fallback(op, &stack_);
  auto returns = torch::jit::last(stack_, num_returns);
  TORCH_CHECK(stack->size()==args_size, "Original stack has been changed!");
  return returns.vec();
}

torch::jit::Stack call_cpu_fallback(const c10::OperatorHandle& op, c10::ArrayRef<const c10::IValue>* stack) {
  const auto& schema_returns = op.schema().returns();
  const auto& num_returns = schema_returns.size();
  const auto& num_arguments = (*stack).size();
  torch::jit::Stack stack_;
  for (const auto idx: c10::irange(num_arguments)) {
    stack_.push_back((*stack)[idx]);
  }
  at::native::cpu_fallback(op, &stack_);
  auto returns = torch::jit::last(stack_, num_returns);
  TORCH_CHECK(stack->size()==num_arguments, "Original stack has been changed!");
  return returns.vec();
}

torch::jit::Stack copy_to_cpu(const torch::jit::Stack* stack) {
  torch::jit::Stack cpu_stack(*stack);
  for (const auto idx: c10::irange(stack->size())) {
    const auto& ivalue = (*stack)[idx];
    if (ivalue.isTensor() && ivalue.toTensor().defined()) {
      auto cpu_ivalue = c10::IValue(ivalue.toTensor().cpu());
      cpu_stack[idx] = std::move(cpu_ivalue);
    } else if (ivalue.isTensorList()) {
      auto cpu_ivalue = c10::IValue(c10::List<at::Tensor>(at::native::to_cpu(ivalue.toTensorList().vec())));
      cpu_stack[idx] = std::move(cpu_ivalue);
    }
  }
  return cpu_stack;
}

torch::jit::Stack copy_to_cpu(c10::ArrayRef<const c10::IValue>* stack) {
  const auto& num_arguments = (*stack).size();
  torch::jit::Stack cpu_stack;

  for (const auto idx: c10::irange(num_arguments)) {
    const auto& ivalue = (*stack)[idx];
    if (ivalue.isTensor() && ivalue.toTensor().defined()) {
      auto cpu_ivalue = c10::IValue(ivalue.toTensor().cpu());
      cpu_stack.push_back(cpu_ivalue);
    } 
    else if (ivalue.isTensorList()) {
      auto cpu_ivalue = c10::IValue(c10::List<at::Tensor>(at::native::to_cpu(ivalue.toTensorList().vec())));
      cpu_stack.push_back(cpu_ivalue);
    } 
    else {
      cpu_stack.push_back(*(const_cast<c10::IValue*>(&ivalue)));
    }
  }
  return cpu_stack;
}

c10::DeviceType extract_device(torch::jit::Stack* stack) {
  c10::DeviceType device = at::kCPU;
  for (const auto& ivalue : *stack) {
    if (ivalue.isTensor()) return ivalue.toTensor().device().type();
    if (ivalue.isList()) {
      auto sub_stack = ivalue.toList().vec();
      return extract_device(&sub_stack);
    }
    if (ivalue.isTensorList()) {
      return ivalue.toTensorList().vec()[0].device().type();
    }
  }
  return device;
}

c10::DeviceType extract_device(c10::ArrayRef<const c10::IValue>* stack) {
  c10::DeviceType device = at::kCPU;
  for (const auto& ivalue : *stack) {
    if (ivalue.isTensor()) return ivalue.toTensor().device().type();
    if (ivalue.isList()) {
      auto sub_stack = ivalue.toList().vec();
      return extract_device(&sub_stack);
    }
    if (ivalue.isTensorList()) {
      return ivalue.toTensorList().vec()[0].device().type();
    }
  }
  return device;
}

// push op into call_stack and create directory in the call_stack.
bool push_and_mkdir(const char* op_name_char) {
  std::string dir_s = getDir();
  std::string op_name(op_name_char);
  int idx = 0;
  dir_s += op_name;
  std::string dir_full = dir_s + "_" + std::to_string(idx);
  while(true) {
    if (access(dir_full.c_str(), 0) == 0) {
      idx++;
      dir_full = dir_s + "_" + std::to_string(idx);
    } else {
      break;
    }
  }
  op_name = op_name + "_" + std::to_string(idx);
  call_stack.emplace_back(op_name);
  int ret = mkdir(dir_full.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
  if (ret == 0) {
    return true;
  } else {
    return false;
  }
}

torch::jit::Stack remove_undef(const torch::jit::Stack* stack) {
  torch::jit::Stack dump_stack;
  for (const auto& ivalue : *stack) {
    if ((!ivalue.isNone()) && (!ivalue.isTensor() || ivalue.toTensor().defined())) {
      dump_stack.emplace_back(ivalue);
    }
  }
  return dump_stack;
}

void dump_json(torch::jit::Stack* stack, const std::string& file_name) {
  std::string dir_s = getDir();
  dir_s += file_name;
  const auto dump_stack = remove_undef(stack);
  at::dump::IvalueVecToJson(dump_stack, dir_s);
}

void dump_bin(torch::jit::Stack* stack, const std::string& dir) {
  std::string dir_s = getDir();
  dir_s += dir;

  int ret = mkdir(dir_s.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
  TORCH_CHECK(ret==0, "mkdir error!");
  const auto dump_stack = remove_undef(stack);
  at::dump::IvalueVecToBin(dump_stack, dir_s);
}

void dump_err(const std::string& err) {
  std::string dir_s = getDir();
  dir_s += "/err_info.txt";
  std::ofstream f;
  f.open(dir_s, std::ios::out | std::ios::app);
  TORCH_CHECK(f.is_open(), dir_s, " open or create failed!");
  f << err << std::endl; 
  f.close();
}

bool check_nan(const torch::jit::Stack* stack) {
  bool has_nan = false;
  for (const auto idx: c10::irange(stack->size())) {
    const auto& ivalue = (*stack)[idx];
    if (ivalue.isTensor() && ivalue.toTensor().defined()) {
      has_nan = ivalue.toTensor().isnan().any().item<bool>();
      if (has_nan) return has_nan;
    } else if (ivalue.isTensorList()) {
      auto tensor_vec = ivalue.toTensorList().vec();
      for (const auto& tensor : tensor_vec) {
        if (tensor.defined()) {
          has_nan = tensor.isnan().any().item<bool>();
          if (has_nan) return has_nan;
        }
      }
    }
  }
  return has_nan;
}


at::CallbackHandle getTLSHandle() {
  return handle_tls;
}

// push accuracy callbacks into CallbackManager when enableProfiler.
void pushAccuracyCallbacks(const std::unordered_set<at::RecordScope>& scopes) {
  auto handle = at::addThreadLocalCallback(at::RecordFunctionCallback(
    // callback for start of op.
    [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
      bool is_op = fn.operator_name().has_value(); // e.g. XXXBackward is not op.
      if (!is_op) return nullptr;
      auto op_name = fn.operator_name()->name;
      op_name = op_name.substr(6,op_name.length());
      const char* record_op = getenv("PYTORCH_ACC_CHECK_OP"); 
      if ((record_op != nullptr) && (strcmp(record_op, op_name.c_str()) == 0 
            || strcmp(record_op, "all") == 0) && !checkNameInBlackLst(op_name) && !skip_cpu) {
        // init base dir in call stack.
        if (call_stack.empty()) {
          std::string base_dir = "op_logs_" + std::to_string(torch::autograd::profiler::getTime());
          TORCH_CHECK(push_and_mkdir(base_dir.c_str()), "Accuracy tool cannot find correct directory!");
        }
        TORCH_CHECK(push_and_mkdir(op_name.c_str()), "Accuracy tool cannot find correct directory!");
      }
      return nullptr;
    },
    // callback for end of op.
    [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
      bool is_op = fn.operator_name().has_value();
      if (!is_op) return;
      auto op_name = fn.operator_name();
      auto op_name_str = op_name->name;
      auto op_name_substr = op_name_str.substr(6,op_name_str.length());
      char *record_op = getenv("PYTORCH_ACC_CHECK_OP"); 
      if ((record_op != nullptr) && (strcmp(record_op, op_name_substr.c_str()) == 0 
            || strcmp(record_op, "all") == 0) && !checkNameInBlackLst(op_name_substr) && !skip_cpu) {
        auto op = c10::Dispatcher::singleton()
          .findSchemaOrThrow(op_name_str.c_str(), op_name->overload_name.c_str());
        auto inputs = fn.inputsVec();
        auto outputs = fn.outputs();
        auto device = extract_device(&inputs);

        // check if the kernel has CPU dispatch.
        auto has_cpu = op.checkValidKernel(c10::DispatchKey::CPU);

        // get acc check level.
        static const int level = ([]()->int{
          const char* env = getenv("PYTORCH_ACC_CHECK_LEVEL");
          if(env) return atoi(env);
          else return 0; // default level 0.
        })();

        // initialize CPU guard.
        ToCPUGuard guard;
        torch::jit::Stack inputs_;
        torch::jit::Stack outputs_;

        // dump operator arguments if level > 0
        if (level > 0) {
          inputs_ = copy_to_cpu(&inputs);
          outputs_ = copy_to_cpu(&outputs);
          char * dumJson = std::getenv("DUMP_JSON");
          if(dumJson){
            dump_json(&inputs_, "inputs.json");
            dump_json(&outputs_, "outputs.json");
          }else{
            dump_bin(&inputs_, "inputs");
            dump_bin(&outputs_, "outputs");
          }
          if (!has_cpu) {
            std::string warn_info = "Operator has not CPU dispatch!";
            dump_err(warn_info);
            TORCH_WARN(op_name_str, ": ", warn_info);
          }
        }

        // check if output has nan value.
        bool has_nan = check_nan(&outputs);
        if (has_nan) {
          if (level == 0) {
            inputs_ = copy_to_cpu(&inputs);
            outputs_ = copy_to_cpu(&outputs);
            char * dumJson = std::getenv("DUMP_JSON");
            if(dumJson){
              dump_json(&inputs_, "inputs.json");
              dump_json(&outputs_, "outputs.json");
            }else{
              dump_bin(&inputs_, "inputs");
              dump_bin(&outputs_, "outputs");
            }
          }
          std::string err_info = "Operator has nan outputs!";
          dump_err(err_info);
          TORCH_WARN(op_name_str, ": ", err_info);
        }

        // comparing with CPU ops if level > 1.
        if ( level > 1 && has_cpu && inputs.size() && device == at::kCUDA) {
          auto cpu_outputs = call_cpu_fallback(op, &inputs_);
          auto err_str = at::dump::CompareAcc(outputs_, cpu_outputs);
          if (err_str != "true") {
            dump_err(err_str);
            // N.B. TORCH_CHECK will exit current thread, so pop the stack before check.
            call_stack.pop_back();
            TORCH_WARN(op_name_str," in CUDA is not equal to CPU result! ", err_str);
          }
        }
        call_stack.pop_back();
      }
    }
  )
  .needsInputs(true)
  .needsOutputs(true)
  .scopes(scopes)
  );
  handle_tls = handle;
}

}} // namespace torch::autograd
