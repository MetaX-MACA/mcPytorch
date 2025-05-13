#pragma once

#include <ATen/dump/json.h>
#include <ATen/dump/binary.h>
#include <ATen/core/ivalue.h>

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <bitset>
using namespace std;


namespace at{
namespace dump{
namespace base{


enum class C10_API_ENUM DKind : uint8_t {
    INT,
    DOUBLE,
    BOOL,
    INT_LIST,
    DOUBLE_LIST,
    BOOL_LIST,
    STRING,
    STORAGE,
    TENSOR,
    TENSOR_LIST,
    TUPLE,
    GENERIC_DICT,
    GENERIC_LIST,
    BLOB,
    CAPSULE,
    CUSTOMCLASS,
    COMPLEX_DOUBLE,
    FUTURE,
    RREF,
    QUANTIZER,
    COMPLEX_DOUBLE_LIST,
    OBJECT,
    MODULE,
    PYOBJECT,
    ENUM,
    NONE,
    DEVICE,
    STREAM,
    GENERATOR
};


namespace DataTrans {
    static union{
        uint32_t i;
        float f;
        uint64_t ll;
        double d;
    } u;
     
    static string obj2str(double d){
        u.d = d;
        std::bitset<64> bt(u.ll);
        return bt.to_string();
    }
    static string obj2str(float f){
        u.f = f;
        std::bitset<32> bt(u.i);
        return bt.to_string();
    }
    static string obj2str(int64_t i){
        return to_string(i);
    }
    static string obj2str(int32_t i){
        return to_string(i);
    }
    static string obj2str(bool b){
        return to_string(static_cast<int64_t>(b));
    }

    static string obj2str(uint8_t b){
        return to_string(static_cast<int64_t>(b));
    }

    static string obj2str(c10::ScalarType b){
        return to_string(static_cast<int64_t>(b));
    }
    
    static string obj2str(DKind b){
        return to_string(static_cast<int64_t>(b));
    }

    // 64 bit string to double
    static double bit2d(string bit){
        std::bitset<64> bt(bit);
        u.ll = bt.to_ullong();
        return u.d;
    }
    //32 bit string to float
    static float bit2f(string bit){
        std::bitset<32> bt(bit);
        u.i = bt.to_ulong();
        return u.f;
    }
    static int64_t toInt(string s){
        return atoi(s.c_str());
    }
    static bool toBool(string s){
        return bool(toInt(s));
    }
    static DKind toDKind(string s){
        return DKind(toInt(s));
    }
    static ScalarType toScalarType(string s){
        return ScalarType(toInt(s));
    }

    //to decimal string
    template<typename T, class = typename std::enable_if<std::is_floating_point<T>::value>::type>
    static string toDecStr(T v){
        std::stringstream ss;
        ss << setprecision(20) << v;
        return ss.str();
    }

}

}  //namespace base


using namespace base;


struct TORCH_API DStorage{
    void setDStorage(const at::Storage& s);

    Json::Value setJson();
    void setBin(const string& dir);

    void json2DStorage(const Json::Value& jsonStorage);
    void bin2DStorage(const string& dir);

    at::Storage loadStorage();

    int64_t size_;
    uint8_t* data_;
};

struct TORCH_API DTensor{
    void setDTensor(const at::Tensor& t);

    Json::Value setJson();
    void setBin(const string& dir);

    void json2DTensor(const Json::Value& jsonTensor);
    void bin2DTensor(const string& dir);

    at::Tensor loadTensor();
    
    ScalarType srcT_;
    int64_t dim_;
    int64_t offset_;
    bool requires_grad_;
    
    std::vector<int64_t> shape_;
    std::vector<int64_t> stride_;
    DStorage DStorage_;
};


struct TORCH_API DIValue final{
    void setDIValue(const c10::IValue& ivalue);

    Json::Value setJson();
    void setBin(const string& dir);

    void json2DIValue(const Json::Value& jsonIvalue);
    void bin2DIValue(const string& dir);

    c10::IValue loadIValue();

    DKind DKind_;
    int64_t dim_ = 0;
    
    int64_t i_;
    double d_;
    bool b_;
    std::string str_;
    DStorage DStorage_;
    DTensor DTensor_;

    c10::List<int64_t> ilist_;
    c10::List<double> dlist_;
    c10::List<bool> blist_;
    std::vector<DTensor> DTensorList_;

    std::vector<DIValue> DIValueList_;
};


TORCH_API void IvalueVecToJson(const std::vector<c10::IValue>& IData_, const string& SavePath);
TORCH_API std::vector<c10::IValue>  jsonToIValueVec(const string& SavePath_);

TORCH_API void IvalueVecToBin(const std::vector<c10::IValue>& IData_, const string& SaveDir);
TORCH_API std::vector<c10::IValue> binToIValueVec(const string& SaveDir);

TORCH_API string CompareAcc(const std::vector<c10::IValue>& cpu_, const std::vector<c10::IValue>& gpu_);


}  //namespace at
}  //namespace dump