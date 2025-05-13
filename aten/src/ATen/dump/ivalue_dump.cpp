#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/dump/ivalue_dump.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorCompare.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/allclose.h>
#endif

#include <utility>
#include <cstdlib>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

namespace at{

namespace dump{

#define PITCH_DKIND(kd1, kd2)  \
    DIValueList_[0].DKind_ == kd1 && DIValueList_[1].DKind_ == kd2


#define DICT_TO_IVLALUE(type1, type2, k, v)  \
    c10::Dict<type1, type2> dic;                                      \
    for(int index = 0; index < DIValueList_.size() - 1; index += 2){  \
        type1 key = DIValueList_[index].k;                         \
        type2 value = DIValueList_[index + 1].v;                    \
        dic.insert(key, value);                                     \
    }                                                               \
    c10::IValue res(dic);                                              \
    return res;                                                  \


namespace base{namespace DataTrans{
    union u;
}}

void DStorage::setDStorage(const at::Storage& s){
    size_ = static_cast<int64_t>(s.nbytes());
    data_ = s.data<uint8_t>();
}


Json::Value DStorage::setJson(){
    Json::Value jsonStorage;
    jsonStorage["size_"] = size_;

    for(int i = 0; i < size_; i++){
        jsonStorage["data_"].append(DataTrans::obj2str(data_[i]));
    }
    return jsonStorage;
}

void DStorage::setBin(const string& dir){
    const string size_file = dir + "/size_";
    const string data_file = dir + "/data_";
    DumpBinaryData<int64_t>(size_file.c_str(), &size_, 1);
    DumpBinaryData<uint8_t>(data_file.c_str(), data_, size_);
}

void DStorage::json2DStorage(const Json::Value& jsonStorage){
    size_ = jsonStorage["size_"].asInt();
    
    data_ = new uint8_t[size_];
    for(int i = 0; i < size_; i++){
        data_[i] = static_cast<uint8_t>(DataTrans::toInt(jsonStorage["data_"][i].asString()));
    }
}

void DStorage::bin2DStorage(const string& dir){
    const string size_file = dir + "/size_";
    const string data_file = dir + "/data_";

    ReadBufferFromBinaryFile<int64_t>(size_file.c_str(), &size_, 1);
    data_ = new uint8_t[size_];
    ReadBufferFromBinaryFile<uint8_t>(data_file.c_str(), data_, size_);
}

at::Storage DStorage::loadStorage(){
    at::Storage st(Storage::use_byte_size_t(),
                   size_,
                   at::DataPtr(data_, DeviceType::CPU));
    return st;
}


void DTensor::setDTensor(const at::Tensor& t){
    TORCH_CHECK(t.defined(), "DTensor can only dump defined tensor!");
    srcT_ = t.scalar_type();
    dim_ = t.dim();
    offset_ = t.storage_offset();
    requires_grad_ = t.requires_grad();

    shape_ = t.sizes().vec();
    stride_ = t.strides().vec();
    DStorage_.setDStorage(t.storage());   
}


Json::Value DTensor::setJson(){
    Json::Value jsonTensor;
    jsonTensor["srcT_"] = DataTrans::obj2str(srcT_);
    jsonTensor["dim_"] = DataTrans::obj2str(dim_);
    jsonTensor["offset_"] = DataTrans::obj2str(offset_);
    jsonTensor["requires_grad_"] = DataTrans::obj2str(requires_grad_);
    
    for(int i = 0; i < dim_; i++){
        jsonTensor["shape_"].append(shape_[i]);
    }
    for(int i = 0; i < dim_; i++){
        jsonTensor["stride_"].append(stride_[i]);
    }
    jsonTensor["DStorage_"] = DStorage_.setJson();
    return jsonTensor;
}

void DTensor::setBin(const string& dir){
    string srcT_file = dir + "/srcT_";
    string dim_file = dir + "/dim_";
    string offset_file = dir + "/offset_";
    string requires_grad_file = dir + "/requires_grad_";
    string shape_file = dir + "/shape_";
    string stride_file = dir + "/stride_";
    string DStorage_dir = dir + "/DStorage_";
    mkdir(DStorage_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

    DumpBinaryData<c10::ScalarType>(srcT_file.c_str(), &srcT_, 1);
    DumpBinaryData<int64_t>(dim_file.c_str(), &dim_, 1);
    DumpBinaryData<int64_t>(offset_file.c_str(), &offset_, 1);
    DumpBinaryData<bool>(requires_grad_file.c_str(), &requires_grad_, 1);

    int64_t* shape_ptr = new int64_t[dim_];
    int64_t* stride_ptr = new int64_t[dim_];
    for(int i = 0; i < dim_; i++){
        shape_ptr[i] = shape_[i];
        stride_ptr[i] = stride_[i];
    }
    DumpBinaryData<int64_t>(shape_file.c_str(), shape_ptr, dim_);
    DumpBinaryData<int64_t>(stride_file.c_str(), stride_ptr, dim_);

    DStorage_.setBin(DStorage_dir);

    delete []shape_ptr;
    delete []stride_ptr;
}

void DTensor::json2DTensor(const Json::Value& jsonTensor){
    srcT_ = DataTrans::toScalarType(jsonTensor["srcT_"].asString());
    dim_ = DataTrans::toInt(jsonTensor["dim_"].asString());
    offset_ = DataTrans::toInt(jsonTensor["offset_"].asString());
    requires_grad_ = DataTrans::toBool(jsonTensor["requires_grad_"].asString());

    shape_.resize(jsonTensor["shape_"].size());
    for(int i = 0; i < shape_.size(); i++){
        shape_[i] = jsonTensor["shape_"][i].asInt();
    }
    
    stride_.resize(jsonTensor["stride_"].size());
    for(int i = 0; i < stride_.size(); i++){
        stride_[i] = jsonTensor["stride_"][i].asInt();
    }
    DStorage_.json2DStorage(jsonTensor["DStorage_"]);
}


void DTensor::bin2DTensor(const string& dir){
    string srcT_file = dir + "/srcT_";
    string dim_file = dir + "/dim_";
    string offset_file = dir + "/offset_";
    string requires_grad_file = dir + "/requires_grad_";
    string shape_file = dir + "/shape_";
    string stride_file = dir + "/stride_";
    string DStorage_dir = dir + "/DStorage_";

    ReadBufferFromBinaryFile<c10::ScalarType>(srcT_file.c_str(), &srcT_, 1);
    ReadBufferFromBinaryFile<int64_t>(dim_file.c_str(), &dim_, 1);
    ReadBufferFromBinaryFile<int64_t>(offset_file.c_str(), &offset_, 1);
    ReadBufferFromBinaryFile<bool>(requires_grad_file.c_str(), &requires_grad_, 1);

    int64_t* shape_ptr = new int64_t[dim_];
    int64_t* stride_ptr = new int64_t[dim_];
    ReadBufferFromBinaryFile<int64_t>(shape_file.c_str(), shape_ptr, dim_);
    ReadBufferFromBinaryFile<int64_t>(stride_file.c_str(), stride_ptr, dim_);
    shape_.resize(dim_);
    stride_.resize(dim_);
    for(int i = 0; i < dim_; i++){
        shape_[i] = shape_ptr[i];
        stride_[i] = stride_ptr[i];
    }

    DStorage_.bin2DStorage(DStorage_dir);
    
    delete []shape_ptr;
    delete []stride_ptr;
}


at::Tensor DTensor::loadTensor(){
    at::Tensor t = at::empty({}, srcT_);

    t.set_(DStorage_.loadStorage(), offset_, shape_, stride_);
    t.set_requires_grad(requires_grad_);
    
    return t;
}


void DIValue::setDIValue(const c10::IValue& ivalue){
    if(ivalue.isInt()){
        DKind_ = DKind::INT;
        i_ = ivalue.toInt();
    }else if(ivalue.isDouble()){
        DKind_ = DKind::DOUBLE;
        d_ = ivalue.toDouble();
    }else if(ivalue.isBool()){
        DKind_ = DKind::BOOL;
        b_ = ivalue.toBool();
    }else if(ivalue.isString()){
        DKind_ = DKind::STRING;
        str_ = ivalue.toStringRef();
    }else if(ivalue.isStorage()){
        DKind_ = DKind::STORAGE;
        DStorage_.setDStorage(ivalue.toStorage());
    }else if(ivalue.isTensor()){
        DKind_ = DKind::TENSOR;
        DTensor_.setDTensor(ivalue.toTensor());
    }else if(ivalue.isList()){
        if(ivalue.isIntList()){
            DKind_ = DKind::INT_LIST;
            ilist_ = ivalue.toIntList();
            dim_ = ilist_.size();
        }else if(ivalue.isDoubleList()){
            DKind_ = DKind::DOUBLE_LIST;
            dlist_ = ivalue.toDoubleList();
            dim_ = dlist_.size();
        }else if(ivalue.isBoolList()){
            DKind_ = DKind::BOOL_LIST;
            blist_ = ivalue.toBoolList();
            dim_ = blist_.size();
        }else if(ivalue.isTensorList()){
            DKind_ = DKind::TENSOR_LIST;
            std::vector<at::Tensor> TensorList = ivalue.toTensorVector();
            DTensorList_.resize(TensorList.size());
            for(int i = 0; i < TensorList.size(); i++){
                DTensorList_[i].setDTensor(TensorList[i]);
            }
            dim_ = DTensorList_.size();
        }else{
            AT_ERROR("unkonw generice list");
        }
    }else if(ivalue.isTuple()){
        DKind_ = DKind::TUPLE;
        const std::vector<c10::IValue>& IValueList = ivalue.toTuple()->elements();
        DIValueList_.resize(IValueList.size());
        for(int i = 0; i < IValueList.size(); i++){
            DIValueList_[i].setDIValue(IValueList[i]);
        }
        dim_ = DIValueList_.size();
    }else if(ivalue.isGenericDict()){
        DKind_ = DKind::GENERIC_DICT;
        const c10::Dict<c10::IValue, c10::IValue>& IValueDict = ivalue.toGenericDict();
        DIValueList_.resize(IValueDict.size() * 2);
        int index = 0;
        for(auto it = IValueDict.begin(); it != IValueDict.end(); it++){
            c10::IValue key = it->key();
            c10::IValue value = it->value();
            DIValueList_[index].setDIValue(key);
            DIValueList_[index + 1].setDIValue(value);
            index += 2;
        }
        dim_ = DIValueList_.size();
    }else if(ivalue.isBlob()){
        AT_ERROR("not suport dump ivalue Blob %d\n");
    }else if(ivalue.isCapsule()){
        AT_ERROR("not suport dump ivalue Capsule %d\n");
    }else if(ivalue.isCustomClass()){
        AT_ERROR("not suport dump ivalue CustomClass %d\n");
    }else if(ivalue.isComplexDouble()){
        AT_ERROR("not suport dump ivalue ComplexDouble %d\n");
    }else if(ivalue.isFuture()){
        AT_ERROR("not suport dump ivalue Future %d\n");
    }else if(ivalue.isRRef()){
        AT_ERROR("not suport dump ivalue RRef %d\n");
    }else if(ivalue.isQuantizer()){
        AT_ERROR("not suport dump ivalue Quantizer %d\n");
    }else if(ivalue.isComplexDoubleList()){
        AT_ERROR("not suport dump ivalue ComplexDoubleList %d\n");
    }else if(ivalue.isObject()){
        AT_ERROR("not suport dump ivalue Object %d\n");
    }else if(ivalue.isModule()){
        AT_ERROR("not suport dump ivalue Module %d\n");
    }else if(ivalue.isPyObject()){
        AT_ERROR("not suport dump ivalue PyObject %d\n");
    }else if(ivalue.isEnum()){
        AT_ERROR("not suport dump ivalue Enum %d\n");
    }else if(ivalue.isNone()){
        AT_ERROR("not suport dump ivalue None %d\n");
    }else if(ivalue.isDevice()){
        AT_ERROR("not suport dump ivalue Device %d\n");
    }else if(ivalue.isStream()){
        AT_ERROR("not suport dump ivalue Stream %d\n");
    }else if(ivalue.isGenerator()){
        AT_ERROR("not suport dump ivalue Generator %d\n");
    }else{
        AT_ERROR("not suport dump ivalue unknown type %d\n");
    }

}


Json::Value DIValue::setJson(){
    Json::Value jsonIvalue;
    jsonIvalue["DKind_"] = DataTrans::obj2str(DKind_);
    jsonIvalue["dim_"] = DataTrans::obj2str(dim_);

    if(DKind_ == DKind::INT){
        jsonIvalue["i_"] = DataTrans::obj2str(i_);
    }else if(DKind_ == DKind::DOUBLE){
        jsonIvalue["d_"] = DataTrans::obj2str(d_);
    }else if(DKind_ == DKind::BOOL){
        jsonIvalue["b_"] = DataTrans::obj2str(b_);
    }else if(DKind_ == DKind::INT_LIST){
        for(int i = 0; i < ilist_.size(); i++){
            jsonIvalue["ilist_"].append(DataTrans::obj2str(ilist_[i]));
        }
    }else if(DKind_ == DKind::DOUBLE_LIST){
        for(int i = 0; i < dlist_.size(); i++){
            jsonIvalue["dlist_"].append(DataTrans::obj2str(dlist_[i]));
        }
    }else if(DKind_ == DKind::BOOL_LIST){
        for(int i = 0; i < blist_.size(); i++){
            jsonIvalue["blist_"].append(DataTrans::obj2str(blist_[i]));
        }
    }else if(DKind_ == DKind::STRING){
        jsonIvalue["str_"] = str_;
    }else if(DKind_ == DKind::STORAGE){
        jsonIvalue["DStorage_"] = DStorage_.setJson();
    }else if(DKind_ == DKind::TENSOR){
        jsonIvalue["DTensor_"] = DTensor_.setJson();
    }else if(DKind_ == DKind::TENSOR_LIST){
        for(int i = 0; i < DTensorList_.size(); i++){
            jsonIvalue["DTensorList_"].append(DTensorList_[i].setJson());
        }
    }else if(DKind_ == DKind::TUPLE || DKind_ == DKind::GENERIC_DICT){
        for(int i = 0; i < DIValueList_.size(); i++){
            jsonIvalue["DIValueList_"].append(DIValueList_[i].setJson());
        }
    }

    return jsonIvalue;
}


void DIValue::setBin(const string& dir){
    string DKind_file = dir + "/DKind_";
    string dim_file = dir + "/dim_";
    DumpBinaryData<DKind>(DKind_file.c_str(), &DKind_, 1);
    DumpBinaryData<int64_t>(dim_file.c_str(), &dim_, 1);

    if(DKind_ == DKind::INT){
        string i_file = dir + "/i_";
        DumpBinaryData<int64_t>(i_file.c_str(), &i_, 1);
    }else if(DKind_ == DKind::DOUBLE){
        string d_file = dir + "/d_";
        DumpBinaryData<double>(d_file.c_str(), &d_, 1);
    }else if(DKind_ == DKind::BOOL){
        string b_file = dir + "/b_";
        DumpBinaryData<bool>(b_file.c_str(), &b_, 1);
    }else if(DKind_ == DKind::INT_LIST){
        int64_t* ilist_ptr = new int64_t[dim_];
        for(int i = 0; i < dim_; i++){
            ilist_ptr[i] = ilist_[i];
        }
        string ilist_file = dir + "/ilist_";
        DumpBinaryData<int64_t>(ilist_file.c_str(), ilist_ptr, dim_);
        delete[] ilist_ptr;
    }else if(DKind_ == DKind::DOUBLE_LIST){
        double* dlist_ptr = new double[dim_];
        for(int i = 0; i < dim_; i++){
            dlist_ptr[i] = dlist_[i];
        }
        string dlist_file = dir + "/dlist_";
        DumpBinaryData<double>(dlist_file.c_str(), dlist_ptr, dim_);

        delete[] dlist_ptr;
    }else if(DKind_ == DKind::BOOL_LIST){
        bool* blist_ptr = new bool[dim_];
        for(int i = 0; i < dim_; i++){
            blist_ptr[i] = blist_[i];
        }
        string blist_file = dir + "/blist_";
        DumpBinaryData<bool>(blist_file.c_str(), blist_ptr, dim_);

        delete[] blist_ptr;
    }else if(DKind_ == DKind::STRING){
        string str_file = dir + "/str_";
        DumpBinaryData<char>(str_file.c_str(), str_.c_str(), (int64_t)str_.length());
    }else if(DKind_ == DKind::STORAGE){
        string DStorage_dir = dir + "/DStorage_";
        mkdir(DStorage_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
        DStorage_.setBin(DStorage_dir);
    }else if(DKind_ == DKind::TENSOR){
        string DTensor_dir = dir + "/DTensor_";
        mkdir(DTensor_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
        DTensor_.setBin(DTensor_dir);
    }else if(DKind_ == DKind::TENSOR_LIST){
        for(int i = 0; i < dim_; i++){
            string DTensorList_dir = dir + "/DTensorList_" + to_string(i);
            mkdir(DTensorList_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
            DTensorList_[i].setBin(DTensorList_dir);
        }
    }else if(DKind_ == DKind::TUPLE || DKind_ == DKind::GENERIC_DICT){
        for(int i = 0; i < dim_; i++){
            string DIValueList_dir = dir + "/DIValueList_" + to_string(i);
            mkdir(DIValueList_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
            DIValueList_[i].setBin(DIValueList_dir);
        }
    }
}


void DIValue::json2DIValue(const Json::Value& jsonIvalue){
    DKind_ = DataTrans::toDKind(jsonIvalue["DKind_"].asString());
    dim_ = DataTrans::toInt(jsonIvalue["dim_"].asString());

    if(DKind_ == DKind::INT){
        i_ = DataTrans::toInt(jsonIvalue["i_"].asString());
    }else if(DKind_ == DKind::DOUBLE){
        d_ = DataTrans::bit2d(jsonIvalue["d_"].asString());
    }else if(DKind_ == DKind::BOOL){
        b_ = DataTrans::toBool(jsonIvalue["b_"].asString());
    }else if(DKind_ == DKind::INT_LIST){
        ilist_.resize(dim_);
        for(int i = 0; i < dim_; i++){
            ilist_[i] = DataTrans::toInt(jsonIvalue["ilist_"][i].asString());
        }
    }else if(DKind_ == DKind::DOUBLE_LIST){
        dlist_.resize(dim_);
        for(int i = 0; i < dim_; i++){
            dlist_[i] = DataTrans::bit2d(jsonIvalue["dlist_"][i].asString());
        }
    }else if(DKind_ == DKind::BOOL_LIST){
        blist_.resize(dim_);
        for(int i = 0; i < dim_; i++){
            blist_[i] = DataTrans::toBool(jsonIvalue["blist_"][i].asString());
        }
    }else if(DKind_ == DKind::STRING){
        str_ = jsonIvalue["str_"].asString();
    }else if(DKind_ == DKind::STORAGE){
         DStorage_.json2DStorage(jsonIvalue["DStorage_"]);
    }else if(DKind_ == DKind::TENSOR){
        DTensor_.json2DTensor(jsonIvalue["DTensor_"]);
    }else if(DKind_ == DKind::TENSOR_LIST){
        DTensorList_.resize(dim_);
        for(int i = 0; i < dim_; i++){
            DTensorList_[i].json2DTensor(jsonIvalue["DTensorList_"][i]);
        }
    }else if(DKind_ == DKind::TUPLE || DKind_ == DKind::GENERIC_DICT){
        DIValueList_.resize(dim_);
        for(int i = 0; i < dim_; i++){
            DIValueList_[i].json2DIValue(jsonIvalue["DIValueList_"][i]);
        }
    }

}


void DIValue::bin2DIValue(const string& dir){
    string DKind_file = dir + "/DKind_";
    string dim_file = dir + "/dim_";
    ReadBufferFromBinaryFile<DKind>(DKind_file.c_str(), &DKind_, 1);
    ReadBufferFromBinaryFile<int64_t>(dim_file.c_str(), &dim_, 1);


    if(DKind_ == DKind::INT){
        std::string i_file = dir + "/i_";
        ReadBufferFromBinaryFile<int64_t>(i_file.c_str(), &i_, 1);
    }else if(DKind_ == DKind::DOUBLE){
        std::string d_file = dir + "/d_";
        ReadBufferFromBinaryFile<double>(d_file.c_str(), &d_, 1);
    }else if(DKind_ == DKind::BOOL){
        std::string b_file = dir + "/b_";
        ReadBufferFromBinaryFile<bool>(b_file.c_str(), &b_, 1);
    }else if(DKind_ == DKind::INT_LIST){
        int64_t* ilist_ptr = new int64_t[dim_];
        string ilist_file = dir + "/ilist_";
        ReadBufferFromBinaryFile<int64_t>(ilist_file.c_str(), ilist_ptr, dim_);
        ilist_.resize(dim_);
        for(int i = 0; i < dim_; i++){
            ilist_[i] = ilist_ptr[i];
        }

        delete[] ilist_ptr;
    }else if(DKind_ == DKind::DOUBLE_LIST){
        double* dlist_ptr = new double[dim_];
        string dlist_file = dir + "/dlist_";
        ReadBufferFromBinaryFile<double>(dlist_file.c_str(), dlist_ptr, dim_);
        dlist_.resize(dim_);
        for(int i = 0; i < dim_; i++){
            dlist_[i] = dlist_ptr[i];
        }

        delete[] dlist_ptr;
    }else if(DKind_ == DKind::BOOL_LIST){
        bool* blist_ptr = new bool[dim_];
        string blist_file = dir + "/blist_";
        ReadBufferFromBinaryFile<bool>(blist_file.c_str(), blist_ptr, dim_);
        blist_.resize(dim_);
        for(int i = 0; i < dim_; i++){
            blist_[i] = blist_ptr[i];
        }

        delete[] blist_ptr;
    }else if(DKind_ == DKind::STRING){
        char * str_ptr = new char[dim_];
        string str_file = dir + "/str_";
        ReadBufferFromBinaryFile<char>(str_file.c_str(), str_ptr, dim_);
        str_ = str_ptr;
    }else if(DKind_ == DKind::STORAGE){
        string DStorage_dir = dir + "/DStorage_";
        DStorage_.bin2DStorage(DStorage_dir);
    }else if(DKind_ == DKind::TENSOR){
        string DTensor_dir = dir + "/DTensor_";
        DTensor_.bin2DTensor(DTensor_dir);
    }else if(DKind_ == DKind::TENSOR_LIST){
        DTensorList_.resize(dim_);
        for(int i = 0; i < dim_; i++){
            string DTensorList_dir = dir + "/DTensorList_" + to_string(i);
            DTensorList_[i].bin2DTensor(DTensorList_dir);
        }
    }else if(DKind_ == DKind::TUPLE || DKind_ == DKind::GENERIC_DICT){
        DIValueList_.resize(dim_);
        for(int i = 0; i < dim_; i++){
            string DIValueList_dir = dir + "/DIValueList_" + to_string(i);
            DIValueList_[i].bin2DIValue(DIValueList_dir);
        }
    }
}


c10::IValue DIValue::loadIValue(){
    
    if(DKind_ == DKind::INT){
        c10::IValue res(i_);
        return res;
    }else if(DKind_ == DKind::DOUBLE){
        c10::IValue res(d_);
        return res;
    }else if(DKind_ == DKind::BOOL){
        c10::IValue res(b_);
        return res;
    }else if(DKind_ == DKind::INT_LIST){
        c10::IValue res(ilist_);
        return res;
    }else if(DKind_ == DKind::DOUBLE_LIST){
        c10::IValue res(dlist_);
        return res;
    }else if(DKind_ == DKind::BOOL_LIST){
        c10::IValue res(blist_);
        return res;
    }else if(DKind_ == DKind::STRING){
        c10::IValue res(str_);
        return res;
    }else if(DKind_ == DKind::STORAGE){
        at::Storage st = DStorage_.loadStorage();
        c10::IValue res(st);
        return st;
    }else if(DKind_ == DKind::TENSOR){
        at::Tensor t = DTensor_.loadTensor();
        c10::IValue res(t);
        return res;
    }else if(DKind_ == DKind::TENSOR_LIST){
        c10::List<at::Tensor> TensorList;
        for(int i = 0; i < DTensorList_.size(); i++){
            at::Tensor t = DTensorList_[i].loadTensor();
            TensorList.emplace_back(t);
        }
        c10::IValue res(TensorList);
        return res;
    }else if(DKind_ == DKind::TUPLE){
        std::vector<c10::IValue> IValueVec;
        for(int i = 0; i < DIValueList_.size(); i++){
            IValueVec.emplace_back(DIValueList_[i].loadIValue());
        }
        c10::intrusive_ptr<c10::ivalue::Tuple> IValuePtr = c10::ivalue::Tuple::create(IValueVec);
        c10::IValue res(IValuePtr);
        return res;
    }else if(DKind_ == DKind::GENERIC_DICT){
        AT_ASSERT(DIValueList_.size() % 2 == 0, "GENERIC_DICT's size must be 2");
        // ska_ordered::order_preserving_flat_hash_map<c10::IValue, c10::IValue, 
        //                 c10::detail::DictKeyHash, c10::detail::DictKeyEqualTo> IValueMap;
        // for(int i = 0; i < DIValueList_.size() - 1; i += 2){
        //     c10::IValue key = DIValueList_[0].loadIValue();
        //     c10::IValue value = DIValueList_[1].loadIValue();
        //     IValueMap.insert(std::pair<c10::IValue, c10::IValue>(key, value));
        // }
        // c10::impl::GenericDict dict(c10::intrusive_ptr<c10::detail::DictImpl>(c10::detail::DictImpl(IValueMap)));
        
        AT_ASSERT(DIValueList_.size() > 0, "GENERIC_DICT SIZE must more than 0");

        if(PITCH_DKIND(DKind::INT, DKind::INT)){
            DICT_TO_IVLALUE(int64_t, int64_t, i_, i_);
        }else if(PITCH_DKIND(DKind::INT, DKind::DOUBLE)){
            DICT_TO_IVLALUE(int64_t, double, i_, d_);
        }else if(PITCH_DKIND(DKind::INT, DKind::BOOL)){
            DICT_TO_IVLALUE(int64_t, bool, i_, b_);
        }else if(PITCH_DKIND(DKind::INT, DKind::STRING)){
            DICT_TO_IVLALUE(int64_t, std::string, i_, str_);
        }else if(PITCH_DKIND(DKind::DOUBLE, DKind::INT)){
            DICT_TO_IVLALUE(double, int64_t, d_, i_);
        }else if(PITCH_DKIND(DKind::DOUBLE, DKind::DOUBLE)){
            DICT_TO_IVLALUE(double, double, d_, d_);
        }else if(PITCH_DKIND(DKind::DOUBLE, DKind::BOOL)){
            DICT_TO_IVLALUE(double, bool, d_, b_);
        }else if(PITCH_DKIND(DKind::DOUBLE, DKind::STRING)){
            DICT_TO_IVLALUE(double, std::string, d_, str_);
        }else if(PITCH_DKIND(DKind::BOOL, DKind::INT)){
            DICT_TO_IVLALUE(bool, int64_t, b_, i_);
        }else if(PITCH_DKIND(DKind::BOOL, DKind::DOUBLE)){
            DICT_TO_IVLALUE(bool, double, b_, d_);
        }else if(PITCH_DKIND(DKind::BOOL, DKind::BOOL)){
            DICT_TO_IVLALUE(bool, bool, b_, b_);
        }else if(PITCH_DKIND(DKind::BOOL, DKind::STRING)){
            DICT_TO_IVLALUE(bool, std::string, b_, str_);
        }else if(PITCH_DKIND(DKind::STRING, DKind::INT)){
            DICT_TO_IVLALUE(std::string, int64_t, str_, i_);
        }else if(PITCH_DKIND(DKind::STRING, DKind::DOUBLE)){
            DICT_TO_IVLALUE(std::string, double, str_, d_);
        }else if(PITCH_DKIND(DKind::STRING, DKind::BOOL)){
            DICT_TO_IVLALUE(std::string, bool, str_, b_);
        }else if(PITCH_DKIND(DKind::STRING, DKind::STRING)){
            DICT_TO_IVLALUE(std::string, std::string, str_, str_);
        }else{
            AT_ERROR("genericDict is not support");
            c10::IValue res("");
            return res;
        }
                                
    }else{
        AT_ERROR("unable to load");      
        c10::IValue res("");
        return res;
    }
}


TORCH_API void IvalueVecToJson(const std::vector<c10::IValue>& IData_, const string& SavePath){
    std::vector<DIValue> DIValueVec(IData_.size());
    Json::Value jsonRoot;

    for(int i = 0; i < IData_.size(); i++){
        DIValueVec[i].setDIValue(IData_[i]);
    }
    
    for(int i = 0; i < IData_.size(); i++){
        jsonRoot.append(DIValueVec[i].setJson());
    }

    ofstream ofs;
    ofs.open(SavePath);
    ofs << jsonRoot.toStyledString();
    ofs.close();
}


TORCH_API std::vector<c10::IValue>  jsonToIValueVec(const string& SavePath) {
    Json::Value jsonRoot;
    ifstream ifs;
    ifs.open(SavePath);
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;

    if(!parseFromStream(builder, ifs, &jsonRoot, &errs)){
         exit(1);
    }

    int n = jsonRoot.size();
    std::vector<DIValue> DIValueVec_(n);
    for(int i = 0; i < n; i++){
        DIValueVec_[i].json2DIValue(jsonRoot[i]);
    }
    
    std::vector<c10::IValue> IValueVec_(n);
    for(int i = 0; i < n; i++){
        IValueVec_[i] = DIValueVec_[i].loadIValue();
    }
    return IValueVec_;
}

void IvalueVecToBin(const std::vector<c10::IValue>& IData_, const string& SaveDir){
    int dim = IData_.size();
    string dim_file = SaveDir + "/dim";
    DumpBinaryData<int>(dim_file.c_str(), &dim, 1);
    
    std::vector<DIValue> DIValueVec(dim);
    for(int i = 0; i < dim; i++){
        DIValueVec[i].setDIValue(IData_[i]);
    }
    
    for(int i = 0; i < dim; i++){
        string dir = SaveDir + '/' + to_string(i);
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
        DIValueVec[i].setBin(dir);
    }
}

std::vector<c10::IValue> binToIValueVec(const string& SaveDir){
    int dim;
    string dim_file = SaveDir + "/dim";
    ReadBufferFromBinaryFile<int>(dim_file.c_str(), &dim, 1);

    std::vector<DIValue> DIValueVec_(dim);
    for(int i = 0; i < dim; i++){
        string dir = SaveDir + '/' + to_string(i);
        DIValueVec_[i].bin2DIValue(dir);
    }
    
    std::vector<c10::IValue> IValueVec_(dim);
    for(int i = 0; i < dim; i++){
        IValueVec_[i] = DIValueVec_[i].loadIValue();
    }
    return IValueVec_;
}

TORCH_API string CompareAcc(const std::vector<c10::IValue>& cpu_, const std::vector<c10::IValue>& gpu_){
    string msg;
    if(cpu_.size() != gpu_.size()){
        msg = "CompareAcc: cpu_ and gpu_ ivalue size_ must be equal";
        return msg;
    }
    double rtol = 1e-5, atol = 1e-8;
    double ftol = 1e-9;
    
    char * rtolEnv = std::getenv("RTOL");
    char * atolEnv = std::getenv("ATOL");
    if(rtolEnv){
        rtol = std::atof(rtolEnv);
    }
    if(atolEnv){
        atol = std::atof(atolEnv);
    }


    for(int i = 0; i < cpu_.size(); i++){
        c10::IValue cpuIValue_ = cpu_[i];
        c10::IValue gpuIValue_ = gpu_[i];

        if(cpuIValue_.isInt()){
            if(!gpuIValue_.isInt()) {
                msg = "CompareAcc: expect gpu Dkind is Int";
                return msg;
            }
            if(cpuIValue_.toInt() != gpuIValue_.toInt()){
                msg = "CompareAcc: cpu and gpu int value is not equal";
                return msg;
            }
        }else if(cpuIValue_.isDouble()){
            if(!gpuIValue_.isDouble()) {
                msg = "CompareAcc: expect gpu Dkind is Double";
                return msg;
            }
            if(std::abs(cpuIValue_.toDouble() - gpuIValue_.toDouble()) > ftol){
                msg = "CompareAcc: cpu and gpu double value is not equal";
                return msg;
            }
        }else if(cpuIValue_.isBool()){
            if(!gpuIValue_.isBool()) {
                msg = "CompareAcc: expect gpu Dkind is Bool";
                return msg;
            }
            if(cpuIValue_.toBool() != gpuIValue_.toBool()){
                msg = "CompareAcc: cpu and gpu bool value is not equal";
                return msg;
            }
        }else if(cpuIValue_.isIntList()){
            if(!gpuIValue_.isIntList()) {
                msg = "CompareAcc: expect gpu Dkind is IntList";
                return msg;
            }
            c10::List<int64_t> cpuIntList_ = cpuIValue_.toIntList();
            c10::List<int64_t> gpuIntList_ = gpuIValue_.toIntList();
            if(cpuIntList_.size() != gpuIntList_.size()){
                msg = "CompareAcc: expect cpu_ and gpu_ Intlist size is equal";
                return msg;
            }
            for(int i = 0; i < cpuIntList_.size(); i++){
                if(cpuIntList_[i] != gpuIntList_[i]){
                    msg = "CompareAcc: cpu_ and gpu_ Intlist vlaue is not equal";
                    return msg;
                }
            }
        }else if(cpuIValue_.isDoubleList()){
            if(!gpuIValue_.isDoubleList()) {
                msg = "CompareAcc: expect gpu Dkind is DoubleList";
                return msg;
            }
            c10::List<double> cpuDoubleList_ = cpuIValue_.toDoubleList();
            c10::List<double> gpuDoubleList_ = gpuIValue_.toDoubleList();
            if(cpuDoubleList_.size() != gpuDoubleList_.size()){
                msg = "CompareAcc: expect cpu_ and gpu_ DoubleList size is equal";
                return msg;
            }
            for(int i = 0; i < cpuDoubleList_.size(); i++){
                if(std::abs(cpuDoubleList_[i] - gpuDoubleList_[i]) > ftol){
                    msg = "CompareAcc: cpu_ and gpu_ DoubleList vlaue is not equal";
                    return msg;
                }
            }
        }else if(cpuIValue_.isBoolList()){
            if(!gpuIValue_.isBoolList()) {
                msg = "CompareAcc: expect gpu Dkind is BoolList";
                return msg;
            }
            c10::List<bool> cpuBoolList_ = cpuIValue_.toBoolList();
            c10::List<bool> gpuBoolList_ = gpuIValue_.toBoolList();
            if(cpuBoolList_.size() != gpuBoolList_.size()){
                msg = "CompareAcc: expect cpu_ and gpu_ BoolList size is equal";
                return msg;
            }
            for(int i = 0; i < cpuBoolList_.size(); i++){
                if(cpuBoolList_[i] != gpuBoolList_[i]){
                    msg = "CompareAcc: cpu_ and gpu_ BoolList vlaue is not equal";
                    return msg;
                }
            }
        }else if(cpuIValue_.isString()){
            if(!gpuIValue_.isString()) {
                msg = "CompareAcc: expect gpu Dkind is String";
                return msg;
            }
            std::string cpuStr_ = cpuIValue_.toStringRef();
            std::string gpuStr_ = gpuIValue_.toStringRef();
            if(cpuStr_ != gpuStr_){
                msg = "CompareAcc: cpu and gpu string value is not equal";
                return msg;
            }
        }else if(cpuIValue_.isStorage()){
            if(!gpuIValue_.isStorage()) {
                msg = "CompareAcc: expect gpu Dkind is Storage";
                return msg;
            }
            const at::Storage& cpuStorage_ = cpuIValue_.toStorage();
            const at::Storage& gpuStorage_ = gpuIValue_.toStorage();
            int64_t cpuStorageSize_ = static_cast<int64_t>(cpuStorage_.nbytes());
            int64_t gpuStorageSize_ = static_cast<int64_t>(gpuStorage_.nbytes());
            if(cpuStorageSize_ != gpuStorageSize_){
                msg = "CompareAcc: cpu and gpu Storage size is not equal";
                return msg;
            }
            for(int i = 0; i < cpuStorageSize_; i++){
                bool isEqual = cpuStorage_.data<uint8_t>()[i] == gpuStorage_.data<uint8_t>()[i];
                if(!isEqual){
                    msg = "CompareAcc: cpu and gpu Storage value is not equal";
                    return msg;
                }
            }
        }else if(cpuIValue_.isTensor()){
            if(!gpuIValue_.isTensor()) {
                msg = "CompareAcc: expect gpu Dkind is Tensor";
                return msg;
            }
            const at::Tensor& cpuTensor_ = cpuIValue_.toTensor();
            const at::Tensor& gpuTensor_ = gpuIValue_.toTensor();
            bool isEqual = at::allclose(cpuTensor_, gpuTensor_, rtol, atol, false);
            if(!isEqual){
                msg = "CompareAcc: cpu and gpu Tensor value is not equal";
                return msg;
            }
        }else if(cpuIValue_.isTensorList()){
            if(!gpuIValue_.isTensorList()) {
                msg = "CompareAcc: expect gpu Dkind is TensorList";
                return msg;
            }
            const c10::List<at::Tensor>& cpuTensorList_ = cpuIValue_.toTensorList();
            const c10::List<at::Tensor>& gpuTensorList_ = gpuIValue_.toTensorList();
            if(cpuTensorList_.size() != gpuTensorList_.size()){
                msg = "CompareAcc: cpu and gpu TensorList size is not equal";
                return msg;
            }
            for(int i = 0; i < cpuTensorList_.size(); i++){
                bool isEqual = at::allclose(cpuTensorList_[i], gpuTensorList_[i], rtol, atol, false);
                if(!isEqual){
                    msg = "CompareAcc: cpu and gpu TensorList value is not equal";
                    return msg;
                }
            }
        }else if(cpuIValue_.isGenericDict()){
            if(!gpuIValue_.isGenericDict()) {
                msg = "CompareAcc: expect gpu Dkind is GenericDict";
                return msg;
            }
            if(!(cpuIValue_ == gpuIValue_)){
                msg = "CompareAcc:  cpu and gpu GenericDict value is not equal";
                return msg;
            }
        }else if(cpuIValue_.isTuple()){
            if(!gpuIValue_.isTuple()) {
                msg = "CompareAcc: expect gpu Dkind is Tuple";
                return msg;
            }
            const std::vector<c10::IValue>& cpuIValueList_ = cpuIValue_.toTuple()->elements();
            const std::vector<c10::IValue>& gpuIValueList_ = gpuIValue_.toTuple()->elements();
            msg = CompareAcc(cpuIValueList_, gpuIValueList_);
            if(msg != "true"){
                return msg;
            }
        }else{
            msg = "CompareAcc:  unsupported ivalue type compare";
            return msg;
        }
        
    }
    msg = "true";
    return msg;
}


}  //namespace at
}  //namespace dump
