#include <ATen/ATen.h>
#include <ATen/dump/json.h>
#include <ATen/dump/ivalue_dump.h>

#include <gtest/gtest.h>
#include <iostream>

TEST(DIValueTest, BinaryString){
    const char* file = "/tmp/file_test.txt";
    string s = "123456";
    int len = s.length();
    at::dump::DumpBinaryData<char>(file, s.c_str(), len);

    char* res_ptr = new char[6];
    at::dump::ReadBufferFromBinaryFile<char>(file, res_ptr, len);
    string res = res_ptr;
    delete []res_ptr;

    EXPECT_EQ(s, res);
}


TEST(DIValueTest, BinaryEnum){
    const char* file = "/tmp/file_test.txt";
    at::dump::base::DKind s = at::dump::base::DKind::INT;
    at::dump::base::DKind d;
    at::dump::DumpBinaryData<at::dump::base::DKind>(file, &s, 1);
    at::dump::ReadBufferFromBinaryFile<at::dump::base::DKind>(file, &d, 1);

    EXPECT_EQ(d, at::dump::base::DKind::INT);
}

TEST(DIValueTest, BinaryFloat){
    const char* file = "/tmp/file_test.txt";
    float a = 3.1415974659201;
    
    at::dump::DumpBinaryData<float>(file, &a, 1);

    float b;
    at::dump::ReadBufferFromBinaryFile<float>(file, &b, 1);
    EXPECT_EQ(a, b);
}


TEST(DIValueTest, BinaryFloatList){
    const char* file = "/tmp/file_test.txt";
    double a[3] = {3.1415974659201,-98.83835, 1111.111222};
    double b[3];

    at::dump::DumpBinaryData<double>(file, a, 3);
    at::dump::ReadBufferFromBinaryFile<double>(file, b, 3);
    for(int i = 0; i < 3; i++){
        EXPECT_EQ(a[i], b[i]);
    }
}


TEST(DIValueTest, DKIND_INT){
    int64_t s = 34576;
    c10::IValue iv(s);
    
    at::dump::DIValue div;
    div.setDIValue(iv);
    Json::Value js = div.setJson();
    std::cout << js.toStyledString() << std::endl;

    at::dump::DIValue div1;
    div1.json2DIValue(js);

    c10::IValue iv1 = div1.loadIValue();
    EXPECT_TRUE(iv1.isInt());
    EXPECT_EQ(iv1.toInt(), s);
}


TEST(DIValueTest, DKIND_DOUBLE){
    double s = 356.248533;
    c10::IValue iv(s);
    
    at::dump::DIValue div;
    div.setDIValue(iv);
    Json::Value js = div.setJson();
    std::cout << js.toStyledString() << std::endl;

    at::dump::DIValue div1;
    div1.json2DIValue(js);

    c10::IValue iv1 = div1.loadIValue();
    EXPECT_TRUE(iv1.isDouble());
    EXPECT_TRUE( abs(iv1.toDouble() - s) < 1e-9 );
}


TEST(DIValueTest, DKIND_BOOL){
    bool s = true;
    c10::IValue iv(s);
    
    at::dump::DIValue div;
    div.setDIValue(iv);
    Json::Value js = div.setJson();
    std::cout << js.toStyledString() << std::endl;

    at::dump::DIValue div1;
    div1.json2DIValue(js);

    c10::IValue iv1 = div1.loadIValue();
    EXPECT_TRUE(iv1.isBool());
    EXPECT_TRUE(iv1.toBool() == s);
}


TEST(DIValueTest, DKIND_INT_LIST){
    c10::List<int64_t> s{10, 9, 2, 7};
    c10::IValue iv(s);
    
    at::dump::DIValue div;
    div.setDIValue(iv);
    Json::Value js = div.setJson();
    std::cout << js.toStyledString() << std::endl;

    at::dump::DIValue div1;
    div1.json2DIValue(js);

    c10::IValue iv1 = div1.loadIValue();
    std::vector<int64_t> res = iv1.toIntVector();
    
    EXPECT_TRUE(iv1.isIntList());
    EXPECT_EQ(res.size(), s.size());
    for(int i = 0;i < res.size(); i++){
        EXPECT_EQ(res[i], s[i]);
    }  
}


TEST(DIValueTest, DKIND_DOUBLE_LIST){
    c10::List<double> s{10.234, 9.7492586, 2.284859382, 74.23469596};
    c10::IValue iv(s);
    
    at::dump::DIValue div;
    div.setDIValue(iv);
    Json::Value js = div.setJson();
    std::cout << js.toStyledString() << std::endl;

    at::dump::DIValue div1;
    div1.json2DIValue(js);

    c10::IValue iv1 = div1.loadIValue();
    std::vector<double> res = iv1.toDoubleVector();
    
    EXPECT_TRUE(iv1.isDoubleList());
    EXPECT_EQ(res.size(), s.size());
    for(int i = 0;i < res.size(); i++){
        EXPECT_TRUE( abs(res[i] - s[i]) < 1e-9 );
    }  
}


TEST(DIValueTest, DKIND_BOOL_LIST){
    c10::List<bool> s{true, false, false, false};
    c10::IValue iv(s);
    
    at::dump::DIValue div;
    div.setDIValue(iv);
    Json::Value js = div.setJson();
    std::cout << js.toStyledString() << std::endl;

    at::dump::DIValue div1;
    div1.json2DIValue(js);

    c10::IValue iv1 = div1.loadIValue();
    const c10::List<bool> res = iv1.toBoolList();
    
    EXPECT_TRUE(iv1.isBoolList());
    EXPECT_EQ(res.size(), s.size());
    for(int i = 0;i < res.size(); i++){
        EXPECT_EQ(res[i], s[i]);
    }  
}


TEST(DIValueTest, DKIND_STRING){
    string s = "ab2359nbt";
    c10::IValue iv(s);
    
    at::dump::DIValue div;
    div.setDIValue(iv);
    Json::Value js = div.setJson();
    std::cout << js.toStyledString() << std::endl;

    at::dump::DIValue div1;
    div1.json2DIValue(js);

    c10::IValue iv1 = div1.loadIValue();
    string res = iv1.toStringRef();
    std::cout<<res<<endl;
    EXPECT_TRUE(iv1.isString());
    EXPECT_TRUE(res == s);
}


TEST(DIValueTest, DKIND_STORAGE){
    const int64_t size = 12;
    uint8_t data[size];
    for(int i = 0; i < size; i++){
        data[i] = static_cast<uint8_t>(i + 14);
    }
    at::Storage s(at::Storage::use_byte_size_t(),
                   size,
                   at::DataPtr(data, c10::DeviceType::CPU));
    c10::IValue iv(s);
    
    at::dump::DIValue div;
    div.setDIValue(iv);
    Json::Value js = div.setJson();
    std::cout << js.toStyledString() << std::endl;

    at::dump::DIValue div1;
    div1.json2DIValue(js);

    c10::IValue iv1 = div1.loadIValue();
    at::Storage res = iv1.toStorage();
    
    uint8_t* res_data = res.data<uint8_t>();
    int64_t res_size = static_cast<int64_t>(res.nbytes());

    EXPECT_TRUE(iv1.isStorage());
    EXPECT_EQ(size, res_size);
    for(int i = 0; i < size; i++){
        EXPECT_EQ(data[i], res_data[i]);
    }
    
}


bool is_equal_vec(const std::vector<int64_t>& a,const std::vector<int64_t>& b){
    int s1 = a.size();
    int s2 = b.size();
    if(s1 != s2) return false;
    for(int i = 0; i < s1; i++){
        if(a[i] != b[i]) return false;
    }
    return true;
}


void is_equal_tensor(const at::Tensor& a, const at::Tensor& b){
    EXPECT_TRUE(is_equal_vec(a.sizes().vec(), b.sizes().vec()));
    EXPECT_TRUE(is_equal_vec(a.strides().vec(), b.strides().vec()));
    
    EXPECT_EQ(a.numel(), b.numel());
    EXPECT_EQ(a.storage_offset(), b.storage_offset());
    EXPECT_EQ(a.requires_grad() ,b.requires_grad());

    EXPECT_EQ(a.storage().nbytes(), b.storage().nbytes());
    EXPECT_EQ(a.numel() * 4, a.storage().nbytes());

    for(int i = 0; i < a.storage().nbytes(); i++){
        EXPECT_EQ(a.storage().data<uint8_t>()[i], a.storage().data<uint8_t>()[i]);
    }
    
}



TEST(DIValueTest, DKIND_TENSOR){
    at::Tensor s = at::rand({2, 3, 4}, c10::ScalarType::Float);
    s = s.transpose(2, 0);
    c10::IValue iv(s);
    
    at::dump::DIValue div;
    div.setDIValue(iv);
    Json::Value js = div.setJson();
    std::cout << js.toStyledString() << std::endl;

    at::dump::DIValue div1;
    div1.json2DIValue(js);

    c10::IValue iv1 = div1.loadIValue();
    at::Tensor res = iv1.toTensor();

    EXPECT_TRUE(iv1.isTensor());
    is_equal_tensor(s, res);
}


TEST(DIValueTest, DKIND_TENSOR_LIST){
    at::Tensor s1 = at::rand({3, 1}, c10::ScalarType::Float);
    at::Tensor s2 = at::rand({2, 3}, c10::ScalarType::Float);
    c10::List<at::Tensor> sv;
    sv.emplace_back(s1);
    sv.emplace_back(s2);

    c10::IValue iv(sv);
    
    at::dump::DIValue div;
    div.setDIValue(iv);
    Json::Value js = div.setJson();
    std::cout << js.toStyledString() << std::endl;

    at::dump::DIValue div1;
    div1.json2DIValue(js);

    c10::IValue iv1 = div1.loadIValue();
    c10::List<at::Tensor> sv1 = iv1.toTensorList();

    EXPECT_TRUE(iv1.isTensorList());
    EXPECT_EQ(sv.size(), sv1.size());

    for(int i = 0; i < sv.size(); i++){
        is_equal_tensor(sv[i], sv1[i]);
    }
}



TEST(DIValueTest, DKIND_TUPLE){
    c10::IValue a("123");
    c10::IValue b(13);
    c10::IValue c(false);

    std::vector<c10::IValue> IValueVec{a, b, c};
    c10::intrusive_ptr<c10::ivalue::Tuple> IValuePtr = c10::ivalue::Tuple::create(IValueVec);
    c10::IValue iv(IValuePtr);
    
    at::dump::DIValue div;
    div.setDIValue(iv);
    Json::Value js = div.setJson();
    std::cout << js.toStyledString() << std::endl;

    at::dump::DIValue div1;
    div1.json2DIValue(js);

    c10::IValue iv1 = div1.loadIValue();
    EXPECT_TRUE(iv1.isTuple());

    const std::vector<c10::IValue>& ivRes = iv1.toTuple()->elements();
    EXPECT_TRUE(ivRes.size() == 3);

    EXPECT_TRUE(ivRes[0].isString());
    EXPECT_EQ(ivRes[0].toStringRef(),  IValueVec[0].toStringRef());

    EXPECT_TRUE(ivRes[1].isInt());
    EXPECT_EQ(ivRes[1].toInt(),  IValueVec[1].toInt());

    EXPECT_TRUE(ivRes[2].isBool());
    EXPECT_EQ(ivRes[2].toBool(),  IValueVec[2].toBool());
}


TEST(DIValueTest, DKIND_GENERIC_DICT){
    c10::Dict<std::string, int64_t> dic;
    dic.insert("444" ,5);
    dic.insert("222" ,777);
    dic.insert("333" ,777);
    c10::IValue iv(dic);

    at::dump::DIValue div;
    div.setDIValue(iv);
    Json::Value js = div.setJson();
    std::cout << js.toStyledString() << std::endl;

    at::dump::DIValue div1;
    div1.json2DIValue(js);

    c10::IValue iv1 = div1.loadIValue();

    EXPECT_TRUE(iv1.isGenericDict());
    EXPECT_TRUE(iv1.toGenericDict().size() == 3);
    
    at::dump::DIValue div2;
    div2.setDIValue(iv1);
    Json::Value js1 = div2.setJson();
    std::cout << js1.toStyledString() << std::endl;
}

TEST(DIValueTest, DKIND_GENERIC_DICT_1){
    c10::Dict<std::string, double> dic1;
    dic1.insert("444" ,5.0);
    dic1.insert("222" ,777.1);
    dic1.insert("333" ,777.2);
    c10::IValue iv1(dic1);

    c10::Dict<std::string, double> dic2;
    dic2.insert("222" ,777.1);
    dic2.insert("444" ,5.0);
    dic2.insert("333" ,777.2);
    c10::IValue iv2(dic2);

    EXPECT_TRUE(iv1 == iv2);
}


TEST(DIValueTest, DKIND_GENERIC_DICT_2){
    c10::Dict<std::string, double> dic1;
    dic1.insert("444" ,5.01);
    dic1.insert("222" ,777.1);
    dic1.insert("333" ,777.2);
    c10::IValue iv1(dic1);

    c10::Dict<std::string, double> dic2;
    dic2.insert("222" ,777.1);
    dic2.insert("444" ,5.0);
    dic2.insert("333" ,777.2);
    c10::IValue iv2(dic2);

    EXPECT_FALSE(iv1 == iv2);
}


TEST(DIValueTest, DKIND_TENSOR_EQUAL){
    int64_t size = 16;
    uint8_t* data1 = new uint8_t[size];
    uint8_t* data2 = new uint8_t[size];

    float* fdata1 = (float*)data1;
    float* fdata2 = (float*)data2;

    fdata1[0] = 12.1234;  fdata1[1] = -12.1234; fdata1[2] = 22.034; fdata1[3] = -22.034; 
    fdata2[0] = 12.1234;  fdata2[1] = -12.1234; fdata2[2] = 22.034; fdata2[3] = -22.034; 

    at::Storage st1(at::Storage::use_byte_size_t(),
                   size,
                   at::DataPtr(data1, c10::DeviceType::CPU));
    at::Storage st2(at::Storage::use_byte_size_t(),
                   size,
                   at::DataPtr(data2, c10::DeviceType::CPU));
    
    at::Tensor t1 = at::empty({}, c10::ScalarType::Float);
    t1.set_(st1, 0, {2, 2}, {1, 2});
    t1.set_requires_grad(false);

    at::Tensor t2 = at::empty({}, c10::ScalarType::Float);
    t2.set_(st2, 0, {2, 2}, {1, 2});
    t2.set_requires_grad(false);
    
    bool isEqual = at::allclose(t1, t2, 10e-4, 10e-4, false);
    EXPECT_TRUE(isEqual);

}

