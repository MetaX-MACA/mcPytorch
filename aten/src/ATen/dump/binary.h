#pragma once

#include<c10/util/Exception.h>

#include<iostream>
#include<fstream>

namespace at{
namespace dump{

template<typename T>
void DumpBinaryData(const char* filename, const T* data, size_t data_length){
    std::ofstream out(filename, std::ios::binary);
    if(out){
        out.write(reinterpret_cast<const char*>(data), data_length * sizeof(T));
        out.close();
    }else{
        AT_ERROR("dump binary data error");
    }
}


template<typename T>
bool ReadBufferFromBinaryFile(const char* filename, T* data, size_t data_length){
    std::ifstream in(filename, std::ios::binary);
    if(in){
        in.read(reinterpret_cast<char*>(data), data_length * sizeof(T));
        in.close();
    }else{
        AT_ERROR("Read binary data error");
        return false;
    }
    return true;
}

}
}
