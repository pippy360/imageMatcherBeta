#include <vector>

#pragma once
class FragmentHash
{
private:
    std::vector<bool> hash_;
public:
    inline FragmentHash(std::vector<bool> hash){
        hash_ = hash;
    };
    inline std::vector<bool> getHash(){
        return hash_;
    }
};
