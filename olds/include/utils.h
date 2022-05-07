#include <string>
#include <vector>
#include "iostream"
#include <fstream>
#include <iostream>
#include <sstream>
#include <glob.h>
std::vector<std::string> glob(const std::string pattern)
{
    std::vector<std::string> filenames;
    using namespace std;
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0){
        globfree(&glob_result);
        return filenames;
    }
    for(auto idx =0; idx <glob_result.gl_pathc; idx++){
        filenames.push_back(string(glob_result.gl_pathv[idx]));
    }
    globfree(&glob_result);
    return filenames;
}



