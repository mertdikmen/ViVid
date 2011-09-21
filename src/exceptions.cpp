#include "exceptions.hpp"

#include <stdexcept>
#include <sstream>

/**
 * Thin wrapper to "throw std::runtime_error"
 */
void throw_runtime_error(const char* error)
{
    throw std::runtime_error(error);
}

void throw_cuda_error(cudaError_t err, const char* file, int line)
{
    std::stringstream message;
    message << "CUDA error at " << file << ":" << line << "  "
            << cudaGetErrorString(err);
    
    throw std::runtime_error(message.str().c_str());
}

