#ifndef CL_EXCEPTIONS_HPP
#define CL_EXCEPTIONS_HPP

#include <string>
#include <CL/cl.h>

#define CHECK_CL_ERROR(err) do {\
	if (err != CL_SUCCESS){ \
	vivid::print_cl_error(err, __FILE__, __LINE__);\
	exit(1);\
	} } while(0)

#define OPENCL_CALL(call) do {\
cl_int err = call; \
if(CL_SUCCESS!= err) { \
	vivid::print_cl_error(err, __FILE__, __LINE__);\
	exit(1);\
} } while (0)

namespace vivid
{
	void print_cl_error(cl_int errorcode, std::string file = "", int line = -1);
}

#endif
