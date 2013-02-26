#ifndef CL_EXCEPTIONS_HPP
#define CL_EXCEPTIONS_HPP

#include <CL/cl.h>

#define OPENCL_CALL(call) do {\
cl_int err = call; \
if(CL_SUCCESS!= err) { \
	vivid::print_cl_error(err);	\
} } while (0)

namespace vivid
{
	void print_cl_error(cl_int errorcode);
}

#endif