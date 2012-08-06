#ifndef _FLEXIBLE_FILTER_WRAPPER_HPP_
#define _FLEXIBLE_FILTER_WRAPPER_HPP_ 1

#include "FlexibleFilter.hpp"
#include "NumPyWrapper.hpp"
#include <boost/python.hpp>

using namespace boost::python;

void export_FlexibleFilter();

int update_filter_bank(object& filterbank_array);

#endif /*_FLEXIBLE_FILTER_WRAPPER_HPP_*/
