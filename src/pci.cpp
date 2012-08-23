/*
Copyright (c) 2009-2012, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// written by Roman Dementiev,
//            Pat Fay
//

#include <iostream>
#include <sstream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "pci.h"

#ifndef _MSC_VER
#include <sys/mman.h>
#include <errno.h>
#endif

#ifdef _MSC_VER

#include <windows.h>
#include "Winmsrdriver\win7\msrstruct.h"
#include "winring0/OlsDef.h"
#include "winring0/OlsApiInitExt.h"

extern HMODULE hOpenLibSys;

PciHandle::PciHandle(uint32 bus_, uint32 device_, uint32 function_) :
    bus(bus_),
    device(device_),
    function(function_),
	pciAddress(PciBusDevFunc(bus_, device_, function_))
{
    hDriver = CreateFile(L"\\\\.\\RDMSR", GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);

    if (hDriver == INVALID_HANDLE_VALUE && hOpenLibSys == NULL)
        throw std::exception();
}

bool PciHandle::exists(uint32 bus_, uint32 device_, uint32 function_)
{
	if(hOpenLibSys != NULL) return true;

    HANDLE tempHandle = CreateFile(L"\\\\.\\RDMSR", GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
    if (tempHandle == INVALID_HANDLE_VALUE)
		return false;

    // TODO: check device availability

    CloseHandle(tempHandle);

    return true;
}

int32 PciHandle::read32(uint64 offset, uint32 * value)
{
	if(hDriver != INVALID_HANDLE_VALUE)
	{
		PCICFG_Request req;
		ULONG64 result = 0;
		DWORD reslength = 0;
		req.bus = bus;
		req.dev = device;
		req.func = function;
		req.bytes = sizeof(uint32);
		req.reg = (uint32)offset;

		BOOL status = DeviceIoControl(hDriver, IO_CTL_PCICFG_READ, &req, sizeof(PCICFG_Request), &result, sizeof(uint64), &reslength, NULL);
		*value = (uint32)result;
		if (!status)
		{
			//std::cerr << "Error reading PCI Config space at bus "<<bus<<" dev "<< device<<" function "<< function <<" offset "<< offset << " size "<< req.bytes  << ". Windows error: "<<GetLastError()<<std::endl;
		}
		return reslength;
	}
	DWORD result = 0;
	if(ReadPciConfigDwordEx(pciAddress,(DWORD)offset,&result))
	{
		*value = result;
		return sizeof(uint32);
	}
	return 0;
}

int32 PciHandle::write32(uint64 offset, uint32 value)
{
	if(hDriver != INVALID_HANDLE_VALUE)
	{
		PCICFG_Request req;
		ULONG64 result;
		DWORD reslength = 0;
		req.bus = bus;
		req.dev = device;
		req.func = function;
		req.bytes = sizeof(uint32);
		req.reg = (uint32)offset;
		req.write_value = value;

		BOOL status = DeviceIoControl(hDriver, IO_CTL_PCICFG_WRITE, &req, sizeof(PCICFG_Request), &result, sizeof(uint64), &reslength, NULL);
		if (!status)
		{
			//std::cerr << "Error writing PCI Config space at bus "<<bus<<" dev "<< device<<" function "<< function <<" offset "<< offset << " size "<< req.bytes  << ". Windows error: "<<GetLastError()<<std::endl;
		}
		return reslength;
	}

	return (WritePciConfigDwordEx(pciAddress,(DWORD)offset,value))?sizeof(uint32):0;
}

int32 PciHandle::read64(uint64 offset, uint64 * value)
{
	if(hDriver != INVALID_HANDLE_VALUE)
	{
		PCICFG_Request req;
		// ULONG64 result;
		DWORD reslength = 0;
		req.bus = bus;
		req.dev = device;
		req.func = function;
		req.bytes = sizeof(uint64);
		req.reg = (uint32)offset;

		BOOL status = DeviceIoControl(hDriver, IO_CTL_PCICFG_READ, &req, sizeof(PCICFG_Request), value, sizeof(uint64), &reslength, NULL);
		if (!status)
		{
			//std::cerr << "Error reading PCI Config space at bus "<<bus<<" dev "<< device<<" function "<< function <<" offset "<< offset << " size "<< req.bytes  << ". Windows error: "<<GetLastError()<<std::endl;
		}
		return reslength;
	}

	cvt_ds cvt;
	cvt.ui64 = 0;

	BOOL status = ReadPciConfigDwordEx(pciAddress, (DWORD)offset, &(cvt.ui32.low));
	status &= ReadPciConfigDwordEx(pciAddress, ((DWORD)offset) + sizeof(uint32), &(cvt.ui32.high));

	if(status)
	{
		*value = cvt.ui64;
		return sizeof(uint64);
	}
	return 0;
}

int32 PciHandle::write64(uint64 offset, uint64 value)
{
	if(hDriver != INVALID_HANDLE_VALUE)
	{
		PCICFG_Request req;
		ULONG64 result;
		DWORD reslength = 0;
		req.bus = bus;
		req.dev = device;
		req.func = function;
		req.bytes = sizeof(uint64);
		req.reg = (uint32)offset;
		req.write_value = value;

		BOOL status = DeviceIoControl(hDriver, IO_CTL_PCICFG_WRITE, &req, sizeof(PCICFG_Request), &result, sizeof(uint64), &reslength, NULL);
		if (!status)
		{
			//std::cerr << "Error writing PCI Config space at bus "<<bus<<" dev "<< device<<" function "<< function <<" offset "<< offset << " size "<< req.bytes  << ". Windows error: "<<GetLastError()<<std::endl;
		}
		return reslength;
	}
	cvt_ds cvt;
	cvt.ui64 = value;
	BOOL status = WritePciConfigDwordEx(pciAddress,(DWORD)offset,cvt.ui32.low);
	status &= WritePciConfigDwordEx(pciAddress,((DWORD)offset)+sizeof(uint32),cvt.ui32.high);

	return status?sizeof(uint64):0;
}

PciHandle::~PciHandle()
{
    if(hDriver != INVALID_HANDLE_VALUE) CloseHandle(hDriver);
}

#else


// Linux implementation

PciHandle::PciHandle(uint32 bus_, uint32 device_, uint32 function_) :
    fd(-1),
    bus(bus_),
    device(device_),
    function(function_)
{
    std::ostringstream path(std::ostringstream::out);

    path << std::hex << "/proc/bus/pci/" << std::setw(2) << std::setfill('0') << bus << "/" << std::setw(2) << std::setfill('0') << device << "." << function;

    // std::cout << path.str().c_str() << std::endl;

    int handle = ::open(path.str().c_str(), O_RDWR);
    if (handle < 0) throw std::exception();
    fd = handle;

    // std::cout << "Opened "<< path.str().c_str() << " on handle "<< fd << std::endl;
}


bool PciHandle::exists(uint32 bus_, uint32 device_, uint32 function_)
{
    std::ostringstream path(std::ostringstream::out);

    path << std::hex << "/proc/bus/pci/" << std::setw(2) << std::setfill('0') << bus_ << "/" << std::setw(2) << std::setfill('0') << device_ << "." << function_;

    int handle = ::open(path.str().c_str(), O_RDWR);

    if (handle < 0) return false;

    ::close(handle);

    return true;
}

int32 PciHandle::read32(uint64 offset, uint32 * value)
{
    return ::pread(fd, (void *)value, sizeof(uint32), offset); 
}

int32 PciHandle::write32(uint64 offset, uint32 value)
{
    return ::pwrite(fd, (const void *)&value, sizeof(uint32), offset);
}

int32 PciHandle::read64(uint64 offset, uint64 * value)
{
    return ::pread(fd, (void *)value, sizeof(uint64), offset);
}

int32 PciHandle::write64(uint64 offset, uint64 value)
{
    return ::pwrite(fd, (const void *)&value, sizeof(uint64), offset);
}

PciHandle::~PciHandle()
{
    if (fd >= 0) ::close(fd);
}

#ifndef PCM_USE_PCI_MM_LINUX

PciHandleM::PciHandleM(uint32 bus_, uint32 device_, uint32 function_) :
    fd(-1),
    bus(bus_),
    device(device_),
    function(function_),
    base_addr(0)
{
    int handle = ::open("/dev/mem", O_RDWR);
    if (handle < 0) throw std::exception();
    fd = handle;

    int mcfg_handle = ::open("/sys/firmware/acpi/tables/MCFG", O_RDONLY);

    if (mcfg_handle < 0) throw std::exception();

    int32 result = ::pread(mcfg_handle, (void *)&base_addr, sizeof(uint64), 44);

    if (result != sizeof(uint64))
    {
        ::close(mcfg_handle);
        throw std::exception();
    }

    unsigned char max_bus = 0;

    result = ::pread(mcfg_handle, (void *)&max_bus, sizeof(unsigned char),55);

    if (result != sizeof(unsigned char))
    {
        ::close(mcfg_handle);
        throw std::exception();
    }

    ::close(mcfg_handle);

    if(bus > (unsigned)max_bus)
    {
       std::cout << "ERROR: Requested bus number "<< bus<< " is larger than the max bus number " << (unsigned)max_bus << std::endl;
       throw std::exception();
    }

    // std::cout << "PCI config base addr: "<< std::hex << base_addr<< std::endl;

    base_addr += (bus * 1024 * 1024 + device * 32 * 1024 + function * 4 * 1024);
}


bool PciHandleM::exists(uint32 /* bus_*/, uint32 /* device_ */, uint32 /* function_ */)
{
    int handle = ::open("/dev/mem", O_RDWR);

    if (handle < 0) return false;

    ::close(handle);

    handle = ::open("/sys/firmware/acpi/tables/MCFG", O_RDONLY);

    if (handle < 0) return false;

    ::close(handle);

    return true;
}

int32 PciHandleM::read32(uint64 offset, uint32 * value)
{
    return ::pread(fd, (void *)value, sizeof(uint32), offset + base_addr);
}

int32 PciHandleM::write32(uint64 offset, uint32 value)
{
    return ::pwrite(fd, (const void *)&value, sizeof(uint32), offset + base_addr);
}

int32 PciHandleM::read64(uint64 offset, uint64 * value)
{
    return ::pread(fd, (void *)value, sizeof(uint64), offset + base_addr);
}

int32 PciHandleM::write64(uint64 offset, uint64 value)
{
    return ::pwrite(fd, (const void *)&value, sizeof(uint64), offset + base_addr);
}

PciHandleM::~PciHandleM()
{
    if (fd >= 0) ::close(fd);
}

#endif // PCM_USE_PCI_MM_LINUX

// mmaped I/O version

PciHandleMM::PciHandleMM(uint32 bus_, uint32 device_, uint32 function_) :
    fd(-1),
    mmapAddr(NULL),
    bus(bus_),
    device(device_),
    function(function_),
    base_addr(0)
{
    int handle = ::open("/dev/mem", O_RDWR);
    if (handle < 0) throw std::exception();
    fd = handle;

    int mcfg_handle = ::open("/sys/firmware/acpi/tables/MCFG", O_RDONLY);

    if (mcfg_handle < 0) throw std::exception();

    int32 result = ::pread(mcfg_handle, (void *)&base_addr, sizeof(uint64), 44);

    if (result != sizeof(uint64))
    {
        ::close(mcfg_handle);
        throw std::exception();
    }

    unsigned char max_bus = 0;

    result = ::pread(mcfg_handle, (void *)&max_bus, sizeof(unsigned char),55);

    if (result != sizeof(unsigned char))
    {
        ::close(mcfg_handle);
        throw std::exception();
    }

    ::close(mcfg_handle);

    if(bus > (unsigned)max_bus)
    {
       std::cout << "ERROR: Requested bus number "<< bus<< " is larger than the max bus number " << (unsigned)max_bus << std::endl;
       throw std::exception();
    }

    // std::cout << "PCI config base addr: "<< std::hex << base_addr<< std::endl;

    base_addr += (bus * 1024 * 1024 + device * 32 * 1024 + function * 4 * 1024);
    
    mmapAddr = (char*) mmap(NULL, 4096, PROT_READ| PROT_WRITE, MAP_SHARED , fd, base_addr);
    
    if(mmapAddr == MAP_FAILED)
    {
      std::cout << "mmap failed: errno is "<< errno<<  std::endl;
      throw std::exception();
    }
    
}


bool PciHandleMM::exists(uint32 bus_, uint32 device_, uint32 function_)
{
    int handle = ::open("/dev/mem", O_RDWR);

    if (handle < 0) return false;

    ::close(handle);

    handle = ::open("/sys/firmware/acpi/tables/MCFG", O_RDONLY);

    if (handle < 0) return false;

    ::close(handle);

    return true;
}

int32 PciHandleMM::read32(uint64 offset, uint32 * value)
{
    *value = *((uint32*)(mmapAddr+offset));

    return sizeof(uint32);
}

int32 PciHandleMM::write32(uint64 offset, uint32 value)
{
    *((uint32*)(mmapAddr+offset)) = value;

    return sizeof(uint32);
}

int32 PciHandleMM::read64(uint64 offset, uint64 * value)
{
    *value = *((uint64*)(mmapAddr+offset));

    return sizeof(uint64);
}

int32 PciHandleMM::write64(uint64 offset, uint64 value)
{
    *((uint64*)(mmapAddr+offset)) = value;

    return sizeof(uint64);
}

PciHandleMM::~PciHandleMM()
{
    if (mmapAddr) munmap(mmapAddr, 4096);
    if (fd >= 0) ::close(fd);
}


#endif
