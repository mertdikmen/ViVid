#include "ConvolutionMCuda.hpp"
#include "exceptions.hpp"

typedef bool (*try_convolution_algorithm) (const MCudaMatrix3D::Ptr&,
                                           const MCudaMatrix3D::Ptr&,
                                           MCudaMatrix3D::Ptr&);
/** This "try_convolution" algorithm always fails */
static bool placeholder(const MCudaMatrix3D::Ptr& video,
                        const MCudaMatrix3D::Ptr& kernel,
                        MCudaMatrix3D::Ptr& output)
{
    return false;
}

//! A list of algorithms to try
static const try_convolution_algorithm convolution_algorithm[] = {
    &try_convolution0_mcuda,
    &placeholder,
    &try_convolution2_mcuda,
    &placeholder,
    &try_convolution4_mcuda,
};

// stolen from http://en.wikibooks.org/wiki/C_Programming/Pointers_and_arrays
#define NUM_ELEM(x) (sizeof (x) / sizeof (*(x)))

//! An internal debugging flag
static unsigned int debug_algorithm_used;

void convolve3d_mcuda(const MCudaMatrix3D::Ptr& video,
                      const MCudaMatrix3D::Ptr& kernel,
                      MCudaMatrix3D::Ptr& output,
                      unsigned int algorithm)
{
    assert(algorithm < NUM_ELEM(convolution_algorithm));
    if (convolution_algorithm[algorithm](video, kernel, output)) {
            debug_algorithm_used = algorithm;
            return;
        }

    throw_runtime_error("Unable to find convolution algorithm");
}

MCudaMatrix3D::Ptr convolve3d_mcuda(const MCudaMatrix3D::Ptr& video,
                                    const MCudaMatrix3D::Ptr& kernel,
                                    int algorithm)
{
    MCudaMatrix3D::Ptr retval
        = makeMCudaMatrix3D(video->dim_t - kernel->dim_t + 1,
                            video->dim_y - kernel->dim_y + 1,
                            video->dim_x - kernel->dim_x + 1);

    convolve3d_mcuda(video, kernel, retval, algorithm);
    return retval;
}
