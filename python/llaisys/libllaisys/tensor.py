from ctypes import POINTER, c_uint8, c_void_p, c_size_t, c_ssize_t, c_int
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t

# Handle type
llaisysTensor_t = c_void_p


def load_tensor(lib):
    """Configure tensor function signatures for the C library."""
    
    # Core tensor lifecycle functions
    lib.tensorCreate.argtypes = [POINTER(c_size_t), c_size_t, llaisysDataType_t, llaisysDeviceType_t, c_int]
    lib.tensorCreate.restype = llaisysTensor_t

    lib.tensorDestroy.argtypes = [llaisysTensor_t]
    lib.tensorDestroy.restype = None

    # Tensor property accessors
    lib.tensorGetData.argtypes = [llaisysTensor_t]
    lib.tensorGetData.restype = c_void_p

    lib.tensorGetNdim.argtypes = [llaisysTensor_t]
    lib.tensorGetNdim.restype = c_size_t

    lib.tensorGetShape.argtypes = [llaisysTensor_t, POINTER(c_size_t)]
    lib.tensorGetShape.restype = None

    lib.tensorGetStrides.argtypes = [llaisysTensor_t, POINTER(c_ssize_t)]
    lib.tensorGetStrides.restype = None

    lib.tensorGetDataType.argtypes = [llaisysTensor_t]
    lib.tensorGetDataType.restype = llaisysDataType_t

    lib.tensorGetDeviceType.argtypes = [llaisysTensor_t]
    lib.tensorGetDeviceType.restype = llaisysDeviceType_t

    lib.tensorGetDeviceId.argtypes = [llaisysTensor_t]
    lib.tensorGetDeviceId.restype = c_int

    lib.tensorIsContiguous.argtypes = [llaisysTensor_t]
    lib.tensorIsContiguous.restype = c_uint8

    # Data manipulation functions
    lib.tensorLoad.argtypes = [llaisysTensor_t, c_void_p]
    lib.tensorLoad.restype = None

    lib.tensorDebug.argtypes = [llaisysTensor_t]
    lib.tensorDebug.restype = None

    # Tensor transformation functions
    lib.tensorView.argtypes = [llaisysTensor_t, POINTER(c_size_t), c_size_t]
    lib.tensorView.restype = llaisysTensor_t

    lib.tensorReshape.argtypes = [llaisysTensor_t, POINTER(c_size_t), c_size_t]
    lib.tensorReshape.restype = llaisysTensor_t

    lib.tensorPermute.argtypes = [llaisysTensor_t, POINTER(c_size_t)]
    lib.tensorPermute.restype = llaisysTensor_t

    lib.tensorSlice.argtypes = [llaisysTensor_t, c_size_t, c_size_t, c_size_t]
    lib.tensorSlice.restype = llaisysTensor_t
