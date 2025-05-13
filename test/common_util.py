import os
import torch
from typing import List, Any, ClassVar, Optional, Sequence, Tuple, Union, cast
from functools import wraps
import copy
import inspect
import threading
import unittest
from collections import namedtuple
import math
import numpy as np
import contextlib

torch.backends.disable_global_flags()

TEST_MULTIGPU = torch.cuda.is_available() and torch.cuda.device_count() >= 2
TEST_MULTIGPU_HINT = "only runs when multiple gpus detected"

# these tests are bases on implementation in PyTorch and the
# tested behaviour is not guranteed in official document.
TEST_EMPIRICAL = os.getenv("PYTORCH_MACA_TESTS_ENABLE_EMPIRICAL")
TEST_EMPIRICAL_HINT = "run empirical tests cases"

BASIC_FLOATING_TYPES = [torch.float, torch.double]
HALF_TYPES = [torch.half, torch.bfloat16]
ALL_FLOATING_TYPES = [torch.float, torch.double, torch.half, torch.bfloat16]
ALL_COMPLEX_TYPES = [torch.complex32, torch.complex64, torch.complex128]
ALL_INTEGER_TYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
# torch.bool
# dnn and blas has tf32

# norm x (e.g. a list/tuple) to standard test name(without ',' ' ')
NORM_NAME = lambda *args: "".join(str(args).replace(' ', '')).replace(',', '_').replace('(', '[').replace(')', ']')

def make_noncontig(tensor):
    ndim = tensor.dim()
    return torch.stack([tensor.clone().zero_(), tensor], ndim).select(ndim, 1)

# Function that instantiates a template class and its tests
def dtype_name(dtype):
    """ Returns the pretty name of the dtype (e.g. torch.int64 -> int64). """
    return str(dtype).split('.')[1]

def _update_param_kwargs(param_kwargs, name, value):
    """ Adds a kwarg with the specified name and value to the param_kwargs dict. """
    if isinstance(value, list) or isinstance(value, tuple):
        # Make name plural (e.g. devices / dtypes) if the value is composite.
        param_kwargs['{}s'.format(name)] = value
    elif value:
        param_kwargs[name] = value

def _dtype_test_suffix(dtypes):
    """ Returns the test suffix for a dtype, sequence of dtypes, or None. """
    if isinstance(dtypes, list) or isinstance(dtypes, tuple):
        if len(dtypes) == 0:
            return ''
        return '_' + '_'.join((dtype_name(d) for d in dtypes))
    elif dtypes:
        return '_{}'.format(dtype_name(dtypes))
    else:
        return ''


tol = namedtuple('tol', ['atol', 'rtol'])


class DeviceTypeTestBase(unittest.TestCase):
    device_type: str = 'generic_device_type'

    # Flag to disable test suite early due to unrecoverable error such as CUDA error.
    _stop_test_suite = False

    # Precision is a thread-local setting since it may be overridden per test
    _tls = threading.local()
    # _tls.precision = TestCase._precision
    # _tls.rel_tol = TestCase._rel_tol
    _tls.precision = 0
    _tls.rel_tol = 0

    @property
    def precision(self):
        return self._tls.precision

    @precision.setter
    def precision(self, prec):
        self._tls.precision = prec

    @property
    def rel_tol(self):
        return self._tls.rel_tol

    @rel_tol.setter
    def rel_tol(self, prec):
        self._tls.rel_tol = prec

    # Returns a string representing the device that single device tests should use.
    # Note: single device tests use this device exclusively.
    @classmethod
    def get_primary_device(cls):
        return cls.device_type

    # Returns a list of strings representing all available devices of this
    # device type. The primary device must be the first string in the list
    # and the list must contain no duplicates.
    # Note: UNSTABLE API. Will be replaced once PyTorch has a device generic
    #   mechanism of acquiring all available devices.
    @classmethod
    def get_all_devices(cls):
        return [cls.get_primary_device()]

    # Returns the dtypes the test has requested.
    # Prefers device-specific dtype specifications over generic ones.
    @classmethod
    def _get_dtypes(cls, test):
        if not hasattr(test, 'dtypes'):
            return None
        return test.dtypes.get(cls.device_type, test.dtypes.get('all', None))

    def _get_precision_override(self, test, dtype):
        if not hasattr(test, 'precision_overrides'):
            return self.precision
        return test.precision_overrides.get(dtype, self.precision)

    def _get_tolerance_override(self, test, dtype):
        if not hasattr(test, 'tolerance_overrides'):
            return self.precision, self.rel_tol
        return test.tolerance_overrides.get(dtype, tol(self.precision, self.rel_tol))

    def _apply_precision_override_for_test(self, test, param_kwargs):
        dtype = param_kwargs['dtype'] if 'dtype' in param_kwargs else None
        dtype = param_kwargs['dtypes'] if 'dtypes' in param_kwargs else dtype
        if dtype:
            self.precision = self._get_precision_override(test, dtype)
            self.precision, self.rel_tol = self._get_tolerance_override(test, dtype)

    def _should_stop_test_suite(self):
        if torch.cuda.is_initialized():
            # CUDA device side error will cause subsequence test cases to fail.
            # stop entire test suite if catches RuntimeError during torch.cuda.synchronize().
            try:
                torch.cuda.synchronize()
            except RuntimeError as rte:
                return True
            return False
        else:
            return False

    # Creates device-specific tests.
    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):

        def instantiate_test_helper(cls, name, *, test, param_kwargs=None):
            # Constructs the test
            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                # Add the device param kwarg if the test needs device or devices.
                param_kwargs = {} if param_kwargs is None else param_kwargs
                test_sig_params = inspect.signature(test).parameters
                if 'device' in test_sig_params or 'devices' in test_sig_params:
                    device_arg: str = cls.get_primary_device()
                    if hasattr(test, 'num_required_devices'):
                        device_arg = cls.get_all_devices()
                    _update_param_kwargs(param_kwargs, 'device', device_arg)

                # Sets precision and runs test
                # Note: precision is reset after the test is run
                guard_precision = self.precision
                guard_rel_tol = self.rel_tol
                try:
                    self._apply_precision_override_for_test(test, param_kwargs)
                    result = test(self, **param_kwargs)
                except RuntimeError as rte:
                    # check if rte should stop entire test suite.
                    self._stop_test_suite = self._should_stop_test_suite()
                    # raise the runtime error as is for the test suite to record.
                    raise rte
                finally:
                    self.precision = guard_precision
                    self.rel_tol = guard_rel_tol

                return result

            assert not hasattr(cls, name), "Redefinition of test {0}".format(name)
            setattr(cls, name, instantiated_test)

        # Handles tests that need parametrization (e.g. those that run across a set of
        # ops / modules using the @ops or @modules decorators).

        def default_parametrize_fn(test, generic_cls, cls):
            # By default, parametrize only over device.
            test_suffix = cls.device_type
            yield (test, test_suffix, {})

        parametrize_fn = test.parametrize_fn if hasattr(test, 'parametrize_fn') else default_parametrize_fn
        for (test, test_suffix, param_kwargs) in parametrize_fn(test, generic_cls, cls):
            if hasattr(test, 'handles_dtypes') and test.handles_dtypes:
                full_name = '{}_{}'.format(name, test_suffix)
                instantiate_test_helper(cls=cls, name=full_name, test=test, param_kwargs=param_kwargs)
            else:
                # The parametrize_fn doesn't handle dtypes internally; handle them here instead by generating
                # a test per dtype.
                dtypes = cls._get_dtypes(test)
                dtypes = tuple(dtypes) if dtypes is not None else (None,)
                for dtype in dtypes:
                    all_param_kwargs = dict(param_kwargs)
                    _update_param_kwargs(all_param_kwargs, 'dtype', dtype)
                    full_name = '{}_{}{}'.format(name, test_suffix, _dtype_test_suffix(dtype))
                    instantiate_test_helper(cls=cls, name=full_name, test=test, param_kwargs=all_param_kwargs)

    def run(self, result=None):
        super().run(result=result)
        # Early terminate test if _stop_test_suite is set.
        if self._stop_test_suite:
            result.stop()

class CUDATestBase(DeviceTypeTestBase):
    device_type = 'cuda'
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    primary_device: ClassVar[str]
    cudnn_version: ClassVar[Any]
    no_magma: ClassVar[bool]
    no_cudnn: ClassVar[bool]

    def has_cudnn(self):
        return not self.no_cudnn

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        primary_device_idx = int(cls.get_primary_device().split(':')[1])
        num_devices = torch.cuda.device_count()

        prim_device = cls.get_primary_device()
        cuda_str = 'cuda:{0}'
        non_primary_devices = [cuda_str.format(idx) for idx in range(num_devices) if idx != primary_device_idx]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        # has_magma shows up after cuda is initialized
        t = torch.ones(1).cuda()
        cls.no_magma = not torch.cuda.has_magma

        # Determines if cuDNN is available and its version
        cls.no_cudnn = not torch.backends.cudnn.is_acceptable(t)
        cls.cudnn_version = None if cls.no_cudnn else torch.backends.cudnn.version()

        # Acquires the current device as the primary (test) device
        cls.primary_device = 'cuda:{0}'.format(torch.cuda.current_device())

def get_device_type_test_bases():
    # set type to List[Any] due to mypy list-of-union issue:
    # https://github.com/python/mypy/issues/3351
    test_bases: List[Any] = list()

    if torch.cuda.is_available():
        test_bases.append(CUDATestBase)

    return test_bases


device_type_test_bases = get_device_type_test_bases()

# Decorator that instantiates a variant of the test for each given dtype.
# Notes:
#   (1) Tests that accept the dtype argument MUST use this decorator.
#   (2) Can be overridden for the CPU or CUDA, respectively, using dtypesIfCPU
#       or dtypesIfCUDA.
#   (3) Can accept an iterable of dtypes or an iterable of tuples
#       of dtypes.
# Examples:
# @dtypes(torch.float32, torch.float64)
# @dtypes((torch.long, torch.float32), (torch.int, torch.float64))
class dtypes(object):

    def __init__(self, *args, device_type="all"):
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            for arg in args:
                assert isinstance(arg, (list, tuple)), \
                    "When one dtype variant is a tuple or list, " \
                    "all dtype variants must be. " \
                    "Received non-list non-tuple dtype {0}".format(str(arg))
                assert all(isinstance(dtype, torch.dtype) for dtype in arg), "Unknown dtype in {0}".format(str(arg))
        else:
            assert all(isinstance(arg, torch.dtype) for arg in args), "Unknown dtype in {0}".format(str(args))

        self.args = args
        self.device_type = device_type

    def __call__(self, fn):
        d = getattr(fn, 'dtypes', {})
        assert self.device_type not in d, "dtypes redefinition for {0}".format(self.device_type)
        d[self.device_type] = self.args
        fn.dtypes = d
        return fn

# Overrides specified dtypes on the CPU.
class dtypesIfCPU(dtypes):

    def __init__(self, *args):
        super().__init__(*args, device_type='cpu')


# Overrides specified dtypes on CUDA.
class dtypesIfCUDA(dtypes):

    def __init__(self, *args):
        super().__init__(*args, device_type='cuda')


def onlyCPU(fn):
    return onlyOn('cpu')(fn)


def onlyCUDA(fn):
    return onlyOn('cuda')(fn)

class onlyOn(object):

    def __init__(self, device_type):
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def only_fn(slf, *args, **kwargs):
            if self.device_type != slf.device_type:
                reason = "Only runs on {0}".format(self.device_type)
                raise unittest.SkipTest(reason)
            return fn(slf, *args, **kwargs)

        return only_fn



# The functions below are used for convenience in our test suite and thus have no corresponding C++ dispatch macro

def get_all_dtypes(include_half=True,
                   include_bfloat16=True,
                   include_bool=True,
                   include_complex=True,
                   include_complex32=False
                   ) -> List[torch.dtype]:
    dtypes = get_all_int_dtypes() + get_all_fp_dtypes(include_half=include_half, include_bfloat16=include_bfloat16)
    if include_bool:
        dtypes.append(torch.bool)
    if include_complex:
        dtypes += get_all_complex_dtypes(include_complex32)
    return dtypes

def get_all_math_dtypes(device) -> List[torch.dtype]:
    return get_all_int_dtypes() + get_all_fp_dtypes(include_half=device.startswith('cuda'),
                                                    include_bfloat16=False) + get_all_complex_dtypes()

def get_all_complex_dtypes(include_complex32=False) -> List[torch.dtype]:
    return [torch.complex32, torch.complex64, torch.complex128] if include_complex32 else [torch.complex64, torch.complex128]


def get_all_int_dtypes() -> List[torch.dtype]:
    return [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def get_all_fp_dtypes(include_half=True, include_bfloat16=True) -> List[torch.dtype]:
    dtypes = [torch.float32, torch.float64]
    if include_half:
        dtypes.append(torch.float16)
    if include_bfloat16:
        dtypes.append(torch.bfloat16)
    return dtypes

# Adds 'instantiated' device-specific test cases to the given scope.
# The tests in these test cases are derived from the generic tests in
# generic_test_class.
# See note "Generic Device Type Testing."
def instantiate_device_type_tests(generic_test_class, scope, except_for=None, only_for=None):
    # Removes the generic test class from its enclosing scope so its tests
    # are not discoverable.
    del scope[generic_test_class.__name__]

    # Creates an 'empty' version of the generic_test_class
    # Note: we don't inherit from the generic_test_class directly because
    #   that would add its tests to our test classes and they would be
    #   discovered (despite not being runnable). Inherited methods also
    #   can't be removed later, and we can't rely on load_tests because
    #   pytest doesn't support it (as of this writing).
    empty_name = generic_test_class.__name__ + "_base"
    empty_class = type(empty_name, generic_test_class.__bases__, {})

    # Acquires members names
    # See Note [Overriding methods in generic tests]
    generic_members = set(generic_test_class.__dict__.keys()) - set(empty_class.__dict__.keys())
    generic_tests = [x for x in generic_members if x.startswith('test')]

    # Filter out the device types based on user inputs
    # desired_device_type_test_bases = filter_desired_device_types(device_type_test_bases,
    #                                                              except_for, only_for)
    desired_device_type_test_bases = device_type_test_bases


    def split_if_not_empty(x: str):
        return x.split(",") if len(x) != 0 else []

    # Filter out the device types based on environment variables if available
    # Usage:
    # export PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda,cpu
    # export PYTORCH_TESTING_DEVICE_EXCEPT_FOR=xla
    # env_only_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, ''))
    # env_except_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, ''))

    # desired_device_type_test_bases = filter_desired_device_types(desired_device_type_test_bases,
    #                                                              env_except_for, env_only_for)


    # Creates device-specific test cases
    for base in desired_device_type_test_bases:
        # Special-case for ROCm testing -- only test for 'cuda' i.e. ROCm device by default
        # The except_for and only_for cases were already checked above. At this point we only need to check 'cuda'.
        if base.device_type != 'cuda':
            continue

        class_name = generic_test_class.__name__ + base.device_type.upper()

        # type set to Any and suppressed due to unsupport runtime class:
        # https://github.com/python/mypy/wiki/Unsupported-Python-Features
        device_type_test_class: Any = type(class_name, (base, empty_class), {})

        for name in generic_members:
            if name in generic_tests:  # Instantiates test member
                test = getattr(generic_test_class, name)
                # XLA-compat shim (XLA's instantiate_test takes doesn't take generic_cls)
                sig = inspect.signature(device_type_test_class.instantiate_test)
                if len(sig.parameters) == 3:
                    # Instantiates the device-specific tests
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test), generic_cls=generic_test_class)
                else:
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test))
            else:  # Ports non-test member
                assert name not in device_type_test_class.__dict__, "Redefinition of directly defined member {0}".format(name)
                nontest = getattr(generic_test_class, name)
                setattr(device_type_test_class, name, nontest)

        # Mimics defining the instantiated class in the caller's file
        # by setting its module to the given class's and adding
        # the module to the given scope.
        # This lets the instantiated class be discovered by unittest.
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class


def checkclose(infer_result_data, golden_data, eps=1e-4, err_msg="", compare_type="numpy"):  #type:"torch" or "numpy"
    assert infer_result_data.shape == golden_data.shape, f"the shape of golden and result should be same: golden shape {golden_data.shape} v.s. result shape {infer_result_data.shape}"
    if golden_data.dtype in [torch.bfloat16] or compare_type=="torch":
        status = torch.allclose(infer_result_data.cpu(), golden_data.cpu(), atol=eps, rtol=eps)
        if not status:
            print("golden: ", golden_data.cpu())
            print("output: ", infer_result_data.cpu())
        assert status, err_msg
    else:
        np.testing.assert_allclose(infer_result_data.cpu().detach().numpy(), golden_data.cpu().detach().numpy(), atol=eps, rtol=eps, err_msg=err_msg)


def check_close_relative(infer_result_data, golden_data, eps=1e-3, exact_dtype=True):
    '''
    function to measure the difference between two tensors/ndarrays/ints/dtypes/Size/
    '''
    if type(infer_result_data) in [torch.Tensor, np.ndarray]:
        if isinstance(infer_result_data, torch.Tensor) and infer_result_data.dtype == torch.bfloat16:
            infer_result_data = infer_result_data.to(torch.double).cpu().numpy()
            if isinstance(golden_data, np.ndarray):
                golden_data = golden_data.astype(np.double)
        elif isinstance(infer_result_data, torch.Tensor):
            infer_result_data = infer_result_data.cpu().numpy()
        if isinstance(golden_data, torch.Tensor) and golden_data.dtype == torch.bfloat16:
            golden_data = golden_data.to(torch.double).cpu().numpy()
        elif isinstance(golden_data, torch.Tensor):
            golden_data = golden_data.cpu().numpy()
        if infer_result_data.dtype == np.bool_ or golden_data.dtype == np.bool_:
            return (infer_result_data == golden_data).all()
    elif type(infer_result_data) == tuple and type(golden_data) == torch.Size:
        infer_result_data = torch.Size(infer_result_data)
        return infer_result_data == golden_data
    elif type(infer_result_data) == torch.dtype or type(golden_data) == torch.dtype:
        return infer_result_data == golden_data
    diff = infer_result_data - golden_data
    diff_square = (diff * diff)
    infer_result_square_double = (2 * infer_result_data * infer_result_data)
    sum_diff_square = np.sum(diff_square)
    sum_infer_result_square_double = np.sum(infer_result_square_double)
    result = np.sqrt(sum_diff_square / (sum_infer_result_square_double + 1e-12))
    print("result:", result)
    return result < eps


# make tensor
def make_tensor(
    shape: Union[torch.Size, List[int], Tuple[int, ...]],
    device: Union[str, torch.device],
    dtype: torch.dtype,
    *,
    low: Optional[float] = None,
    high: Optional[float] = None,
    requires_grad: bool = False,
    noncontiguous: bool = False,
    exclude_zero: bool = False
) -> torch.Tensor:
    r"""Creates a tensor with the given :attr:`shape`, :attr:`device`, and :attr:`dtype`, and filled with
    values uniformly drawn from ``[low, high)``.

    If :attr:`low` or :attr:`high` are specified and are outside the range of the :attr:`dtype`'s representable
    finite values then they are clamped to the lowest or highest representable finite value, respectively.
    If ``None``, then the following table describes the default values for :attr:`low` and :attr:`high`,
    which depend on :attr:`dtype`.

    +---------------------------+------------+----------+
    | ``dtype``                 | ``low``    | ``high`` |
    +===========================+============+==========+
    | boolean type              | ``0``      | ``2``    |
    +---------------------------+------------+----------+
    | unsigned integral type    | ``0``      | ``10``   |
    +---------------------------+------------+----------+
    | signed integral types     | ``-9``     | ``10``   |
    +---------------------------+------------+----------+
    | floating types            | ``-9``     | ``9``    |
    +---------------------------+------------+----------+
    | complex types             | ``-9``     | ``9``    |
    +---------------------------+------------+----------+

    Args:
        shape (Tuple[int, ...]): A sequence of integers defining the shape of the output tensor.
        device (Union[str, torch.device]): The device of the returned tensor.
        dtype (:class:`torch.dtype`): The data type of the returned tensor.
        low (Optional[Number]): Sets the lower limit (inclusive) of the given range. If a number is provided it is
            clamped to the least representable finite value of the given dtype. When ``None`` (default),
            this value is determined based on the :attr:`dtype` (see the table above). Default: ``None``.
        high (Optional[Number]): Sets the upper limit (exclusive) of the given range. If a number is provided it is
            clamped to the greatest representable finite value of the given dtype. When ``None`` (default) this value
            is determined based on the :attr:`dtype` (see the table above). Default: ``None``.
        requires_grad (Optional[bool]): If autograd should record operations on the returned tensor. Default: ``False``.
        noncontiguous (Optional[bool]): If `True`, the returned tensor will be noncontiguous. This argument is
            ignored if the constructed tensor has fewer than two elements.
        exclude_zero (Optional[bool]): If ``True`` then zeros are replaced with the dtype's small positive value
            depending on the :attr:`dtype`. For bool and integer types zero is replaced with one. For floating
            point types it is replaced with the dtype's smallest positive normal number (the "tiny" value of the
            :attr:`dtype`'s :func:`~torch.finfo` object), and for complex types it is replaced with a complex number
            whose real and imaginary parts are both the smallest positive normal number representable by the complex
            type. Default ``False``.

    Raises:
        ValueError: If ``low > high``.
        ValueError: If either :attr:`low` or :attr:`high` is ``nan``.
        TypeError: If :attr:`dtype` isn't supported by this function.

    Examples:
        >>> from torch.testing import make_tensor
        >>> # Creates a float tensor with values in [-1, 1)
        >>> make_tensor((3,), device='cpu', dtype=torch.float32, low=-1, high=1)
        tensor([ 0.1205, 0.2282, -0.6380])
        >>> # Creates a bool tensor on CUDA
        >>> make_tensor((2, 2), device='cuda', dtype=torch.bool)
        tensor([[False, False],
                [False, True]], device='cuda:0')
    """
    def _modify_low_high(low, high, lowest, highest, default_low, default_high, dtype):
        """
        Modifies (and raises ValueError when appropriate) low and high values given by the user (input_low, input_high) if required.
        """
        def clamp(a, l, h):
            return min(max(a, l), h)

        low = low if low is not None else default_low
        high = high if high is not None else default_high

        # Checks for error cases
        if low != low or high != high:
            raise ValueError("make_tensor: one of low or high was NaN!")
        if low > high:
            raise ValueError("make_tensor: low must be weakly less than high!")

        low = clamp(low, lowest, highest)
        high = clamp(high, lowest, highest)

        if dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            return math.floor(low), math.ceil(high)

        return low, high

    _integral_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    _floating_types = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    _complex_types = [torch.cfloat, torch.cdouble]

    if dtype is torch.bool:
        result = torch.randint(0, 2, shape, device=device, dtype=dtype)
    elif dtype is torch.uint8:
        ranges = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
        low, high = cast(Tuple[int, int], _modify_low_high(low, high, ranges[0], ranges[1], 0, 10, dtype))
        result = torch.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype in _integral_types:
        ranges = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges[0], ranges[1], -9, 10, dtype)
        result = torch.randint(low, high, shape, device=device, dtype=dtype)  # type: ignore[call-overload]
    elif dtype in _floating_types:
        ranges_floats = (torch.finfo(dtype).min, torch.finfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges_floats[0], ranges_floats[1], -9, 9, dtype)
        rand_val = torch.rand(shape, device=device, dtype=dtype)
        result = high * rand_val + low * (1 - rand_val)
    elif dtype in _complex_types:
        float_dtype = torch.float if dtype is torch.cfloat else torch.double
        ranges_floats = (torch.finfo(float_dtype).min, torch.finfo(float_dtype).max)
        low, high = _modify_low_high(low, high, ranges_floats[0], ranges_floats[1], -9, 9, dtype)
        real_rand_val = torch.rand(shape, device=device, dtype=float_dtype)
        imag_rand_val = torch.rand(shape, device=device, dtype=float_dtype)
        real = high * real_rand_val + low * (1 - real_rand_val)
        imag = high * imag_rand_val + low * (1 - imag_rand_val)
        result = torch.complex(real, imag)
    else:
        raise TypeError(f"The requested dtype '{dtype}' is not supported by torch.testing.make_tensor()."
                        " To request support, file an issue at: https://github.com/pytorch/pytorch/issues")

    if noncontiguous and result.numel() > 1:
        result = torch.repeat_interleave(result, 2, dim=-1)
        result = result[..., ::2]

    if exclude_zero:
        if dtype in _integral_types or dtype is torch.bool:
            replace_with = torch.tensor(1, device=device, dtype=dtype)
        elif dtype in _floating_types:
            replace_with = torch.tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)
        else:  # dtype in _complex_types:
            float_dtype = torch.float if dtype is torch.cfloat else torch.double
            float_eps = torch.tensor(torch.finfo(float_dtype).tiny, device=device, dtype=float_dtype)
            replace_with = torch.complex(float_eps, float_eps)
        result[result == 0] = replace_with

    if dtype in _floating_types + _complex_types:
        result.requires_grad = requires_grad

    return result


def gendata(shape_list, type_list=ALL_FLOATING_TYPES, rand_algo="randn", lower=0.0, upper=0.0, memory_formats=[torch.contiguous_format]):
    r"""generate data according to different parameters
    The output format is a list, [tensor1, tensor2, ...] where data_info is a dict.

    Args:
    shape_list is all the shape needed, e.g. [(1,2), (3,4,5)]
    type_list is all the data type needed, e.g. [torch.float, torch.double]
    rand_algo is used to gen random data, support "randn", "rand", "uniform"
    lower & upper is only needed if rand_algo is "uniform
    """
    output = []
    int_types = ALL_INTEGER_TYPES
    for shape in shape_list:
        for type in type_list:
            if rand_algo is "randn":
                if type in int_types:
                    data = torch.randn(shape).to(type)
                else:
                    data = torch.randn(shape, dtype=type)
            elif rand_algo is "rand":
                if type in int_types:
                    data = torch.randn(shape).to(type)
                else:
                    data = torch.rand(shape, dtype=type)
            elif rand_algo is "randint":
                data = torch.randint(lower, upper, shape).to(type)
            elif rand_algo is "uniform":
                data = torch.rand(shape, dtype=type).uniform_(lower, upper)
            else:
                assert 0, f"invalid rand_algo: {rand_algo} not supported"

            for format in memory_formats:
                data = data.to(memory_format=format)
                output.append(data)
    return output

# recursive sort or copy-to-cpu data(could be torch.Tensor, tuple, list, torch.return_types etc.)
def recursive_sort(data_lst, output_lst, copy_to_cpu=False, dtype=None) -> list:
    if torch.is_tensor(data_lst):
        if copy_to_cpu:
            data_lst = data_lst.cpu()
        if dtype:
            output_lst.append(data_lst.to(dtype))
        else:
            output_lst.append(data_lst)
        return output_lst
    elif type(data_lst)==dict:
        data_lst = list(data_lst.values())
    length = data_lst.__len__()
    for i in range(length):
        output_lst = recursive_sort(data_lst[i], output_lst, copy_to_cpu, dtype)
    return output_lst

def runtest(func, fwd_input_list, bwd_input_list=[], fwd_golden=[],
            bwd_golden_list=[], param_name_list=[], bww_golden_list=[],
            device="cuda", enable_backward=False, input_grad_skip_idx=[],
            fwd_tol=1e-4, bwd_tol=1e-4, bww_tol=1e-4, low=None, high=None, compare_type="numpy"):
    r"""run forward & backward(bwd & bww) on specified device
    return fwd_output, bwd_input, bwd_output, bww_output.
    all the output tensor is in cpu.
    bwd_input is a list, [tensor1], normally its lenght should be 1
    bwd_output is a list, [tensor1, tensor2, ...], its squence is same with fwd_input_list
    bww_output is a list, [tensor1, tensor2, ...], its squence is same with param_name_list

    Args:
    func is the test function, e.g. torch.add, or nn.Linear(32, 64)
    fwd_input_list is all the inputs needed by forward, it's a list or dict, e.g. [data1, data2, ...] or {"data1":data1, "data2":data2, ...}
    bwd_input_list is all the inputs needed by backward, it's a list, e.g. [tensor1)]. If it is None and enable_backward is True, we would gen random data as backward input.
    fwd_golden is forward golden data, if it is not None, we would check tolerance between forward golden and forward output. It is a tensor.
    bwd_golden_list is bwd golden, if it is not None, we would check tolerance between bwd golden and bwd output. It is a tensor list, [tensor1, tensor2, ...].
    param_name_list includes all parameters need calculate backward grad. if it is not NOne, we would calculate all parameters's grad. it is string list, ["weight", "bias", ...]
    bww_golden_list is bww golden, if it is not None, we would check tolerance between bww golden and bww output . It's sequence should be same with param_name_list. It is a tensor list, [weight_grad, bias_grad, ...]
    device set the test run on which device, cpu or gpu
    enable_backward use to set run backward or not
    input_grad_skip_idx is a list which include forward input index you do not want to calucate grad. The index start from 0, and should not large than lenght of fwd_input_list.
    """
    run_bwd = False
    run_bww = False
    check_fwd = False
    check_bwd = False
    check_bww = False
    if enable_backward:
        run_bwd = True
    if param_name_list != []:
        run_bww = True
    if fwd_golden != []:
        check_fwd = True
    if bwd_golden_list != []:
        assert run_bwd, "enable_backward should be set True, or it could not check bww output"
        check_bwd = True
    if bww_golden_list != []:
        assert run_bwd, "enable_backward should be set True, or it could not check bww output"
        assert run_bww, "param_name_list should not be None if you want to check bww golden"
        assert len(bww_golden_list) == len(param_name_list), "param_name_list should have one golden data in bww_golden_list"
        check_bww = True

    if hasattr(func, "to"):
        func = func.to(device, dtype=fwd_input_list[0].dtype)

    # run forward
    fwd_inputs = {} if type(fwd_input_list)==dict else []
    fwd_inputs_info = []    # [{}, {}, ...]
    input_grad_count = 0

    if type(fwd_input_list)==dict:
        for key, input in fwd_input_list.items():
            if torch.is_tensor(input):  # if it is tensor, we should copy before backward
                input_t = input.detach().clone().to(device)
            else:
                input_t = input
            idx = list(fwd_input_list.keys()).index(key)
            if idx not in input_grad_skip_idx:
                if (run_bwd or run_bww) and torch.is_tensor(input_t) and ((input_t.is_floating_point() or input_t.is_complex())):
                    input_t = input_t.requires_grad_(True)
                    input_grad_count += 1
            fwd_inputs[key] = input_t
        fwd_output = func(**fwd_inputs)
    else:
        for idx, input in enumerate(fwd_input_list):
            if torch.is_tensor(input):  # if it is tensor, we should copy before backward
                input_t = input.detach().clone().to(device)
            else:
                input_t = input
            if idx not in input_grad_skip_idx:
                if (run_bwd or run_bww) and torch.is_tensor(input_t) and ((input_t.is_floating_point() or input_t.is_complex())):
                    input_t = input_t.requires_grad_(True)
                    input_grad_count += 1
            fwd_inputs.append(input_t)
        fwd_output = func(*fwd_inputs)

    fwd_output_lst = recursive_sort(fwd_output, [])
    fwd_output_lst_cpu = recursive_sort(fwd_output, [], True)

    # check forward result
    if check_fwd:
        for output,golden_output in zip(fwd_output_lst_cpu, fwd_golden):
            checkclose(output.to(golden_output.dtype), golden_output.cpu(), eps=fwd_tol, compare_type=compare_type)

    bwd_output = []
    bww_output = []
    if not input_grad_count:    # if all input have no grad, no need run backward
        run_bwd = False

    #XXX(yuliu): fwd_output_lst tuple cases only considered maxpoolingwithindices,
    #only consider output's grad!
    # run backward
    if run_bwd:
        if bwd_input_list == []:
            # gen bwd_input_list if bwd_input_list is None
            bwd_input_list = [make_tensor(fwd_output_lst[0].shape, device="cpu", dtype=fwd_output_lst[0].dtype, low=low, high=high)]

        for bwd_input in bwd_input_list:    # normally there should be only one bwd input
            if torch.is_tensor(bwd_input):
                bwd_input = bwd_input.to(device)
            fwd_output_lst[0].backward(bwd_input)

            idx = 0
            for i, fwd_input in enumerate(fwd_inputs):
                if torch.is_tensor(fwd_input)  and fwd_input.is_floating_point() and fwd_input.requires_grad:
                    output = fwd_input.grad.cpu()
                    bwd_output.append(output)
                    # check backward output
                    if check_bwd:
                        assert_info = f"{func} backward, input index: {i}, bwd fail"
                        checkclose(output.to(bwd_golden_list[idx].dtype), bwd_golden_list[idx].cpu(), eps=bwd_tol, err_msg=assert_info, compare_type=compare_type)
                        idx += 1

        # get bww
        if run_bww:
            param_dict = {}
            for name, param in func.named_parameters():
                param_dict[name] = param
            for idx, name in enumerate(param_name_list):
                if hasattr(func, name):
                    grad = param_dict[name].grad.cpu()
                    bww_output.append(grad)
                    # check bww
                    if check_bww:
                        assert_info = f"{func} param index: {idx} bww fail"
                        checkclose(grad.to(bww_golden_list[idx].dtype), bww_golden_list[idx].cpu(), eps=bww_tol, err_msg=assert_info, compare_type=compare_type)
    return fwd_output_lst_cpu, bwd_input_list, bwd_output, bww_output

def runtestapi(func, fwd_input_list, bwd_input_list=[], fwd_golden=[],
            bwd_golden_list=[], param_name_list=[], bww_golden_list=[],
            type_dict={}, enable_backward=False, input_grad_skip_idx=[],
            fwd_tol=1e-4, bwd_tol=1e-4, bww_tol=1e-4, low=None, high=None, compare_type="numpy"):
    r"""run test on gpu and compare result with golden, if golden is [], would gen golden on cpu
    Args:
    type_dict, e.g. {torch.float: {torch.float16, torch.bfloat16}} use to map torch.float16, torch.bfloat16 to torch.float during gen golden on cpu.
    """
    cpu_golden = False
    if fwd_golden == []:
        cpu_golden = True

    src_types = []
    dst_types = []
    for k, v in type_dict.items():
        for i in v:
            src_types.append(i)
            dst_types.append(k)
    assert len(src_types) == len(set(src_types)), "each src type should only have one map to dst type"
    dtype = None
    if cpu_golden:
        if type(fwd_input_list)==dict:
            fwd_input_cpu = {}
            for key, input in fwd_input_list.items():
                if torch.is_tensor(input) and input.dtype in src_types:
                    dtype = input.dtype
                    fwd_input_cpu[key] = input.to(dst_types[src_types.index(dtype)])
                else:
                    fwd_input_cpu[key] = input
        else:
            fwd_input_cpu =[]
            for input in fwd_input_list:
                if torch.is_tensor(input) and input.dtype in src_types:
                    dtype = input.dtype
                    fwd_input_cpu += [input.to(dst_types[src_types.index(dtype)])]
                else:
                    fwd_input_cpu += [input]
        fwd_golden, bwd_input_list, bwd_golden_list, bww_golden_list = runtest(func=func, fwd_input_list=fwd_input_cpu, bwd_input_list=bwd_input_list,
                                                                                param_name_list=param_name_list, device="cpu", enable_backward=enable_backward,
                                                                                input_grad_skip_idx=input_grad_skip_idx, low=low, high=high,compare_type=compare_type)

    if dtype:
        bwd_input_list = [input.to(dtype) for input in bwd_input_list]
    runtest(func=func, fwd_input_list=fwd_input_list, bwd_input_list=bwd_input_list, fwd_golden=fwd_golden,
                bwd_golden_list=bwd_golden_list, param_name_list=param_name_list, bww_golden_list=bww_golden_list,
                device="cuda", enable_backward=enable_backward, input_grad_skip_idx=input_grad_skip_idx,
                fwd_tol=fwd_tol, bwd_tol=bwd_tol, bww_tol=bww_tol, low=low, high=high,compare_type=compare_type)


def gettol(fn_dict, dtype, key):
    r"""get tol value from fn_dict. If the key is not exist, it would return the default value.

    Args:
    fn_dict is a dict. Its format as follow:
    {"fn": torch.erf, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2, torch.float32: 1e-4},
                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2, torch.float32: 1e-4},
                            "bww_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2, torch.float32: 1e-4}},
     "enable_backward":True}
    dtype is data type
    key is the key in "tol"
    """
    val = 1e-4
    if "tol" in fn_dict.keys():
        if key in fn_dict["tol"].keys():
            tols = fn_dict["tol"][key]
            if dtype in tols.keys():
                val = tols[dtype]
    return val

def getval(fn_dict, key, recursive= True, default=None):
    r""" get value of key from fn_dict, if key is not exist, it would return the default value

    Args:
    fn_dict is a dict. Its format as follow:
    {"fn": torch.erf, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2, torch.float32: 1e-4},
                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2, torch.float32: 1e-4},
                            "bww_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2, torch.float32: 1e-4}},
    "enable_backward":True}

    key is the key in fn_dict.
    recursive use to control search key by recursive method or not.
    default use to set default value, which would be return if key is not exist.
    """
    assert isinstance(fn_dict, dict), "fn_dict should be a dict"
    assert key is not None, "key should be not None"
    val = default
    if key in fn_dict.keys():
        val = fn_dict[key]
    else:
        if recursive:
            for v in fn_dict.values():
                if isinstance(v, dict):
                    return getval(v, key, val)
    return val


class _TestParametrizer(object):
    """
    Decorator class for parametrizing a test function, yielding a set of new tests spawned
    from the original generic test, each specialized for a specific set of test inputs. For
    example, parametrizing a test across the set of ops will result in a test function per op.

    The decision of how to parametrize / what to parametrize over is intended to be implemented
    by each derived class.

    In the details, the decorator adds a 'parametrize_fn' property to the test function that is called
    during device-specific test instantiation performed in instantiate_device_type_tests(). Because of this,
    there is no need to parametrize over device type, as that is already handled separately.

    If the decorator is applied to a test function that already has a 'parametrize_fn' property, a new
    composite 'parametrize_fn' will be created that generates tests with the product of the parameters
    generated by the old and new parametrize_fns. This allows for convenient composability of decorators.

    Args:
        handles_dtypes (bool): If True, indicates that it is the responsibility of the decorator to handle
            dtypes internally. This allows for more flexibility when needed (e.g. for op-specific dtype handling).
            Default: True
    """
    def __init__(self, handles_dtypes=True):
        self.handles_dtypes = handles_dtypes

    def _parametrize_test(self, test, generic_cls, device_cls):
        """
        Parametrizes the given test function across whatever dimension is specified by the derived class.
        Tests can be parametrized over any arbitrary dimension or combination of dimensions, such as all
        ops, all modules, or all ops + their associated dtypes.

        Args:
            test (fn): Test function to parametrize over
            generic_cls (class): Generic test class object containing tests (e.g. TestFoo)
            device_cls (class): Device-specialized test class object (e.g. TestFooCPU); set to None
                if the tests are not part of a device-specific set

        Returns:
            Generator object returning 3-tuples of:
                test (fn): Parametrized test function; must support a device arg and args for any params
                test_name (str): Parametrized suffix for the test (e.g. opname_int64); will be appended to
                    the base name of the test
                param_kwargs (dict): Param kwargs to pass to the test (e.g. {'op': 'add', 'dtype': torch.int64})
        """
        raise NotImplementedError

    def __call__(self, fn):
        if hasattr(fn, 'parametrize_fn'):
            # Do composition with the product of args.
            old_parametrize_fn = fn.parametrize_fn
            new_parametrize_fn = self._parametrize_test

            def composite_fn(test, generic_cls, device_cls,
                             old_parametrize_fn=old_parametrize_fn,
                             new_parametrize_fn=new_parametrize_fn):
                old_tests = [(test, test_name, param_kwargs) for (test, test_name, param_kwargs) in
                             old_parametrize_fn(test, generic_cls, device_cls)]
                for (old_test, old_test_name, old_param_kwargs) in old_tests:
                    for (new_test, new_test_name, new_param_kwargs) in \
                            new_parametrize_fn(old_test, generic_cls, device_cls):
                        full_param_kwargs = {**old_param_kwargs, **new_param_kwargs}
                        yield (new_test, '{}_{}'.format(new_test_name, old_test_name), full_param_kwargs)

            fn.parametrize_fn = composite_fn
            old_handles_dtypes = fn.handles_dtypes if hasattr(fn, 'handles_dtypes') else False
            if self.handles_dtypes and old_handles_dtypes:
                raise RuntimeError('Cannot compose multiple parametrization decorators that handle dtypes; '
                                   'their dtype handling conflicts')
            fn.handles_dtypes = self.handles_dtypes or old_handles_dtypes
        else:
            fn.parametrize_fn = self._parametrize_test
            fn.handles_dtypes = self.handles_dtypes
        return fn

class subtest(object):
    """
    Explicit subtest case for use with test parametrization.
    Allows for explicit naming of individual subtest cases as well as applying
    decorators to the parametrized test.

    Args:
        arg_values (iterable): Iterable of arg values (e.g. range(10)) or
            tuples of arg values (e.g. [(1, 2), (3, 4)]).
        name (str): Optional name to use for the test.
        decorators (iterable): Iterable of decorators to apply to the generated test.
    """
    __slots__ = ['arg_values', 'name', 'decorators']

    def __init__(self, arg_values, name=None, decorators=None):
        self.arg_values = arg_values
        self.name = name
        self.decorators = decorators if decorators else []


class parametrize(_TestParametrizer):
    """
    Decorator for applying generic test parametrizations.

    The interface for this decorator is modeled after `@pytest.mark.parametrize`.
    Basic usage between this decorator and pytest's is identical. The first argument
    should be a string containing comma-separated names of parameters for the test, and
    the second argument should be an iterable returning values or tuples of values for
    the case of multiple parameters.

    Beyond this basic usage, the decorator provides some additional functionality that
    pytest does not.

    1. Parametrized tests end up as generated test functions on unittest test classes.
    Since this differs from how pytest works, this decorator takes on the additional
    responsibility of naming these test functions. The default test names consists of
    the test's base name followed by each parameter name + value (e.g. "test_bar_x_1_y_foo"),
    but custom names can be defined using `name_fn` or the `subtest` structure (see below).

    2. The decorator specially handles parameter values of type `subtest`, which allows for
    more fine-grained control over both test naming and test execution. In particular, it can
    be used to tag subtests with explicit test names or apply arbitrary decorators (see examples
    below).

    Examples::

        @parametrize("x", range(5))
        def test_foo(self, x):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')])
        def test_bar(self, x, y):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')],
                     name_fn=lambda x, y: '{}_{}'.format(x, y))
        def test_bar_custom_names(self, x, y):
            ...

        @parametrize("x, y", [subtest((1, 2), name='double'),
                              subtest((1, 3), name='triple', decorators=[unittest.expectedFailure]),
                              subtest((1, 4), name='quadruple')])
        def test_baz(self, x, y):
            ...

    Args:
        arg_str (str): String of arg names separate by commas (e.g. "x,y").
        arg_values (iterable): Iterable of arg values (e.g. range(10)) or
            tuples of arg values (e.g. [(1, 2), (3, 4)]).
        name_fn (callable): Optional function that takes in parameters and returns subtest name.
    """
    def __init__(self, arg_str, arg_values, name_fn=None):
        super().__init__(handles_dtypes=False)
        self.arg_names = arg_str.split(',')
        self.arg_values = arg_values
        self.name_fn = name_fn

    def _formatted_str_repr(self, name, value):
        """ Returns a string representation for the given arg that is suitable for use in test function names. """
        if isinstance(value, torch.dtype):
            return dtype_name(value)
        elif isinstance(value, torch.device):
            return str(value)
        # Can't use isinstance as it would cause a circular import
        elif value.__class__.__name__ == 'OpInfo' or value.__class__.__name__ == 'ModuleInfo':
            return value.formatted_name
        else:
            # Include name and value separated by underscore.
            return '{}_{}'.format(name, str(value).replace('.', '_'))

    def _default_subtest_name(self, values):
        return '_'.join([self._formatted_str_repr(a, v) for a, v in zip(self.arg_names, values)])

    def _get_subtest_name(self, values, keys=None, explicit_name=None):
        if explicit_name:
            subtest_name = explicit_name
        elif self.name_fn:
            if len(keys) >= 1:
                keys_str = ""
                for key in keys:
                    keys_str += key + "_"
                subtest_name = keys_str + self.name_fn(*values)
            else:
                subtest_name = self.name_fn(*values)
        else:
            subtest_name = self._default_subtest_name(values)
        return subtest_name

    def _parametrize_test(self, test, generic_cls, device_cls):
        if len(self.arg_names) == 0:
            # No additional parameters needed for the test.
            test_name = device_cls.device_type if device_cls else ''
            yield (test, test_name, {})
        else:
            # Each "values" item is expected to be either:
            # * A tuple of values with one for each arg. For a single arg, a single item is expected.
            # * A subtest instance with arg_values matching the previous.
            for values in self.arg_values:
                maybe_name = None
                if isinstance(values, subtest):
                    sub = values
                    values = sub.arg_values
                    maybe_name = sub.name

                    # Apply decorators.
                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    for decorator in sub.decorators:
                        test_wrapper = decorator(test_wrapper)

                    gen_test = test_wrapper
                else:
                    gen_test = test

                values = list(values) if len(self.arg_names) > 1 else [values]
                if len(values) != len(self.arg_names):
                    raise RuntimeError('Expected # values == # arg names, but got: {} '
                                       'values and {} names for test "{}"'.format(
                                           len(values), len(self.arg_names), test.__name__))

                param_kwargs = {
                    name: value for name, value in zip(self.arg_names, values)
                }

                subtest_name = self._get_subtest_name(values, keys=self.arg_names, explicit_name=maybe_name)
                test_name = '{}{}'.format(subtest_name, '_' + device_cls.device_type if device_cls else '')
                # if '.' in test_name:
                #     raise RuntimeError('Test name cannot contain periods, but got: {}'.format(test_name))

                yield (gen_test, test_name, param_kwargs)

# Specifies per-dtype precision overrides.
# Ex.
#
# @precisionOverride({torch.half : 1e-2, torch.float : 1e-4})
# @dtypes(torch.half, torch.float, torch.double)
# def test_X(self, device, dtype):
#   ...
#
# When the test is instantiated its class's precision will be set to the
# corresponding override, if it exists.
# self.precision can be accessed directly, and it also controls the behavior of
# functions like self.assertEqual().
#
# Note that self.precision is a scalar value, so if you require multiple
# precisions (or are working with multiple dtypes) they should be specified
# explicitly and computed using self.precision (e.g.
# self.precision *2, max(1, self.precision)).
class precisionOverride(object):

    def __init__(self, d):
        assert isinstance(d, dict), "precisionOverride not given a dtype : precision dict!"
        for dtype, prec in d.items():
            assert isinstance(dtype, torch.dtype), "precisionOverride given unknown dtype {0}".format(dtype)

        self.d = d

    def __call__(self, fn):
        fn.precision_overrides = self.d
        return fn

# Specifies per-dtype tolerance overrides tol(atol, rtol). It has priority over
# precisionOverride.
# Ex.
#
# @toleranceOverride({torch.float : tol(atol=1e-2, rtol=1e-3},
#                     torch.double : tol{atol=1e-4, rtol = 0})
# @dtypes(torch.half, torch.float, torch.double)
# def test_X(self, device, dtype):
#   ...
#
# When the test is instantiated its class's tolerance will be set to the
# corresponding override, if it exists.
# self.rtol and self.precision can be accessed directly, and they also control
# the behavior of functions like self.assertEqual().
#
# The above example sets atol = 1e-2 and rtol = 1e-3 for torch.float and
# atol = 1e-4 and rtol = 0 for torch.double.
tol = namedtuple('tol', ['atol', 'rtol'])

class toleranceOverride(object):
    def __init__(self, d):
        assert isinstance(d, dict), "toleranceOverride not given a dtype : tol dict!"
        for dtype, prec in d.items():
            assert isinstance(dtype, torch.dtype), "toleranceOverride given unknown dtype {0}".format(dtype)
            assert isinstance(prec, tol), "toleranceOverride not given a dtype : tol dict!"

        self.d = d

    def __call__(self, fn):
        fn.tolerance_overrides = self.d
        return fn

@contextlib.contextmanager
def cudnn_off_helper():
    try:
        with torch.backends.cudnn.flags(
                enabled=False,
                benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic,
                allow_tf32=torch.backends.cudnn.allow_tf32):
            yield
    finally:
        pass

def with_cudnn_off_helper(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        with cudnn_off_helper():
            return f(*args, **kwargs)

    return wrapped

@contextlib.contextmanager
def tf32_off_helper():
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic,
                allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul

def with_tf32_off_helper(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        with tf32_off_helper():
            return f(*args, **kwargs)

    return wrapped