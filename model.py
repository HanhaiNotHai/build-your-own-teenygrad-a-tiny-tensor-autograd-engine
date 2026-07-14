"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
from functools import reduce
from operator import mul


def prod(shape: tuple[int, ...]):
    '''Multiply together the elements of a shape tuple to get the total number of elements.'''

    return reduce(mul, shape, 1)

# Step 2 - argsort
def argsort(values: list[int]):
    '''Return the indices that would sort values in ascending order.'''

    return sorted(range(len(values)), key=lambda i: values[i])

# Step 3 - make_op_enums
from enum import Enum

UnaryOps = Enum('UnaryOps', ['NEG', 'RELU', 'LOG', 'EXP', 'SQRT', 'SIGMOID'])
BinaryOps = Enum('BinaryOps', ['ADD', 'SUB', 'MUL', 'DIV', 'CMPLT', 'MAX'])
ReduceOps = Enum('ReduceOps', ['SUM', 'MAX'])
MovementOps = Enum('MovementOps', ['RESHAPE', 'EXPAND', 'PERMUTE'])


def make_op_enums():
    '''create four enum classes naming every supported operation kind'''

    return UnaryOps, BinaryOps, ReduceOps, MovementOps

# Step 4 - LazyBuffer
import numpy as np


class LazyBuffer:
    def __init__(self, np_array):
        '''wrap np_array as an ndarray and expose shape and dtype'''

        self._np = np.asarray(np_array)
        self.shape = tuple(int(d) for d in self._np.shape)
        self.dtype = self._np.dtype

    def __array__(self, dtype=None):
        return np.asarray(self._np, dtype)

    def __float__(self):
        return float(self._np)

    def __repr__(self):
        return self._np.__repr__()

    def __str__(self):
        return self._np.__str__()

# Step 5 - lazybuffer_const
def const(value: float, shape: tuple[int, ...]):
    '''Create a new LazyBuffer of the given shape filled with a constant value.'''

    return LazyBuffer(np.full(shape, value, dtype=np.float32))


LazyBuffer.const = staticmethod(const)

# Step 6 - rand
def rand(shape: tuple[int, ...], seed=None):
    '''return a LazyBuffer of uniform random floats in [0, 1) with given shape'''

    return LazyBuffer(np.random.RandomState(seed).random(shape).astype(np.float32))

# Step 7 - lazybuffer_unary_e
def e(self: LazyBuffer, op: UnaryOps):
    '''apply a unary elementwise op (NEG, RELU, LOG, EXP, SQRT, SIGMOID)'''

    x = self._np

    if op is UnaryOps.NEG:
        out = -x
    elif op is UnaryOps.RELU:
        out = np.maximum(x, 0)
    elif op is UnaryOps.LOG:
        out = np.log(x)
    elif op is UnaryOps.EXP:
        out = np.exp(x)
    elif op is UnaryOps.SQRT:
        out = np.sqrt(x)
    elif op is UnaryOps.SIGMOID:
        out = 1.0 / (1.0 + np.exp(-x))
    else:
        raise ValueError(...)

    return LazyBuffer(out)


LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
def lazybuffer_binary_e(self: LazyBuffer, op: BinaryOps, other: LazyBuffer):
    '''apply a binary elementwise op between two LazyBuffers, return a new LazyBuffer'''

    a = self._np
    b = other._np

    if op is BinaryOps.ADD:
        out = a + b
    elif op is BinaryOps.SUB:
        out = a - b
    elif op is BinaryOps.MUL:
        out = a * b
    elif op is BinaryOps.DIV:
        out = a / b
    elif op is BinaryOps.CMPLT:
        out = (a < b).astype(a.dtype)
    elif op is BinaryOps.MAX:
        out = np.maximum(a, b)
    else:
        raise ValueError

    return LazyBuffer(out)

# Step 9 - lazybuffer_r
def r(self: LazyBuffer, op: ReduceOps, axis: int | tuple[int, ...] | None = None):
    '''reduce the underlying array along axis (SUM or MAX), keeping reduced dims as size 1'''

    x = self._np

    if op is ReduceOps.SUM:
        out = x.sum(axis, keepdims=True)
    elif op is ReduceOps.MAX:
        out = x.max(axis, keepdims=True)
    else:
        raise ValueError

    return LazyBuffer(out)

# Step 10 - lazybuffer_reshape
def reshape(self: LazyBuffer, new_shape: tuple[int, ...]):
    '''return a new LazyBuffer with the array reshaped to new_shape'''

    return LazyBuffer(self._np.reshape(new_shape))

# Step 11 - lazybuffer_expand
def expand(self: LazyBuffer, new_shape: tuple[int, ...]):
    '''broadcast this buffer's size-1 dims out to new_shape'''

    return LazyBuffer(np.array(np.broadcast_to(self._np, tuple(int(d) for d in new_shape))))

# Step 12 - lazybuffer_permute
def permute(self: LazyBuffer, order: tuple[int, ...]):
    '''return a new LazyBuffer with axes reordered according to order'''

    return LazyBuffer(self._np.transpose(order))

# Step 13 - Function
class Function:
    def __init__(self, *tensors):
        '''record needs_input_grad, requires_grad, and parents for backprop'''

        self.needs_input_grad = [t.requires_grad for t in tensors]
        self.requires_grad = any(self.needs_input_grad) or (
            None if None in self.needs_input_grad else False
        )
        if self.requires_grad:
            self.parents = tensors

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    '''attach forward and backward stubs to Function that raise NotImplementedError'''

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self).__name__}")

    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self).__name__}")

    Function.forward = forward
    Function.backward = backward

    return Function

# Step 15 - apply
@classmethod
def apply(cls: type[Function], *tensors, **kwargs):
    '''build the Function, run forward on the input buffers, wrap in a
    Tensor, and link out._ctx when a gradient is needed.
    '''

    ctx = cls(*tensors)

    bufs = [t.lazydata for t in tensors]
    out_buf = ctx.forward(*bufs, **kwargs)

    result = Tensor(out_buf, requires_grad=ctx.requires_grad)

    if ctx.requires_grad:
        result._ctx = ctx

    return result


# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg
class Neg(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        '''return a LazyBuffer holding the elementwise negation of x'''

        return x.e(UnaryOps.NEG)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        '''return the negated incoming gradient'''

        return grad_output.e(UnaryOps.NEG)

# Step 17 - Relu
class Relu(Function):
    def forward(self, x: LazyBuffer):
        '''apply the rectified linear unit to lazy buffer x and cache the result'''

        self.y: LazyBuffer = x.e(UnaryOps.RELU)
        return self.y

    def backward(self, grad_output: LazyBuffer):
        '''route the upstream gradient only through positions that were positive'''

        zero: LazyBuffer = LazyBuffer.const(0, self.y.shape)
        mask = lazybuffer_binary_e(zero, BinaryOps.CMPLT, self.y)
        return lazybuffer_binary_e(mask, BinaryOps.MUL, grad_output)

# Step 18 - Log
class Log(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        '''return the natural log of x and save x for backward'''

        self.x = x
        return x.e(UnaryOps.LOG)

    def backward(self, grad_output: LazyBuffer):
        '''return the gradient of log with respect to its input'''

        return lazybuffer_binary_e(grad_output, BinaryOps.DIV, self.x)

# Step 19 - Exp
class Exp(Function):
    def forward(self, x: LazyBuffer):
        '''compute the elementwise exponential and keep what backward needs'''

        self.y: LazyBuffer = x.e(UnaryOps.EXP)
        return self.y

    def backward(self, grad_output: LazyBuffer):
        '''turn the upstream gradient into the gradient w.r.t. the input'''

        return lazybuffer_binary_e(grad_output, BinaryOps.MUL, self.y)

# Step 20 - Sqrt
class Sqrt(Function):
    def forward(self, x: LazyBuffer):
        '''compute the elementwise square root and cache it for backward'''

        self.y: LazyBuffer = x.e(UnaryOps.SQRT)
        return self.y

    def backward(self, grad_output: LazyBuffer):

        two = LazyBuffer.const(2, self.y.shape)
        two_mul_y = lazybuffer_binary_e(two, BinaryOps.MUL, self.y)
        return lazybuffer_binary_e(grad_output, BinaryOps.DIV, two_mul_y)

# Step 21 - Sigmoid
class Sigmoid(Function):
    def forward(self, x: LazyBuffer):
        '''return the elementwise logistic activation of LazyBuffer x'''

        self.y: LazyBuffer = x.e(UnaryOps.SIGMOID)
        return self.y

    def backward(self, grad_output: LazyBuffer):
        '''return grad_output times the sigmoid derivative'''

        one = LazyBuffer.const(1, self.y.shape)
        one_sub_y = lazybuffer_binary_e(one, BinaryOps.SUB, self.y)
        y_mul_one_sub_y = lazybuffer_binary_e(self.y, BinaryOps.MUL, one_sub_y)
        return lazybuffer_binary_e(grad_output, BinaryOps.MUL, y_mul_one_sub_y)

# Step 22 - Add
class Add(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        '''return the elementwise sum of LazyBuffers x and y'''

        return lazybuffer_binary_e(x, BinaryOps.ADD, y)

    def backward(self, grad_output: LazyBuffer):
        '''route grad_output to each input that requires a gradient'''

        gx = grad_output if self.needs_input_grad[0] else None
        gy = grad_output if self.needs_input_grad[1] else None
        return gx, gy

# Step 23 - Sub
class Sub(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        '''return the elementwise difference x - y as a LazyBuffer'''

        return lazybuffer_binary_e(x, BinaryOps.SUB, y)

    def backward(self, grad_output: LazyBuffer):
        '''return gradients for x and y (None where grad is not needed)'''

        gx: LazyBuffer | None = grad_output if self.needs_input_grad[0] else None
        gy: LazyBuffer | None = grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None
        return gx, gy

# Step 24 - Mul
class Mul(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        '''compute the elementwise product and save what backward needs'''

        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.MUL, y)

    def backward(self, grad_output: LazyBuffer):
        '''return the gradient w.r.t. each input (None if not needed)'''

        gx = (
            lazybuffer_binary_e(grad_output, BinaryOps.MUL, self.y)
            if self.needs_input_grad[0]
            else None
        )
        gy = (
            lazybuffer_binary_e(grad_output, BinaryOps.MUL, self.x)
            if self.needs_input_grad[1]
            else None
        )
        return gx, gy

# Step 25 - Div
class Div(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        '''divide LazyBuffer x by y and cache inputs for backward'''

        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.DIV, y)

    def backward(self, grad_output: LazyBuffer):
        '''return gradients w.r.t. x and y via the quotient rule'''

        gx = (
            lazybuffer_binary_e(grad_output, BinaryOps.DIV, self.y)
            if self.needs_input_grad[0]
            else None
        )

        if self.needs_input_grad[1]:
            y2 = lazybuffer_binary_e(self.y, BinaryOps.MUL, self.y)
            x_div_y2 = lazybuffer_binary_e(self.x, BinaryOps.DIV, y2)
            gy: LazyBuffer = lazybuffer_binary_e(grad_output, BinaryOps.MUL, x_div_y2)
            gy = gy.e(UnaryOps.NEG)
        else:
            gy = None

        return gx, gy

# Step 26 - sum_function_forward
class Sum(Function):
    def forward(self, x: LazyBuffer, axis: int | tuple[int, ...] | None = None):
        '''Reduce x with ReduceOps.SUM over axis (keepdims) and cache shape/axis.'''

        self.input_shape = x.shape
        self.axis = axis
        return r(x, ReduceOps.SUM, axis)

# Step 27 - sum_function_backward
def backward(self: Sum, grad_output: LazyBuffer):
    '''broadcast the summed gradient back to the original input shape'''

    return expand(grad_output, self.input_shape)

# Step 28 - max_function_forward
class Max(Function):
    def forward(self, x: LazyBuffer, axis: int | tuple[int, ...] | None = None):
        '''reduce x with the MAX reduce op along axis and cache for backward'''

        self.x = x
        self.axis = axis
        self.y = r(x, ReduceOps.MAX, axis)
        return self.y

# Step 29 - max_function_backward
def backward(self: Max, grad_output: LazyBuffer):
    '''route grad_output back to the input elements that were the maximum'''

    y = expand(self.y, self.x.shape)

    x_lt_y = lazybuffer_binary_e(self.x, BinaryOps.CMPLT, y)
    one: LazyBuffer = LazyBuffer.const(1, y.shape)
    max_is_1s = lazybuffer_binary_e(one, BinaryOps.SUB, x_lt_y)

    k = r(max_is_1s, ReduceOps.SUM, self.axis)
    k = expand(k, y.shape)

    max_is_1s = lazybuffer_binary_e(max_is_1s, BinaryOps.DIV, k)

    g = expand(grad_output, y.shape)

    return lazybuffer_binary_e(max_is_1s, BinaryOps.MUL, g)


Max.backward = backward

# Step 30 - Reshape
class Reshape(Function):
    def forward(self, x: LazyBuffer, shape: tuple[int, ...]):
        '''cache the input shape and return x reshaped to shape'''

        self.input_shape = x.shape
        return reshape(x, shape)

    def backward(self, grad_output: LazyBuffer):
        '''reshape the gradient back to the cached input shape'''

        return reshape(grad_output, self.input_shape)

# Step 31 - expand_function_forward
def expand_function_forward(ctx, x: LazyBuffer, shape: tuple[int, ...]):
    '''cache x.shape on ctx, then broadcast x to the target shape'''

    ctx.input_shape = x.shape
    return expand(x, shape)

# Step 32 - expand_function_backward
def expand_function_backward(ctx, grad_output: LazyBuffer):
    '''Sum grad_output over the broadcast axes back to ctx.input_shape...'''

    axis = tuple(
        i
        for i, (di, dg) in enumerate(zip(ctx.input_shape, grad_output.shape))
        if di == 1 and dg != 1
    )
    return r(grad_output, ReduceOps.SUM, axis)

# Step 33 - permute_function_forward_backward
def permute_function_forward_backward():
    '''return (forward, backward); forward reorders axes, backward inverts the order'''

    def permute_function_forward(ctx, x: LazyBuffer, order: tuple[int, ...]):
        ctx.order = order
        return permute(x, order)

    def permute_function_backward(ctx, grad_output: LazyBuffer):
        return permute(grad_output, argsort(ctx.order))

    return permute_function_forward, permute_function_backward

# Step 34 - Tensor
class Tensor:
    def __init__(self, data, requires_grad: bool = False):
        '''wrap data in a LazyBuffer and store grad/ctx bookkeeping'''

        self.lazydata = (
            data
            if isinstance(data, LazyBuffer)
            else LazyBuffer(np.asarray(data, dtype=np.float32))
        )
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None

    @property
    def data(self):
        '''return the underlying LazyBuffer'''

        return self.lazydata

    @data.setter
    def data(self, value):
        '''replace the underlying LazyBuffer'''

        self.lazydata = value

    @property
    def shape(self):
        return self.lazydata.shape

    @property
    def dtype(self):
        return self.lazydata.dtype

    def numpy(self):
        return self.lazydata._np

    def __repr__(self):
        return f'Tensor(shape={self.shape}, requires_grad={self.requires_grad})'

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

