from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes
        self.M = FileNotFoundError

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        self.M = array_api.max(Z, axis=self.axes, keepdims=True)
        sum_exp_sub_Z = array_api.sum(array_api.exp(Z - self.M), axis=self.axes)
        log_sum_exp_sub_Z = array_api.log(sum_exp_sub_Z)
        return log_sum_exp_sub_Z + array_api.squeeze(self.M, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        lhs_shape_len = len(Z.shape)
        axes = self.axes
        if axes is None:
            axes = tuple(range(lhs_shape_len))
        shape = tuple([1 if i in axes else Z.shape[i] for i in range(lhs_shape_len)])    
        
        exp_Z = exp(Z - self.M)
        sum_exp = summation(exp_Z, axes=self.axes).reshape(shape).broadcast_to(Z.shape)
        softmax = exp_Z / sum_exp
        return (out_grad.reshape(shape).broadcast_to(Z.shape) * softmax, )
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)