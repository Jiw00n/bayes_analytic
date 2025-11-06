# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Initializer of parameters."""
from functools import reduce
import numpy as np

import tvm
from tvm import relay


class Initializer(object):
    """The base class of an initializer."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __call__(self, desc, arr):
        """Initialize an array

        Parameters
        ----------
        desc : str
            Initialization pattern descriptor.

        arr : NDArray
            The array to be initialized.
        """
        if desc.endswith("weight"):
            self._init_weight(desc, arr)
        elif desc.endswith("bias"):
            self._init_bias(desc, arr)
        elif desc.endswith("gamma"):
            self._init_gamma(desc, arr)
        elif desc.endswith("beta"):
            self._init_beta(desc, arr)
        elif desc.endswith("mean"):
            self._init_mean(desc, arr)
        elif desc.endswith("var"):
            self._init_var(desc, arr)
        else:
            self._init_default(desc, arr)

    def _init_bias(self, _, arr):
        arr[:] = 0.0

    def _init_gamma(self, _, arr):
        arr[:] = 1.0

    def _init_beta(self, _, arr):
        arr[:] = 0.0

    def _init_mean(self, _, arr):
        arr[:] = 0.0

    def _init_var(self, _, arr):
        arr[:] = 1.0

    def _init_weight(self, name, arr):
        """Abstract method to Initialize weight."""
        raise NotImplementedError("Must override it")

    def _init_default(self, name, _):
        raise ValueError(
            f"Unknown initialization pattern for {name}. "
            f"Default initialization is now limited to "
            f'"weight", "bias", "gamma" (1.0), and "beta" (0.0).'
            f"Please use mx.sym.Variable(init=mx.init.*) to set initialization pattern"
        )


class Xavier(Initializer):
    """ "Xavier" initialization for weights

    Parameters
    ----------
    rnd_type: str, optional
        Random generator type, can be ``'gaussian'`` or ``'uniform'``.

    factor_type: str, optional
        Can be ``'avg'``, ``'in'``, or ``'out'``.

    magnitude: float, optional
        Scale of random number.
    """

    def __init__(self, rnd_type="uniform", factor_type="avg", magnitude=3):
        super(Xavier, self).__init__(
            rnd_type=rnd_type, factor_type=factor_type, magnitude=magnitude
        )
        self.rnd_type = rnd_type
        self.factor_type = factor_type
        self.magnitude = float(magnitude)

    def _init_weight(self, name, arr):
        shape = arr.shape
        hw_scale = 1.0
        if len(shape) < 2:
            raise ValueError(
                f"Xavier initializer cannot be applied to vector {name}. It requires at least 2D."
            )
        if len(shape) > 2:
            hw_scale = np.prod(shape[2:])
        fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
        factor = 1.0
        if self.factor_type == "avg":
            factor = (fan_in + fan_out) / 2.0
        elif self.factor_type == "in":
            factor = fan_in
        elif self.factor_type == "out":
            factor = fan_out
        else:
            raise ValueError("Incorrect factor type")
        # Hack for mobilenet, because there is less connectivity
        if "depthwise" in name:
            factor = hw_scale
        scale = np.sqrt(self.magnitude / factor)
        if self.rnd_type == "uniform":
            arr[:] = np.random.uniform(-scale, scale, size=arr.shape)
        else:
            raise ValueError("Unknown random type")


class Constant(Initializer):
    """Constant initialization of weights. Sum of weights in the matrix is 1."""

    def _init_weight(self, name, arr):
        num_elements = reduce(lambda x, y: x * y, arr.shape)
        arr[:] = 1.0 / num_elements


def create_workload(net, single=False, initializer=None, seed=0):
    """Helper function to create benchmark image classification workload.

    Parameters
    ----------
    net : tvm.relay.Function
        The selected function of the network.

    initializer : Initializer
        The initializer used

    seed : int
        The seed used in initialization.

    Returns
    -------
    mod : tvm.IRModule
        The created relay module.

    params : dict of str to NDArray
        The parameters.
    """
    mod = tvm.IRModule.from_expr(net)
    mod = relay.transform.InferType()(mod)
    if single is True:
        mod = inner_convert(net)
    # breakpoint()
    shape_dict = {v.name_hint: v.checked_type for v in mod["main"].params}
    np.random.seed(seed)
    initializer = initializer if initializer else Xavier()
    params = {}
    for k, v in shape_dict.items():
        if k == "data":
            continue
        init_value = np.zeros(v.concrete_shape).astype(v.dtype)
        initializer(k, init_value)
        params[k] = tvm.nd.array(init_value, device=tvm.cpu(0))
    return mod, params


def inner_convert(net):
    inner_gv = relay.GlobalVar("inner")
    mod = tvm.IRModule({inner_gv: net})

    # 2) 타입 정보를 먼저 채운다 (checked_type을 얻기 위함)
    mod = relay.transform.InferType()(mod)
    inner_ty = mod[inner_gv].checked_type  # relay.FuncType

    # 3) inner의 파라미터 타입을 바탕으로 main의 파라미터를 새로 만든다
    #    (이때 name_hint는 유지해서 가독성을 살릴 수 있다)
    new_params = []
    for i, (p, ty) in enumerate(zip(mod[inner_gv].params, inner_ty.arg_types)):
        name = p.name_hint if hasattr(p, "name_hint") else f"p{i}"
        new_params.append(relay.var(name, type_annotation=ty))

    # 4) main은 단순히 inner를 호출만 한다
    call_inner = relay.Call(inner_gv, new_params)
    main = relay.Function(new_params, call_inner)

    # 5) 모듈에 main을 등록하고 다시 타입 정리
    mod.update_func(relay.GlobalVar("main"), main)
    mod = relay.transform.InferType()(mod)
    return mod