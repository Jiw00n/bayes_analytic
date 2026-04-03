import os
from datetime import datetime
import pickle
import tvm
from tvm.relay import testing
from tvm.relay.testing.init import create_workload
from tvm import relay
import json
from tvm import auto_scheduler
import re
import random
import numpy as np
import torch
import time




def get_network(name, batch_size, layout="NHWC", dtype="float32", num_classes=1000):
    """Get the symbol definition and random weight of a network"""

    print(f"Getting network {name}...\n")

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, num_classes)

    if name.startswith("resnet_"):
        n_layer = int(name.split("_")[1])
        mod, params = testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )

    elif name.startswith("tiny_conv"):
        layer_n = int(name.split("_")[-1])
        net = testing.resnet.tiny_convnet_nhwc(layer_n=layer_n, batch_size=batch_size, num_classes=num_classes)        
        mod, params = create_workload(net)

    elif name.startswith("tiny_res"):
        net = testing.resnet.tiny_resnet_nhwc()
        mod, params = create_workload(net)

    elif name.startswith("resnet3d_"):
        n_layer = int(name.split("_")[1])
        mod, params = testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)

    return mod, params, input_shape, output_shape


def get_tasks(mod, params, network_name, input_shape, target, get_pkl=True):
    # breakpoint()
    input_shape = str(input_shape).replace(" ", "")
    network_pkl = f"/root/work/tenset/scripts/ansor_tasks_pkl/{network_name}-{input_shape}.pkl"

    if os.path.exists(network_pkl):
        with open(network_pkl, "rb") as f:
            tasks, task_weights = pickle.load(f)
    else:
        os.makedirs(os.path.dirname(network_pkl), exist_ok=True)
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        with open(network_pkl, "wb") as f:
            pickle.dump((tasks, task_weights), f)
    
    print(f"Total tasks length : {len(tasks)}")
    return tasks, task_weights


def get_arg(parser):
    
    parser.add_argument("--network", type=str, help="Name of the network", default="resnet_18")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=1)
    parser.add_argument("--layout", type=str, help="Layout of the input data", default="NHWC")
    parser.add_argument("--timenow", type=str, help="Time string to identify the log file")
    parser.add_argument("--num-trials", type=str, help="Time string to identify the log file", default="2000")
    parser.add_argument("--json", type=str, help="Time string to identify the log file")
    
    args = parser.parse_args()

    return args





class PathManager:
    def __init__(self, network, input_shape, args, gdb_mode=None, json=None):
        self.tvm_home = os.environ.get("TVM_HOME")
        self.gallery_path = f"{self.tvm_home}/gallery"

        self.network = network
        self.inp_shape_str = str(input_shape).replace(" ", "")
        self.json = json

        self.get_dirs()
        self.get_paths(json, args)




    def get_dirs(self):
        self.dirs = {}
        dirs = self.dirs

        

        dirs["json"] = f"{self.gallery_path}/logs_json/{self.network}"
        dirs["tsv"] = f"{self.gallery_path}/logs_tsv/{self.network}"
        os.makedirs(dirs["json"], exist_ok=True)
        os.makedirs(dirs["tsv"], exist_ok=True)



    def get_paths(self, json, args):
        self.paths = {}
        paths = self.paths

        json_tmp = f'{self.dirs["json"]}/{self.inp_shape_str}'
        tsv_tmp = f'{self.dirs["tsv"]}/{self.inp_shape_str}'

        if args.timenow is not None:
            timenow = args.timenow
            paths["json"] = f"{json_tmp}-{timenow}.json"
            paths["tsv"] = f"{tsv_tmp}-{timenow}.tsv"
        else:
            timenow = datetime.now().strftime("%m%d_%H%M")
            paths["json"] = f"{json_tmp}-{timenow}.json"
            paths["tsv"] = f"{tsv_tmp}-{timenow}.tsv"
            self.timenow = timenow
        
        if args.json:
            paths["json"] = args.json
        if json:
            paths['json'] = json
        print("Using json :", paths["json"])

        
    def use_json(self, json_path):
        self.paths["json"] = json_path


    def tasks_pkl_check(self):
        self.dirs["tasks_pkl"] = f"{self.gallery_path}/ansor_tasks_pkl"
        self.paths["tasks_pkl"] = f'{self.dirs["tasks_pkl"]}/{self.network}-{self.inp_shape_str}.pkl'

        return os.path.exists(self.paths["tasks_pkl"])


    def tasks_pkl_use(self):
        

        if self.tasks_pkl_check():
            print(f'Load tasks from {self.paths["tasks_pkl"]}')
            with open(self.paths["tasks_pkl"], "rb") as f:
                tasks, task_weights = pickle.load(f)
        else:
            tasks, task_weights = None, None

        return tasks, task_weights

        

    def tasks_pkl_save(self, tasks, task_weights):
        os.makedirs(self.dirs["tasks_pkl"], exist_ok=True)

        print(f'Saved tasks to {self.paths["tasks_pkl"]}')
        with open(self.paths["tasks_pkl"], "wb") as f:
            pickle.dump((tasks, task_weights), f)
        
        
