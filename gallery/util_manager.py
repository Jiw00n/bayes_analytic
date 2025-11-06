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

def get_network(name, batch_size, layout="NHWC", dtype="float32", num_classes=1000):
    """Get the symbol definition and random weight of a network"""

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
    def __init__(self, get_pkl, network, input_shape, args, gdb_mode=None):
        self.tvm_home = os.environ.get("TVM_HOME")
        self.gallery_path = f"{self.tvm_home}/gallery"

        self.network = network
        self.inp_shape_str = str(input_shape).replace(" ", "")
        self.args = args

        self.get_pkl = get_pkl

        self.paths = {}
        self.dirs = {}

        self.get_dirs()
        self.get_paths()


        if gdb_mode == "1":
            GDBManager.save_cur_info(network, self.inp_shape_str, self.timenow)


    def get_dirs(self):

        dirs = self.dirs
        dirs["json"] = f"{self.gallery_path}/logs_json/{self.network}"
        dirs["tsv"] = f"{self.gallery_path}/logs_tsv/{self.network}"
        os.makedirs(dirs["json"], exist_ok=True)
        os.makedirs(dirs["tsv"], exist_ok=True)


        dirs["tasks"] = f"{self.gallery_path}/ansor_tasks_pkl"

        if self.get_pkl:
            os.makedirs(dirs["tasks"], exist_ok=True)
            self.paths["tasks"] = f"{dirs['tasks']}/{self.network}-{self.inp_shape_str}.pkl"



    def get_paths(self):

        paths = self.paths

        json_tmp = f"{self.dirs['json']}/{self.inp_shape_str}"
        tsv_tmp = f"{self.dirs['tsv']}/{self.inp_shape_str}"

        if self.args.timenow is not None:
            timenow = self.args.timenow
            paths["json"] = f"{json_tmp}-{timenow}.json"
            paths["tsv"] = f"{tsv_tmp}-{timenow}.tsv"
        else:
            timenow = datetime.now().strftime("%m%d_%H%M")
            paths["json"] = f"{json_tmp}-{timenow}.json"
            paths["tsv"] = f"{tsv_tmp}-{timenow}.tsv"
            self.timenow = timenow
        
        if self.args.json:
            paths["json"] = self.args.json
        

        
    def use_json(self, json_path):
        self.paths["json"] = json_path


    def use_tasks_pkl(self):
        if os.path.exists(self.paths["tasks"]):
            print(f"Load tasks from {self.paths['tasks']}")
            with open(self.paths["tasks"], "rb") as f:
                tasks, task_weights = pickle.load(f)
            return tasks, task_weights
        else:
            return None
        
    def save_tasks_pkl(self, tasks, task_weights):
        with open(self.paths["tasks"], "wb") as f:
            pickle.dump((tasks, task_weights), f)






class GDBManager:
    def __init__(self):
        self.tvm_home = os.environ.get("TVM_HOME")
        self.gallery_path = f"{self.tvm_home}/gallery"
        self.gdb_dir = f"{self.gallery_path}/logs/gdb"
        self.cur_gdb_path = f"{self.gdb_dir}/current.log"

        print("GDB Mode")
        

    def gdb_log_set(self):
        
        if os.path.exists(self.cur_gdb_path):
            import shutil

            # 예전 json 정보 읽고 복사
            if os.path.exists("/tmp/last_info.json"):
                with open(f"/tmp/last_info.json", "r") as f:
                    try:
                        last_info = json.load(f)
                        gdb_log_dir = f"{self.gdb_dir}/{last_info['network']}"
                        os.makedirs(gdb_log_dir, exist_ok=True)
                        move_path = f"{gdb_log_dir}/{last_info['input_shape']}-{last_info['timenow']}.log"
                        shutil.copy2(self.cur_gdb_path, move_path)
                        open(self.cur_gdb_path, "w").close()
                    except Exception as e:
                        print(e)
        else:
            print("Set GDB dir")


    @staticmethod
    def save_cur_info(network, inp_shape_str, timenow):
        current_info = {
            "network": network,
            "input_shape": inp_shape_str,
            "timenow": timenow
        }

        print("Save current info to /tmp/last_info.json")
        with open("/tmp/last_info.json", "w") as f:
            json.dump(current_info, f)




class ScheduleSelector:
    def __init__(self, tasks_info, path_manager):
        self.tasks_info = tasks_info
        self.paths = path_manager.paths
        self.dirs = path_manager.dirs


    def load_rec_only_high(self, percent=0.05):
        tasks_info = self.tasks_info
        
        records = {}            # records[wk] = [(json_data, mean_cost), ...]
        
        with open(self.paths['json'], "r") as f:
            lines = f.readlines()

            for wk_idx, wk in enumerate(tasks_info):
                records[wk] = []
                
                for line_idx, line in enumerate(lines):
                    json_data = json.loads(line)
                    json_workload = json_data['i'][0][0]
                    cost = json_data['r'][0]
                    mean_cost = sum(cost) / len(cost)

                    if wk in json_workload and mean_cost < 1000:
                        records[wk].append((json_data, mean_cost, line_idx))

                records[wk].sort(key=lambda x: x[1])
                print(len(records[wk]), end=" - ")
                records[wk] = records[wk][:int(len(records[wk]) * percent)]  # 상위 percent%만 사용
                print(len(records[wk]))

        breakpoint()
        return records


    def random_look4_better(self, records, seen=None, best=False):
        tasks_info = self.tasks_info

        selected_rec_cost  = 0
        line_indices = []

        tmp_json = os.path.dirname(self.dirs['json'])+f"/tmp.json"
        print("Selected workloads recorded cost (ms) : ", end="")
        
        if seen is not None:
            seen_indices = [exp["Line_indices"] for exp in seen]
        else:
            seen_indices = []
        
        while True:
            with open(tmp_json, "w") as f:
                for task_idx, wk in enumerate(tasks_info):
                    # breakpoint()
                    rd_rec = random.choice(records[wk])
                    if best:
                        rd_rec = records[wk][0]  # best 스케줄
                    sch = rd_rec[0]  # 랜덤 스케줄
                    cost = rd_rec[1]
                    line_idx = rd_rec[2]

                    print(f"{cost*1000:.4f}", end="")
                    if task_idx != len(tasks_info)-1:
                        print(" + ", end="")
                    selected_rec_cost += cost
                    line_indices.append(line_idx)
                        

                    json.dump(sch, f)
                    f.write("\n")
            if line_indices not in seen_indices:
                break


        selected_total_cost = selected_rec_cost*1000
        print(f" = {selected_total_cost:.4f} ms")
        return tmp_json, selected_total_cost, line_indices

    def all_look4_better(self, records, cnt):
        
        progress = 0
        schedule_indices = {}
        tasks_info = self.tasks_info
        for task_idx, wk in enumerate(tasks_info):
            schedule_indices[wk] = 0

        selected_rec_cost  = 0
        line_indices = []

        tmp_json = os.path.dirname(self.dirs['json'])+f"/tmp.json"
        print("Selected workloads recorded cost (ms) : ", end="")
        with open(tmp_json, "w") as f:
            for task_idx, wk in enumerate(tasks_info):
                
                while len(records[wk]) - 1 > schedule_indices[wk] and progress < cnt:
                    schedule_indices[wk] += 1
                    progress += 1
                rd_rec = records[wk][schedule_indices[wk]]
                sch = rd_rec[0]
                cost = rd_rec[1]
                line_idx = rd_rec[2]

                print(f"{cost*1000:.4f}", end="")
                if task_idx != len(tasks_info)-1:
                    print(" + ", end="")
                selected_rec_cost += cost
                line_indices.append(line_idx)
                    

                json.dump(sch, f)
                f.write("\n")
        # breakpoint()

        selected_total_cost = selected_rec_cost*1000
        print(f" = {selected_total_cost:.4f} ms")
        return tmp_json, selected_total_cost, line_indices


    # 가장 변화율 높은 k개만 랜덤으로 선택
    # 나머지는 best 스케줄로 선택
    def load_rec(self, select_type="all"):
        
        # 1. 각 워크로드별로 비용 변화율 구하기 (최대 - 최소)
            # a. 각 워크로드에서 모든 (스케줄, cost) 수집
            # b. cost를 기준으로 정렬 후 변화율 계산
        # 2. 해당 인덱스 워크로드는 랜덤 스케줄 선택, 나머지는 best 스케줄 선택

        tasks_info = self.tasks_info
        if select_type.startswith("top-"):
            select_num = int(select_type.split("-")[1])
        elif select_type.startswith("random-"):
            select_num = int(select_type.split("-")[1])
        elif select_type == "all":
            select_num = len(tasks_info)
        
        
        records = {}            # records[wk] = [(json_data, mean_cost), ...]
        cost_diff = {}
        
        with open(self.paths['json'], "r") as f:
            lines = f.readlines()

            for wk_idx, wk in enumerate(tasks_info):
                records[wk] = []
                
                for line in lines:
                    json_data = json.loads(line)
                    json_workload = json_data['i'][0][0]
                    cost = json_data['r'][0]
                    mean_cost = sum(cost) / len(cost)

                    if wk in json_workload and mean_cost < 1000:
                        records[wk].append((json_data, mean_cost))

                records[wk].sort(key=lambda x: x[1])
                cost_diff[wk] = records[wk][-1][1] - records[wk][1][1]

        if select_type.startswith("random-"):
            selected_wks = random.sample(list(tasks_info.keys()), select_num)
            selected_diff = {}
            for wk in selected_wks:
                selected_diff[wk] = cost_diff[wk]
        else:
            selected_diff = dict(sorted(cost_diff.items(), key=lambda x: x[1], reverse=True)[:select_num])

        print("\n"+"="*40)
        print(f"{select_type} schedule selection")
        print("Workloads with highest cost differences between best and worst schedules : ")
        for wk, diff in selected_diff.items():
            print(f"{wk}({tasks_info[wk]}) : {diff * 1000:.4f} ms")
            print("Schedule length : ", len(records[wk]))

        return records, selected_diff


    def random_sch(self, rec):

        tasks_info = self.tasks_info
        records, diff_selected_wk = rec
        rec_costs = []
        selected_rec_costs = []


        tmp_json = os.path.dirname(self.dirs['json'])+f"/tmp.json"
        with open(tmp_json, "w") as f:
            for wk in tasks_info:
                # best 스케줄 idx = 0
                sch_idx = 0
                
                if wk in diff_selected_wk:
                    sch_idx = random.randint(0, len(records[wk])-1)
                    selected_rec_costs.append(records[wk][sch_idx][1])
                    

                sch = records[wk][sch_idx][0]  # best 스케줄
                cost = records[wk][sch_idx][1]
                rec_costs.append(cost)

                json.dump(sch, f)
                f.write("\n")

        print("Selected workloads recorded cost (ms) : ", end="")
        for idx, c in enumerate(selected_rec_costs):
            print(f"{c*1000:.4f}", end="")
            if idx != len(selected_rec_costs)-1:
                print(" + ", end="")

        selected_total_cost = sum(selected_rec_costs)*1000
        print(f" = {selected_total_cost:.4f} ms")
        return tmp_json, rec_costs, selected_total_cost