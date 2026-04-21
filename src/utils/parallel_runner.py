import os
import json
import math
import random
from multiprocessing import Process, set_start_method

import torch


def split_instances(instances, num_workers):
    if num_workers <= 1 or len(instances) <= 1:
        return [instances]
    chunk_size = math.ceil(len(instances) / num_workers)
    return [instances[i:i + chunk_size] for i in range(0, len(instances), chunk_size)]


def resolve_device(cfg_dict, worker_id):
    if torch.cuda.is_available():
        device_list = cfg_dict.get("device_list", None)
        if device_list:
            return device_list[worker_id % len(device_list)]
        return cfg_dict.get("device", "cuda:0")
    return "cpu"


def get_worker_paths(cfg_dict, worker_id):
    """
    Directory structure：
      log_dir/
        worker_0/
        worker_1/
      result_dir/
        worker_0/
        worker_1/
      scores_dir/
        worker_0/
        worker_1/
    """
    worker_log_root = os.path.join(cfg_dict["log_dir"], f"worker_{worker_id}")
    worker_result_root = os.path.join(cfg_dict["result_dir"], f"worker_{worker_id}")

    worker_scores_root = None
    if cfg_dict.get("scores_dir", None):
        worker_scores_root = os.path.join(cfg_dict["scores_dir"], f"worker_{worker_id}")

    os.makedirs(worker_log_root, exist_ok=True)
    os.makedirs(worker_result_root, exist_ok=True)
    if worker_scores_root is not None:
        os.makedirs(worker_scores_root, exist_ok=True)

    return {
        "worker_log_root": worker_log_root,
        "worker_result_root": worker_result_root,
        "worker_scores_root": worker_scores_root,
    }


def _worker_entry(
    worker_id,
    instances,
    cfg_dict,
    build_model_fn,
    build_eval_config_fn,
    evaluator_cls,
    result_filename="worker_metrics.json",
):
    if not instances:
        return

    seed = int(cfg_dict.get("seed", 0)) + worker_id
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = resolve_device(cfg_dict, worker_id)
    worker_paths = get_worker_paths(cfg_dict, worker_id)

    model = None
    if build_model_fn is not None:
        model = build_model_fn(cfg_dict, device)

    evaluator_config = build_eval_config_fn(
        cfg_dict=cfg_dict,
        device=device,
        worker_id=worker_id,
        worker_log_root=worker_paths["worker_log_root"],
        worker_result_root=worker_paths["worker_result_root"],
        worker_scores_root=worker_paths["worker_scores_root"],
        chunk_size=len(instances),
    )

    if model is None:
        evaluator = evaluator_cls(evaluator_config)
    else:
        evaluator = evaluator_cls(model=model, config=evaluator_config)

    metrics = evaluator.evaluate(instances)

    out_path = os.path.join(worker_paths["worker_result_root"], result_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def merge_basic_metrics(
    cfg_dict,
    actual_workers,
    result_filename="worker_metrics.json",
    output_name="merged_results.json"
):
    merged_instances = []
    total_time = 0.0
    num_success = 0
    num_instances = 0

    method_name = None
    method_type = None
    task_name = None
    difficulty = None

    for worker_id in range(actual_workers):
        fp = os.path.join(cfg_dict["result_dir"], f"worker_{worker_id}", result_filename)
        if not os.path.exists(fp):
            continue

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        merged_instances.extend(data.get("instances", []))
        total_time += data.get("total_time", 0.0)
        num_success += data.get("num_success", 0)
        num_instances += data.get("num_instances", 0)

        method_name = data.get("method_name", method_name)
        method_type = data.get("method_type", method_type)
        task_name = data.get("task_name", task_name)
        difficulty = data.get("difficulty", difficulty)

    merged = {
        "method_name": method_name,
        "method_type": method_type,
        "task_name": task_name,
        "difficulty": difficulty,
        "num_instances": num_instances,
        "num_success": num_success,
        "avg_time": total_time / max(num_success, 1),
        "total_time": total_time,
        "instances": merged_instances,
    }

    merged_path = os.path.join(cfg_dict["result_dir"], output_name)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged results saved to {merged_path}")
    return merged


def merge_apollo_metrics(
    cfg_dict,
    actual_workers,
    result_filename="worker_metrics.json",
    output_name="merged_results.json"
):
    merged_results = []
    total_time = 0.0
    num_success = 0
    num_instances = 0

    weighted_avg_obj = 0.0
    weighted_avg_time = 0.0
    weighted_avg_node = 0.0
    weighted_avg_best_obj = 0.0
    hyperparams = None

    for worker_id in range(actual_workers):
        fp = os.path.join(cfg_dict["result_dir"], f"worker_{worker_id}", result_filename)
        if not os.path.exists(fp):
            continue

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        worker_success = data.get("num_success", 0)
        worker_instances = data.get("num_instances", 0)

        merged_results.extend(data.get("results", []))
        num_success += worker_success
        num_instances += worker_instances

        if hyperparams is None:
            hyperparams = data.get("hyperparams", None)

        if data.get("avg_obj") is not None:
            weighted_avg_obj += data["avg_obj"] * worker_success
        if data.get("avg_time") is not None:
            weighted_avg_time += data["avg_time"] * worker_success
        if data.get("avg_node") is not None:
            weighted_avg_node += data["avg_node"] * worker_success
        if data.get("avg_best_obj") is not None:
            weighted_avg_best_obj += data["avg_best_obj"] * worker_success

        for r in data.get("results", []):
            total_time += r.get("total_time", 0.0)

    merged = {
        "method": "Apollo",
        "num_instances": num_instances,
        "num_success": num_success,
        "avg_obj": (weighted_avg_obj / num_success) if num_success > 0 else None,
        "avg_time": (weighted_avg_time / num_success) if num_success > 0 else None,
        "avg_node": (weighted_avg_node / num_success) if num_success > 0 else None,
        "avg_best_obj": (weighted_avg_best_obj / num_success) if num_success > 0 else None,
        "total_time": total_time,
        "hyperparams": hyperparams,
        "results": merged_results,
    }

    merged_path = os.path.join(cfg_dict["result_dir"], output_name)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged results saved to {merged_path}")
    return merged


def run_parallel_eval(
    cfg_dict,
    instances,
    evaluator_cls,
    build_model_fn,
    build_eval_config_fn,
    merge_fn,
    result_filename="worker_metrics.json",
):
    test_num = int(cfg_dict.get("test_num", len(instances)))
    instances = instances[:test_num]

    os.makedirs(cfg_dict["log_dir"], exist_ok=True)
    os.makedirs(cfg_dict["result_dir"], exist_ok=True)
    if cfg_dict.get("scores_dir", None):
        os.makedirs(cfg_dict["scores_dir"], exist_ok=True)

    num_workers = int(cfg_dict.get("num_workers", 1))
    if num_workers <= 1:
        _worker_entry(
            worker_id=0,
            instances=instances,
            cfg_dict=cfg_dict,
            build_model_fn=build_model_fn,
            build_eval_config_fn=build_eval_config_fn,
            evaluator_cls=evaluator_cls,
            result_filename=result_filename,
        )
        return merge_fn(cfg_dict, 1, result_filename=result_filename)

    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    chunks = split_instances(instances, num_workers)
    actual_workers = len(chunks)

    processes = []
    for worker_id, chunk in enumerate(chunks):
        p = Process(
            target=_worker_entry,
            args=(
                worker_id,
                chunk,
                cfg_dict,
                build_model_fn,
                build_eval_config_fn,
                evaluator_cls,
                result_filename,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    bad = [p.exitcode for p in processes if p.exitcode != 0]
    if bad:
        raise RuntimeError(f"Some worker processes failed, exit codes: {bad}")

    return merge_fn(cfg_dict, actual_workers, result_filename=result_filename)