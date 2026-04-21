from __future__ import annotations

import importlib
from typing import Type, Union

from .base import BaseGenerator

GeneratorSpec = Union[str, Type[BaseGenerator]]

_GENERATOR_REGISTRY: dict[str, GeneratorSpec] = {}
_CLASS_CACHE: dict[str, Type[BaseGenerator]] = {}

_GENERATOR_REGISTRY: dict[str, GeneratorSpec] = {
    "bp": ".BP:BinPackingGenerator",
    "ca": ".CA:CombinatorialAuctionsGenerator",
    "cflp": ".CFLP:FacilityLocationGenerator",
    "gc": ".GC:GraphColoringGenerator",
    "gisp_dimacs": ".GISP_dimacs:GISPDimacsGenerator",
    "gisp_erdos": ".GISP_erdos_renyi:GISPErdosRenyiGenerator",
    "gisp_erdos_renyi": ".GISP_erdos_renyi:GISPErdosRenyiGenerator",
    "js": ".JS:JobSchedulingGenerator",
    "ks": ".KS:MultipleKnapsackGenerator",
    "ls": ".LS:MultiItemLotSizingGenerator",
    "mc": ".MC:MaxCutGenerator",
    "mis": ".MIS:IndependentSetGenerator",
    "mis_erdos": ".MIS_erdos:MISErdosRenyiGenerator",
    "mvc": ".MVC:MinimumVertexCoverGenerator",
    "nf": ".NF:FCMCNFGenerator",
    "pf": ".PF:ProteinFoldingGenerator",
    "sat": ".SAT:MaxSatisfiabilityGenerator",
    "sc": ".SC:SetCoverGenerator",
}

_CLASS_CACHE: dict[str, Type[BaseGenerator]] = {}

def _resolve_spec(spec: GeneratorSpec) -> Type[BaseGenerator]:
    if isinstance(spec, type):
        cls = spec
    else:
        module_path, class_name = spec.split(":")
        module = importlib.import_module(module_path, package=__package__)
        cls = getattr(module, class_name)

    if not issubclass(cls, BaseGenerator):
        raise TypeError(f"{cls} 不是 BaseGenerator 的子类。")
    return cls


def register_generator(name: str, spec: GeneratorSpec) -> None:
    key = name.lower().strip()
    if not key:
        raise ValueError("生成器名称不能为空。")
    _GENERATOR_REGISTRY[key] = spec
    _CLASS_CACHE.pop(key, None)


def unregister_generator(name: str) -> None:
    key = name.lower().strip()
    _GENERATOR_REGISTRY.pop(key, None)
    _CLASS_CACHE.pop(key, None)


def get_generator_class(name: str) -> Type[BaseGenerator]:
    key = name.lower().strip()
    if key in _CLASS_CACHE:
        return _CLASS_CACHE[key]

    if key not in _GENERATOR_REGISTRY:
        available = ", ".join(sorted(_GENERATOR_REGISTRY.keys()))
        raise ValueError(f"未知生成器 '{name}'。可用生成器: {available}")

    cls = _resolve_spec(_GENERATOR_REGISTRY[key])
    _CLASS_CACHE[key] = cls
    return cls

def create_generator(name: str, **kwargs) -> BaseGenerator:
    cls = get_generator_class(name)
    return cls(**kwargs)


def generate_batch_by_name(
    name: str,
    *,
    n_instances: int,
    output_dir: str,
    **kwargs,
):
    generator = create_generator(name, **kwargs)
    return generator.generate_batch(
        n_instances=n_instances,
        output_dir=output_dir,
    )


def list_generators() -> list[str]:
    return sorted(_GENERATOR_REGISTRY.keys())