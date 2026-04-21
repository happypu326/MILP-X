from __future__ import annotations

import importlib

from .base import BaseGenerator
from .registry import (
    register_generator,
    unregister_generator,
    get_generator_class,
    create_generator,
    generate_batch_by_name,
    list_generators,
)

generate_batch = generate_batch_by_name

_LAZY_CLASS_EXPORTS = {
    "BinPackingGenerator": ".BP:BinPackingGenerator",
    "CombinatorialAuctionsGenerator": ".CA:CombinatorialAuctionsGenerator",
    "FacilityLocationGenerator": ".CFLP:FacilityLocationGenerator",
    "GraphColoringGenerator": ".GC:GraphColoringGenerator",
    "GISPDimacsGenerator": ".GISP_dimacs:GISPDimacsGenerator",
    "GISPErdosRenyiGenerator": ".GISP_erdos_renyi:GISPErdosRenyiGenerator",
    "JobSchedulingGenerator": ".JS:JobSchedulingGenerator",
    "MultipleKnapsackGenerator": ".KS:MultipleKnapsackGenerator",
    "MultiItemLotSizingGenerator": ".LS:MultiItemLotSizingGenerator",
    "MaxCutGenerator": ".MC:MaxCutGenerator",
    "IndependentSetGenerator": ".MIS:IndependentSetGenerator",
    "MISErdosRenyiGenerator": ".MIS_erdos:MISErdosRenyiGenerator",
    "MinimumVertexCoverGenerator": ".MVC:MinimumVertexCoverGenerator",
    "FCMCNFGenerator": ".NF:FCMCNFGenerator",
    "ProteinFoldingGenerator": ".PF:ProteinFoldingGenerator",
    "MaxSatisfiabilityGenerator": ".SAT:MaxSatisfiabilityGenerator",
    "SetCoverGenerator": ".SC:SetCoverGenerator",
}


def __getattr__(name: str):
    if name not in _LAZY_CLASS_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, class_name = _LAZY_CLASS_EXPORTS[name].split(":")
    module = importlib.import_module(module_path, package=__package__)
    obj = getattr(module, class_name)

    globals()[name] = obj
    return obj


__all__ = [
    "BaseGenerator",
    "register_generator",
    "unregister_generator",
    "get_generator_class",
    "create_generator",
    "generate_batch_by_name",
    "generate_batch",
    "list_generators",
] + list(_LAZY_CLASS_EXPORTS.keys())