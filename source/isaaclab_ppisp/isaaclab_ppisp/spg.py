# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD SPG graph helpers for native RTX/OVRTX PPISP execution."""

from __future__ import annotations

from typing import Any

from .cfg import PPISP_SHADER_NAMES


def ppisp_uses_native_spg(ppisp_cfg: Any) -> bool:
    """Return whether ``ppisp_cfg`` carries an authored USD SPG graph."""
    return bool(getattr(ppisp_cfg, "spg_render_product_prim_path", None))


def copy_ppisp_spg_to_render_product(
    stage: Any, source_render_product_path: str, target_render_product_path: str
) -> None:
    """Copy a composed PPISP SPG graph onto a backend-generated RenderProduct."""
    from pxr import Sdf

    source_render_product = stage.GetPrimAtPath(source_render_product_path)
    if not source_render_product or not source_render_product.IsValid():
        return
    spg_prim_paths = ppisp_spg_prim_paths(source_render_product)
    if not spg_prim_paths:
        return
    if source_render_product_path == target_render_product_path:
        return
    target_rp = stage.GetPrimAtPath(target_render_product_path)
    if not target_rp or not target_rp.IsValid():
        return

    copied_render_vars = []
    for source_prim_path in spg_prim_paths:
        source_prim = stage.GetPrimAtPath(source_prim_path)
        if not source_prim or not source_prim.IsValid():
            continue
        target_child_path = Sdf.Path(target_render_product_path).AppendChild(source_prim.GetName())
        _copy_prim_composed_values(
            stage, source_prim, target_child_path, source_render_product_path, target_render_product_path
        )
        if source_prim.GetTypeName() == "RenderVar":
            copied_render_vars.append(target_child_path)
    ordered_vars_rel = target_rp.CreateRelationship("orderedVars")
    if ordered_vars_rel and copied_render_vars:
        copied_source_names = {_render_var_source_name(stage.GetPrimAtPath(path)) for path in copied_render_vars}
        copied_source_names.discard(None)
        targets = [
            target
            for target in ordered_vars_rel.GetTargets()
            if _render_var_source_name(stage.GetPrimAtPath(target)) not in copied_source_names
        ]
        for path in copied_render_vars:
            if path not in targets:
                targets.append(path)
        ordered_vars_rel.SetTargets(targets)


def find_ppisp_shader_prim(stage: Any, render_product_prim: Any) -> Any | None:
    """Return the PPISP/PPISPAuto shader child on ``render_product_prim``, if present."""
    for shader_name in PPISP_SHADER_NAMES:
        shader_prim = stage.GetPrimAtPath(render_product_prim.GetPath().AppendChild(shader_name))
        if shader_prim and shader_prim.IsValid():
            return shader_prim
    return None


def ppisp_spg_prim_paths(container_prim: Any, shader_prim: Any | None = None) -> tuple[str, ...]:
    """Return direct child prim paths that belong to the authored PPISP SPG graph.

    The exporter contract is that the final PPISP shader is a direct child named
    ``PPISP`` or ``PPISPAuto``. From there, follow authored child-to-child USD
    connections instead of relying on every intermediate node keeping a fixed
    name.
    """
    stage = container_prim.GetStage()
    source_container_path = str(container_prim.GetPath())
    shader_prim = shader_prim or find_ppisp_shader_prim(stage, container_prim)
    if shader_prim is None:
        return ()

    children_by_name = {child.GetName(): child for child in container_prim.GetChildren()}
    names = {shader_prim.GetName()}
    changed = True
    while changed:
        changed = False
        for child_name in list(names):
            child = children_by_name.get(child_name)
            if child is None:
                continue
            for connected_name in _direct_connected_child_names(child, source_container_path):
                if connected_name in children_by_name and connected_name not in names:
                    names.add(connected_name)
                    changed = True
        for child_name, child in children_by_name.items():
            if child_name in names:
                continue
            if _direct_connected_child_names(child, source_container_path).intersection(names):
                names.add(child_name)
                changed = True

    graph_children = [child for child in container_prim.GetChildren() if child.GetName() in names]
    if not any(child.GetTypeName() == "RenderVar" for child in graph_children):
        return ()
    return tuple(str(child.GetPath()) for child in graph_children)


def _direct_connected_child_names(prim: Any, source_container_path: str) -> set[str]:
    names = set()
    for attr in prim.GetAttributes():
        names.update(_direct_child_names_from_paths(attr.GetConnections(), source_container_path))
    for rel in prim.GetRelationships():
        names.update(_direct_child_names_from_paths(rel.GetTargets(), source_container_path))
    return names


def _direct_child_names_from_paths(paths: Any, source_container_path: str) -> set[str]:
    names = set()
    prefix = source_container_path + "/"
    for path in paths:
        prim_path = path.GetPrimPath() if hasattr(path, "GetPrimPath") else path
        path_str = str(prim_path)
        if path_str.startswith("../"):
            sibling_name = path_str[len("../") :]
            if sibling_name and "/" not in sibling_name:
                names.add(sibling_name)
            continue
        if not path_str.startswith(prefix):
            continue
        suffix = path_str[len(prefix) :]
        if suffix and "/" not in suffix:
            names.add(suffix)
    return names


def _render_var_source_name(render_var_prim: Any) -> str | None:
    if not render_var_prim or not render_var_prim.IsValid() or render_var_prim.GetTypeName() != "RenderVar":
        return None
    source_name_attr = render_var_prim.GetAttribute("sourceName")
    source_name = source_name_attr.Get() if source_name_attr else None
    return str(source_name) if source_name is not None else None


def _copy_prim_composed_values(
    stage: Any,
    source_prim: Any,
    target_path: Any,
    source_container_path: str,
    target_render_product_path: str,
) -> None:
    target_prim = stage.DefinePrim(target_path, source_prim.GetTypeName())
    for attr in source_prim.GetAttributes():
        target_attr = target_prim.CreateAttribute(
            attr.GetName(),
            attr.GetTypeName(),
            custom=attr.IsCustom(),
            variability=attr.GetVariability(),
        )
        connections = attr.GetConnections()
        if connections:
            target_attr.SetConnections(
                [
                    _rewrite_connection_path(path, source_container_path, target_render_product_path)
                    for path in connections
                ]
            )
            continue
        value = attr.Get()
        if value is not None:
            target_attr.Set(value)
    for rel in source_prim.GetRelationships():
        target_rel = target_prim.CreateRelationship(rel.GetName(), custom=rel.IsCustom())
        targets = rel.GetTargets()
        if targets:
            target_rel.SetTargets(
                [_rewrite_connection_path(path, source_container_path, target_render_product_path) for path in targets]
            )


def _rewrite_connection_path(path: Any, source_container_path: str, target_render_product_path: str) -> Any:
    from pxr import Sdf

    path_str = str(path)
    if path_str == source_container_path or path_str.startswith(source_container_path + "/"):
        return Sdf.Path(target_render_product_path + path_str[len(source_container_path) :])
    return path
