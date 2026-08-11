"""Open an explicitly requested USD stage during Isaac Sim startup."""

from __future__ import annotations

import asyncio
import bisect
import functools
import json
import math
import os
import re
import struct
import time
import zlib

import carb
import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


_SCENE_HEADER = struct.Struct("<IHHQ32fQII")
_SCENE_LIGHT_RECORD = struct.Struct("<II16fQ9fIQQffII")
_SCENE_TEXTURE_RECORD = struct.Struct("<QIII")
_SCENE_MESH_RECORD = struct.Struct("<QII26f28I")
_SCENE_MAGIC = 0x31434D49
_SCENE_VERSION = 20
_CAMERA_VALID_PERSPECTIVE = 1
_SCENE_VALID_SPHERE_LIGHT = 2
_SCENE_HAS_MESH_MANIFEST = 4
_SCENE_VALID_DISTANT_LIGHT = 8
_SCENE_VALID_DOME_LIGHT = 16
_SCENE_HAS_MESH_GEOMETRY = 32
_SCENE_HAS_CORNER_NORMALS = 64
_SCENE_HAS_FILE_TEXTURES = 128
_SCENE_HAS_MATERIAL_PARAMETERS = 256
_SCENE_HAS_EMISSION = 512
_SCENE_HAS_PARAMETER_TEXTURES = 1024
_SCENE_HAS_NORMAL_TEXTURES = 2048
_LIGHT_SHAPING_APPLIED = 1
_LIGHT_TEXTURE_RECT_EMISSION = 1
_LIGHT_TEXTURE_IES_PROFILE = 2
_LIGHT_TEXTURE_IES_NORMALIZED = 4
_LIGHT_TEXTURE_DOME_ENVIRONMENT = 8
_LIGHT_TEXTURE_DOME_RGBE = 16
_MESH_HAS_BOUND_MATERIAL = 1
_MESH_HAS_BASE_COLOR = 2
_MESH_HAS_CONNECTED_BASE_COLOR = 4
_MESH_HAS_FILE_TEXTURE = 8
_MESH_HAS_MATERIAL_PARAMETERS = 16
_MESH_HAS_EMISSION = 32
_MESH_HAS_ALPHA_CUTOUT = 64
_MESH_HAS_STANDARD_OPACITY = 128
_TEXTURE_HAS_ROUGHNESS = 1
_TEXTURE_HAS_METALLIC = 2
_TEXTURE_HAS_EMISSION = 4
_TEXTURE_HAS_NORMAL = 8
_TEXTURE_HAS_OPACITY = 16
_MAX_SCENE_MESHES = 4096
_MAX_SCENE_GEOMETRY_BYTES = 320 * 1024 * 1024
_MAX_SCENE_TEXTURE_BYTES = 128 * 1024 * 1024
_MAX_SCENE_TEXTURE_DIMENSION = 4096
_MAX_SCENE_TEXTURE_TRANSPORT_DIMENSION = 512
_MAX_SCENE_MATERIAL_TEXTURES = 126
_MAX_SCENE_LIGHT_TEXTURES = 32
_MAX_SCENE_LIGHT_TEXTURE_BYTES = 16 * 1024 * 1024
_MAX_SCENE_POSITIONAL_LIGHTS = 8
_MAX_SCENE_DISTANT_LIGHTS = 4
_MAX_SCENE_DOME_LIGHTS = 4
_NO_SCENE_TEXTURE = 0xFFFFFFFF
_NO_SCENE_LIGHT_TEXTURE = 0xFFFFFFFFFFFFFFFF


def _scene_flags(sphere_light_path, distant_light_path, dome_light_path):
    flags = (
        _CAMERA_VALID_PERSPECTIVE
        | _SCENE_HAS_MESH_MANIFEST
        | _SCENE_HAS_MESH_GEOMETRY
        | _SCENE_HAS_CORNER_NORMALS
        | _SCENE_HAS_FILE_TEXTURES
        | _SCENE_HAS_MATERIAL_PARAMETERS
        | _SCENE_HAS_EMISSION
        | _SCENE_HAS_PARAMETER_TEXTURES
        | _SCENE_HAS_NORMAL_TEXTURES
    )
    if sphere_light_path:
        flags |= _SCENE_VALID_SPHERE_LIGHT
    if distant_light_path:
        flags |= _SCENE_VALID_DISTANT_LIGHT
    if dome_light_path:
        flags |= _SCENE_VALID_DOME_LIGHT
    return flags


def _stage_prims(stage: Usd.Stage):
    """Traverse ordinary prims and native USD scenegraph instance proxies."""

    return Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())


def _path_is_at_or_below(path, root_path) -> bool:
    value = str(path)
    root = str(root_path).rstrip("/")
    return value == root or value.startswith(root + "/")


def _mesh_material_parts(prim: Usd.Prim, sample_time: Usd.TimeCode):
    """Return disjoint face groups with their real USD bound materials.

    USD commonly assigns several materials to one Mesh through
    ``materialBind`` GeomSubsets.  A ray-tracing geometry has one material
    descriptor in the bridge protocol, so preserve the authored result by
    emitting one geometry record per subset instead of applying an arbitrary
    first subset to the whole Mesh.  Faces outside valid subsets retain the
    direct/ancestor Mesh binding.
    """

    try:
        mesh = UsdGeom.Mesh(prim)
        face_counts = mesh.GetFaceVertexCountsAttr().Get(sample_time) or ()
        face_count = len(face_counts)
        mesh_material, _relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
        if not (mesh_material and mesh_material.GetPrim().IsValid()):
            mesh_material = None

        assigned_faces = set()
        subset_parts = []
        for child in prim.GetChildren():
            if not child.IsA(UsdGeom.Subset):
                continue
            subset = UsdGeom.Subset(child)
            if subset.GetFamilyNameAttr().Get() != "materialBind":
                continue
            element_type = subset.GetElementTypeAttr().Get()
            if element_type not in (None, "", UsdGeom.Tokens.face):
                continue
            material, _relationship = UsdShade.MaterialBindingAPI(
                child
            ).ComputeBoundMaterial()
            if not (material and material.GetPrim().IsValid()):
                continue
            authored_faces = subset.GetIndicesAttr().Get(sample_time) or ()
            faces = tuple(
                sorted(
                    {
                        int(face)
                        for face in authored_faces
                        if 0 <= int(face) < face_count
                        and int(face) not in assigned_faces
                    }
                )
            )
            if not faces:
                continue
            assigned_faces.update(faces)
            subset_parts.append(
                (
                    f"/__IMBMaterialSubset_{child.GetName()}",
                    frozenset(faces),
                    material,
                )
            )

        if not subset_parts:
            return (("", None, mesh_material),)

        remaining_faces = frozenset(
            face for face in range(face_count) if face not in assigned_faces
        )
        if remaining_faces:
            return (("", remaining_faces, mesh_material), *subset_parts)
        return tuple(subset_parts)
    except Exception:
        try:
            material, _relationship = UsdShade.MaterialBindingAPI(
                prim
            ).ComputeBoundMaterial()
            if not (material and material.GetPrim().IsValid()):
                material = None
        except Exception:
            material = None
        return (("", None, material),)


def _scene_mesh_occurrences(
    stage: Usd.Stage,
    xform_cache: UsdGeom.XformCache,
    sample_time: Usd.TimeCode,
):
    """Yield visible Mesh sources plus bounded PointInstancer expansions."""

    point_instancers = []
    embedded_prototype_paths = []
    for prim in _stage_prims(stage):
        if not prim.IsA(UsdGeom.PointInstancer):
            continue
        instancer = UsdGeom.PointInstancer(prim)
        try:
            targets = tuple(instancer.GetPrototypesRel().GetTargets())
        except Exception:
            targets = ()
        if targets:
            point_instancers.append((prim, instancer, targets))
            embedded_prototype_paths.extend(
                target
                for target in targets
                if _path_is_at_or_below(target, prim.GetPath())
            )

    # Prototypes nested under a PointInstancer are not independently drawn.
    # External defined prototype sources retain their own ordinary appearance;
    # class/over prototypes are already omitted by the default stage predicate.
    for prim in _stage_prims(stage):
        if not prim.IsA(UsdGeom.Mesh) or any(
            _path_is_at_or_below(prim.GetPath(), root)
            for root in embedded_prototype_paths
        ):
            continue
        for material_suffix, included_faces, material in _mesh_material_parts(
            prim, sample_time
        ):
            yield (
                prim,
                str(prim.GetPath()) + material_suffix,
                xform_cache.GetLocalToWorldTransform(prim),
                prim.IsInstanceProxy(),
                None,
                included_faces,
                material,
            )

    for instancer_prim, instancer, targets in point_instancers:
        try:
            imageable = UsdGeom.Imageable(instancer_prim)
            if (
                imageable
                and imageable.ComputeVisibility(sample_time)
                == UsdGeom.Tokens.invisible
            ):
                continue
            proto_indices = tuple(
                int(value)
                for value in (
                    instancer.GetProtoIndicesAttr().Get(sample_time) or ()
                )
            )
            transforms = instancer.ComputeInstanceTransformsAtTime(
                sample_time,
                sample_time,
                UsdGeom.PointInstancer.IncludeProtoXform,
                UsdGeom.PointInstancer.IgnoreMask,
            )
            mask = tuple(instancer.ComputeMaskAtTime(sample_time))
            if not proto_indices or len(transforms) != len(proto_indices):
                continue
            if mask and len(mask) != len(proto_indices):
                continue
            instancer_world = xform_cache.GetLocalToWorldTransform(instancer_prim)
        except Exception:
            continue

        prototype_meshes = {}
        for instance_index, (prototype_index, instance_transform) in enumerate(
            zip(proto_indices, transforms)
        ):
            if mask and not mask[instance_index]:
                continue
            if prototype_index < 0 or prototype_index >= len(targets):
                continue
            if prototype_index not in prototype_meshes:
                prototype_path = targets[prototype_index]
                prototype_prim = stage.GetPrimAtPath(prototype_path)
                meshes = []
                if prototype_prim.IsValid():
                    try:
                        for candidate in Usd.PrimRange(
                            prototype_prim,
                            Usd.TraverseInstanceProxies(
                                Usd.PrimAllPrimsPredicate
                            ),
                        ):
                            if not candidate.IsA(UsdGeom.Mesh):
                                continue
                            candidate_visible = True
                            visibility_prim = candidate
                            while visibility_prim.IsValid():
                                visibility_imageable = UsdGeom.Imageable(
                                    visibility_prim
                                )
                                if (
                                    visibility_imageable
                                    and visibility_imageable.GetVisibilityAttr().Get(
                                        sample_time
                                    )
                                    == UsdGeom.Tokens.invisible
                                ):
                                    candidate_visible = False
                                    break
                                if visibility_prim == prototype_prim:
                                    break
                                visibility_prim = visibility_prim.GetParent()
                            if not candidate_visible:
                                continue
                            relative_transform, _reset_stack = (
                                xform_cache.ComputeRelativeTransform(
                                    candidate, prototype_prim
                                )
                            )
                            meshes.append((candidate, relative_transform))
                    except Exception:
                        meshes = []
                prototype_meshes[prototype_index] = (
                    prototype_path,
                    tuple(meshes),
                )

            prototype_path, meshes = prototype_meshes[prototype_index]
            for mesh_prim, relative_transform in meshes:
                source_path = str(mesh_prim.GetPath())
                prototype_prefix = str(prototype_path).rstrip("/")
                relative_path = source_path[len(prototype_prefix) :]
                if not relative_path:
                    relative_path = "/__PrototypeRoot"
                synthetic_path = (
                    f"{instancer_prim.GetPath()}/__IMBPointInstance_"
                    f"{instance_index}{relative_path}"
                )
                # Gf matrices use row vectors. ComputeInstanceTransformsAtTime
                # returns PointInstancer-relative transforms with the prototype
                # root transform already included.
                world_transform = (
                    relative_transform * instance_transform * instancer_world
                )
                for (
                    material_suffix,
                    included_faces,
                    material,
                ) in _mesh_material_parts(mesh_prim, sample_time):
                    yield (
                        mesh_prim,
                        synthetic_path + material_suffix,
                        world_transform,
                        mesh_prim.IsInstanceProxy(),
                        f"{instancer_prim.GetPath()}[{instance_index}]",
                        included_faces,
                        material,
                    )


def _fnv1a_64(value: str) -> int:
    result = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        result ^= byte
        result = (result * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return result


def _world_transform_3x4(world: Gf.Matrix4d) -> tuple[float, ...]:
    # USD's Gf matrices use row-vector convention. Vulkan/Metal instance
    # transforms are a row-major 3x4 matrix for column vectors, so transpose
    # the affine part while serializing.
    return (
        float(world[0][0]),
        float(world[1][0]),
        float(world[2][0]),
        float(world[3][0]),
        float(world[0][1]),
        float(world[1][1]),
        float(world[2][1]),
        float(world[3][1]),
        float(world[0][2]),
        float(world[1][2]),
        float(world[2][2]),
        float(world[3][2]),
    )


def _color3(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        result = tuple(float(value[index]) for index in range(3))
    except (IndexError, TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in result):
        return None
    return tuple(min(max(component, 0.0), 1.0) for component in result)


def _color4(value) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        result = tuple(float(value[index]) for index in range(4))
    except (IndexError, TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in result):
        return None
    return result


def _finite_scalar(value, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _scalar01(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return min(max(result, 0.0), 1.0)


def _nonnegative_scalar(value, maximum: float = 1_000_000.0) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return min(max(result, 0.0), maximum)


def _light_shaping_values(prim, world: Gf.Matrix4d, sample_time, schema_code: int):
    """Return the bounded Protocol shaping tail for an applied ShapingAPI."""
    empty = ((0.0,) * 9, 0, "", 0.0, False)
    try:
        if not prim.HasAPI(UsdLux.ShapingAPI):
            return empty
        shaping = UsdLux.ShapingAPI(prim)
        cone_angle = _finite_scalar(
            shaping.GetShapingConeAngleAttr().Get(sample_time), 90.0
        )
        cone_softness = _finite_scalar(
            shaping.GetShapingConeSoftnessAttr().Get(sample_time), 0.0
        )
        focus = _finite_scalar(
            shaping.GetShapingFocusAttr().Get(sample_time), 0.0
        )
        focus_tint_value = shaping.GetShapingFocusTintAttr().Get(sample_time)
        focus_tint = tuple(
            min(max(float(focus_tint_value[index]), 0.0), 1_000_000.0)
            for index in range(3)
        )
        ies_value = shaping.GetShapingIesFileAttr().Get(sample_time)
        ies_angle_scale = _finite_scalar(
            shaping.GetShapingIesAngleScaleAttr().Get(sample_time), 0.0
        )
        ies_normalize = bool(
            shaping.GetShapingIesNormalizeAttr().Get(sample_time) or False
        )
    except Exception:
        return empty

    scalar_values = (
        cone_angle,
        cone_softness,
        focus,
        *focus_tint,
        ies_angle_scale,
    )
    if not all(value is not None and math.isfinite(value) for value in scalar_values):
        return empty

    # CylinderLight's primary emission axis is local +X. Other supported
    # positional lights follow the UsdLux convention and emit along local -Z.
    local_axis = (
        Gf.Vec3d(1.0, 0.0, 0.0)
        if schema_code == 6
        else Gf.Vec3d(0.0, 0.0, -1.0)
    )
    world_axis = world.TransformDir(local_axis)
    axis_length = math.sqrt(sum(float(value) ** 2 for value in world_axis))
    if not math.isfinite(axis_length) or axis_length <= 0.000001:
        return empty
    axis = tuple(float(value) / axis_length for value in world_axis)
    values = (
        *axis,
        min(max(float(cone_angle), 0.0), 180.0),
        min(max(float(cone_softness), 0.0), 1.0),
        min(max(float(focus), 0.0), 1_000_000.0),
        *focus_tint,
    )
    ies_path = _resolve_prim_asset_path(prim, ies_value)
    return (
        values,
        _LIGHT_SHAPING_APPLIED,
        ies_path,
        min(max(float(ies_angle_scale), -1_000_000.0), 1_000_000.0),
        ies_normalize,
    )


def _resolve_prim_asset_path(prim: Usd.Prim, asset_value) -> str:
    """Resolve one SdfAssetPath relative to the stage root layer."""

    try:
        if not asset_value:
            return ""
        resolved = str(getattr(asset_value, "resolvedPath", "") or "")
        authored = str(getattr(asset_value, "path", "") or "")
        if not resolved and authored:
            resolved = prim.GetStage().GetRootLayer().ComputeAbsolutePath(authored)
        return resolved
    except Exception:
        return ""


def _connected_texture_shader(shade_property, visited: set[tuple[str, str]]):
    """Follow NodeGraph outputs to one connected UsdUVTexture shader."""
    try:
        source_info = shade_property.GetConnectedSource()
    except Exception:
        return False, None, ""
    if not source_info:
        return False, None, ""
    try:
        connectable, source_name, _source_type = source_info
        source_prim = connectable.GetPrim()
        source_name = str(source_name)
    except Exception:
        return True, None, ""
    key = (str(source_prim.GetPath()), source_name)
    if key in visited:
        return True, None, ""
    visited.add(key)

    try:
        if source_prim.IsA(UsdShade.NodeGraph):
            output = UsdShade.NodeGraph(source_prim).GetOutput(source_name)
            if not output:
                return True, None, ""
            _connected, shader, output_name = _connected_texture_shader(
                output, visited
            )
            return True, shader, output_name
        if not source_prim.IsA(UsdShade.Shader):
            return True, None, ""
        source_shader = UsdShade.Shader(source_prim)
        shader_id = str(source_shader.GetIdAttr().Get() or "")
        if shader_id != "UsdUVTexture" or source_name not in (
            "r", "g", "b", "a", "rgb", "rgba"
        ):
            return True, None, ""
        return True, source_shader, source_name
    except Exception:
        return True, None, ""


def _connected_source_color(shade_property, visited: set[tuple[str, str]]):
    """Resolve the standard no-file UsdUVTexture fallback color subset."""

    connected, source_shader, _source_name = _connected_texture_shader(
        shade_property, visited
    )
    if not connected:
        return False, None
    if source_shader is None:
        return True, None
    try:

        # UsdUVTexture defines inputs:fallback as the sampled result when no
        # texture asset is supplied. File-backed textures are handled by the
        # separate bounded UV/image transport path, not approximated here.
        file_input = source_shader.GetInput("file")
        file_value = file_input.Get() if file_input else None
        asset_path = getattr(file_value, "path", "") if file_value else ""
        if asset_path:
            return True, None
        fallback_input = source_shader.GetInput("fallback")
        fallback = _color3(fallback_input.Get()) if fallback_input else None
        return True, fallback
    except Exception:
        return True, None


def _bound_material_base_color(material: UsdShade.Material):
    """Return supported Preview/OmniPBR base color and connection status."""

    input_names = (
        "diffuseColor",
        "diffuse_color_constant",
        "diffuse_color",
        "base_color",
        "albedo",
        # The official Simple Warehouse assets use source-asset MDL shaders
        # instead of a UsdPreviewSurface network.  ColorAlbedo is their
        # authored constant fallback before AlbedoTexture/MaskSelection are
        # evaluated by MDL.
        "ColorAlbedo",
        "BaseColor",
    )
    try:
        for prim in Usd.PrimRange(material.GetPrim()):
            if not prim.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(prim)
            for input_name in input_names:
                shader_input = shader.GetInput(input_name)
                if not shader_input:
                    continue
                connected, color = _connected_source_color(shader_input, set())
                if connected:
                    if color is not None:
                        return color, True
                    continue
                color = _color3(shader_input.Get())
                if color is not None:
                    return color, False
    except Exception:
        return None, False
    return None, False


def _bound_material_opacity(material: UsdShade.Material):
    """Resolve standard scalar opacity, cutout threshold, and texture source.

    UsdPreviewSurface permits ``opacity`` to be either a finite scalar or one
    channel of a connected UsdUVTexture.  Preserve the authored
    ``opacityThreshold`` independently; it is not the fixed Warehouse MDL
    threshold.  OmniPBR's direct scalar spellings are accepted as the same
    bounded data model without attempting to execute its MDL implementation.
    """

    opacity_names = ("opacity", "opacity_constant", "opacity_amount")
    threshold_names = ("opacityThreshold", "opacity_threshold")
    channels = {"r": 0, "g": 1, "b": 2, "a": 3}
    try:
        for prim in Usd.PrimRange(material.GetPrim()):
            if not prim.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(prim)
            if not _supported_surface_shader(shader):
                continue

            opacity_input = next(
                (
                    shader.GetInput(name)
                    for name in opacity_names
                    if shader.GetInput(name)
                ),
                None,
            )
            threshold_input = next(
                (
                    shader.GetInput(name)
                    for name in threshold_names
                    if shader.GetInput(name)
                ),
                None,
            )
            if opacity_input is None and threshold_input is None:
                continue

            threshold = _scalar01(
                threshold_input.Get() if threshold_input else 0.0
            )
            if threshold is None:
                continue
            opacity = 1.0
            opacity_texture = None
            if opacity_input is not None and opacity_input.HasConnectedSource():
                _connected, texture_shader, source_name = (
                    _connected_texture_shader(opacity_input, set())
                )
                if texture_shader is None or source_name not in channels:
                    continue
                asset = _texture_asset_and_primvar(material, texture_shader)
                if asset is not None:
                    opacity_texture = (*asset, channels[source_name])
                else:
                    file_input = texture_shader.GetInput("file")
                    file_value = file_input.Get() if file_input else None
                    file_path = str(
                        getattr(file_value, "path", "") or ""
                    ) if file_value else ""
                    fallback_input = texture_shader.GetInput("fallback")
                    fallback = (
                        _color4(fallback_input.Get())
                        if fallback_input else None
                    )
                    if file_path or fallback is None:
                        continue
                    opacity = _scalar01(fallback[channels[source_name]])
                    if opacity is None:
                        continue
            elif opacity_input is not None:
                opacity = _scalar01(opacity_input.Get())
                if opacity is None:
                    continue

            return {
                "opacity": opacity,
                "threshold": threshold,
                "texture": opacity_texture,
                "relevant": (
                    opacity_texture is not None
                    or opacity < 1.0
                    or threshold > 0.0
                ),
            }
    except Exception:
        return None
    return None


def _supported_surface_shader(shader: UsdShade.Shader) -> bool:
    """Recognize the bounded Preview/OmniPBR/source-asset MDL subset."""

    try:
        shader_id = str(shader.GetIdAttr().Get() or "")
        if shader_id == "UsdPreviewSurface" or "OmniPBR" in shader_id:
            return True
        source_asset_attr = shader.GetPrim().GetAttribute("info:mdl:sourceAsset")
        source_asset = source_asset_attr.Get() if source_asset_attr else None
        source_path = str(getattr(source_asset, "path", "") or source_asset or "")
        source_name = source_path.rsplit("/", 1)[-1]
        if source_name == "OmniPBR.mdl":
            return True
        # General MDL execution is deliberately outside this bridge.  Accept
        # only source-asset shaders that expose the concrete input vocabulary
        # used by the official Simple Warehouse materials; the helpers below
        # read those USD-authored values and local assets without executing
        # the MDL program.
        if source_name in {
            "MI_Barcode_0001.mdl",
            "MI_CeilingA_06b.mdl",
            "MI_Floor_01.mdl",
            "MI_FrameA_01.mdl",
            "MI_LampCeilingA.mdl",
            "MI_PushcartA_01.mdl",
            "MI_RackShield_01.mdl",
            "MI_SignB.mdl",
            "MI_WallB_01.mdl",
            "M_AisleSign.mdl",
            "M_Glow.mdl",
            "M_WallBoard_01.mdl",
        }:
            return True
        return source_name.endswith(".mdl") and any(
            shader.GetInput(name)
            for name in (
                "ColorAlbedo",
                "AlbedoTexture",
                "MainNormalInput",
                "MergeMapInput",
            )
        )
    except Exception:
        return False


def _shader_source_asset_info(shader: UsdShade.Shader):
    """Return one MDL source basename and resolved local path."""

    try:
        source_asset_attr = shader.GetPrim().GetAttribute(
            "info:mdl:sourceAsset"
        )
        source_asset = source_asset_attr.Get() if source_asset_attr else None
        authored_path = str(getattr(source_asset, "path", "") or "")
        resolved_path = str(
            getattr(source_asset, "resolvedPath", "") or ""
        )
        if not resolved_path and authored_path:
            layer = shader.GetPrim().GetStage().GetRootLayer()
            resolved_path = layer.ComputeAbsolutePath(authored_path)
        source_name = (resolved_path or authored_path).rsplit("/", 1)[-1]
        return source_name, resolved_path
    except Exception:
        return "", ""


def _direct_asset_path(
    material: UsdShade.Material,
    shader: UsdShade.Shader,
    input_name: str,
):
    """Resolve an authored texture input without requiring a UV network."""

    try:
        texture_input = shader.GetInput(input_name)
        texture_value = texture_input.Get() if texture_input else None
        if not texture_value:
            return ""
        resolved_path = str(
            getattr(texture_value, "resolvedPath", "") or ""
        )
        authored_path = str(getattr(texture_value, "path", "") or "")
        if not resolved_path and authored_path:
            root_layer = material.GetPrim().GetStage().GetRootLayer()
            resolved_path = root_layer.ComputeAbsolutePath(authored_path)
        return resolved_path
    except Exception:
        return ""


def _direct_texture_asset_and_primvar(
    material: UsdShade.Material,
    shader: UsdShade.Shader,
    input_name: str,
):
    """Resolve one source-asset texture input and its authored UV transform."""

    try:
        if not _supported_surface_shader(shader):
            return None
        texture_input = shader.GetInput(input_name)
        texture_value = texture_input.Get() if texture_input else None
        if not texture_value:
            return None
        texture_path = str(getattr(texture_value, "resolvedPath", "") or "")
        authored_path = str(getattr(texture_value, "path", "") or "")
        if not texture_path and authored_path:
            root_layer = material.GetPrim().GetStage().GetRootLayer()
            texture_path = root_layer.ComputeAbsolutePath(authored_path)
        if not texture_path:
            return None

        transforms = ()
        tiling_input = shader.GetInput("MainTiling")
        tiling = tiling_input.Get() if tiling_input else None
        if tiling is not None:
            scale = _float2_or_default(tiling, (1.0, 1.0))
            if scale is None:
                return None
            transforms = ((scale[0], scale[1], 0.0, 0.0, 0.0),)
        else:
            texture_scale_input = shader.GetInput("texture_scale")
            texture_scale = (
                texture_scale_input.Get() if texture_scale_input else None
            )
            scale = _float2_or_default(texture_scale, (1.0, 1.0))
            u_tiling_input = shader.GetInput("U_Tiling")
            v_tiling_input = shader.GetInput("V_Tiling")
            if u_tiling_input:
                u_tiling = _nonnegative_scalar(u_tiling_input.Get())
                if u_tiling is not None:
                    scale = (u_tiling, scale[1])
            if v_tiling_input:
                v_tiling = _nonnegative_scalar(v_tiling_input.Get())
                if v_tiling is not None:
                    scale = (scale[0], v_tiling)
            texture_translation_input = shader.GetInput("texture_translate")
            translation = _float2_or_default(
                texture_translation_input.Get()
                if texture_translation_input else None,
                (0.0, 0.0),
            )
            texture_rotation_input = shader.GetInput("texture_rotate")
            rotation_value = (
                texture_rotation_input.Get() if texture_rotation_input else None
            )
            try:
                rotation = float(rotation_value or 0.0)
            except (TypeError, ValueError):
                return None
            if (
                scale is None
                or translation is None
                or not math.isfinite(rotation)
            ):
                return None
            if scale != (1.0, 1.0) or translation != (0.0, 0.0) or rotation:
                transforms = ((
                    scale[0], scale[1], math.fmod(rotation, 360.0),
                    translation[0], translation[1],
                ),)
        return texture_path, ("st", transforms)
    except Exception:
        return None


def _texture_asset_and_primvar(
    material: UsdShade.Material, texture_shader: UsdShade.Shader
):
    """Resolve one local texture asset and its bounded float2 UV network."""

    try:
        file_input = texture_shader.GetInput("file")
        file_value = file_input.Get() if file_input else None
        if not file_value:
            return None
        texture_path = str(getattr(file_value, "resolvedPath", "") or "")
        authored_path = str(getattr(file_value, "path", "") or "")
        if not texture_path and authored_path:
            root_layer = material.GetPrim().GetStage().GetRootLayer()
            texture_path = root_layer.ComputeAbsolutePath(authored_path)
        if not texture_path:
            return None

        st_input = texture_shader.GetInput("st")
        if not st_input or not st_input.HasConnectedSource():
            uv_source = ("st", ())
        else:
            uv_source = _connected_uv_source(st_input, set())
            if uv_source is None:
                return None
        return texture_path, uv_source
    except Exception:
        return None


def _float2_or_default(value, default: tuple[float, float]):
    """Return one finite float2 without changing authored transform values."""

    if value is None:
        return default
    try:
        result = (float(value[0]), float(value[1]))
    except (IndexError, TypeError, ValueError):
        return None
    return result if all(math.isfinite(component) for component in result) else None


def _connected_uv_source(shade_property, visited: set[tuple[str, str]]):
    """Resolve PrimvarReader plus a bounded chain of standard UsdTransform2d."""

    try:
        source_info = shade_property.GetConnectedSource()
    except Exception:
        return None
    if not source_info:
        return None
    try:
        connectable, source_name, _source_type = source_info
        source_prim = connectable.GetPrim()
        source_name = str(source_name)
    except Exception:
        return None
    key = (str(source_prim.GetPath()), source_name)
    if key in visited:
        return None
    visited.add(key)

    try:
        if source_prim.IsA(UsdShade.NodeGraph):
            output = UsdShade.NodeGraph(source_prim).GetOutput(source_name)
            return _connected_uv_source(output, visited) if output else None
        if not source_prim.IsA(UsdShade.Shader):
            return None
        source_shader = UsdShade.Shader(source_prim)
        shader_id = str(source_shader.GetIdAttr().Get() or "")
        if shader_id == "UsdPrimvarReader_float2" and source_name == "result":
            varname_input = source_shader.GetInput("varname")
            varname = varname_input.Get() if varname_input else None
            return (str(varname or "st"), ())
        if shader_id != "UsdTransform2d" or source_name != "result":
            return None

        input_property = source_shader.GetInput("in")
        uv_source = (
            _connected_uv_source(input_property, visited)
            if input_property and input_property.HasConnectedSource()
            else None
        )
        if uv_source is None:
            return None
        scale_input = source_shader.GetInput("scale")
        rotation_input = source_shader.GetInput("rotation")
        translation_input = source_shader.GetInput("translation")
        scale = _float2_or_default(
            scale_input.Get() if scale_input else None, (1.0, 1.0)
        )
        translation = _float2_or_default(
            translation_input.Get() if translation_input else None, (0.0, 0.0)
        )
        rotation_value = rotation_input.Get() if rotation_input else None
        rotation = 0.0 if rotation_value is None else float(rotation_value)
        if scale is None or translation is None or not math.isfinite(rotation):
            return None
        transform = (
            scale[0],
            scale[1],
            math.fmod(rotation, 360.0),
            translation[0],
            translation[1],
        )
        return (uv_source[0], uv_source[1] + (transform,))
    except Exception:
        return None


def _bound_material_file_texture(material: UsdShade.Material):
    """Return a connected UsdUVTexture file and its bounded UV source."""

    input_names = (
        "diffuseColor",
        "diffuse_texture",
        "diffuse_color_texture",
        "base_color_texture",
        "albedo_texture",
    )
    try:
        for prim in Usd.PrimRange(material.GetPrim()):
            if not prim.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(prim)
            for input_name in input_names:
                shader_input = shader.GetInput(input_name)
                if not shader_input:
                    continue
                _connected, texture_shader, source_name = (
                    _connected_texture_shader(shader_input, set())
                )
                if texture_shader is None or source_name not in ("rgb", "rgba"):
                    continue
                asset = _texture_asset_and_primvar(material, texture_shader)
                if asset is not None:
                    return asset
            for direct_name in (
                "AlbedoTexture",
                "diffuse_texture",
                "BaseColor_Texture",
                "base_color_texture",
                "albedo_texture",
            ):
                direct_asset = _direct_texture_asset_and_primvar(
                    material, shader, direct_name
                )
                if direct_asset is not None:
                    return direct_asset
    except Exception:
        return None
    return None


def _bound_material_parameter_texture(
    material: UsdShade.Material,
    input_names: tuple[str, ...],
    color_output: bool = False,
):
    """Return a file, shared primvar, and sampled channel for one parameter."""

    channels = {"r": 0, "g": 1, "b": 2, "a": 3, "rgb": 0, "rgba": 0}
    try:
        for prim in Usd.PrimRange(material.GetPrim()):
            if not prim.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(prim)
            if not _supported_surface_shader(shader):
                continue
            for input_name in input_names:
                shader_input = shader.GetInput(input_name)
                if not shader_input:
                    continue
                _connected, texture_shader, source_name = (
                    _connected_texture_shader(shader_input, set())
                )
                if texture_shader is None:
                    continue
                if color_output and source_name not in ("rgb", "rgba"):
                    continue
                if not color_output and source_name not in channels:
                    continue
                asset = _texture_asset_and_primvar(material, texture_shader)
                if asset is not None:
                    texture_path, uv_source = asset
                    return (
                        texture_path,
                        uv_source,
                        4 if color_output else channels[source_name],
                    )
            requested_names = set(input_names)
            direct_name = None
            direct_channel = 4 if color_output else 0
            if requested_names & {"roughness", "reflection_roughness_constant"}:
                # Simple Warehouse MergeMapInput follows the authored ORM
                # convention: occlusion=R, roughness=G, metallic=B.
                direct_name, direct_channel = "MergeMapInput", 1
            elif requested_names & {"metallic", "metallic_constant"}:
                direct_name, direct_channel = "MergeMapInput", 2
            elif color_output and requested_names & {
                "normal", "normalmap", "normal_map"
            }:
                direct_name, direct_channel = "MainNormalInput", 4
            direct_candidates = []
            if direct_name is not None:
                direct_candidates.append((direct_name, direct_channel))
            if requested_names & {"roughness", "reflection_roughness_constant"}:
                direct_candidates.extend((
                    ("ORM_texture", 1),
                    ("reflectionroughness_texture", 0),
                    ("roughness_texture", 0),
                ))
            elif requested_names & {"metallic", "metallic_constant"}:
                direct_candidates.extend((
                    ("ORM_texture", 2),
                    ("metallic_texture", 0),
                ))
            elif color_output and requested_names & {
                "normal", "normalmap", "normal_map"
            }:
                direct_candidates.extend((
                    ("normalmap_texture", 4),
                    ("normal_texture", 4),
                ))
            for candidate_name, candidate_channel in direct_candidates:
                direct_asset = _direct_texture_asset_and_primvar(
                    material, shader, candidate_name
                )
                if direct_asset is not None:
                    return (*direct_asset, candidate_channel)
    except Exception:
        return None
    return None


def _bound_material_parameters(material: UsdShade.Material):
    """Return bounded direct roughness and metallic values when supported."""

    roughness_names = ("roughness", "reflection_roughness_constant")
    metallic_names = ("metallic", "metallic_constant")
    try:
        for prim in Usd.PrimRange(material.GetPrim()):
            if not prim.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(prim)
            if not _supported_surface_shader(shader):
                continue
            roughness = None
            metallic = None
            for name in roughness_names:
                shader_input = shader.GetInput(name)
                if shader_input and not shader_input.HasConnectedSource():
                    roughness = _scalar01(shader_input.Get())
                    if roughness is not None:
                        break
            for name in metallic_names:
                shader_input = shader.GetInput(name)
                if shader_input and not shader_input.HasConnectedSource():
                    metallic = _scalar01(shader_input.Get())
                    if metallic is not None:
                        break
            if roughness is None:
                roughness_min_input = shader.GetInput("RoughnessMin")
                roughness_max_input = shader.GetInput("RoughnessMax")
                roughness_min = _scalar01(
                    roughness_min_input.Get() if roughness_min_input else None
                )
                roughness_max = _scalar01(
                    roughness_max_input.Get() if roughness_max_input else None
                )
                if roughness_min is not None and roughness_max is not None:
                    roughness = (roughness_min + roughness_max) * 0.5
            if roughness is not None or metallic is not None:
                return (
                    roughness if roughness is not None else 0.5,
                    metallic if metallic is not None else 0.0,
                )
    except Exception:
        return None
    return None


def _bound_material_emission(material: UsdShade.Material):
    """Return bounded direct emissive color and intensity when supported."""

    color_names = (
        "emissiveColor",
        "emissive_color",
        "emissive_color_constant",
        "emission_color",
    )
    intensity_names = ("emissive_intensity", "emission_intensity")
    try:
        for prim in Usd.PrimRange(material.GetPrim()):
            if not prim.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(prim)
            if not _supported_surface_shader(shader):
                continue
            enabled_input = shader.GetInput("enable_emission")
            if enabled_input and enabled_input.Get() is False:
                continue
            emission_color = None
            for name in color_names:
                shader_input = shader.GetInput(name)
                if shader_input and not shader_input.HasConnectedSource():
                    emission_color = _color3(shader_input.Get())
                    if emission_color is not None:
                        break
            if emission_color is None:
                continue
            intensity = 1.0
            for name in intensity_names:
                shader_input = shader.GetInput(name)
                if shader_input and not shader_input.HasConnectedSource():
                    value = _nonnegative_scalar(shader_input.Get())
                    if value is not None:
                        intensity = value
                        break
            return emission_color, intensity
    except Exception:
        return None
    return None


@functools.lru_cache(maxsize=256)
def _load_rgba_texture(path: str):
    """Decode a bounded local material asset into tightly packed RGBA8."""

    try:
        from PIL import Image

        with Image.open(path) as source:
            source.load()
            if (
                source.width <= 0
                or source.height <= 0
                or source.width > _MAX_SCENE_TEXTURE_DIMENSION
                or source.height > _MAX_SCENE_TEXTURE_DIMENSION
            ):
                return None
            rgba = source.convert("RGBA")
            if (
                rgba.width > _MAX_SCENE_TEXTURE_TRANSPORT_DIMENSION
                or rgba.height > _MAX_SCENE_TEXTURE_TRANSPORT_DIMENSION
            ):
                rgba.thumbnail(
                    (
                        _MAX_SCENE_TEXTURE_TRANSPORT_DIMENSION,
                        _MAX_SCENE_TEXTURE_TRANSPORT_DIMENSION,
                    ),
                    Image.Resampling.LANCZOS,
                )
            pixels = rgba.tobytes()
            if len(pixels) != rgba.width * rgba.height * 4:
                return None
            return rgba.width, rgba.height, pixels
    except Exception:
        return None


@functools.lru_cache(maxsize=32)
def _load_radiance_rgbe(path: str):
    """Decode one bounded Radiance HDR latlong image without clipping HDR.

    The four transported bytes remain Radiance RGBE.  Metal expands them to
    linear radiance at sample time, so values above one survive the RGBA8
    scene sideband.
    """

    try:
        with open(path, "rb") as source:
            signature = source.readline().rstrip(b"\r\n")
            if signature not in (b"#?RADIANCE", b"#?RGBE"):
                return None
            has_rgbe_format = False
            while True:
                line = source.readline()
                if not line:
                    return None
                stripped = line.strip()
                if stripped == b"FORMAT=32-bit_rle_rgbe":
                    has_rgbe_format = True
                if not stripped:
                    break
            if not has_rgbe_format:
                return None
            resolution = source.readline().decode("ascii", "strict").strip()
            match = re.fullmatch(
                r"([+-])Y\s+(\d+)\s+([+-])X\s+(\d+)", resolution
            )
            if match is None:
                return None
            y_sign, height_text, x_sign, width_text = match.groups()
            width = int(width_text)
            height = int(height_text)
            if (
                width <= 0
                or height <= 0
                or width > _MAX_SCENE_TEXTURE_DIMENSION
                or height > _MAX_SCENE_TEXTURE_DIMENSION
            ):
                return None

            rows = []
            for _ in range(height):
                prefix = source.read(4)
                if (
                    len(prefix) != 4
                    or width < 8
                    or width > 32767
                    or prefix[0] != 2
                    or prefix[1] != 2
                    or ((prefix[2] << 8) | prefix[3]) != width
                ):
                    return None
                channels = [bytearray() for _ in range(4)]
                for channel in channels:
                    while len(channel) < width:
                        packet = source.read(2)
                        if len(packet) != 2 or packet[0] == 0:
                            return None
                        if packet[0] > 128:
                            run = packet[0] - 128
                            if run == 0 or len(channel) + run > width:
                                return None
                            channel.extend(bytes((packet[1],)) * run)
                        else:
                            literal = packet[0]
                            if len(channel) + literal > width:
                                return None
                            channel.append(packet[1])
                            remainder = source.read(literal - 1)
                            if len(remainder) != literal - 1:
                                return None
                            channel.extend(remainder)
                row = bytearray(width * 4)
                for x in range(width):
                    for channel_index in range(4):
                        row[x * 4 + channel_index] = channels[channel_index][x]
                if x_sign == "-":
                    row = bytearray(
                        b"".join(
                            row[x * 4 : x * 4 + 4]
                            for x in range(width - 1, -1, -1)
                        )
                    )
                rows.append(row)
            if y_sign == "+":
                rows.reverse()

        target_scale = min(
            1.0,
            _MAX_SCENE_TEXTURE_TRANSPORT_DIMENSION / width,
            _MAX_SCENE_TEXTURE_TRANSPORT_DIMENSION / height,
        )
        target_width = max(1, int(width * target_scale))
        target_height = max(1, int(height * target_scale))
        if target_width == width and target_height == height:
            pixels = b"".join(rows)
        else:
            resized = bytearray(target_width * target_height * 4)
            for target_y in range(target_height):
                source_y = min(int(target_y * height / target_height), height - 1)
                source_row = rows[source_y]
                for target_x in range(target_width):
                    source_x = min(int(target_x * width / target_width), width - 1)
                    source_offset = source_x * 4
                    target_offset = (target_y * target_width + target_x) * 4
                    resized[target_offset : target_offset + 4] = source_row[
                        source_offset : source_offset + 4
                    ]
            pixels = bytes(resized)
        return target_width, target_height, pixels
    except Exception:
        return None


def _load_dome_texture(path: str):
    """Return a bounded latlong image and whether its bytes are linear RGBE."""

    extension = os.path.splitext(path)[1].lower()
    if extension == ".exr":
        # Pillow availability varies across Kit builds, and an implicit
        # float-to-RGBA8 conversion would silently clip high-range radiance.
        return None
    if extension in (".hdr", ".pic"):
        decoded = _load_radiance_rgbe(path)
        return None if decoded is None else (*decoded, True)
    decoded = _load_rgba_texture(path)
    return None if decoded is None else (*decoded, False)


def _ies_axis_interval(values: tuple[float, ...], value: float):
    """Return bounded interpolation indices and weight for a sorted axis."""

    if not values:
        return None
    if len(values) == 1 or value <= values[0]:
        return 0, 0, 0.0
    if value >= values[-1]:
        last = len(values) - 1
        return last, last, 0.0
    upper = bisect.bisect_right(values, value)
    lower = upper - 1
    span = values[upper] - values[lower]
    if span <= 0.0:
        return lower, lower, 0.0
    return lower, upper, (value - values[lower]) / span


def _ies_horizontal_angle(angle: float, angles: tuple[float, ...]) -> float:
    """Apply the common LM-63 Type-C horizontal symmetry conventions."""

    if len(angles) <= 1:
        return angles[0] if angles else 0.0
    angle = math.fmod(angle, 360.0)
    if angle < 0.0:
        angle += 360.0
    maximum = angles[-1]
    if maximum <= 90.0001:
        quadrant = math.fmod(angle, 180.0)
        return min(quadrant, 180.0 - quadrant)
    if maximum <= 180.0001:
        return angle if angle <= 180.0 else 360.0 - angle
    return angle


def _sample_ies_type_c(
    vertical_angles: tuple[float, ...],
    horizontal_angles: tuple[float, ...],
    candela_rows: tuple[tuple[float, ...], ...],
    theta_degrees: float,
    phi_degrees: float,
) -> float:
    """Bilinearly sample a bounded LM-63 Type-C candela table."""

    vertical = _ies_axis_interval(vertical_angles, theta_degrees)
    horizontal_value = _ies_horizontal_angle(phi_degrees, horizontal_angles)
    horizontal = _ies_axis_interval(horizontal_angles, horizontal_value)
    if vertical is None or horizontal is None:
        return 0.0
    v0, v1, vertical_weight = vertical
    h0, h1, horizontal_weight = horizontal
    lower = candela_rows[h0][v0] + (
        candela_rows[h0][v1] - candela_rows[h0][v0]
    ) * vertical_weight
    upper = candela_rows[h1][v0] + (
        candela_rows[h1][v1] - candela_rows[h1][v0]
    ) * vertical_weight
    return max(lower + (upper - lower) * horizontal_weight, 0.0)


@functools.lru_cache(maxsize=64)
def _load_ies_profile(path: str, normalize: bool):
    """Decode an LM-63 Type-C profile into a linear angular RGBA8 LUT.

    The LUT stores candela divided by the profile maximum.  A separate finite
    multiplier restores authored candela response, or applies the OpenUSD
    energy-preserving normalization requested by shaping:ies:normalize.
    """

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as source:
            lines = source.read().replace("\r", "").split("\n")
        tilt_index = next(
            index
            for index, line in enumerate(lines)
            if line.strip().upper().startswith("TILT=")
        )
        tilt_mode = lines[tilt_index].split("=", 1)[1].strip().upper()
        if tilt_mode not in ("NONE", "INCLUDE"):
            return None
        numeric_text = " ".join(lines[tilt_index + 1 :])
        number_strings = re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?",
            numeric_text,
        )
        numbers = [float(value) for value in number_strings]
        offset = 0
        if tilt_mode == "INCLUDE":
            if len(numbers) < 2:
                return None
            pair_count = int(numbers[1])
            if pair_count < 0 or pair_count > 4096:
                return None
            offset = 2 + pair_count * 2
        if len(numbers) < offset + 13:
            return None
        header = numbers[offset : offset + 13]
        offset += 13
        candela_multiplier = float(header[2])
        vertical_count = int(header[3])
        horizontal_count = int(header[4])
        photometric_type = int(header[5])
        if (
            photometric_type != 1
            or vertical_count <= 0
            or horizontal_count <= 0
            or vertical_count > 721
            or horizontal_count > 721
            or not math.isfinite(candela_multiplier)
            or candela_multiplier <= 0.0
        ):
            return None
        required = vertical_count + horizontal_count
        required += vertical_count * horizontal_count
        if len(numbers) < offset + required:
            return None
        vertical_angles = tuple(
            float(value) for value in numbers[offset : offset + vertical_count]
        )
        offset += vertical_count
        horizontal_angles = tuple(
            float(value) for value in numbers[offset : offset + horizontal_count]
        )
        offset += horizontal_count
        if (
            not all(math.isfinite(value) for value in vertical_angles)
            or not all(math.isfinite(value) for value in horizontal_angles)
            or any(
                vertical_angles[index] >= vertical_angles[index + 1]
                for index in range(len(vertical_angles) - 1)
            )
            or any(
                horizontal_angles[index] >= horizontal_angles[index + 1]
                for index in range(len(horizontal_angles) - 1)
            )
        ):
            return None
        rows = []
        maximum = 0.0
        for horizontal_index in range(horizontal_count):
            row = tuple(
                max(float(value) * candela_multiplier, 0.0)
                for value in numbers[
                    offset
                    + horizontal_index * vertical_count : offset
                    + (horizontal_index + 1) * vertical_count
                ]
            )
            if not all(math.isfinite(value) for value in row):
                return None
            maximum = max(maximum, max(row))
            rows.append(row)
        if maximum <= 0.0:
            return None
        candela_rows = tuple(rows)
        width = 128
        height = 64
        sampled_values = []
        integral = 0.0
        theta_step = math.pi / height
        phi_step = 2.0 * math.pi / width
        for y in range(height):
            theta = (y + 0.5) * theta_step
            for x in range(width):
                phi = (x + 0.5) * phi_step
                sample = _sample_ies_type_c(
                    vertical_angles,
                    horizontal_angles,
                    candela_rows,
                    math.degrees(theta),
                    math.degrees(phi),
                )
                sampled_values.append(sample)
                integral += sample * math.sin(theta) * theta_step * phi_step
        if not math.isfinite(integral) or integral <= 0.0:
            return None
        multiplier = (
            maximum * 4.0 * math.pi / integral if normalize else maximum
        )
        if not math.isfinite(multiplier) or multiplier < 0.0:
            return None
        multiplier = min(multiplier, 1_000_000.0)
        pixels = bytearray(width * height * 4)
        for index, sample in enumerate(sampled_values):
            encoded = int(round(min(max(sample / maximum, 0.0), 1.0) * 255.0))
            pixel_offset = index * 4
            pixels[pixel_offset : pixel_offset + 4] = bytes(
                (encoded, encoded, encoded, 255)
            )
        return width, height, bytes(pixels), multiplier
    except Exception:
        return None


def _srgb_to_linear(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    if value <= 0.04045:
        return value / 12.92
    return math.pow((value + 0.055) / 1.055, 2.4)


def _linear_to_srgb(value: float) -> int:
    value = min(max(value, 0.0), 1.0)
    encoded = (
        value * 12.92
        if value <= 0.0031308
        else 1.055 * math.pow(value, 1.0 / 2.4) - 0.055
    )
    return int(round(min(max(encoded, 0.0), 1.0) * 255.0))


@functools.lru_cache(maxsize=256)
def _resized_rgba_texture(path: str, width: int, height: int):
    decoded = _load_rgba_texture(path)
    if decoded is None:
        return None
    source_width, source_height, pixels = decoded
    if source_width == width and source_height == height:
        return pixels
    try:
        from PIL import Image

        image = Image.frombytes(
            "RGBA", (source_width, source_height), pixels
        )
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image.tobytes()
    except Exception:
        return None


@functools.lru_cache(maxsize=256)
def _compose_standard_opacity_texture(
    base_path: str,
    opacity_path: str,
    opacity_channel: int,
    base_color: tuple[float, float, float],
):
    """Bake one opacity channel into base alpha without changing base RGB."""

    if opacity_channel < 0 or opacity_channel > 3:
        return None
    opacity_texture = _load_rgba_texture(opacity_path)
    if opacity_texture is None:
        return None
    base_texture = _load_rgba_texture(base_path) if base_path else None
    reference = base_texture or opacity_texture
    width, height, _pixels = reference
    opacity_pixels = _resized_rgba_texture(opacity_path, width, height)
    if opacity_pixels is None:
        return None
    if base_texture is not None:
        base_pixels = _resized_rgba_texture(base_path, width, height)
        if base_pixels is None:
            return None
    else:
        encoded = bytes(
            _linear_to_srgb(component) for component in base_color
        ) + b"\xff"
        base_pixels = encoded * (width * height)
    output = bytearray(base_pixels)
    for offset in range(0, len(output), 4):
        output[offset + 3] = opacity_pixels[offset + opacity_channel]
    return width, height, bytes(output)


@functools.lru_cache(maxsize=128)
def _compose_base_texture(
    mode: str,
    albedo_path: str,
    mask_path: str,
    color_a: tuple[float, float, float, float],
    color_b: tuple[float, float, float, float],
    color_c: tuple[float, float, float, float],
    scalar: float,
):
    """Bake the exact bounded color algebra used by Warehouse MDL assets."""

    albedo = _load_rgba_texture(albedo_path) if albedo_path else None
    mask = _load_rgba_texture(mask_path) if mask_path else None
    reference = albedo or mask
    if reference is None:
        return None
    width, height, _pixels = reference
    albedo_pixels = (
        _resized_rgba_texture(albedo_path, width, height)
        if albedo_path
        else bytes((0, 0, 0, 255)) * (width * height)
    )
    mask_pixels = (
        _resized_rgba_texture(mask_path, width, height)
        if mask_path
        else bytes((255, 255, 255, 255)) * (width * height)
    )
    if albedo_pixels is None or mask_pixels is None:
        return None

    output = bytearray(width * height * 4)
    for offset in range(0, len(output), 4):
        albedo_linear = tuple(
            _srgb_to_linear(albedo_pixels[offset + channel] / 255.0)
            for channel in range(3)
        )
        mask_value = tuple(
            mask_pixels[offset + channel] / 255.0 for channel in range(4)
        )
        if mode == "channel-mask":
            result = tuple(
                color_a[channel]
                + (albedo_linear[channel] - color_a[channel])
                    * mask_value[channel]
                for channel in range(3)
            )
            alpha = 1.0
        elif mode == "pushcart-mask":
            result_values = []
            for channel in range(3):
                value = color_a[channel] * mask_value[0]
                value += (color_b[channel] - value) * mask_value[1]
                value += (color_c[channel] - value) * mask_value[2]
                value += (albedo_linear[channel] - value) * mask_value[3]
                result_values.append(value)
            result = tuple(result_values)
            alpha = 1.0
        elif mode == "tint-desaturate":
            grayscale = (
                albedo_linear[0] * 0.3
                + albedo_linear[1] * 0.59
                + albedo_linear[2] * 0.11
            )
            amount = min(max(scalar, 0.0), 1.0)
            result = tuple(
                (value + (grayscale - value) * amount) * color_a[channel]
                for channel, value in enumerate(albedo_linear)
            )
            alpha = 1.0
        elif mode == "multiply-tint":
            result = tuple(
                albedo_linear[channel] * color_a[channel]
                for channel in range(3)
            )
            alpha = albedo_pixels[offset + 3] / 255.0
        elif mode == "alpha-mask":
            result = albedo_linear
            # OmniUe4Function::greyscale_texture_lookup replicates the red
            # channel, and M_WallBoard_01 applies this exact 0.3333 cutout.
            alpha = 1.0 if mask_value[0] >= 0.3333 else 0.0
        elif mode == "aisle-sign":
            # Text is declared gamma_linear while the sign diffuse image is
            # gamma_srgb. Reproduce the source MDL's per-channel overlay.
            text_value = tuple(mask_value[channel] for channel in range(3))
            result_values = []
            for channel in range(3):
                text = text_value[channel]
                diffuse = albedo_linear[channel]
                inverse_branch = 1.0 - (1.0 - text) * 2.0 * (1.0 - diffuse)
                multiply_branch = text * 2.0 * diffuse
                result_values.append(
                    inverse_branch if text >= 0.5 else multiply_branch
                )
            result = tuple(result_values)
            alpha = 1.0
        else:
            return None
        for channel in range(3):
            output[offset + channel] = _linear_to_srgb(result[channel])
        output[offset + 3] = int(round(min(max(alpha, 0.0), 1.0) * 255.0))
    return width, height, bytes(output)


@functools.lru_cache(maxsize=128)
def _remap_orm_texture(path: str, roughness_min: float, roughness_max: float):
    decoded = _load_rgba_texture(path)
    if decoded is None:
        return None
    width, height, pixels = decoded
    output = bytearray(pixels)
    for offset in range(0, len(output), 4):
        source = pixels[offset + 1] / 255.0
        roughness = roughness_min + (roughness_max - roughness_min) * source
        output[offset + 1] = int(
            round(min(max(roughness, 0.0), 1.0) * 255.0)
        )
    return width, height, bytes(output)


@functools.lru_cache(maxsize=128)
def _reconstruct_normal_texture(
    path: str, strength: tuple[float, float, float, float]
):
    decoded = _load_rgba_texture(path)
    if decoded is None:
        return None
    width, height, pixels = decoded
    output = bytearray(pixels)
    for offset in range(0, len(output), 4):
        x = pixels[offset] / 255.0 * 2.0 - 1.0
        y = pixels[offset + 1] / 255.0 * 2.0 - 1.0
        z = math.sqrt(max(1.0 - x * x - y * y, 0.0))
        x *= strength[0]
        y *= strength[1]
        z *= strength[2]
        length = math.sqrt(x * x + y * y + z * z)
        if length > 0.000001:
            x, y, z = x / length, y / length, z / length
        output[offset] = int(round((min(max(x, -1.0), 1.0) * 0.5 + 0.5) * 255.0))
        output[offset + 1] = int(round((min(max(y, -1.0), 1.0) * 0.5 + 0.5) * 255.0))
        output[offset + 2] = int(round((min(max(z, -1.0), 1.0) * 0.5 + 0.5) * 255.0))
        output[offset + 3] = 255
    return width, height, bytes(output)


def _warehouse_mdl_material_bundle(material: UsdShade.Material):
    """Normalize the official Simple Warehouse MDL materials into v13 maps.

    This is intentionally a named, bounded source-asset subset.  It evaluates
    only the concrete color/ORM/normal algebra present in NVIDIA's shipped
    Simple Warehouse files; it is not a general MDL interpreter.
    """

    official_names = {
        "MI_Barcode_0001.mdl",
        "MI_CeilingA_06b.mdl",
        "MI_Floor_01.mdl",
        "MI_FrameA_01.mdl",
        "MI_LampCeilingA.mdl",
        "MI_PushcartA_01.mdl",
        "MI_RackShield_01.mdl",
        "MI_SignB.mdl",
        "MI_WallB_01.mdl",
        "M_AisleSign.mdl",
        "M_Glow.mdl",
        "M_WallBoard_01.mdl",
    }
    try:
        shader = None
        source_name = ""
        source_path = ""
        for prim in Usd.PrimRange(material.GetPrim()):
            if not prim.IsA(UsdShade.Shader):
                continue
            candidate = UsdShade.Shader(prim)
            candidate_name, candidate_path = _shader_source_asset_info(
                candidate
            )
            if candidate_name in official_names:
                shader = candidate
                source_name = candidate_name
                source_path = candidate_path
                break
        if shader is None:
            return None

        source_directory = os.path.dirname(source_path)
        texture_directory = os.path.join(source_directory, "Textures")

        def input_value(name, default=None):
            shader_input = shader.GetInput(name)
            value = shader_input.Get() if shader_input else None
            return default if value is None else value

        def input_color(name, default):
            return _color4(input_value(name)) or default

        def input_scalar(name, default):
            return _finite_scalar(input_value(name), default)

        def input_asset(name, default_name=""):
            path = _direct_asset_path(material, shader, name)
            if path:
                return path
            return (
                os.path.join(texture_directory, default_name)
                if texture_directory and default_name
                else ""
            )

        def uv_for(name):
            direct = _direct_texture_asset_and_primvar(
                material, shader, name
            )
            return direct[1] if direct is not None else ("st", ())

        bundle = {
            "base_color": None,
            "material_parameters": None,
            "material_emission": None,
            "alpha_cutout": False,
            "texture_specs": {},
            "decoded": {},
        }

        def add_original(kind, path, uv_source, channel):
            if path and _load_rgba_texture(path) is not None:
                bundle["texture_specs"][kind] = (
                    path, uv_source, channel
                )

        def add_derived(kind, key, uv_source, channel, decoded):
            if decoded is None:
                return
            bundle["texture_specs"][kind] = (key, uv_source, channel)
            bundle["decoded"][key] = decoded

        if source_name == "M_Glow.mdl":
            color = input_color(
                "EmissiveColor", (0.28835, 0.365, 0.365, 1.0)
            )
            strength = max(input_scalar("EmissiveStrength", 10.0), 0.0)
            bundle["base_color"] = (0.0, 0.0, 0.0)
            bundle["material_parameters"] = (0.5, 0.0)
            bundle["material_emission"] = (color[:3], strength)
            return bundle

        if source_name == "MI_SignB.mdl":
            path = input_asset("TextureSelection", "T_SignsA_D.png")
            add_original("base", path, uv_for("TextureSelection"), 4)
            bundle["material_parameters"] = (0.125, 0.0)
            return bundle

        if source_name == "MI_Barcode_0001.mdl":
            albedo_path = input_asset("BaseColor_Texture", "0001.png")
            tint = input_color(
                "BaseColor_Tint", (1.0, 1.0, 1.0, 1.0)
            )
            decoded = _compose_base_texture(
                "multiply-tint", albedo_path, "", tint,
                (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 0.0,
            )
            key = f"imb-derived:barcode:{albedo_path}:{tint}"
            add_derived(
                "base", key, uv_for("BaseColor_Texture"), 4, decoded
            )
            bundle["alpha_cutout"] = True
            bundle["material_parameters"] = (
                min(max(input_scalar("Roughness", 0.3), 0.0), 1.0),
                min(max(input_scalar("Metallic", 0.05), 0.0), 1.0),
            )
            return bundle

        if source_name == "M_AisleSign.mdl":
            albedo_path = os.path.join(
                texture_directory, "T_AisleSign_D.png"
            )
            text_path = input_asset("Text", "AisleSign_Text_01.png")
            normal_path = os.path.join(
                texture_directory, "T_AisleSign_N.png"
            )
            orm_path = os.path.join(
                texture_directory, "T_AisleSign_ORM.png"
            )
            decoded = _compose_base_texture(
                "aisle-sign", albedo_path, text_path,
                (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0), 0.0,
            )
            key = f"imb-derived:aisle:{albedo_path}:{text_path}"
            add_derived("base", key, ("st", ()), 4, decoded)
            add_original("roughness", orm_path, ("st", ()), 1)
            add_original("metallic", orm_path, ("st", ()), 2)
            normal = _reconstruct_normal_texture(
                normal_path, (1.0, 1.0, 1.0, 1.0)
            )
            normal_key = f"imb-derived:normal:{normal_path}:1,1,1"
            add_derived("normal", normal_key, ("st", ()), 4, normal)
            bundle["material_parameters"] = (0.5, 0.0)
            return bundle

        albedo_path = input_asset("AlbedoTexture")
        normal_path = input_asset("MainNormalInput")
        orm_path = input_asset("MergeMapInput")
        base_uv = uv_for("AlbedoTexture")
        shared_uv = base_uv

        if source_name in {
            "MI_CeilingA_06b.mdl",
            "MI_Floor_01.mdl",
            "MI_FrameA_01.mdl",
            "MI_WallB_01.mdl",
        }:
            mask_path = input_asset("MaskSelection")
            constant = input_color(
                "ColorAlbedo", (0.145, 0.145, 0.145, 0.0)
            )
            decoded = _compose_base_texture(
                "channel-mask", albedo_path, mask_path, constant,
                (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 0.0,
            )
            key = (
                f"imb-derived:channel-mask:{albedo_path}:{mask_path}:"
                f"{constant}"
            )
            add_derived("base", key, shared_uv, 4, decoded)
        elif source_name == "MI_PushcartA_01.mdl":
            mask_path = input_asset("MaskSelection")
            body = input_color("Body", (0.128, 0.128, 0.128, 1.0))
            handle = input_color("Handle", (0.128, 0.128, 0.128, 1.0))
            cap = input_color("Cap", (0.128, 0.128, 0.128, 1.0))
            decoded = _compose_base_texture(
                "pushcart-mask", albedo_path, mask_path,
                body, handle, cap, 0.0,
            )
            key = (
                f"imb-derived:pushcart:{albedo_path}:{mask_path}:"
                f"{body}:{handle}:{cap}"
            )
            add_derived("base", key, shared_uv, 4, decoded)
        elif source_name in {
            "MI_LampCeilingA.mdl", "MI_RackShield_01.mdl"
        }:
            tint = input_color(
                "BaseColor_Tint", (1.0, 1.0, 1.0, 1.0)
            )
            desaturation = input_scalar("Desaturation", 0.0)
            decoded = _compose_base_texture(
                "tint-desaturate", albedo_path, "", tint,
                (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0),
                desaturation,
            )
            key = (
                f"imb-derived:tint:{albedo_path}:{tint}:{desaturation}"
            )
            add_derived("base", key, shared_uv, 4, decoded)
        elif source_name == "M_WallBoard_01.mdl":
            alpha_path = input_asset("AlphaSelection")
            decoded = _compose_base_texture(
                "alpha-mask", albedo_path, alpha_path,
                (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0), 0.0,
            )
            key = f"imb-derived:wallboard:{albedo_path}:{alpha_path}"
            add_derived("base", key, shared_uv, 4, decoded)
            bundle["alpha_cutout"] = True

        roughness_min = input_scalar("RoughnessMin", 0.1)
        roughness_max = input_scalar("RoughnessMax", 0.9)
        remapped_orm = _remap_orm_texture(
            orm_path, roughness_min, roughness_max
        )
        orm_key = (
            f"imb-derived:orm:{orm_path}:{roughness_min}:{roughness_max}"
        )
        add_derived("roughness", orm_key, shared_uv, 1, remapped_orm)
        add_derived("metallic", orm_key, shared_uv, 2, remapped_orm)

        normal_strength = input_color(
            "MainNormalStrenght", (1.0, 1.0, 1.0, 1.0)
        )
        reconstructed_normal = _reconstruct_normal_texture(
            normal_path, normal_strength
        )
        normal_key = (
            f"imb-derived:normal:{normal_path}:{normal_strength}"
        )
        add_derived(
            "normal", normal_key, shared_uv, 4, reconstructed_normal
        )
        bundle["material_parameters"] = (0.5, 0.0)
        return bundle
    except Exception:
        return None


def _pack_mesh_base_color(color: tuple[float, float, float]) -> int:
    red, green, blue = (int(round(component * 255.0)) for component in color)
    return (red << 8) | (green << 16) | (blue << 24)


def _triangulated_corner_normals(
    mesh: UsdGeom.Mesh,
    corner_sources: list[tuple[int, int, int]],
    sample_time: Usd.TimeCode,
) -> tuple[float, ...]:
    """Flatten authored USD normals into one normal per triangle corner."""

    try:
        normals = mesh.GetNormalsAttr().Get(sample_time)
        interpolation = str(mesh.GetNormalsInterpolation())
    except Exception:
        return ()
    if not normals or not corner_sources:
        return ()

    values = []
    for face_index, point_index, face_vertex_index in corner_sources:
        if interpolation == "constant":
            normal_index = 0
        elif interpolation == "uniform":
            normal_index = face_index
        elif interpolation in ("vertex", "varying"):
            normal_index = point_index
        elif interpolation == "faceVarying":
            normal_index = face_vertex_index
        else:
            return ()
        if normal_index < 0 or normal_index >= len(normals):
            return ()
        try:
            normal = tuple(float(normals[normal_index][axis]) for axis in range(3))
        except (IndexError, TypeError, ValueError):
            return ()
        length_squared = sum(component * component for component in normal)
        if (
            not all(math.isfinite(component) for component in normal)
            or not math.isfinite(length_squared)
            or length_squared <= 0.000000000001
        ):
            return ()
        inverse_length = 1.0 / math.sqrt(length_squared)
        values.extend(component * inverse_length for component in normal)
    return tuple(values)


def _triangulated_corner_uvs(
    mesh: UsdGeom.Mesh,
    uv_source,
    corner_sources: list[tuple[int, int, int]],
    sample_time: Usd.TimeCode,
) -> tuple[float, ...]:
    """Flatten one float2 primvar and bake its standard SRT UV chain."""

    try:
        primvar_name, transforms = uv_source
        primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar(primvar_name)
        if not primvar:
            return ()
        values = primvar.ComputeFlattened(sample_time)
        interpolation = str(primvar.GetInterpolation())
        operations = tuple(
            (
                transform[0],
                transform[1],
                math.cos(math.radians(transform[2])),
                math.sin(math.radians(transform[2])),
                transform[3],
                transform[4],
            )
            for transform in transforms
        )
    except Exception:
        return ()
    if not values or not corner_sources:
        return ()

    result = []
    for face_index, point_index, face_vertex_index in corner_sources:
        if interpolation == "constant":
            value_index = 0
        elif interpolation == "uniform":
            value_index = face_index
        elif interpolation in ("vertex", "varying"):
            value_index = point_index
        elif interpolation == "faceVarying":
            value_index = face_vertex_index
        else:
            return ()
        if value_index < 0 or value_index >= len(values):
            return ()
        try:
            uv = (float(values[value_index][0]), float(values[value_index][1]))
        except (IndexError, TypeError, ValueError):
            return ()
        for scale_x, scale_y, cosine, sine, translate_x, translate_y in operations:
            scaled_u = uv[0] * scale_x
            scaled_v = uv[1] * scale_y
            uv = (
                scaled_u * cosine - scaled_v * sine + translate_x,
                scaled_u * sine + scaled_v * cosine + translate_y,
            )
        if not all(math.isfinite(component) for component in uv):
            return ()
        result.extend(uv)
    return tuple(result)


class StartupStageExtension(omni.ext.IExt):
    """Open an explicit stage and publish Kit's active viewport camera."""

    def on_startup(self, ext_id: str) -> None:
        del ext_id
        self._task = None
        self._camera_state_path = os.environ.get("IMB_CAMERA_STATE_FILE", "")
        self._restore_full_layout = (
            os.environ.get("IMB_RESTORE_FULL_LAYOUT", "0") == "1"
        )
        self._full_layout_ready_file = os.environ.get(
            "IMB_FULL_LAYOUT_READY_FILE", ""
        )
        if self._full_layout_ready_file:
            try:
                os.unlink(self._full_layout_ready_file)
            except FileNotFoundError:
                pass
        self._layout_restored = not self._restore_full_layout
        self._camera_sensor_output = os.environ.get(
            "IMB_CAMERA_SENSOR_OUTPUT", ""
        )
        self._camera_sensor_frame_file = os.environ.get(
            "IMB_CAMERA_SENSOR_FRAME_FILE", ""
        )
        self._camera_sensor_width = int(
            os.environ.get("IMB_CAMERA_SENSOR_WIDTH", "640")
        )
        self._camera_sensor_height = int(
            os.environ.get("IMB_CAMERA_SENSOR_HEIGHT", "480")
        )
        self._physics_smoke_output = os.environ.get(
            "IMB_PHYSICS_SMOKE_OUTPUT", ""
        )
        self._timeline_autoplay = (
            os.environ.get("IMB_TIMELINE_AUTOPLAY", "0") == "1"
        )
        self._center_stage_origin = (
            os.environ.get("IMB_CENTER_STAGE_ORIGIN", "0") == "1"
        )
        self._frame_prim_path = os.environ.get("IMB_FRAME_PRIM_PATH", "")
        self._camera_framed = not self._frame_prim_path
        self._origin_centered = not self._center_stage_origin
        self._origin_center_error_reported = False
        self._startup_reference_url = os.environ.get(
            "IMB_STARTUP_REFERENCE_URL", ""
        )
        self._startup_reference_path = os.environ.get(
            "IMB_STARTUP_REFERENCE_PATH", "/World/Reference"
        )
        self._add_startup_key_light = (
            os.environ.get("IMB_ADD_STARTUP_KEY_LIGHT", "0") == "1"
        )
        self._camera_sequence = 0
        self._static_scene_sequence = 0
        self._last_scene_values = None
        self._static_scene_cache = None
        self._static_scene_dirty = True
        self._scene_notice_key = None
        self._scene_notice_stage = None
        self._active_camera_path = ""
        self._camera_error_reported = False
        self._app_ready_updates = 0
        self._camera_subscription = (
            omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                self._on_update, name="isaacmetalbridge.stage.camera"
            )
            if self._camera_state_path
            else None
        )
        stage_url = os.environ.get("IMB_STARTUP_STAGE_URL", "")
        if not stage_url:
            carb.log_warn(
                "isaacmetalbridge.stage: retaining Kit's normal empty stage; "
                "no startup USD will be opened"
            )
            if self._restore_full_layout:
                self._task = asyncio.ensure_future(
                    self._restore_default_full_layout()
                )
            return

        if self._camera_state_path:
            carb.log_warn(
                "isaacmetalbridge.stage: publishing active Kit camera to "
                f"{self._camera_state_path}"
            )
        if self._camera_sensor_output:
            carb.log_warn(
                "isaacmetalbridge.stage: Isaac Camera RGB capture requested: "
                f"{self._camera_sensor_width}x{self._camera_sensor_height} -> "
                f"{self._camera_sensor_output}"
            )
        if self._physics_smoke_output:
            carb.log_warn(
                "isaacmetalbridge.stage: real Kit timeline/CPU PhysX smoke "
                f"requested -> {self._physics_smoke_output}"
            )
        if self._timeline_autoplay:
            carb.log_warn(
                "isaacmetalbridge.stage: live USD timeline autoplay requested"
            )
        if self._center_stage_origin:
            carb.log_warn(
                "isaacmetalbridge.stage: active viewport will target the XYZ world origin"
            )
        if self._frame_prim_path:
            carb.log_warn(
                "isaacmetalbridge.stage: active viewport will frame "
                f"{self._frame_prim_path}"
            )
        if self._startup_reference_url:
            carb.log_warn(
                "isaacmetalbridge.stage: startup reference requested: "
                f"{self._startup_reference_url} -> {self._startup_reference_path}"
            )
        carb.log_warn(
            f"isaacmetalbridge.stage: opening startup stage: {stage_url}"
        )
        self._task = asyncio.ensure_future(self._open_stage(stage_url))

    def on_shutdown(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self._camera_subscription = None
        self._last_scene_values = None
        self._static_scene_cache = None
        self._static_scene_dirty = True
        if self._scene_notice_key is not None:
            self._scene_notice_key.Revoke()
        self._scene_notice_key = None
        self._scene_notice_stage = None
        self._active_camera_path = ""
        self._app_ready_updates = 0
        self._layout_restored = False
        self._camera_framed = False
        self._origin_centered = False

    def _on_usd_objects_changed(self, notice, _stage) -> None:
        changed_paths = tuple(notice.GetResyncedPaths()) + tuple(
            notice.GetChangedInfoOnlyPaths()
        )
        if changed_paths and self._active_camera_path:
            camera_root = self._active_camera_path.rstrip("/")
            if all(
                str(path) == camera_root
                or str(path).startswith(camera_root + "/")
                or str(path).startswith(camera_root + ".")
                for path in changed_paths
            ):
                return
        self._static_scene_dirty = True

    def _on_update(self, _event) -> None:
        try:
            # Full constructs and docks its workspace late in startup.
            # get_active_viewport() before app-ready can cause Kit to create a
            # separate maximized "Viewport" window, replacing the expected
            # Stage/Property/Content layout in the bridged UI. Wait for two
            # completed app-ready updates before resolving the active camera.
            app = omni.kit.app.get_app()
            if not app.is_app_ready() or not self._layout_restored:
                self._app_ready_updates = 0
                return
            self._app_ready_updates += 1
            if self._app_ready_updates < 2:
                return
            # Importing viewport.utility while Full is still constructing its
            # quick layout creates an undocked maximized Viewport surface.
            # Keep this import lazy as well as the call itself.
            from omni.kit.viewport.utility import get_active_viewport

            stage = omni.usd.get_context().get_stage()
            viewport = get_active_viewport()
            if stage is None or viewport is None:
                return
            if self._scene_notice_stage is not stage:
                if self._scene_notice_key is not None:
                    self._scene_notice_key.Revoke()
                self._scene_notice_key = Tf.Notice.Register(
                    Usd.Notice.ObjectsChanged,
                    self._on_usd_objects_changed,
                    stage,
                )
                self._scene_notice_stage = stage
                self._static_scene_cache = None
                self._static_scene_dirty = True
            camera_path = getattr(viewport, "camera_path", None)
            if callable(camera_path):
                camera_path = camera_path()
            if not camera_path:
                return
            self._active_camera_path = str(camera_path)
            camera_prim = stage.GetPrimAtPath(camera_path)
            if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
                return

            if not self._camera_framed:
                frame_prim = stage.GetPrimAtPath(self._frame_prim_path)
                if not frame_prim.IsValid():
                    return
                from omni.kit.viewport.utility import frame_viewport_prims

                frame_viewport_prims(viewport, [self._frame_prim_path])
                self._camera_framed = True
                carb.log_warn(
                    "isaacmetalbridge.stage: active viewport framed real USD prim: "
                    f"{self._frame_prim_path}"
                )
                return

            if not self._origin_centered:
                try:
                    # Use Kit's own viewport-camera command path so the USD
                    # camera, center-of-interest metadata, gizmo, and native
                    # navigation controller all agree. This is intentionally a
                    # one-shot retarget: subsequent viewer input is free to
                    # orbit, pan, and zoom without being overwritten.
                    from omni.kit.viewport.utility.camera_state import (
                        ViewportCameraState,
                    )

                    camera_state = ViewportCameraState(
                        camera_path=str(camera_path), viewport=viewport
                    )
                    camera_state.set_target_world(Gf.Vec3d(0.0, 0.0, 0.0), True)
                    self._origin_centered = True
                    self._origin_center_error_reported = False
                    carb.log_warn(
                        "isaacmetalbridge.stage: active viewport now targets "
                        f"XYZ origin (0,0,0): path={camera_path}"
                    )
                    # Let TransformPrimCommand finish propagating before the
                    # camera/scene-state snapshot is serialized.
                    return
                except Exception as error:
                    if not self._origin_center_error_reported:
                        carb.log_error(
                            "isaacmetalbridge.stage: failed to center active "
                            f"viewport on XYZ origin: {error}"
                        )
                        self._origin_center_error_reported = True
                    return

            camera = UsdGeom.Camera(camera_prim)
            timeline = omni.timeline.get_timeline_interface()
            timeline_seconds = float(timeline.get_current_time())
            time_code_value = float(timeline.time_to_time_code(timeline_seconds))
            if not math.isfinite(time_code_value):
                return
            sample_time = Usd.TimeCode(time_code_value)
            if (
                camera.GetProjectionAttr().Get(sample_time)
                != UsdGeom.Tokens.perspective
            ):
                return
            xform_cache = UsdGeom.XformCache(sample_time)
            world = xform_cache.GetLocalToWorldTransform(camera_prim)
            position = world.ExtractTranslation()
            forward = world.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
            up = world.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)).GetNormalized()
            focal_length = float(camera.GetFocalLengthAttr().Get(sample_time))
            vertical_aperture = float(
                camera.GetVerticalApertureAttr().Get(sample_time)
            )
            clipping_range = camera.GetClippingRangeAttr().Get(sample_time)
            if focal_length <= 0.0 or vertical_aperture <= 0.0:
                return
            vertical_fov = 2.0 * math.atan(vertical_aperture / (2.0 * focal_length))
            camera_values = (
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(forward[0]),
                float(forward[1]),
                float(forward[2]),
                float(up[0]),
                float(up[1]),
                float(up[2]),
                float(vertical_fov),
                max(float(clipping_range[0]), 0.0001),
                max(float(clipping_range[1]), float(clipping_range[0]) + 0.001),
            )
            if not all(math.isfinite(value) for value in camera_values):
                return

            if timeline.is_playing():
                # Animated transforms and time-sampled materials can change
                # without authoring a new USD value or emitting a notice.
                self._static_scene_dirty = True
            if self._static_scene_cache is not None and not self._static_scene_dirty:
                static_scene = self._static_scene_cache
                scene_values = (
                    camera_values
                    + static_scene["sphere_light_values"]
                    + static_scene["distant_light_values"]
                    + static_scene["dome_light_values"]
                    + (
                        static_scene["additional_light_payload"],
                        static_scene["light_texture_payload"],
                        static_scene["mesh_payload"],
                    )
                )
                if scene_values == self._last_scene_values:
                    return

                self._camera_sequence += 1
                payload = _SCENE_HEADER.pack(
                    _SCENE_MAGIC,
                    _SCENE_VERSION,
                    static_scene["flags"],
                    self._camera_sequence,
                    *camera_values,
                    *static_scene["sphere_light_values"],
                    *static_scene["distant_light_values"],
                    *static_scene["dome_light_values"],
                    static_scene["sequence"],
                    static_scene["mesh_count"],
                    static_scene["additional_light_count"],
                )
                payload += (
                    static_scene["additional_light_payload"]
                    + static_scene["light_texture_payload"]
                    + static_scene["mesh_payload"]
                )
                temporary_path = f"{self._camera_state_path}.tmp.{os.getpid()}"
                with open(temporary_path, "wb", buffering=0) as camera_file:
                    camera_file.write(payload)
                os.replace(temporary_path, self._camera_state_path)
                self._last_scene_values = scene_values
                self._camera_error_reported = False
                if self._camera_sequence <= 3 or self._camera_sequence % 30 == 0:
                    carb.log_warn(
                        "isaacmetalbridge.stage: cached USD scene update published: "
                        f"sequence={self._camera_sequence} "
                        f"timeSeconds={timeline_seconds:.4f} "
                        f"timeCode={time_code_value:.4f} "
                        f"meshes={static_scene['mesh_count']} "
                        f"pointInstances={static_scene['point_instance_count']}"
                    )
                return

            mesh_records = []
            mesh_paths = []
            file_texture_mesh_count = 0
            parameter_mesh_count = 0
            emission_mesh_count = 0
            parameter_texture_count = 0
            normal_texture_count = 0
            alpha_cutout_mesh_count = 0
            standard_opacity_mesh_count = 0
            uv_transform_mesh_count = 0
            instance_proxy_mesh_count = 0
            point_instance_mesh_count = 0
            point_instance_keys = set()
            scene_geometry_bytes = 0
            scene_texture_bytes = 0
            scene_material_texture_paths = set()
            scene_texture_indices = {}
            scene_textures = []
            for (
                prim,
                mesh_path,
                mesh_world_transform,
                is_instance_proxy,
                point_instance_key,
                included_faces,
                bound_material,
            ) in _scene_mesh_occurrences(stage, xform_cache, sample_time):
                if len(mesh_records) >= _MAX_SCENE_MESHES:
                    break
                imageable = UsdGeom.Imageable(prim)
                if (
                    point_instance_key is None
                    and imageable
                    and imageable.ComputeVisibility(sample_time)
                    == UsdGeom.Tokens.invisible
                ):
                    continue
                mesh = UsdGeom.Mesh(prim)
                face_counts = mesh.GetFaceVertexCountsAttr().Get(sample_time)
                face_indices = mesh.GetFaceVertexIndicesAttr().Get(sample_time)
                points = mesh.GetPointsAttr().Get(sample_time)
                if not face_counts or not face_indices or not points:
                    continue
                if sum(int(count) for count in face_counts) != len(face_indices):
                    continue
                holes = set(
                    int(value)
                    for value in (
                        mesh.GetHoleIndicesAttr().Get(sample_time) or ()
                    )
                )
                triangle_indices = []
                triangle_corner_sources = []
                face_offset = 0
                valid_topology = True
                for face_index, raw_count in enumerate(face_counts):
                    count = int(raw_count)
                    if count < 0 or face_offset + count > len(face_indices):
                        valid_topology = False
                        break
                    if (
                        face_index not in holes
                        and count >= 3
                        and (
                            included_faces is None
                            or face_index in included_faces
                        )
                    ):
                        first = int(face_indices[face_offset])
                        for corner in range(1, count - 1):
                            corner_offsets = (
                                face_offset,
                                face_offset + corner,
                                face_offset + corner + 1,
                            )
                            triangle_indices.extend(
                                int(face_indices[offset])
                                for offset in corner_offsets
                            )
                            triangle_corner_sources.extend(
                                (
                                    face_index,
                                    int(face_indices[offset]),
                                    offset,
                                )
                                for offset in corner_offsets
                            )
                    face_offset += count
                if not valid_topology or face_offset != len(face_indices):
                    continue
                triangle_count = len(triangle_indices) // 3
                if triangle_count <= 0:
                    continue
                if any(index < 0 or index >= len(points) for index in triangle_indices):
                    continue
                point_values = tuple(
                    float(component)
                    for point in points
                    for component in (point[0], point[1], point[2])
                )
                if not all(math.isfinite(value) for value in point_values):
                    continue
                corner_normal_values = _triangulated_corner_normals(
                    mesh, triangle_corner_sources, sample_time
                )
                if corner_normal_values and (
                    len(corner_normal_values) != len(triangle_indices) * 3
                ):
                    corner_normal_values = ()
                geometry_bytes = (
                    len(point_values) * 4
                    + len(triangle_indices) * 4
                    + len(corner_normal_values) * 4
                )
                if (
                    geometry_bytes <= 0
                    or geometry_bytes
                    > _MAX_SCENE_GEOMETRY_BYTES - _MAX_SCENE_TEXTURE_BYTES
                    or scene_geometry_bytes
                    > _MAX_SCENE_GEOMETRY_BYTES
                        - _MAX_SCENE_TEXTURE_BYTES
                        - geometry_bytes
                ):
                    continue
                extent = mesh.GetExtentAttr().Get(sample_time)
                if extent is None or len(extent) != 2:
                    minimum = Gf.Vec3d(points[0])
                    maximum = Gf.Vec3d(points[0])
                    for point in points[1:]:
                        point_value = Gf.Vec3d(point)
                        minimum = Gf.Vec3d(
                            min(minimum[0], point_value[0]),
                            min(minimum[1], point_value[1]),
                            min(minimum[2], point_value[2]),
                        )
                        maximum = Gf.Vec3d(
                            max(maximum[0], point_value[0]),
                            max(maximum[1], point_value[1]),
                            max(maximum[2], point_value[2]),
                        )
                else:
                    minimum = Gf.Vec3d(extent[0])
                    maximum = Gf.Vec3d(extent[1])
                local_bounds = (
                    float(minimum[0]),
                    float(minimum[1]),
                    float(minimum[2]),
                    float(maximum[0]),
                    float(maximum[1]),
                    float(maximum[2]),
                )
                world_transform = _world_transform_3x4(
                    mesh_world_transform
                )
                values = local_bounds + world_transform
                if not all(math.isfinite(value) for value in values):
                    continue
                material_flags = 0
                base_color = None
                connected_base_color = False
                file_texture = None
                material_parameters = None
                material_emission = None
                roughness_file_texture = None
                metallic_file_texture = None
                emission_file_texture = None
                normal_file_texture = None
                warehouse_bundle = None
                standard_opacity = None
                standard_opacity_decoded = {}
                try:
                    material = bound_material
                    if material and material.GetPrim().IsValid():
                        material_flags |= _MESH_HAS_BOUND_MATERIAL
                        base_color, connected_base_color = (
                            _bound_material_base_color(material)
                        )
                        file_texture = _bound_material_file_texture(material)
                        material_parameters = _bound_material_parameters(material)
                        material_emission = _bound_material_emission(material)
                        roughness_file_texture = _bound_material_parameter_texture(
                            material,
                            ("roughness", "reflection_roughness_constant"),
                        )
                        metallic_file_texture = _bound_material_parameter_texture(
                            material,
                            ("metallic", "metallic_constant"),
                        )
                        emission_file_texture = _bound_material_parameter_texture(
                            material,
                            (
                                "emissiveColor",
                                "emissive_color",
                                "emissive_color_constant",
                                "emission_color",
                            ),
                            color_output=True,
                        )
                        normal_file_texture = _bound_material_parameter_texture(
                            material,
                            ("normal", "normalmap", "normal_map"),
                            color_output=True,
                        )
                        standard_opacity = _bound_material_opacity(material)
                        warehouse_bundle = _warehouse_mdl_material_bundle(
                            material
                        )
                        if warehouse_bundle is not None:
                            if warehouse_bundle["alpha_cutout"]:
                                material_flags |= _MESH_HAS_ALPHA_CUTOUT
                            if warehouse_bundle["base_color"] is not None:
                                base_color = warehouse_bundle["base_color"]
                                connected_base_color = False
                            if (
                                warehouse_bundle["material_parameters"]
                                is not None
                            ):
                                material_parameters = warehouse_bundle[
                                    "material_parameters"
                                ]
                            if (
                                warehouse_bundle["material_emission"]
                                is not None
                            ):
                                material_emission = warehouse_bundle[
                                    "material_emission"
                                ]
                        if (
                            standard_opacity is not None
                            and standard_opacity["relevant"]
                        ):
                            opacity_texture = standard_opacity["texture"]
                            if opacity_texture is None:
                                material_flags |= _MESH_HAS_STANDARD_OPACITY
                            else:
                                (
                                    opacity_path,
                                    opacity_uv_source,
                                    opacity_channel,
                                ) = opacity_texture
                                base_path = file_texture[0] if file_texture else ""
                                base_uv_source = (
                                    file_texture[1]
                                    if file_texture else opacity_uv_source
                                )
                                if base_uv_source == opacity_uv_source:
                                    if base_color is None:
                                        base_color = (0.18, 0.18, 0.18)
                                    decoded_opacity = (
                                        _compose_standard_opacity_texture(
                                            base_path,
                                            opacity_path,
                                            opacity_channel,
                                            base_color,
                                        )
                                    )
                                    if decoded_opacity is not None:
                                        derived_key = (
                                            "imb-derived:standard-opacity:"
                                            f"{base_path}:{opacity_path}:"
                                            f"{opacity_channel}:{base_color}"
                                        )
                                        file_texture = (
                                            derived_key,
                                            opacity_uv_source,
                                        )
                                        standard_opacity_decoded[
                                            derived_key
                                        ] = decoded_opacity
                                        material_flags |= (
                                            _MESH_HAS_STANDARD_OPACITY
                                        )
                except Exception:
                    # A missing or unsupported binding must not suppress valid
                    # geometry from the scene manifest.
                    pass
                if base_color is None:
                    try:
                        display_colors = (
                            mesh.GetDisplayColorPrimvar().ComputeFlattened(
                                sample_time
                            )
                        )
                        if display_colors:
                            base_color = _color3(display_colors[0])
                    except Exception:
                        pass
                if base_color is not None:
                    material_flags |= _MESH_HAS_BASE_COLOR
                    if connected_base_color:
                        material_flags |= _MESH_HAS_CONNECTED_BASE_COLOR
                    material_flags |= _pack_mesh_base_color(base_color)
                opacity = 1.0
                opacity_threshold = 0.0
                if material_flags & _MESH_HAS_STANDARD_OPACITY:
                    opacity = standard_opacity["opacity"]
                    opacity_threshold = standard_opacity["threshold"]
                    standard_opacity_mesh_count += 1
                roughness = 0.5
                metallic = 0.0
                if material_parameters is not None:
                    roughness, metallic = material_parameters
                    material_flags |= _MESH_HAS_MATERIAL_PARAMETERS
                    parameter_mesh_count += 1
                emission_color = (0.0, 0.0, 0.0)
                emission_intensity = 0.0
                if material_emission is not None:
                    emission_color, emission_intensity = material_emission
                    material_flags |= _MESH_HAS_EMISSION
                    emission_mesh_count += 1

                texture_spec_map = {}
                if file_texture is not None:
                    texture_spec_map["base"] = (*file_texture, 4)
                if roughness_file_texture is not None:
                    texture_spec_map["roughness"] = roughness_file_texture
                if metallic_file_texture is not None:
                    texture_spec_map["metallic"] = metallic_file_texture
                if emission_file_texture is not None:
                    texture_spec_map["emission"] = emission_file_texture
                if normal_file_texture is not None:
                    texture_spec_map["normal"] = normal_file_texture
                predecoded_material_textures = dict(
                    standard_opacity_decoded
                )
                if warehouse_bundle is not None:
                    texture_spec_map.update(
                        warehouse_bundle["texture_specs"]
                    )
                    predecoded_material_textures.update(
                        warehouse_bundle["decoded"]
                    )
                texture_specs = [
                    (kind, *texture_spec_map[kind])
                    for kind in (
                        "base", "roughness", "metallic", "emission", "normal"
                    )
                    if kind in texture_spec_map
                ]

                decoded_textures = {}
                decoded_texture_paths = {}
                texture_indices = {}
                corner_uv_values = ()
                if texture_specs:
                    shared_uv_source = texture_specs[0][2]
                    for kind, texture_path, uv_source, channel in texture_specs:
                        if uv_source != shared_uv_source:
                            continue
                        decoded_texture = predecoded_material_textures.get(
                            texture_path
                        )
                        if decoded_texture is None:
                            decoded_texture = _load_rgba_texture(texture_path)
                        if decoded_texture is not None:
                            decoded_textures[kind] = (*decoded_texture, channel)
                            decoded_texture_paths[kind] = texture_path
                    if decoded_textures:
                        material_texture_paths = set(
                            decoded_texture_paths.values()
                        )
                        if len(
                            scene_material_texture_paths
                            | material_texture_paths
                        ) > _MAX_SCENE_MATERIAL_TEXTURES:
                            decoded_textures = {}
                            decoded_texture_paths = {}
                    if decoded_textures:
                        corner_uv_values = _triangulated_corner_uvs(
                            mesh,
                            shared_uv_source,
                            triangle_corner_sources,
                            sample_time,
                        )
                        if len(corner_uv_values) != len(triangle_indices) * 2:
                            corner_uv_values = ()
                            decoded_textures = {}

                texture_width = texture_height = 0
                texture_pixels = b""
                roughness_texture_width = roughness_texture_height = 0
                roughness_texture_pixels = b""
                roughness_texture_channel = 0
                metallic_texture_width = metallic_texture_height = 0
                metallic_texture_pixels = b""
                metallic_texture_channel = 0
                emission_texture_width = emission_texture_height = 0
                emission_texture_pixels = b""
                emission_texture_channel = 0
                normal_texture_width = normal_texture_height = 0
                normal_texture_pixels = b""
                texture_flags = 0
                if decoded_textures:
                    new_texture_paths = []
                    new_texture_bytes = 0
                    for kind, texture_path in decoded_texture_paths.items():
                        if (
                            texture_path in scene_texture_indices
                            or texture_path in new_texture_paths
                        ):
                            continue
                        new_texture_paths.append(texture_path)
                        new_texture_bytes += len(decoded_textures[kind][2])
                    texture_bytes = (
                        len(corner_uv_values) * 4 + new_texture_bytes
                    )
                    if (
                        texture_bytes <= _MAX_SCENE_TEXTURE_BYTES
                        and scene_texture_bytes
                        <= _MAX_SCENE_TEXTURE_BYTES - texture_bytes
                        and len(scene_textures) + len(new_texture_paths)
                        <= _MAX_SCENE_MATERIAL_TEXTURES
                    ):
                        scene_texture_bytes += texture_bytes
                        scene_material_texture_paths.update(
                            decoded_texture_paths.values()
                        )
                        for kind, texture_path in decoded_texture_paths.items():
                            if texture_path not in scene_texture_indices:
                                width, height, pixels, _channel = (
                                    decoded_textures[kind]
                                )
                                scene_texture_indices[texture_path] = len(
                                    scene_textures
                                )
                                scene_textures.append(
                                    (texture_path, width, height, pixels)
                                )
                            texture_indices[kind] = scene_texture_indices[
                                texture_path
                            ]
                        if shared_uv_source[1]:
                            uv_transform_mesh_count += 1
                        if "base" in decoded_textures:
                            texture_width, texture_height, texture_pixels, _ = (
                                decoded_textures["base"]
                            )
                            material_flags |= _MESH_HAS_FILE_TEXTURE
                            file_texture_mesh_count += 1
                            if standard_opacity_decoded:
                                texture_flags |= _TEXTURE_HAS_OPACITY
                        if "roughness" in decoded_textures:
                            (
                                roughness_texture_width,
                                roughness_texture_height,
                                roughness_texture_pixels,
                                roughness_texture_channel,
                            ) = decoded_textures["roughness"]
                            texture_flags |= _TEXTURE_HAS_ROUGHNESS
                            parameter_texture_count += 1
                        if "metallic" in decoded_textures:
                            (
                                metallic_texture_width,
                                metallic_texture_height,
                                metallic_texture_pixels,
                                metallic_texture_channel,
                            ) = decoded_textures["metallic"]
                            texture_flags |= _TEXTURE_HAS_METALLIC
                            parameter_texture_count += 1
                        if "emission" in decoded_textures:
                            (
                                emission_texture_width,
                                emission_texture_height,
                                emission_texture_pixels,
                                emission_texture_channel,
                            ) = decoded_textures["emission"]
                            texture_flags |= _TEXTURE_HAS_EMISSION
                            parameter_texture_count += 1
                            if material_emission is None:
                                emission_color = (1.0, 1.0, 1.0)
                                emission_intensity = 1.0
                                material_flags |= _MESH_HAS_EMISSION
                                emission_mesh_count += 1
                        if "normal" in decoded_textures:
                            (
                                normal_texture_width,
                                normal_texture_height,
                                normal_texture_pixels,
                                _normal_texture_channel,
                            ) = decoded_textures["normal"]
                            texture_flags |= _TEXTURE_HAS_NORMAL
                            normal_texture_count += 1
                    else:
                        corner_uv_values = ()
                        decoded_textures = {}
                if (
                    standard_opacity_decoded
                    and texture_flags & _TEXTURE_HAS_OPACITY == 0
                ):
                    # Never claim a texture-driven cutout when its shared UV
                    # or bounded image payload could not be transported.
                    material_flags &= ~_MESH_HAS_STANDARD_OPACITY
                    opacity = 1.0
                    opacity_threshold = 0.0
                    standard_opacity_mesh_count -= 1
                vertex_payload = struct.pack(
                    f"<{len(point_values)}f", *point_values
                )
                index_payload = struct.pack(
                    f"<{len(triangle_indices)}I", *triangle_indices
                )
                normal_payload = (
                    struct.pack(
                        f"<{len(corner_normal_values)}f", *corner_normal_values
                    )
                    if corner_normal_values
                    else b""
                )
                uv_payload = (
                    struct.pack(f"<{len(corner_uv_values)}f", *corner_uv_values)
                    if corner_uv_values
                    else b""
                )
                if material_flags & _MESH_HAS_ALPHA_CUTOUT:
                    alpha_cutout_mesh_count += 1
                mesh_records.append(
                    _SCENE_MESH_RECORD.pack(
                        _fnv1a_64(mesh_path),
                        triangle_count,
                        material_flags,
                        *values,
                        roughness,
                        metallic,
                        *emission_color,
                        emission_intensity,
                        opacity,
                        opacity_threshold,
                        len(points),
                        len(triangle_indices),
                        len(corner_normal_values) // 3,
                        len(corner_uv_values) // 2,
                        texture_width,
                        texture_height,
                        0,
                        texture_flags,
                        roughness_texture_width,
                        roughness_texture_height,
                        0,
                        roughness_texture_channel,
                        metallic_texture_width,
                        metallic_texture_height,
                        0,
                        metallic_texture_channel,
                        emission_texture_width,
                        emission_texture_height,
                        0,
                        emission_texture_channel,
                        normal_texture_width,
                        normal_texture_height,
                        0,
                        texture_indices.get("base", _NO_SCENE_TEXTURE),
                        texture_indices.get("roughness", _NO_SCENE_TEXTURE),
                        texture_indices.get("metallic", _NO_SCENE_TEXTURE),
                        texture_indices.get("emission", _NO_SCENE_TEXTURE),
                        texture_indices.get("normal", _NO_SCENE_TEXTURE),
                    )
                    + vertex_payload
                    + index_payload
                    + normal_payload
                    + uv_payload
                )
                if is_instance_proxy:
                    instance_proxy_mesh_count += 1
                if point_instance_key is not None:
                    point_instance_mesh_count += 1
                    point_instance_keys.add(point_instance_key)
                scene_geometry_bytes += geometry_bytes
                mesh_paths.append(mesh_path)

            positional_lights = []
            light_texture_indices = {}
            light_textures = []
            light_texture_bytes = 0
            rejected_light_assets = []

            def append_light_texture(key, decoded):
                nonlocal light_texture_bytes
                if key in light_texture_indices:
                    return light_texture_indices[key]
                if decoded is None or len(decoded) < 3:
                    return _NO_SCENE_LIGHT_TEXTURE
                width, height, pixels = decoded[:3]
                if (
                    len(light_textures) >= _MAX_SCENE_LIGHT_TEXTURES
                    or width <= 0
                    or height <= 0
                    or len(pixels) != width * height * 4
                    or light_texture_bytes + len(pixels)
                    > _MAX_SCENE_LIGHT_TEXTURE_BYTES
                ):
                    return _NO_SCENE_LIGHT_TEXTURE
                index = len(light_textures)
                light_texture_indices[key] = index
                light_textures.append((key, width, height, pixels))
                light_texture_bytes += len(pixels)
                return index

            for prim in _stage_prims(stage):
                if len(positional_lights) >= _MAX_SCENE_POSITIONAL_LIGHTS:
                    break
                world = xform_cache.GetLocalToWorldTransform(prim)
                if prim.IsA(UsdLux.SphereLight):
                    light = UsdLux.SphereLight(prim)
                    radius = light.GetRadiusAttr().Get(sample_time)
                    schema_name = "SphereLight"
                    schema_code = 1
                    world_u = world.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
                    world_v = world.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
                    scale_u = math.sqrt(sum(float(value) ** 2 for value in world_u))
                    scale_v = math.sqrt(sum(float(value) ** 2 for value in world_v))
                    if scale_u <= 0.000001 or scale_v <= 0.000001:
                        continue
                    ies_axis_u = tuple(float(value) / scale_u for value in world_u)
                    ies_axis_v = tuple(float(value) / scale_v for value in world_v)
                    axis_u = (0.0, 0.0, 0.0)
                    axis_v = (0.0, 0.0, 0.0)
                    half_extent_u = 0.0
                    half_extent_v = 0.0
                elif prim.IsA(UsdLux.RectLight):
                    light = UsdLux.RectLight(prim)
                    width = light.GetWidthAttr().Get(sample_time)
                    height = light.GetHeightAttr().Get(sample_time)
                    if width is None or height is None:
                        continue
                    world_u = world.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
                    world_v = world.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
                    scale_u = math.sqrt(sum(float(value) ** 2 for value in world_u))
                    scale_v = math.sqrt(sum(float(value) ** 2 for value in world_v))
                    if scale_u <= 0.000001 or scale_v <= 0.000001:
                        continue
                    axis_u = tuple(float(value) / scale_u for value in world_u)
                    axis_v = tuple(float(value) / scale_v for value in world_v)
                    ies_axis_u = axis_u
                    ies_axis_v = axis_v
                    half_extent_u = max(float(width), 0.0) * scale_u * 0.5
                    half_extent_v = max(float(height), 0.0) * scale_v * 0.5
                    radius = math.sqrt(
                        max(float(width) * scale_u, 0.0)
                        * max(float(height) * scale_v, 0.0)
                        / math.pi
                    )
                    schema_name = "RectLight"
                    schema_code = 2
                elif prim.IsA(UsdLux.DiskLight):
                    light = UsdLux.DiskLight(prim)
                    radius = light.GetRadiusAttr().Get(sample_time)
                    world_u = world.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
                    world_v = world.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
                    scale_u = math.sqrt(sum(float(value) ** 2 for value in world_u))
                    scale_v = math.sqrt(sum(float(value) ** 2 for value in world_v))
                    if scale_u <= 0.000001 or scale_v <= 0.000001:
                        continue
                    axis_u = tuple(float(value) / scale_u for value in world_u)
                    axis_v = tuple(float(value) / scale_v for value in world_v)
                    ies_axis_u = axis_u
                    ies_axis_v = axis_v
                    half_extent_u = max(float(radius or 0.0), 0.0) * scale_u
                    half_extent_v = max(float(radius or 0.0), 0.0) * scale_v
                    radius = math.sqrt(max(half_extent_u * half_extent_v, 0.0))
                    schema_name = "DiskLight"
                    schema_code = 3
                elif prim.IsA(UsdLux.CylinderLight):
                    light = UsdLux.CylinderLight(prim)
                    authored_radius = light.GetRadiusAttr().Get(sample_time)
                    length = light.GetLengthAttr().Get(sample_time)
                    treat_as_line = light.GetTreatAsLineAttr().Get(sample_time)
                    if authored_radius is None or length is None:
                        continue
                    world_u = world.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
                    world_v = world.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
                    world_w = world.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
                    scale_u = math.sqrt(sum(float(value) ** 2 for value in world_u))
                    scale_v = math.sqrt(sum(float(value) ** 2 for value in world_v))
                    scale_w = math.sqrt(sum(float(value) ** 2 for value in world_w))
                    if (
                        scale_u <= 0.000001
                        or scale_v <= 0.000001
                        or scale_w <= 0.000001
                    ):
                        continue
                    axis_u = tuple(float(value) / scale_u for value in world_u)
                    axis_v = tuple(float(value) / scale_v for value in world_v)
                    ies_axis_u = axis_v
                    ies_axis_v = tuple(
                        float(value)
                        for value in world.TransformDir(
                            Gf.Vec3d(0.0, 0.0, 1.0)
                        ).GetNormalized()
                    )
                    half_extent_u = max(float(length), 0.0) * scale_u * 0.5
                    if bool(treat_as_line):
                        half_extent_v = 0.0
                        radius = 0.0
                    else:
                        half_extent_v = (
                            max(float(authored_radius), 0.0) * scale_v
                        )
                        # The existing radius word carries the transformed
                        # local-Z radial extent for CylinderLight. Together
                        # with axisV/halfExtentV this preserves an ellipse
                        # under ordinary non-uniform scale.
                        radius = max(float(authored_radius), 0.0) * scale_w
                    if half_extent_u <= 0.0:
                        continue
                    schema_name = "CylinderLight"
                    schema_code = 6
                else:
                    continue
                light_position = world.ExtractTranslation()
                light_color = light.GetColorAttr().Get(sample_time)
                intensity = light.GetIntensityAttr().Get(sample_time)
                exposure = light.GetExposureAttr().Get(sample_time)
                if light_color is None or intensity is None:
                    continue
                effective_intensity = float(intensity) * math.pow(
                    2.0, float(exposure or 0.0)
                )
                values = (
                    float(light_position[0]),
                    float(light_position[1]),
                    float(light_position[2]),
                    max(float(light_color[0]), 0.0),
                    max(float(light_color[1]), 0.0),
                    max(float(light_color[2]), 0.0),
                    max(effective_intensity, 0.0),
                    max(
                        float(radius or 0.0),
                        0.0 if schema_code == 6 else 0.0001,
                    ),
                )
                wire_values = (
                    values[0], values[1], values[2], values[6],
                    values[3], values[4], values[5], values[7],
                    axis_u[0], axis_u[1], axis_u[2], half_extent_u,
                    axis_v[0], axis_v[1], axis_v[2], half_extent_v,
                )
                (
                    shaping_values,
                    shaping_flags,
                    shaping_ies_path,
                    shaping_ies_angle_scale,
                    shaping_ies_normalize,
                ) = (
                    _light_shaping_values(
                        prim, world, sample_time, schema_code
                    )
                )
                if all(math.isfinite(value) for value in wire_values):
                    emission_texture_index = _NO_SCENE_LIGHT_TEXTURE
                    ies_texture_index = _NO_SCENE_LIGHT_TEXTURE
                    ies_multiplier = 0.0
                    light_texture_flags = 0
                    if schema_code == 2:
                        try:
                            texture_value = light.GetTextureFileAttr().Get(
                                sample_time
                            )
                        except Exception:
                            texture_value = None
                        texture_path = _resolve_prim_asset_path(
                            prim, texture_value
                        )
                        if texture_path:
                            emission_texture_index = append_light_texture(
                                "rect:" + texture_path,
                                _load_rgba_texture(texture_path),
                            )
                            if emission_texture_index != _NO_SCENE_LIGHT_TEXTURE:
                                light_texture_flags |= _LIGHT_TEXTURE_RECT_EMISSION
                            else:
                                rejected_light_assets.append(
                                    (str(prim.GetPath()), "RectLight texture", texture_path)
                                )
                    if shaping_ies_path:
                        ies_profile = _load_ies_profile(
                            shaping_ies_path, shaping_ies_normalize
                        )
                        ies_texture_index = append_light_texture(
                            "ies:" + shaping_ies_path,
                            ies_profile,
                        )
                        if (
                            ies_texture_index != _NO_SCENE_LIGHT_TEXTURE
                            and ies_profile is not None
                        ):
                            ies_multiplier = ies_profile[3]
                            light_texture_flags |= _LIGHT_TEXTURE_IES_PROFILE
                            if shaping_ies_normalize:
                                light_texture_flags |= (
                                    _LIGHT_TEXTURE_IES_NORMALIZED
                                )
                            if schema_code == 1:
                                wire_values = (
                                    *wire_values[:8],
                                    *ies_axis_u,
                                    0.0,
                                    *ies_axis_v,
                                    0.0,
                                )
                        else:
                            rejected_light_assets.append(
                                (str(prim.GetPath()), "ShapingAPI IES", shaping_ies_path)
                            )
                    positional_lights.append(
                        (
                            schema_code,
                            schema_name,
                            str(prim.GetPath()),
                            values,
                            wire_values,
                            shaping_values,
                            shaping_flags,
                            emission_texture_index,
                            ies_texture_index,
                            shaping_ies_angle_scale
                            if light_texture_flags & _LIGHT_TEXTURE_IES_PROFILE
                            else 0.0,
                            ies_multiplier,
                            light_texture_flags,
                        )
                    )

            sphere_light_values = (
                positional_lights[0][3] if positional_lights else (0.0,) * 8
            )
            sphere_light_path = positional_lights[0][2] if positional_lights else ""
            sphere_light_schema = positional_lights[0][1] if positional_lights else ""

            distant_lights = []
            for prim in _stage_prims(stage):
                if len(distant_lights) >= _MAX_SCENE_DISTANT_LIGHTS:
                    break
                if not prim.IsA(UsdLux.DistantLight):
                    continue
                light = UsdLux.DistantLight(prim)
                world = xform_cache.GetLocalToWorldTransform(prim)
                # USD distant lights emit along local -Z. Preserve that world
                # direction on the bridge; Metal negates it to obtain the
                # surface-to-light direction used by bounded hit shading.
                light_direction = world.TransformDir(
                    Gf.Vec3d(0.0, 0.0, -1.0)
                ).GetNormalized()
                light_color = light.GetColorAttr().Get(sample_time)
                intensity = light.GetIntensityAttr().Get(sample_time)
                exposure = light.GetExposureAttr().Get(sample_time)
                angle = light.GetAngleAttr().Get(sample_time)
                if light_color is None or intensity is None:
                    continue
                effective_intensity = float(intensity) * math.pow(
                    2.0, float(exposure or 0.0)
                )
                values = (
                    float(light_direction[0]),
                    float(light_direction[1]),
                    float(light_direction[2]),
                    max(float(light_color[0]), 0.0),
                    max(float(light_color[1]), 0.0),
                    max(float(light_color[2]), 0.0),
                    max(effective_intensity, 0.0),
                    min(max(float(angle or 0.0), 0.0), 359.999),
                )
                if (
                    all(math.isfinite(value) for value in values)
                    and sum(value * value for value in values[:3])
                    > 0.000001
                ):
                    distant_lights.append(
                        (4, "DistantLight", str(prim.GetPath()), values)
                    )

            distant_light_values = (
                distant_lights[0][3] if distant_lights else (0.0,) * 8
            )
            distant_light_path = distant_lights[0][2] if distant_lights else ""

            dome_lights = []
            for prim in _stage_prims(stage):
                if len(dome_lights) >= _MAX_SCENE_DOME_LIGHTS:
                    break
                if not prim.IsA(UsdLux.DomeLight):
                    continue
                light = UsdLux.DomeLight(prim)
                world = xform_cache.GetLocalToWorldTransform(prim)
                light_color = light.GetColorAttr().Get(sample_time)
                intensity = light.GetIntensityAttr().Get(sample_time)
                exposure = light.GetExposureAttr().Get(sample_time)
                if light_color is None or intensity is None:
                    continue
                effective_intensity = float(intensity) * math.pow(
                    2.0, float(exposure or 0.0)
                )
                values = (
                    max(float(light_color[0]), 0.0),
                    max(float(light_color[1]), 0.0),
                    max(float(light_color[2]), 0.0),
                    max(effective_intensity, 0.0),
                )
                world_axes = []
                for local_axis in (
                    Gf.Vec3d(1.0, 0.0, 0.0),
                    Gf.Vec3d(0.0, 1.0, 0.0),
                    Gf.Vec3d(0.0, 0.0, 1.0),
                ):
                    world_axis = world.TransformDir(local_axis)
                    axis_length = math.sqrt(
                        sum(float(value) ** 2 for value in world_axis)
                    )
                    if not math.isfinite(axis_length) or axis_length <= 0.000001:
                        world_axes = []
                        break
                    world_axes.append(
                        tuple(float(value) / axis_length for value in world_axis)
                    )
                if not world_axes:
                    continue
                texture_index = _NO_SCENE_LIGHT_TEXTURE
                light_texture_flags = 0
                try:
                    texture_value = light.GetTextureFileAttr().Get(sample_time)
                    texture_format = str(
                        light.GetTextureFormatAttr().Get(sample_time) or "automatic"
                    )
                except Exception:
                    texture_value = None
                    texture_format = "automatic"
                texture_path = _resolve_prim_asset_path(prim, texture_value)
                if texture_path:
                    decoded = _load_dome_texture(texture_path)
                    valid_format = texture_format in ("automatic", "latlong")
                    valid_aspect = (
                        decoded is not None
                        and decoded[0] == decoded[1] * 2
                    )
                    if valid_format and valid_aspect:
                        texture_index = append_light_texture(
                            "dome:" + texture_path,
                            decoded,
                        )
                    if texture_index != _NO_SCENE_LIGHT_TEXTURE:
                        light_texture_flags |= _LIGHT_TEXTURE_DOME_ENVIRONMENT
                        if decoded[3]:
                            light_texture_flags |= _LIGHT_TEXTURE_DOME_RGBE
                    else:
                        rejected_light_assets.append(
                            (
                                str(prim.GetPath()),
                                f"DomeLight {texture_format} 2:1 texture",
                                texture_path,
                            )
                        )
                wire_values = (
                    (
                        *values,
                        *world_axes[0],
                        *world_axes[1],
                        *world_axes[2],
                        0.0, 0.0, 0.0,
                    )
                    if light_texture_flags & _LIGHT_TEXTURE_DOME_ENVIRONMENT
                    else (*values, *((0.0,) * 12))
                )
                if all(math.isfinite(value) for value in wire_values):
                    dome_lights.append(
                        (
                            5,
                            "DomeLight",
                            str(prim.GetPath()),
                            values,
                            wire_values,
                            texture_index,
                            light_texture_flags,
                        )
                    )

            dome_light_values = dome_lights[0][3] if dome_lights else (0.0,) * 4
            dome_light_path = dome_lights[0][2] if dome_lights else ""

            # Scene-state v20 publishes one complete list. The fixed first-light
            # slots remain populated only so older state readers retain a stable
            # header shape; the v20 ICD suppresses those legacy slots when it
            # submits the rich list to Protocol 1.22.
            additional_light_records = []
            additional_light_descriptions = []
            for (
                schema_code,
                schema_name,
                path,
                _,
                wire_values,
                shaping_values,
                shaping_flags,
                emission_texture_index,
                ies_texture_index,
                ies_angle_scale,
                ies_multiplier,
                light_texture_flags,
            ) in positional_lights:
                additional_light_records.append(
                    _SCENE_LIGHT_RECORD.pack(
                        1,
                        schema_code,
                        *wire_values,
                        _fnv1a_64(path),
                        *shaping_values,
                        shaping_flags,
                        emission_texture_index,
                        ies_texture_index,
                        ies_angle_scale,
                        ies_multiplier,
                        light_texture_flags,
                        0,
                    )
                )
                additional_light_descriptions.append(f"{schema_name}:{path}")
            for schema_code, schema_name, path, values in distant_lights:
                additional_light_records.append(
                    _SCENE_LIGHT_RECORD.pack(
                        2,
                        schema_code,
                        values[0], values[1], values[2], values[6],
                        values[3], values[4], values[5], values[7],
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                        _fnv1a_64(path),
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0,
                        0,
                        _NO_SCENE_LIGHT_TEXTURE,
                        _NO_SCENE_LIGHT_TEXTURE,
                        0.0, 0.0, 0, 0,
                    )
                )
                additional_light_descriptions.append(f"{schema_name}:{path}")
            for (
                schema_code,
                schema_name,
                path,
                values,
                wire_values,
                texture_index,
                light_texture_flags,
            ) in dome_lights:
                additional_light_records.append(
                    _SCENE_LIGHT_RECORD.pack(
                        3,
                        schema_code,
                        *wire_values,
                        _fnv1a_64(path),
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0,
                        0,
                        texture_index,
                        _NO_SCENE_LIGHT_TEXTURE,
                        0.0, 0.0, light_texture_flags, 0,
                    )
                )
                additional_light_descriptions.append(f"{schema_name}:{path}")
            additional_light_payload = b"".join(additional_light_records)

            light_texture_records = [struct.pack("<I", len(light_textures))]
            for texture_key, width, height, pixels in light_textures:
                light_texture_records.append(
                    _SCENE_TEXTURE_RECORD.pack(
                        _fnv1a_64(texture_key),
                        width,
                        height,
                        len(pixels),
                    )
                    + pixels
                )
            light_texture_payload = b"".join(light_texture_records)

            texture_table_records = [struct.pack("<I", len(scene_textures))]
            for texture_path, width, height, pixels in scene_textures:
                texture_table_records.append(
                    _SCENE_TEXTURE_RECORD.pack(
                        _fnv1a_64(texture_path),
                        width,
                        height,
                        len(pixels),
                    )
                    + pixels
                )
            mesh_payload = b"".join(texture_table_records + mesh_records)
            flags = _scene_flags(
                sphere_light_path,
                distant_light_path,
                dome_light_path,
            )
            static_scene_values = (
                flags,
                sphere_light_values,
                distant_light_values,
                dome_light_values,
                additional_light_payload,
                light_texture_payload,
                mesh_payload,
                len(mesh_records),
            )
            if (
                self._static_scene_cache is not None
                and self._static_scene_cache["values"] == static_scene_values
            ):
                static_scene_sequence = self._static_scene_cache["sequence"]
            else:
                self._static_scene_sequence += 1
                static_scene_sequence = self._static_scene_sequence
            self._static_scene_cache = {
                "flags": flags,
                "sphere_light_values": sphere_light_values,
                "distant_light_values": distant_light_values,
                "dome_light_values": dome_light_values,
                "additional_light_payload": additional_light_payload,
                "additional_light_count": len(additional_light_records),
                "light_texture_payload": light_texture_payload,
                "mesh_payload": mesh_payload,
                "mesh_count": len(mesh_records),
                "point_instance_count": len(point_instance_keys),
                "sequence": static_scene_sequence,
                "values": static_scene_values,
            }
            self._static_scene_dirty = False
            scene_values = (
                camera_values
                + sphere_light_values
                + distant_light_values
                + dome_light_values
                + (additional_light_payload, light_texture_payload, mesh_payload)
            )
            if scene_values == self._last_scene_values:
                return

            self._camera_sequence += 1
            payload = _SCENE_HEADER.pack(
                _SCENE_MAGIC,
                _SCENE_VERSION,
                flags,
                self._camera_sequence,
                *camera_values,
                *sphere_light_values,
                *distant_light_values,
                *dome_light_values,
                static_scene_sequence,
                len(mesh_records),
                len(additional_light_records),
            )
            payload += (
                additional_light_payload + light_texture_payload + mesh_payload
            )
            temporary_path = (
                f"{self._camera_state_path}.tmp.{os.getpid()}"
            )
            with open(temporary_path, "wb", buffering=0) as camera_file:
                camera_file.write(payload)
            os.replace(temporary_path, self._camera_state_path)
            self._last_scene_values = scene_values
            self._camera_error_reported = False
            if self._camera_sequence == 1:
                reference_prefix = self._startup_reference_path.rstrip("/") + "/"
                reference_mesh_count = sum(
                    1 for path in mesh_paths if path.startswith(reference_prefix)
                )
                if len(mesh_paths) <= 24:
                    mesh_path_sample = mesh_paths
                else:
                    mesh_path_sample = mesh_paths[:12] + ["..."] + mesh_paths[-12:]
                carb.log_warn(
                    "isaacmetalbridge.stage: active Kit camera published: "
                    f"path={camera_path} position=({camera_values[0]:.3f},"
                    f"{camera_values[1]:.3f},{camera_values[2]:.3f}) "
                    f"fov={camera_values[9]:.4f} near={camera_values[10]:.4f} "
                    f"far={camera_values[11]:.1f}"
                )
                if sphere_light_path:
                    carb.log_warn(
                        "isaacmetalbridge.stage: active USD positional light "
                        f"published: schema={sphere_light_schema} "
                        f"path={sphere_light_path} "
                        f"position=({sphere_light_values[0]:.3f},"
                        f"{sphere_light_values[1]:.3f},"
                        f"{sphere_light_values[2]:.3f}) "
                        f"color=({sphere_light_values[3]:.3f},"
                        f"{sphere_light_values[4]:.3f},"
                        f"{sphere_light_values[5]:.3f}) "
                        f"intensity={sphere_light_values[6]:.3f} "
                        f"radius={sphere_light_values[7]:.3f}"
                    )
                for (
                    schema_code,
                    schema_name,
                    path,
                    _,
                    wire_values,
                    shaping_values,
                    shaping_flags,
                    emission_texture_index,
                    ies_texture_index,
                    ies_angle_scale,
                    ies_multiplier,
                    light_texture_flags,
                ) in positional_lights:
                    if schema_code != 1:
                        cylinder_details = (
                            f" radialW={wire_values[7]:.3f}"
                            if schema_code == 6
                            else ""
                        )
                        carb.log_warn(
                            "isaacmetalbridge.stage: oriented USD area/line light published: "
                            f"schema={schema_name} path={path} "
                            f"axisU=({wire_values[8]:.3f},{wire_values[9]:.3f},"
                            f"{wire_values[10]:.3f}) halfExtentU={wire_values[11]:.3f} "
                            f"axisV=({wire_values[12]:.3f},{wire_values[13]:.3f},"
                            f"{wire_values[14]:.3f}) halfExtentV={wire_values[15]:.3f}"
                            f"{cylinder_details}"
                        )
                    if shaping_flags & _LIGHT_SHAPING_APPLIED:
                        carb.log_warn(
                            "isaacmetalbridge.stage: UsdLuxShapingAPI published: "
                            f"schema={schema_name} path={path} "
                            f"axis=({shaping_values[0]:.3f},"
                            f"{shaping_values[1]:.3f},{shaping_values[2]:.3f}) "
                            f"cone={shaping_values[3]:.3f}deg "
                            f"softness={shaping_values[4]:.3f} "
                            f"focus={shaping_values[5]:.3f} "
                            f"focusTint=({shaping_values[6]:.3f},"
                            f"{shaping_values[7]:.3f},{shaping_values[8]:.3f})"
                        )
                    if light_texture_flags & _LIGHT_TEXTURE_RECT_EMISSION:
                        carb.log_warn(
                            "isaacmetalbridge.stage: RectLight emission texture "
                            f"published: path={path} textureIndex="
                            f"{emission_texture_index}"
                        )
                    if light_texture_flags & _LIGHT_TEXTURE_IES_PROFILE:
                        carb.log_warn(
                            "isaacmetalbridge.stage: ShapingAPI IES profile "
                            f"published: path={path} textureIndex="
                            f"{ies_texture_index} angleScale={ies_angle_scale:.3f} "
                            f"multiplier={ies_multiplier:.6f} normalize="
                            f"{bool(light_texture_flags & _LIGHT_TEXTURE_IES_NORMALIZED)}"
                        )
                for light_path, asset_kind, asset_path in rejected_light_assets:
                    carb.log_warn(
                        f"isaacmetalbridge.stage: {asset_kind} rejected by "
                        f"bounded decoder: path={light_path} asset={asset_path}"
                    )
                if distant_light_path:
                    carb.log_warn(
                        "isaacmetalbridge.stage: active USD DistantLight "
                        f"published: path={distant_light_path} "
                        f"direction=({distant_light_values[0]:.3f},"
                        f"{distant_light_values[1]:.3f},"
                        f"{distant_light_values[2]:.3f}) "
                        f"color=({distant_light_values[3]:.3f},"
                        f"{distant_light_values[4]:.3f},"
                        f"{distant_light_values[5]:.3f}) "
                        f"intensity={distant_light_values[6]:.3f} "
                        f"angle={distant_light_values[7]:.3f}deg"
                    )
                if dome_light_path:
                    carb.log_warn(
                        "isaacmetalbridge.stage: active USD DomeLight published: "
                        f"path={dome_light_path} "
                        f"color=({dome_light_values[0]:.3f},"
                        f"{dome_light_values[1]:.3f},"
                        f"{dome_light_values[2]:.3f}) "
                        f"intensity={dome_light_values[3]:.3f}"
                    )
                    for (
                        _schema_code,
                        _schema_name,
                        path,
                        _values,
                        _wire_values,
                        texture_index,
                        light_texture_flags,
                    ) in dome_lights:
                        if light_texture_flags & _LIGHT_TEXTURE_DOME_ENVIRONMENT:
                            carb.log_warn(
                                "isaacmetalbridge.stage: DomeLight latlong "
                                f"environment published: path={path} "
                                f"textureIndex={texture_index} rgbe="
                                f"{bool(light_texture_flags & _LIGHT_TEXTURE_DOME_RGBE)}"
                            )
                if additional_light_descriptions:
                    carb.log_warn(
                        "isaacmetalbridge.stage: complete USD light list published: "
                        f"count={len(additional_light_descriptions)} "
                        f"lights={additional_light_descriptions}"
                    )
                carb.log_warn(
                    "isaacmetalbridge.stage: visible USD Mesh manifest published: "
                    f"count={len(mesh_records)} "
                    f"fileTextures={file_texture_mesh_count} "
                    f"materialParameters={parameter_mesh_count} "
                    f"emissiveMaterials={emission_mesh_count} "
                    f"parameterTextures={parameter_texture_count} "
                    f"normalTextures={normal_texture_count} "
                    f"alphaCutoutMeshes={alpha_cutout_mesh_count} "
                    f"standardOpacityMeshes={standard_opacity_mesh_count} "
                    f"uvTransforms={uv_transform_mesh_count} "
                    f"geometryMiB={scene_geometry_bytes / 1048576.0:.1f} "
                    f"textureMiB={scene_texture_bytes / 1048576.0:.1f} "
                    f"uniqueTextures={len(scene_material_texture_paths)} "
                    f"instanceProxyMeshes={instance_proxy_mesh_count} "
                    f"pointInstances={len(point_instance_keys)} "
                    f"pointInstanceMeshes={point_instance_mesh_count} "
                    f"startupReferenceMeshes={reference_mesh_count} "
                    f"pathSample=[{','.join(mesh_path_sample)}]"
                )
            elif self._camera_sequence <= 3 or self._camera_sequence % 30 == 0:
                carb.log_warn(
                    "isaacmetalbridge.stage: live USD scene update published: "
                    f"sequence={self._camera_sequence} "
                    f"timeSeconds={timeline_seconds:.4f} "
                    f"timeCode={time_code_value:.4f} "
                    f"meshes={len(mesh_records)} "
                    f"pointInstances={len(point_instance_keys)}"
                )
        except Exception as error:
            if not self._camera_error_reported:
                carb.log_error(
                    f"isaacmetalbridge.stage: camera publication failed: {error}"
                )
                self._camera_error_reported = True

    async def _open_stage(self, stage_url: str) -> None:
        try:
            result, error = await omni.usd.get_context().open_stage_async(stage_url)
            if not result:
                carb.log_error(
                    "isaacmetalbridge.stage: startup stage open rejected: "
                    f"{stage_url}: {error}"
                )
                return

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                carb.log_error(
                    "isaacmetalbridge.stage: startup stage open returned no stage: "
                    f"{stage_url}"
                )
                return

            if self._startup_reference_url:
                previous_edit_target = stage.GetEditTarget()
                try:
                    # Keep the downloaded NVIDIA background immutable. The
                    # robot composition belongs in the anonymous session layer
                    # used for this launch only.
                    stage.SetEditTarget(stage.GetSessionLayer())
                    reference_prim = stage.DefinePrim(
                        self._startup_reference_path, "Xform"
                    )
                    if not reference_prim.GetReferences().AddReference(
                        self._startup_reference_url
                    ):
                        raise RuntimeError("USD reference authoring was rejected")
                    if self._add_startup_key_light:
                        key_light = UsdLux.SphereLight.Define(
                            stage, "/World/IsaacMetalBridgeKeyLight"
                        )
                        key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.94, 0.84))
                        key_light.CreateIntensityAttr(45000.0)
                        key_light.CreateRadiusAttr(0.35)
                        UsdGeom.Xformable(key_light.GetPrim()).AddTranslateOp().Set(
                            Gf.Vec3d(2.0, -2.0, 3.0)
                        )
                finally:
                    stage.SetEditTarget(previous_edit_target)
                # Give USD and Kit a few updates to resolve the remote robot's
                # referenced visual and physics layers before framing it.
                app = omni.kit.app.get_app()
                for _attempt in range(8):
                    await app.next_update_async()
                reference_prim = stage.GetPrimAtPath(self._startup_reference_path)
                child_count = sum(1 for _child in reference_prim.GetChildren())
                carb.log_warn(
                    "isaacmetalbridge.stage: real startup reference composed: "
                    f"path={self._startup_reference_path} children={child_count} "
                    f"asset={self._startup_reference_url}"
                )
                if self._add_startup_key_light:
                    carb.log_warn(
                        "isaacmetalbridge.stage: real USD SphereLight composed "
                        "for robot inspection: /World/IsaacMetalBridgeKeyLight"
                    )

            root_layer = stage.GetRootLayer().identifier
            root_prims = ",".join(prim.GetName() for prim in stage.GetPseudoRoot().GetChildren())
            world = stage.GetPrimAtPath("/World")
            world_prims = (
                ",".join(prim.GetName() for prim in world.GetChildren())
                if world.IsValid()
                else ""
            )
            carb.log_warn(
                "isaacmetalbridge.stage: startup stage open completed: "
                f"root={root_layer} prims=[{root_prims}] worldPrims=[{world_prims}]"
            )
            if self._restore_full_layout:
                await self._restore_default_full_layout()
            if self._timeline_autoplay:
                timeline = omni.timeline.get_timeline_interface()
                timeline.set_current_time(timeline.get_start_time())
                timeline.set_looping(True)
                timeline.play()
                carb.log_warn(
                    "isaacmetalbridge.stage: live USD timeline playback started: "
                    f"start={timeline.get_start_time():.4f}s "
                    f"end={timeline.get_end_time():.4f}s "
                    f"timeCodesPerSecond={timeline.get_time_codes_per_second():.4f}"
                )
            if self._physics_smoke_output:
                await self._run_physics_smoke(stage)
            if self._camera_sensor_output:
                await self._capture_camera_sensor(stage)
        except asyncio.CancelledError:
            return
        except Exception as error:
            carb.log_error(
                f"isaacmetalbridge.stage: startup stage open failed: {error}"
            )

    async def _restore_default_full_layout(self) -> None:
        """Keep or reapply Isaac Sim's own default Full workspace."""
        try:
            app = omni.kit.app.get_app()
            for _attempt in range(1200):
                if app.is_app_ready():
                    break
                await app.next_update_async()
            if not app.is_app_ready():
                raise RuntimeError(
                    "Kit did not reach app-ready before Full layout restore"
                )

            import omni.ui as ui

            required_windows = ("Viewport", "Stage", "Property", "Content")
            windows = {
                title: ui.Workspace.get_window(title)
                for title in required_windows
            }
            workspace_is_docked = all(
                window is not None and window.visible and window.docked
                for window in windows.values()
            )

            # In the normal path the late stage open preserves Full's genuine
            # workspace. Only fall back to the official layout loader when a
            # window is actually missing or detached; reloading an already
            # healthy DockSpace can itself create a second UI root headlessly.
            from isaacsim.app.setup.layout import (
                LAYOUTS_PATH,
                dock_windows,
                load_layout,
            )
            if not workspace_is_docked:
                await load_layout(
                    str(LAYOUTS_PATH / "default.json"),
                    keep_windows_open=False,
                )
                for _attempt in range(60):
                    await app.next_update_async()

                async def update_layout() -> None:
                    await app.next_update_async()

                await dock_windows(update_layout)
                windows = {
                    title: ui.Workspace.get_window(title)
                    for title in required_windows
                }
            else:
                carb.log_warn(
                    "isaacmetalbridge.stage: preserving existing docked "
                    "Isaac Sim Full workspace"
                )
            missing_windows = [
                title
                for title, window in windows.items()
                if window is None or not window.visible
            ]
            if missing_windows:
                raise RuntimeError(
                    "default Full layout is missing windows: "
                    + ",".join(missing_windows)
                )
            if not all(
                windows[title].docked
                for title in ("Stage", "Property", "Content")
            ):
                detached = [
                    title
                    for title in ("Stage", "Property", "Content")
                    if not windows[title].docked
                ]
                raise RuntimeError(
                    "default Full layout has detached windows: "
                    + ",".join(detached)
                )
            for title, window in windows.items():
                carb.log_warn(
                    "isaacmetalbridge.stage: Full window "
                    f"title={title} visible={window.visible} "
                    f"position=({float(window.position_x):.1f},"
                    f"{float(window.position_y):.1f}) "
                    f"size={float(window.width):.1f}x{float(window.height):.1f} "
                    f"docked={bool(window.docked)}"
                )
            if self._full_layout_ready_file:
                temporary_path = (
                    f"{self._full_layout_ready_file}.tmp.{os.getpid()}"
                )
                with open(temporary_path, "w", encoding="utf-8") as ready_file:
                    ready_file.write("isaac-full-layout-ready\n")
                os.replace(temporary_path, self._full_layout_ready_file)
            # Trigger a dirty frame after the bridge can see the ready marker.
            # Stage is in the main root; detached Viewport frames are rejected
            # by the Vulkan Full-workspace classifier.
            windows["Stage"].focus()
            for _attempt in range(10):
                await app.next_update_async()
            carb.log_warn(
                "isaacmetalbridge.stage: restored Isaac Sim default Full "
                "workspace with Stage/Property/Content"
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            carb.log_error(
                "isaacmetalbridge.stage: Full workspace restore failed: "
                f"{error}"
            )
        finally:
            # Do not permanently suppress camera publication if a future
            # Isaac release renames a window or layout action.
            self._layout_restored = True

    async def _run_physics_smoke(self, stage: Usd.Stage) -> None:
        """Exercise the real Kit timeline and PhysX without altering the USD file."""
        app = None
        timeline = None
        previous_edit_target = stage.GetEditTarget()
        result = {
            "backend": "Kit timeline plus PhysX CPU scene",
            "passed": False,
        }
        try:
            app = omni.kit.app.get_app()
            for _attempt in range(1200):
                if app.is_app_ready():
                    break
                await app.next_update_async()
            if not app.is_app_ready():
                raise RuntimeError(
                    "Kit did not reach app-ready before the PhysX smoke test"
                )

            # Author only into the anonymous session layer. The checked-in
            # validation stage and the user's normal/default stage stay clean.
            stage.SetEditTarget(stage.GetSessionLayer())
            physics_scene = UsdPhysics.Scene.Define(
                stage, "/World/PhysicsSmokeScene"
            )
            physics_scene.CreateGravityDirectionAttr().Set(
                Gf.Vec3f(0.0, 0.0, -1.0)
            )
            physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

            # Explicitly keep this validation on CPU PhysX. GPU PhysX depends
            # on NVIDIA CUDA behavior that the compatibility shim does not yet
            # implement generally.
            try:
                from pxr import PhysxSchema

                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(
                    physics_scene.GetPrim()
                )
                physx_scene.CreateEnableGPUDynamicsAttr(False)
                physx_scene.CreateBroadphaseTypeAttr("MBP")
                carb.settings.get_settings().set_bool(
                    "/physics/suppressReadback", False
                )
            except Exception as error:
                carb.log_warn(
                    "isaacmetalbridge.stage: could not author the optional "
                    f"PhysX CPU-only schema attribute: {error}"
                )

            # Use Isaac Sim's own high-level object classes. Besides validating
            # a public Isaac API, this avoids relying on a hand-authored subset
            # of the PhysX schemas for contact/collider setup.
            import numpy
            from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid

            DynamicCuboid(
                prim_path="/World/PhysicsSmokeCube",
                name="imb_physics_smoke_cube",
                position=numpy.array([0.0, 0.0, 2.5]),
                scale=numpy.array([1.0, 1.0, 1.0]),
                color=numpy.array([0.95, 0.3, 0.1]),
                mass=1.0,
            )
            FixedCuboid(
                prim_path="/World/PhysicsSmokeGround",
                name="imb_physics_smoke_ground",
                position=numpy.array([0.0, 0.0, -0.5]),
                scale=numpy.array([10.0, 10.0, 1.0]),
                color=numpy.array([0.25, 0.25, 0.25]),
            )
            cube_prim = stage.GetPrimAtPath("/World/PhysicsSmokeCube")
            if not cube_prim.IsValid():
                raise RuntimeError("Isaac DynamicCuboid prim was not created")

            def world_position() -> Gf.Vec3d:
                return UsdGeom.XformCache(
                    Usd.TimeCode.Default()
                ).GetLocalToWorldTransform(cube_prim).ExtractTranslation()

            for _attempt in range(10):
                await app.next_update_async()
            initial = world_position()

            from omni import timeline as omni_timeline

            timeline = omni_timeline.get_timeline_interface()
            timeline.set_start_time(0.0)
            timeline.set_end_time(10.0)
            timeline.set_looping(False)
            timeline.set_current_time(0.0)
            timeline.play()
            timeline_play_observed = False

            updates = 0
            stable_updates = 0
            previous_z = float(initial[2])
            final = initial
            wall_start = time.monotonic()
            # Bound both wall time and update count because a bridged Full UI
            # update can be much slower than a native headless CPU frame.
            for updates in range(1, 241):
                await app.next_update_async()
                final = world_position()
                current_z = float(final[2])
                timeline_position = float(timeline.get_current_time())
                timeline_play_observed = (
                    timeline_play_observed
                    or bool(timeline.is_playing())
                    or timeline_position > 0.0
                )
                drop = float(initial[2]) - current_z
                if drop > 1.0 and abs(current_z - previous_z) < 1.0e-4:
                    stable_updates += 1
                else:
                    stable_updates = 0
                previous_z = current_z
                if stable_updates >= 20 and timeline_position >= 1.0:
                    break
                # The 1 m cube's resting center is z=0.5 on this box floor.
                # At t>=1 s it would be well below the floor without contact,
                # so this bounded range is direct collision evidence.
                if timeline_position >= 1.0 and 0.4 <= current_z <= 0.65:
                    break
                if time.monotonic() - wall_start >= 60.0:
                    break

            timeline_position = float(timeline.get_current_time())
            wall_elapsed = time.monotonic() - wall_start
            drop = float(initial[2]) - float(final[2])
            passed = (
                timeline_play_observed
                and drop >= 1.5
                and timeline_position >= 1.0
                and float(final[2]) >= 0.4
                and float(final[2]) <= 0.65
            )
            result = {
                "backend": "Kit timeline plus PhysX CPU scene",
                "cubePath": "/World/PhysicsSmokeCube",
                "groundPath": "/World/PhysicsSmokeGround",
                "initialPosition": [float(value) for value in initial],
                "finalPosition": [float(value) for value in final],
                "dropDistance": drop,
                "timelinePlayObserved": timeline_play_observed,
                "timelinePositionSeconds": timeline_position,
                "wallSeconds": wall_elapsed,
                "updateCount": updates,
                "settledUpdateCount": stable_updates,
                "passed": passed,
            }
            if not passed:
                result["error"] = (
                    "timeline did not advance a colliding rigid body to a "
                    "physically valid resting height"
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            result["error"] = str(error)
        finally:
            if timeline is not None:
                # Pause preserves the measured resting transform. Calling Stop
                # here asks PhysX to reset authored transforms while the async
                # extension task is still on the update stack; Kit 110.1 can
                # crash in that re-entrant reset path. The explicit smoke run
                # is destroyed with its container, so a reset is unnecessary.
                timeline.pause()
                if app is not None:
                    await app.next_update_async()
                result["timelinePausedAfterTest"] = not timeline.is_playing()
            stage.SetEditTarget(previous_edit_target)

        temporary_path = f"{self._physics_smoke_output}.tmp.{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8") as result_file:
            json.dump(result, result_file, indent=2, sort_keys=True)
            result_file.write("\n")
        os.replace(temporary_path, self._physics_smoke_output)
        if result.get("passed"):
            carb.log_warn(
                "isaacmetalbridge.stage: real Kit timeline/CPU PhysX smoke "
                f"passed drop={result['dropDistance']:.4f}m "
                f"finalZ={result['finalPosition'][2]:.4f}m "
                f"wall={result['wallSeconds']:.4f}s "
                f"updates={result['updateCount']}"
            )
        else:
            carb.log_error(
                "isaacmetalbridge.stage: real Kit timeline/CPU PhysX smoke "
                f"failed: {result.get('error', 'unknown failure')}"
            )

    async def _capture_camera_sensor(self, stage: Usd.Stage) -> None:
        render_product = None
        annotator = None
        try:
            if (
                self._camera_sensor_width < 16
                or self._camera_sensor_width > 8192
                or self._camera_sensor_height < 16
                or self._camera_sensor_height > 8192
            ):
                raise ValueError("camera sensor resolution must be 16..8192")

            app = omni.kit.app.get_app()
            for _attempt in range(1200):
                if app.is_app_ready():
                    break
                await app.next_update_async()
            if not app.is_app_ready():
                raise RuntimeError("Kit did not reach app-ready before sensor capture")

            from omni.kit.viewport.utility import get_active_viewport

            camera_path = None
            for _attempt in range(300):
                viewport = get_active_viewport()
                candidate = (
                    getattr(viewport, "camera_path", None)
                    if viewport is not None
                    else None
                )
                if callable(candidate):
                    candidate = candidate()
                if candidate:
                    camera_prim = stage.GetPrimAtPath(candidate)
                    if camera_prim.IsValid() and camera_prim.IsA(UsdGeom.Camera):
                        camera_path = str(candidate)
                        break
                await app.next_update_async()
            if not camera_path:
                raise RuntimeError("active USD camera was not available")

            import numpy

            rgb = None
            data_source = None
            replicator_rgb_ready = False
            # The Metal bridge already renders the active Kit camera from the
            # real USD stage. Creating an additional Replicator Render Product
            # on the NVIDIA-compat device makes RTX record a second unsupported
            # render graph and causes multiTickRateRender()/Sensor endFrame()
            # failures. Consume the bridge's RGB sensor frame directly instead.
            for _attempt in range(240):
                await app.next_update_async()
                if self._camera_sensor_frame_file:
                    try:
                        with open(
                            self._camera_sensor_frame_file,
                            "rb",
                        ) as bridge_frame:
                            magic = bridge_frame.readline()
                            dimensions = bridge_frame.readline().split()
                            maximum = bridge_frame.readline()
                            frame_bytes = bridge_frame.read()
                        if (
                            magic == b"P6\n"
                            and dimensions
                            == [
                                str(self._camera_sensor_width).encode("ascii"),
                                str(self._camera_sensor_height).encode("ascii"),
                            ]
                            and maximum == b"255\n"
                            and len(frame_bytes)
                            == self._camera_sensor_width
                            * self._camera_sensor_height
                            * 3
                        ):
                            bridge_rgb = numpy.frombuffer(
                                frame_bytes,
                                dtype=numpy.uint8,
                            ).reshape(
                                self._camera_sensor_height,
                                self._camera_sensor_width,
                                3,
                            )
                            if numpy.any(bridge_rgb):
                                rgb = numpy.ascontiguousarray(bridge_rgb)
                                data_source = "metal-camera-sensor"
                                break
                    except FileNotFoundError:
                        pass
            if rgb is None or not numpy.any(rgb):
                raise RuntimeError(
                    "Metal camera sensor returned no non-zero pixels"
                )

            rgb_bytes = rgb.tobytes()
            temporary_path = (
                f"{self._camera_sensor_output}.tmp.{os.getpid()}"
            )
            with open(temporary_path, "wb", buffering=0) as sensor_file:
                sensor_file.write(
                    f"P6\n{self._camera_sensor_width} "
                    f"{self._camera_sensor_height}\n255\n".encode("ascii")
                )
                sensor_file.write(rgb_bytes)
            os.replace(temporary_path, self._camera_sensor_output)

            metadata = {
                "cameraPath": camera_path,
                "width": self._camera_sensor_width,
                "height": self._camera_sensor_height,
                "channels": 3,
                "dtype": "uint8",
                "annotator": "isaacmetalbridge.rgb",
                "dataSource": data_source,
                "replicatorRgbDataReady": replicator_rgb_ready,
                "crc32": f"{zlib.crc32(rgb_bytes) & 0xFFFFFFFF:08x}",
            }
            metadata_path = f"{self._camera_sensor_output}.json"
            metadata_temporary_path = f"{metadata_path}.tmp.{os.getpid()}"
            with open(
                metadata_temporary_path,
                "w",
                encoding="utf-8",
            ) as metadata_file:
                json.dump(metadata, metadata_file, indent=2, sort_keys=True)
                metadata_file.write("\n")
            os.replace(metadata_temporary_path, metadata_path)
            carb.log_warn(
                "isaacmetalbridge.stage: Isaac Camera sensor published: "
                f"camera={camera_path} resolution={self._camera_sensor_width}x"
                f"{self._camera_sensor_height} bytes={len(rgb_bytes)} "
                f"source={data_source} "
                f"crc32={metadata['crc32']} output={self._camera_sensor_output}"
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            carb.log_error(
                "isaacmetalbridge.stage: Isaac Camera RGB capture failed: "
                f"{error}"
            )
        finally:
            if annotator is not None and render_product is not None:
                try:
                    annotator.detach([render_product])
                except Exception:
                    pass
            if render_product is not None:
                try:
                    render_product.destroy()
                except Exception:
                    pass
