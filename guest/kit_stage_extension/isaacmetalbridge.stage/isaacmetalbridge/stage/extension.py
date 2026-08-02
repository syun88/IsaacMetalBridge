"""Open an explicitly requested USD stage during Isaac Sim startup."""

from __future__ import annotations

import asyncio
import functools
import json
import math
import os
import struct
import time
import zlib

import carb
import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


_SCENE_HEADER = struct.Struct("<IHHQ32fI")
_SCENE_MESH_RECORD = struct.Struct("<QII24f23I")
_SCENE_MAGIC = 0x31434D49
_SCENE_VERSION = 11
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
_MESH_HAS_BOUND_MATERIAL = 1
_MESH_HAS_BASE_COLOR = 2
_MESH_HAS_CONNECTED_BASE_COLOR = 4
_MESH_HAS_FILE_TEXTURE = 8
_MESH_HAS_MATERIAL_PARAMETERS = 16
_MESH_HAS_EMISSION = 32
_TEXTURE_HAS_ROUGHNESS = 1
_TEXTURE_HAS_METALLIC = 2
_TEXTURE_HAS_EMISSION = 4
_TEXTURE_HAS_NORMAL = 8
_MAX_SCENE_MESHES = 4096
_MAX_SCENE_GEOMETRY_BYTES = 256 * 1024 * 1024
_MAX_SCENE_TEXTURE_DIMENSION = 4096


def _stage_prims(stage: Usd.Stage):
    """Traverse ordinary prims and native USD scenegraph instance proxies."""

    return Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())


def _path_is_at_or_below(path, root_path) -> bool:
    value = str(path)
    root = str(root_path).rstrip("/")
    return value == root or value.startswith(root + "/")


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
        yield (
            prim,
            str(prim.GetPath()),
            xform_cache.GetLocalToWorldTransform(prim),
            prim.IsInstanceProxy(),
            None,
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
                yield (
                    mesh_prim,
                    synthetic_path,
                    world_transform,
                    mesh_prim.IsInstanceProxy(),
                    f"{instancer_prim.GetPath()}[{instance_index}]",
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
            shader_id = str(shader.GetIdAttr().Get() or "")
            if shader_id != "UsdPreviewSurface" and "OmniPBR" not in shader_id:
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
            shader_id = str(shader.GetIdAttr().Get() or "")
            if shader_id != "UsdPreviewSurface" and "OmniPBR" not in shader_id:
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
            shader_id = str(shader.GetIdAttr().Get() or "")
            if shader_id != "UsdPreviewSurface" and "OmniPBR" not in shader_id:
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


@functools.lru_cache(maxsize=16)
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
            pixels = rgba.tobytes()
            if len(pixels) != rgba.width * rgba.height * 4:
                return None
            return rgba.width, rgba.height, pixels
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
        self._camera_sequence = 0
        self._last_scene_values = None
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
            carb.log_error(
                "isaacmetalbridge.stage: IMB_STARTUP_STAGE_URL is unset; "
                "no startup stage will be opened"
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
        self._app_ready_updates = 0
        self._layout_restored = False

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
            camera_path = getattr(viewport, "camera_path", None)
            if callable(camera_path):
                camera_path = camera_path()
            if not camera_path:
                return
            camera_prim = stage.GetPrimAtPath(camera_path)
            if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
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

            mesh_records = []
            mesh_paths = []
            file_texture_mesh_count = 0
            parameter_mesh_count = 0
            emission_mesh_count = 0
            parameter_texture_count = 0
            normal_texture_count = 0
            uv_transform_mesh_count = 0
            instance_proxy_mesh_count = 0
            point_instance_mesh_count = 0
            point_instance_keys = set()
            scene_geometry_bytes = 0
            for (
                prim,
                mesh_path,
                mesh_world_transform,
                is_instance_proxy,
                point_instance_key,
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
                    if face_index not in holes and count >= 3:
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
                    or geometry_bytes > _MAX_SCENE_GEOMETRY_BYTES
                    or scene_geometry_bytes
                    > _MAX_SCENE_GEOMETRY_BYTES - geometry_bytes
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
                try:
                    material, _relationship = UsdShade.MaterialBindingAPI(
                        prim
                    ).ComputeBoundMaterial()
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

                texture_specs = []
                if file_texture is not None:
                    texture_specs.append(("base", *file_texture, 4))
                if roughness_file_texture is not None:
                    texture_specs.append(("roughness", *roughness_file_texture))
                if metallic_file_texture is not None:
                    texture_specs.append(("metallic", *metallic_file_texture))
                if emission_file_texture is not None:
                    texture_specs.append(("emission", *emission_file_texture))
                if normal_file_texture is not None:
                    texture_specs.append(("normal", *normal_file_texture))

                decoded_textures = {}
                corner_uv_values = ()
                if texture_specs:
                    shared_uv_source = texture_specs[0][2]
                    for kind, texture_path, uv_source, channel in texture_specs:
                        if uv_source != shared_uv_source:
                            continue
                        decoded_texture = _load_rgba_texture(texture_path)
                        if decoded_texture is not None:
                            decoded_textures[kind] = (*decoded_texture, channel)
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
                    texture_bytes = len(corner_uv_values) * 4 + sum(
                        len(value[2]) for value in decoded_textures.values()
                    )
                    if (
                        texture_bytes
                        <= _MAX_SCENE_GEOMETRY_BYTES - geometry_bytes
                        and scene_geometry_bytes
                        <= _MAX_SCENE_GEOMETRY_BYTES
                            - geometry_bytes
                            - texture_bytes
                    ):
                        geometry_bytes += texture_bytes
                        if shared_uv_source[1]:
                            uv_transform_mesh_count += 1
                        if "base" in decoded_textures:
                            texture_width, texture_height, texture_pixels, _ = (
                                decoded_textures["base"]
                            )
                            material_flags |= _MESH_HAS_FILE_TEXTURE
                            file_texture_mesh_count += 1
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
                        len(points),
                        len(triangle_indices),
                        len(corner_normal_values) // 3,
                        len(corner_uv_values) // 2,
                        texture_width,
                        texture_height,
                        len(texture_pixels),
                        texture_flags,
                        roughness_texture_width,
                        roughness_texture_height,
                        len(roughness_texture_pixels),
                        roughness_texture_channel,
                        metallic_texture_width,
                        metallic_texture_height,
                        len(metallic_texture_pixels),
                        metallic_texture_channel,
                        emission_texture_width,
                        emission_texture_height,
                        len(emission_texture_pixels),
                        emission_texture_channel,
                        normal_texture_width,
                        normal_texture_height,
                        len(normal_texture_pixels),
                    )
                    + vertex_payload
                    + index_payload
                    + normal_payload
                    + uv_payload
                    + texture_pixels
                    + roughness_texture_pixels
                    + metallic_texture_pixels
                    + emission_texture_pixels
                    + normal_texture_pixels
                )
                if is_instance_proxy:
                    instance_proxy_mesh_count += 1
                if point_instance_key is not None:
                    point_instance_mesh_count += 1
                    point_instance_keys.add(point_instance_key)
                scene_geometry_bytes += geometry_bytes
                mesh_paths.append(mesh_path)

            sphere_light_values = (0.0,) * 8
            sphere_light_path = ""
            for prim in _stage_prims(stage):
                if not prim.IsA(UsdLux.SphereLight):
                    continue
                light = UsdLux.SphereLight(prim)
                light_position = xform_cache.GetLocalToWorldTransform(
                    prim
                ).ExtractTranslation()
                light_color = light.GetColorAttr().Get(sample_time)
                intensity = light.GetIntensityAttr().Get(sample_time)
                exposure = light.GetExposureAttr().Get(sample_time)
                radius = light.GetRadiusAttr().Get(sample_time)
                if light_color is None or intensity is None:
                    continue
                effective_intensity = float(intensity) * math.pow(
                    2.0, float(exposure or 0.0)
                )
                sphere_light_values = (
                    float(light_position[0]),
                    float(light_position[1]),
                    float(light_position[2]),
                    max(float(light_color[0]), 0.0),
                    max(float(light_color[1]), 0.0),
                    max(float(light_color[2]), 0.0),
                    max(effective_intensity, 0.0),
                    max(float(radius or 0.0), 0.0001),
                )
                if all(math.isfinite(value) for value in sphere_light_values):
                    sphere_light_path = str(prim.GetPath())
                    break
                sphere_light_values = (0.0,) * 8

            distant_light_values = (0.0,) * 8
            distant_light_path = ""
            for prim in _stage_prims(stage):
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
                distant_light_values = (
                    float(light_direction[0]),
                    float(light_direction[1]),
                    float(light_direction[2]),
                    max(float(light_color[0]), 0.0),
                    max(float(light_color[1]), 0.0),
                    max(float(light_color[2]), 0.0),
                    max(effective_intensity, 0.0),
                    max(float(angle or 0.0), 0.0),
                )
                if (
                    all(math.isfinite(value) for value in distant_light_values)
                    and sum(value * value for value in distant_light_values[:3])
                    > 0.000001
                ):
                    distant_light_path = str(prim.GetPath())
                    break
                distant_light_values = (0.0,) * 8

            dome_light_values = (0.0,) * 4
            dome_light_path = ""
            for prim in _stage_prims(stage):
                if not prim.IsA(UsdLux.DomeLight):
                    continue
                light = UsdLux.DomeLight(prim)
                light_color = light.GetColorAttr().Get(sample_time)
                intensity = light.GetIntensityAttr().Get(sample_time)
                exposure = light.GetExposureAttr().Get(sample_time)
                if light_color is None or intensity is None:
                    continue
                effective_intensity = float(intensity) * math.pow(
                    2.0, float(exposure or 0.0)
                )
                dome_light_values = (
                    max(float(light_color[0]), 0.0),
                    max(float(light_color[1]), 0.0),
                    max(float(light_color[2]), 0.0),
                    max(effective_intensity, 0.0),
                )
                if all(math.isfinite(value) for value in dome_light_values):
                    dome_light_path = str(prim.GetPath())
                    break
                dome_light_values = (0.0,) * 4

            mesh_payload = b"".join(mesh_records)
            scene_values = (
                camera_values
                + sphere_light_values
                + distant_light_values
                + dome_light_values
                + (mesh_payload,)
            )
            if scene_values == self._last_scene_values:
                return

            self._camera_sequence += 1
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
            payload = _SCENE_HEADER.pack(
                _SCENE_MAGIC,
                _SCENE_VERSION,
                flags,
                self._camera_sequence,
                *camera_values,
                *sphere_light_values,
                *distant_light_values,
                *dome_light_values,
                len(mesh_records),
            )
            payload += mesh_payload
            temporary_path = (
                f"{self._camera_state_path}.tmp.{os.getpid()}"
            )
            with open(temporary_path, "wb", buffering=0) as camera_file:
                camera_file.write(payload)
            os.replace(temporary_path, self._camera_state_path)
            self._last_scene_values = scene_values
            self._camera_error_reported = False
            if self._camera_sequence == 1:
                carb.log_warn(
                    "isaacmetalbridge.stage: active Kit camera published: "
                    f"path={camera_path} position=({camera_values[0]:.3f},"
                    f"{camera_values[1]:.3f},{camera_values[2]:.3f}) "
                    f"fov={camera_values[9]:.4f} near={camera_values[10]:.4f} "
                    f"far={camera_values[11]:.1f}"
                )
                if sphere_light_path:
                    carb.log_warn(
                        "isaacmetalbridge.stage: active USD SphereLight published: "
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
                carb.log_warn(
                    "isaacmetalbridge.stage: visible USD Mesh manifest published: "
                    f"count={len(mesh_records)} "
                    f"fileTextures={file_texture_mesh_count} "
                    f"materialParameters={parameter_mesh_count} "
                    f"emissiveMaterials={emission_mesh_count} "
                    f"parameterTextures={parameter_texture_count} "
                    f"normalTextures={normal_texture_count} "
                    f"uvTransforms={uv_transform_mesh_count} "
                    f"instanceProxyMeshes={instance_proxy_mesh_count} "
                    f"pointInstances={len(point_instance_keys)} "
                    f"pointInstanceMeshes={point_instance_mesh_count} "
                    f"paths=[{','.join(mesh_paths)}]"
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
