#!/usr/bin/env python3
"""Cache one or more official Isaac USD trees and their local dependencies."""

from __future__ import annotations

import os
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import deque

from pxr import Sdf


_USD_SUFFIXES = {".usd", ".usda", ".usdc"}
_MAX_FILES = 4096
_MAX_FILE_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
_ASSET_TOKEN_PATTERN = re.compile(
    r"@([^@\r\n]+?\.(?:usd|usda|usdc|mdl|png|jpg|jpeg|exr|hdr|tga|bmp))@",
    re.IGNORECASE,
)
_MDL_RELATIVE_MODULE_PATTERN = re.compile(
    r"(?:using|import)\s+\.::([A-Za-z_][A-Za-z0-9_:]*)"
)
_MDL_FILE_PATTERN = re.compile(
    r'"([^"\r\n]+\.(?:png|jpg|jpeg|exr|hdr|tga|bmp))"', re.IGNORECASE
)


def _fail(message: str) -> None:
    raise RuntimeError(f"cache-isaac-usd-tree: {message}")


def _relative_dependency(
    cache_root: pathlib.Path, layer_path: pathlib.Path, dependency: str
) -> pathlib.Path | None:
    value = dependency.strip()
    if not value or "<UDIM>" in value:
        return None
    # Bare standard modules such as OmniPBR.mdl are supplied by Kit's MDL
    # search path. Relative asset modules (for example ../Materials/*.mdl in
    # Simple Warehouse) must be mirrored beside the USD or native RTX emits a
    # shader-resolution error for almost every mesh.
    normalized_value = value.replace("\\", "/")
    if value.endswith(".mdl") and "/" not in normalized_value:
        return None
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme:
        return None
    if value.startswith("/opt/imb-assets/"):
        candidate = cache_root / value.removeprefix("/opt/imb-assets/")
    elif value.startswith("/Isaac/"):
        candidate = cache_root / value.lstrip("/")
    elif os.path.isabs(value):
        return None
    else:
        candidate = layer_path.parent / value
    candidate = candidate.resolve()
    try:
        return candidate.relative_to(cache_root)
    except ValueError:
        _fail(f"dependency escapes cache root: {dependency} from {layer_path}")


def _download(base_url: str, cache_root: pathlib.Path, relative_path: pathlib.Path) -> int:
    target = cache_root / relative_path
    if target.is_file() and target.stat().st_size > 0:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded_path = urllib.parse.quote(relative_path.as_posix(), safe="/")
    url = f"{base_url.rstrip('/')}/{encoded_path}"
    temporary = target.with_name(f"{target.name}.part.{os.getpid()}")
    byte_count = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > _MAX_FILE_BYTES:
                    _fail(f"asset exceeds per-file limit: {relative_path}")
                output.write(chunk)
        if byte_count == 0:
            _fail(f"downloaded an empty asset: {relative_path}")
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    print(f"cache-isaac-usd-tree: downloaded {relative_path} ({byte_count} bytes)")
    return byte_count


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "usage: cache-isaac-usd-tree.py BASE_URL CACHE_ROOT ROOT_USD...",
            file=sys.stderr,
        )
        return 2
    base_url = sys.argv[1]
    cache_root = pathlib.Path(sys.argv[2]).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    roots = [pathlib.Path(value) for value in sys.argv[3:]]
    pending = deque(roots)
    visited: set[pathlib.Path] = set()
    total_downloaded = 0

    while pending:
        relative_path = pathlib.Path(os.path.normpath(pending.popleft().as_posix()))
        if relative_path.is_absolute() or relative_path.as_posix().startswith("../"):
            _fail(f"invalid relative asset path: {relative_path}")
        if relative_path in visited:
            continue
        visited.add(relative_path)
        if len(visited) > _MAX_FILES:
            _fail(f"dependency count exceeds {_MAX_FILES}")
        try:
            total_downloaded += _download(base_url, cache_root, relative_path)
        except urllib.error.HTTPError as error:
            _fail(f"HTTP {error.code} for {relative_path}")
        if total_downloaded > _MAX_TOTAL_BYTES:
            _fail("download total exceeds 4 GiB")
        layer_path = cache_root / relative_path
        dependencies: set[str] = set()
        if relative_path.suffix.lower() in _USD_SUFFIXES:
            layer = Sdf.Layer.FindOrOpen(str(layer_path))
            if layer is None:
                _fail(f"USD layer could not be opened: {relative_path}")
            dependencies.update(layer.GetCompositionAssetDependencies())
            dependencies.update(layer.GetExternalAssetDependencies())
            # Sdf's external dependency list omits unresolved MDL source
            # assets in binary layers. USDA notation exposes each authored
            # @asset@ token without guessing material filenames.
            dependencies.update(
                _ASSET_TOKEN_PATTERN.findall(layer.ExportToString())
            )
        elif relative_path.suffix.lower() == ".mdl":
            mdl_source = layer_path.read_text(encoding="utf-8-sig")
            dependencies.update(_MDL_FILE_PATTERN.findall(mdl_source))
            for module_name in _MDL_RELATIVE_MODULE_PATTERN.findall(mdl_source):
                module_path = module_name.rstrip(":").replace("::", "/")
                dependencies.add(f"./{module_path}.mdl")
        else:
            continue
        for dependency in sorted(dependencies):
            resolved = _relative_dependency(cache_root, layer_path, dependency)
            if resolved is not None and resolved not in visited:
                pending.append(resolved)

    print(
        "cache-isaac-usd-tree: ready "
        f"files={len(visited)} downloadedBytes={total_downloaded} root={cache_root}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
