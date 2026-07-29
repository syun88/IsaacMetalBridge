"""Open an explicit USD stage before Isaac Sim's renderer finishes startup."""

from __future__ import annotations

import asyncio
import os

import carb
import omni.ext
import omni.usd


class StartupStageExtension(omni.ext.IExt):
    """Open IMB_STARTUP_STAGE_URL through the real omni.usd context."""

    def on_startup(self, ext_id: str) -> None:
        del ext_id
        self._task = None
        stage_url = os.environ.get("IMB_STARTUP_STAGE_URL", "")
        if not stage_url:
            carb.log_error(
                "isaacmetalbridge.stage: IMB_STARTUP_STAGE_URL is unset; "
                "no startup stage will be opened"
            )
            return

        carb.log_warn(
            f"isaacmetalbridge.stage: opening startup stage: {stage_url}"
        )
        self._task = asyncio.ensure_future(self._open_stage(stage_url))

    def on_shutdown(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

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
        except asyncio.CancelledError:
            return
        except Exception as error:
            carb.log_error(
                f"isaacmetalbridge.stage: startup stage open failed: {error}"
            )
