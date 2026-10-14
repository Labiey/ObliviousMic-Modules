"""SDK registration for source development; installed bundles load through PluginManager."""

import json
from pathlib import Path

from obliviousmic.sdk import Manifest, ModuleRegistration
from obliviousmic.sdk.actions import ModuleAction

from .bsrnn import runtime as bsrnn
from .campplus import runtime as campplus


def register_models(registry, *, identity_factory=None, separation_factory=None):
    for module, factory in ((campplus, identity_factory), (bsrnn, separation_factory)):
        folder = Path(module.__file__).parent
        actions = json.loads((folder / "actions.schema.json").read_text("utf-8"))["actions"]
        declarations = [
            ModuleAction(
                name,
                item["label"],
                item["settings_schema"],
                tuple(item["permissions"]),
                getattr(module, item["handler"]),
            )
            for name, item in actions.items()
        ]
        registry.register(
            ModuleRegistration(
                Manifest(module.Module.MANIFEST),
                module.Module.SCHEMA,
                lambda cls=module.Module, make=factory: cls(make),
                actions=declarations,
                setup=json.loads((folder / "setup.json").read_text("utf-8")),
                resources=json.loads((folder / "resources.json").read_text("utf-8"))["assets"],
            )
        )
