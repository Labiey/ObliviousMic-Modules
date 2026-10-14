# Module package format

A release ZIP contains manifest.json, settings.schema.json, actions.schema.json,
setup.json, resources.json, runtime.py, actions.py, lib/, LICENSE, NOTICE and README.md.
No model weights or user data are included. Each module is self-contained except the stable
ObliviousMic SDK and declared third-party dependencies; modules do not import another module's package.

The framework checks manifest compatibility, declared dependencies and package hashes before installation.
Official source provenance authorizes official module IDs; core.* remains reserved.
Settings use the existing strict JSON Schema and an optional advanced annotation.
Setup declares settings_bindings/action_bindings, private artifacts and runtime_resources.
Resources identify role, source, license, exact size/hash and permitted download behavior.
A local-only resource with unresolved rights cannot silently become an automatically redistributed model.

An initialize action runs on the control plane. Model preparation never executes on audio callbacks.
Module installation/restoration uses unique Python namespaces; old versions can coexist and active/LKG
references prevent removal. Configuration/user artifacts remain outside installed module code.
