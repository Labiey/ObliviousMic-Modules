# SPDX-License-Identifier: Apache-2.0
"""Combined local first-use enrollment/baking; framework resolves model roles."""

import numpy as np

from .lib.identity_stream import WindowEmbeddingExtractor
from .lib.sessions import make_session


def initialize_action(ctx, args):
    lease = None
    try:
        ctx.job.progress(0.1, "reading_audio")
        samples = ctx.capabilities["audio.read"](args["file_id"])
        minimum = 48000 if ctx.module_id.endswith("campplus") else 64000
        if len(samples) < minimum:
            raise ValueError(
                "enrollment audio is too short; provide at least "
                + str(minimum // 16000)
                + " seconds"
            )
        path = ctx.capabilities["asset.read"](args["speaker_asset"])
        config = {
            "engine": args.get("engine", "onnxruntime"),
            "device": args.get("device", "CPU"),
            "threads": args.get("threads", 1),
        }
        lease = ctx.capabilities["resource.acquire"](
            path,
            config,
            lambda: make_session(
                config["engine"], path, config["threads"], device=config["device"]
            ),
        )
        extractor = WindowEmbeddingExtractor(
            path,
            engine=config["engine"],
            device=config["device"],
            session_factory=lambda *_: lease.value,
            probe_window_samples=48000,
        )
        extractor.initialize()
        ctx.job.progress(0.3, "registering_identity")
        if ctx.module_id.endswith("campplus"):
            embedding = extractor.embed_enrollment(samples)
        else:
            try:
                embedding = ctx.capabilities["profile.read"](args["profile_id"], "embedding")
            except (KeyError, FileNotFoundError):
                embedding = extractor.embed_enrollment(samples)
        vector = ctx.workdir / "embedding.npy"
        np.save(vector, embedding)

        def save_embedding():
            return ctx.capabilities["profile.write"](
                args["profile_id"],
                "org.obliviousmic.campplus",
                "embedding",
                vector,
                {"sample_rate": 16000, "channels": 1, "dimension": len(embedding)},
                args["speaker_asset"],
                "frozen-kaldi-fbank-v1",
            )

        if ctx.module_id.endswith("campplus"):
            artifact = ctx.job.commit(save_embedding)
        else:
            from .lib.bake import rebake_separation_core
            from .lib.excerpt import select_enrollment_excerpt

            excerpt = select_enrollment_excerpt(samples, embedding, embed=extractor.embed)
            template = ctx.capabilities["asset.read"](args["template_asset"])
            output = ctx.workdir / "baked.onnx"
            ctx.job.progress(0.6, "baking")
            rebake_separation_core(template, excerpt.samples, output)

            def publish():
                save_embedding()
                return ctx.capabilities["profile.write"](
                    args["profile_id"],
                    ctx.module_id,
                    "baked_core",
                    output,
                    {
                        "sample_rate": 16000,
                        "channels": 1,
                        "win": 512,
                        "hop": 128,
                        "template_sha256": args["template_asset"],
                    },
                    args["template_asset"],
                    "frozen-excerpt+enroll_mag-v1",
                )

            artifact = ctx.job.commit(publish)
        return {
            "profile_id": args["profile_id"],
            "module_id": ctx.module_id,
            "artifact_kind": artifact["artifact_kind"],
            "sha256": artifact["sha256"],
        }
    finally:
        if lease:
            lease.release()


def validate_action(ctx, args):
    kind = "embedding" if ctx.module_id.endswith("campplus") else "baked_core"
    ctx.capabilities["profile.read"](args["profile_id"], kind)
    return {"valid": True, "profile_id": args["profile_id"]}
