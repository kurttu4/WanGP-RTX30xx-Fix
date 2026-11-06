# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
from typing import Any

from PIL import Image

from postprocessing.face_refiner.bridge import ASSET_DOWNLOAD_DEF, FaceRefineJob, FaceRefinerBridge, FaceSegment, UNDETECTED_FRAMES_CHOICES, face_count_parameter, face_reference_images_parameter, face_strength_parameter
from postprocessing.spatial_upsamplers import PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_MEDIA_FLOW, PARAMETER_UI_POSTPROCESSING, POSTPROCESSING_CATEGORY_REFINER, UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_TYPE_POSTPROCESSING

from .runtime import DEFAULT_MODEL_TYPE, DEFAULT_WINDOW_FRAMES, DEFAULT_WINDOW_OVERLAP, MAX_WINDOW_FRAMES, RUNTIME_NAME, TEMPORAL_STRIDE, TEXT_ENCODER_CONFIG


REFERENCE_BINDING = """subject_definitions:
<Subject 1> is the person in <Picture 1>, preserving their exact identity, facial features, skin tone, hairstyle, and expression."""
DEFAULT_PROMPT = """A stabilized close-up crop of the single tracked person already visible in the control video. Preserve that exact person's identity, facial proportions, expression, gaze, head pose, hair, clothing, motion, framing, lighting, and background. Restore natural, sharp, temporally stable eyes, mouth, teeth, skin, and fine facial detail. Do not introduce any person, duplicate head, object, or scene element."""
INJECT_REFERENCE_IMAGE = True
METHOD = "h3facerefine"


class H3Engine:
    """The H3 Ref2VA model of the running generation when it can refine, or the private pruned model, with the Turbo LoRA."""

    def __init__(self, bridge: "H3FaceRefinerBridge", job: FaceRefineJob):
        from .runtime import RUNTIME, load_model

        self.job = job
        context = job.loaded_model_context
        self.hybrid_h3 = (context is not None and context.model_family == "minimax_h3" and (not context.model.reference_mode or context.model.transformer.pdd_num_steps is not None))
        if context is None or self.hybrid_h3:
            load_model(context if self.hybrid_h3 else None)
            self.pipeline = RUNTIME.model
            self.model_type = DEFAULT_MODEL_TYPE
        else:
            self.pipeline = context.model
            self.model_type = context.base_model_type
            if context.model_family == "minimax_h3":
                print(f"[FaceRefine] Reusing loaded H3 Ref2VA model '{context.model_type}' and its MMGP offload object")
        self.reference_mode = bool(getattr(self.pipeline, "reference_mode", False))
        if self.reference_mode and not INJECT_REFERENCE_IMAGE:
            print("[FaceRefine] Reference image injection disabled; conditioning Ref2VA from the tracked control video and its audio track")
        # Turbo LoRA is opt-in, set in the refiner settings
        self.use_turbo = bool(job.config["use_turbo_lora"])
        self.turbo_multiplier = float(job.config["turbo_lora_strength"])
        self.turbo_path = bridge._ensure_turbo(job.wgp) if self.use_turbo and (context is None or context.model_family == "minimax_h3") else None
        self.prompt = job.prompt or DEFAULT_PROMPT
        print(f"[FaceRefine] Prompt: {self.prompt}")

    def prepare(self, tracked_faces):
        from postprocessing.face_refiner.face import best_reference_crop

        job, track_jobs = self.job, []
        for track_index, (transform, strengths, matched_reference) in enumerate(tracked_faces):
            track_references = [matched_reference] if INJECT_REFERENCE_IMAGE and matched_reference is not None else None
            if INJECT_REFERENCE_IMAGE and track_references is None and self.reference_mode:
                reference_frame, reference_crop = best_reference_crop(job.frames, transform, job.insightface_model_dir)
                track_references = [Image.fromarray(reference_crop.permute(1, 2, 0).numpy())]
                print(f"[FaceRefine] Face track {track_index + 1}: using stabilized single-face crop from pose-aware source frame {reference_frame + 1} as reference")
            track_prompt = self.prompt if not track_references or "<Picture 1>" in self.prompt else REFERENCE_BINDING + "\n" + self.prompt
            track_jobs.append((transform, strengths, (track_references, track_prompt)))
        if hasattr(self.pipeline, "prewarm_refinement_prompt"):
            if callable(job.progress_callback):
                job.progress_callback("Encoding Prompt")
            print(f"[FaceRefine] Pre-encoding prompt/reference conditioning for {len(track_jobs)} face track(s) before denoising")
            for transform, _strengths, (track_references, track_prompt) in track_jobs:
                canvas_width, canvas_height = transform["canvas"]
                self.pipeline.prewarm_refinement_prompt(track_prompt, track_references if self.reference_mode else None, canvas_width, canvas_height,
                                                        image_refs_relative_size=job.image_refs_relative_size)
        return track_jobs

    def window_count(self, frames: int) -> int:
        from .runtime import window_starts

        return len(window_starts(frames, self.job.config["window_size"], self.job.config["window_overlap"]))

    def refine_segment(self, segment: FaceSegment):
        from .runtime import refine_video

        job, config = self.job, self.job.config
        track_references, track_prompt = segment.data
        start, stop = segment.start, segment.stop
        return refine_video(None, segment.strengths.new_full((stop - start,), job.strength), pipeline=self.pipeline, model_type=self.model_type, turbo_path=self.turbo_path,
                            turbo_multiplier=self.turbo_multiplier, prompt=track_prompt,
                            denoising_strength=config["denoising_strength"], sampling_steps=config["sampling_steps"], shift=config["flow_shift"],
                            sample_solver=config["sample_solver"], seed=job.seed + segment.track_index, fps=job.fps, window_size=config["window_size"],
                            window_overlap=config["window_overlap"], frame_offset=job.frame_offset + start, audio_waveform=job.audio_waveform,
                            audio_sample_rate=job.audio_sample_rate, source_audio_path=job.source_audio_path,
                            reference_images=track_references if self.reference_mode else None,
                            image_refs_relative_size=job.image_refs_relative_size, vae_tile_size=job.vae_tile_size,
                            abort_callback=job.abort_callback, progress_callback=segment.progress_callback,
                            frame_count=stop - start, video_window_loader=segment.load_crops,
                            window_index_offset=segment.window_index_offset, window_count_total=segment.window_count_total)

    def close(self):
        from .runtime import RUNTIME

        context = self.job.loaded_model_context
        if context is None or self.hybrid_h3:
            RUNTIME.unload_all()
        else:
            context.offloadobj.unload_all()


class H3FaceRefinerBridge(FaceRefinerBridge):
    METHOD = METHOD
    NAME = RUNTIME_NAME
    CONFIGURABLE_FIELDS = ("model_mode", "use_turbo_lora", "turbo_lora_strength", "sampling_steps", "denoising_strength", "flow_shift", "undetected_frames", "window_size", "window_overlap")
    ENGINE_CONFIG = {
        "model_mode": "current",
        "denoising_strength": 0.45,
        "sampling_steps": 4,
        "flow_shift": 12.0,
        "sample_solver": "er_sde",
        "turbo_lora_strength": 1.0,
        "use_turbo_lora": False,
        "strength": 0.75,
        "window_size": DEFAULT_WINDOW_FRAMES,
        "window_overlap": DEFAULT_WINDOW_OVERLAP,
    }

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        values = super().normalize_config_section(config)
        if values["model_mode"] not in ("h3", "current"):
            values["model_mode"] = "current"
        values["use_turbo_lora"] = bool(values["use_turbo_lora"])
        for key, cast, low, high in (("sampling_steps", int, 1, 50), ("denoising_strength", float, 0.0, 1.0), ("flow_shift", float, 0.1, 50.0)):
            try:
                values[key] = max(low, min(high, cast(values[key])))
            except (TypeError, ValueError):
                values[key] = cls.ENGINE_CONFIG[key]
        try:
            values["turbo_lora_strength"] = max(0.0, min(1.0, float(values["turbo_lora_strength"])))
        except (TypeError, ValueError):
            values["turbo_lora_strength"] = 1.0
        try:
            values["window_size"] = int(values["window_size"])
        except (TypeError, ValueError):
            values["window_size"] = DEFAULT_WINDOW_FRAMES
        values["window_size"] = max(5, min(MAX_WINDOW_FRAMES, values["window_size"]))
        values["window_size"] = ((values["window_size"] - 5) // TEMPORAL_STRIDE) * TEMPORAL_STRIDE + 5
        try:
            values["window_overlap"] = int(values["window_overlap"])
        except (TypeError, ValueError):
            values["window_overlap"] = DEFAULT_WINDOW_OVERLAP
        values["window_overlap"] = max(1, min(values["window_overlap"], values["window_size"] - 1))
        return values

    @classmethod
    def query_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": RUNTIME_NAME,
            "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": ("video",),
            "profile": UPSAMPLER_PROFILE_VIDEO,
            "config_key": "h3_face_refiner",
            "pos": 10_001,
            "methods": [("H3 Face Refiner", METHOD)],
            "vae_methods": [],
            "default_spatial_upsampling": METHOD,
            "postprocessing_category": POSTPROCESSING_CATEGORY_REFINER,
            "progress_label": "Face Refiner",
            "default_prompt": DEFAULT_PROMPT,
            "source_audio_conditioning": True,
            "description": "Restore blurred or damaged faces while keeping the surrounding video and resolution. Tracks up to five faces; more faces take longer. Choose it to keep mouths close to the source, or for small or distant faces; TaoMate-FVSR Face Refiner gives more detail on close-ups and corrects mouths harder.",
            "media_descriptions": {"video": "**Pros**\n\n- Mouths stay close to the source.\n- More natural than TaoMate-FVSR on small or distant faces.\n- Accepts a prompt (Late Post Processing).\n- Reuses H3 when it is already loaded.\n- Faster than TaoMate-FVSR with several faces, and uses a little less VRAM.\n\n**Cons**\n\n- Less skin, eye and teeth detail than TaoMate-FVSR on close-ups.\n- Expressions and the upper face can change.\n- Does little to correct a mouth that looks wrong.\n- Slower than TaoMate-FVSR on a single close-up."},
            "method_parameters": {METHOD: [
                face_count_parameter(),
                face_strength_parameter(0.75, "Face Refinement Strength - lower preserves more of the original face", "Controls how strongly H3 replaces source facial detail. Lower values preserve more of the original face."),
                {"name": "prompt", "setting": "prompt", "type": "string", "component": "textbox", "ui": (PARAMETER_UI_LATE_POSTPROCESSING,), "required": False, "default": "", "label": "Refiner Prompt", "description": "Optional description of the source clip and desired facial restoration. The source metadata prompt or a neutral restoration prompt is used when omitted.", "lines": 1},
                face_reference_images_parameter(),
            ]},
        }

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        defaults = self.default_config()
        interactive = not lock_config

        def reset_button(control, default):
            button = gr.Button("↺", size="sm", scale=0, min_width=44, interactive=interactive)
            button.click(lambda: default, inputs=None, outputs=control)
            return button

        with gr.Group():
            with gr.Row():
                model_mode = gr.Dropdown(choices=[("Force/reuse H3", "h3"), ("Use current compatible model, fall back to H3", "current")], value=config["model_mode"], label="Face Refiner Model", interactive=interactive)
                reset_button(model_mode, defaults["model_mode"])
            with gr.Row():
                with gr.Row():
                    use_turbo_lora = gr.Checkbox(value=config["use_turbo_lora"], label="Use Turbo LoRA", info="Uncheck if your checkpoint already has 8-step acceleration baked in. Only applies to H3 models.", interactive=interactive)
                    reset_button(use_turbo_lora, defaults["use_turbo_lora"])
                with gr.Row():
                    turbo_lora_strength = gr.Slider(0.0, 1.0, value=config["turbo_lora_strength"], step=0.05, label="Turbo LoRA Strength", info="Multiplier of the H3 Turbo acceleration LoRA. 0.0 disables its effect.", interactive=interactive)
                    reset_button(turbo_lora_strength, defaults["turbo_lora_strength"])
            with gr.Row():
                with gr.Row():
                    sampling_steps = gr.Slider(1, 30, value=config["sampling_steps"], step=1, label="Sampling Steps", info="4 is meant for Turbo LoRA or pre-distilled models; use more without acceleration.", interactive=interactive)
                    reset_button(sampling_steps, defaults["sampling_steps"])
                with gr.Row():
                    denoising_strength = gr.Slider(0.0, 1.0, value=config["denoising_strength"], step=0.05, label="Denoising Strength", info="Higher changes the faces more. 0.0 disables the refiner.", interactive=interactive)
                    reset_button(denoising_strength, defaults["denoising_strength"])
                with gr.Row():
                    flow_shift = gr.Slider(1.0, 20.0, value=config["flow_shift"], step=0.5, label="Flow Shift", info="Sampler noise schedule shift.", interactive=interactive)
                    reset_button(flow_shift, defaults["flow_shift"])
            with gr.Row():
                with gr.Row():
                    undetected_frames = gr.Dropdown(choices=UNDETECTED_FRAMES_CHOICES, value=config["undetected_frames"], label="Undetected Frames", interactive=interactive)
                    reset_button(undetected_frames, defaults["undetected_frames"])
                with gr.Row():
                    window_size = gr.Slider(5, MAX_WINDOW_FRAMES, value=config["window_size"], step=TEMPORAL_STRIDE, label="H3 Refiner Sliding Window Size", info="Frames processed per H3 refinement window. Larger windows use more VRAM.", interactive=interactive)
                    reset_button(window_size, defaults["window_size"])
                with gr.Row():
                    window_overlap = gr.Slider(1, config["window_size"] - 1, value=config["window_overlap"], step=1, label="H3 Refiner Sliding Window Overlap", info="Frames shared and crossfaded between consecutive windows.", interactive=interactive)
                    reset_button(window_overlap, defaults["window_overlap"])

        def update_overlap_limit(new_window_size, current_overlap):
            maximum = int(new_window_size) - 1
            return gr.update(maximum=maximum, value=min(int(current_overlap), maximum))

        window_size.change(update_overlap_limit, inputs=[window_size, window_overlap], outputs=window_overlap)
        controls = locals().copy()
        return [(name, controls[name]) for name in self.CONFIGURABLE_FIELDS]

    def refines(self, config: dict[str, Any], strength: float) -> bool:
        return config["denoising_strength"] != 0.0 and super().refines(config, strength)

    def supports_loaded_model(self, spatial_upsampling, context, *, reference_images=None, **_kwargs) -> bool:
        if self.validate_upsampling(spatial_upsampling, 0) or getattr(context.model, "refinement_api", None) != "masked_video_sigma_v1":
            return False
        if self.config()["model_mode"] == "current":
            return True
        return context.model_family == "minimax_h3"

    def query_checkpoint_inventory(self):
        from models.minimax_h3.minimax_h3_handler import REF_TURBO_LORA_KEY
        return [{"model_types": [DEFAULT_MODEL_TYPE], "download_defs": self.query_download_defs(), "model_lora_keys": {DEFAULT_MODEL_TYPE: [REF_TURBO_LORA_KEY]}}]

    def query_download_defs(self) -> list[dict[str, Any]]:
        from models.minimax_h3.minimax_h3_handler import REF2VA_PRUNED_ARCHITECTURE, family_handler

        model_def = family_handler.query_model_def(REF2VA_PRUNED_ARCHITECTURE, {})
        return family_handler.query_model_files(lambda filename: [os.path.basename(filename)], REF2VA_PRUNED_ARCHITECTURE, model_def) + [ASSET_DOWNLOAD_DEF]

    @staticmethod
    def _private_model_def(wgp):
        model_def = wgp.get_model_def(DEFAULT_MODEL_TYPE).copy()
        model_def.update(model_def["system_configs"][TEXT_ENCODER_CONFIG])
        return model_def

    def _turbo_path(self, wgp, model_type=DEFAULT_MODEL_TYPE):
        from models.minimax_h3.minimax_h3_handler import REF_TURBO_LORA_KEY

        model_def = wgp.get_model_def(model_type)
        return wgp.get_lora_local_path(wgp.get_lora_dir(model_type), model_def[REF_TURBO_LORA_KEY])

    def _ensure_turbo(self, wgp):
        from models.minimax_h3.minimax_h3_handler import REF_TURBO_LORA_KEY

        model_def = self._private_model_def(wgp)
        path = self._turbo_path(wgp)
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            wgp.download_file(model_def[REF_TURBO_LORA_KEY], path)
        return path

    def _download_extras(self, wgp):
        model_def = self._private_model_def(wgp)
        transformer = wgp.get_model_filename(DEFAULT_MODEL_TYPE, wgp.transformer_quantization, wgp.transformer_dtype_policy, model_def=model_def)
        wgp.download_models(transformer, DEFAULT_MODEL_TYPE, 0, -1, model_def=model_def)
        text_encoder_urls = wgp.get_model_recursive_prop(DEFAULT_MODEL_TYPE, "text_encoder_URLs", return_list=True, model_def=model_def)
        text_encoder = wgp.get_model_filename(DEFAULT_MODEL_TYPE, wgp.text_encoder_quantization, wgp.transformer_dtype_policy, URLs=text_encoder_urls)
        wgp.download_models(text_encoder, DEFAULT_MODEL_TYPE, 2, -1, force_path=model_def["text_encoder_folder"], model_def=model_def)

    def open_engine(self, job: FaceRefineJob):
        return H3Engine(self, job)

    def release_private_runtime(self) -> None:
        from .runtime import release_model

        release_model()


__all__ = ["H3FaceRefinerBridge"]
