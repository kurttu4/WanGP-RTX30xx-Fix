"""Released PTD RefAsymOPD inference contract for the H3 Ref2VA pipeline."""

PTD_FOLDER = "minimax_h3/ptd_refasymopd"
PTD_HEAD_FILE = "minimax_h3_ref2va_ptd_refasymopd_8step_heads_fp32.safetensors"

PTD_INFOS = """## Reference Images to Video and Audio

Generate a video with synchronized stereo sound from a prompt and one or more reference images. Reference images guide subjects, clothing, objects and scenery; describe their roles using `<Picture 1>`, `<Picture 2>`, and so on.

- **Main Image Reference:** the first image sets the output dimensions; further images add references.
- **Use Reference Images:** keep the selected output dimensions.
- **Eight-Step Generation:** the accelerator is loaded automatically. Keep the fixed eight-step schedule; additional acceleration LoRAs, step skipping and audio refinement are not supported.

This release supports image-reference video generation. Video/audio references, continuation, editing an existing video and still-image output are not exposed. Start with a short scene of about five seconds; fewer steps can reduce detail or motion quality compared with the full model.
"""

PTD_DEEPY_INFOS = """Generate video with stereo audio from `prompt` and `image_refs` (at least one, at most nine). Set `image_mode=0`, `video_prompt_type='KI'` to derive output dimensions from the first reference or `'I'` to keep `resolution`; additional images remain references. Use `num_inference_steps=8`, `sample_solver='euler'`, `flow_shift=12`, `guidance_scale=1`, `guidance_phases=1`, `skip_steps_cache_type=''`, `audio_prompt_type=''`, and a single video window. Eight built-in PTD RefAsymOPD adapters are automatically scheduled one per step. Do not add Turbo/PDD accelerators. This release exposes no video/audio inputs, continuation, video editing or still-image output. Start with `video_length=124` at 24 fps. Prompt enhancement is optional and off by default. Read `prompt_infos` for H3 reference labels and sound/dialogue syntax."""

PTD_PROMPT_INFOS = """## Writing a Reference Prompt

Describe what should happen, which reference supplies each subject or object, and what should remain recognizable. Use `<Picture 1>`, `<Picture 2>`, etc. in upload order. For a short clip, one clear action and one camera movement are a useful starting point.

H3 accepts structured sections such as `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, and `non_diegetic_music`. Reference subjects with `<Subject 1>` and identify shots with `[Shot 1]`. Put spoken words inside `<d>[English] Spoken words.</d>`, with a speaker marker such as `(S1)`; describe background sound separately.

```text
subject_definitions: <Subject 1> is the person in <Picture 1>.
summary: [reference generation] The person walks through a sunlit garden.
retention_analysis: <Subject 1>: preserve identity and clothing.
detailed_description: [Shot 1] The camera follows <Subject 1> past flowering trees. They turn toward the camera, smile and say (S1) <d>[English] What a beautiful morning.</d>
overall_soundscape: Soft footsteps, birds and synchronized speech.
non_diegetic_music: N/A
```
"""

PTD_DEEPY_PROMPT_INFOS = """Write H3 reference-generation prompts using `<Picture 1>`, `<Picture 2>`, etc. in `image_refs` order. Define recurring entities with `<Subject N>` in `subject_definitions`; describe preservation in `retention_analysis`, the scene in `summary`, and action/camera in `detailed_description` with `[Shot N]`. Use `overall_soundscape` for environmental sound and `non_diegetic_music` for music or `N/A`. Dialogue syntax: `(S1) <d>[English] Exact spoken words.</d>`, preserving the requested language and wording. Keep action and speech short enough for `video_length / 24` seconds. Example: `subject_definitions: <Subject 1> is the person in <Picture 1>.\nsummary: [reference generation] A greeting in a garden.\ndetailed_description: [Shot 1] <Subject 1> smiles and says (S1) <d>[English] Good morning.</d>\noverall_soundscape: Birds and clear speech.\nnon_diegetic_music: N/A`. Only image references are supported by this release."""
