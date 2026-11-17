# Custom user variables and functions for Wan2GP
# Kept in a separate file to prevent upstream git merges from overwriting them.

CONTINUE_NO_JOIN = [False]  # state of the Continue without joining checkbox (page only, not saved in settings)

def set_continue_no_join(value):
    CONTINUE_NO_JOIN[0] = bool(value)

def deep_memory_purge(wgp_globals):
    """Deeply purge VRAM and RAM when models are released."""
    import gc
    import torch

    wan_model = wgp_globals.get("wan_model")
    if wan_model is not None:
        if hasattr(wan_model, "detach_shared_components"):
            try:
                wan_model.detach_shared_components()
            except Exception:
                pass
        for attr in ("transformer", "text_encoder", "vae", "video_encoder", "video_decoder", "audio_vae", "latent_upscaler"):
            if hasattr(wan_model, attr):
                try:
                    setattr(wan_model, attr, None)
                except Exception:
                    pass

    # Clear offload shared state
    try:
        from mmgp import offload
        if "_cache" in offload.shared_state:
            del offload.shared_state["_cache"]
        offload.shared_state.clear()
    except Exception:
        pass

    # Unload enhancer offload
    enhancer_offloadobj = wgp_globals.get("enhancer_offloadobj")
    if enhancer_offloadobj is not None:
        try:
            enhancer_offloadobj.unload_all()
            enhancer_offloadobj.release()
        except Exception:
            pass

    # Clear offload registry
    try:
        from shared.utils import offload_registry
        offload_registry.release_all()
    except Exception:
        pass

    # Release VRAM from upsamplers
    try:
        from shared.upsamplers import upsampler_api
        for method in ("flashvsr", "flux_pid", "flux2_pid", "coz"):
            h = upsampler_api.find_upsampler_by_method(method)
            if h is not None and hasattr(h, "release_vram"):
                h.release_vram()
    except Exception:
        pass

    # Multi-pass GC & CUDA IPC purge
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            for dev_idx in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(dev_idx):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
