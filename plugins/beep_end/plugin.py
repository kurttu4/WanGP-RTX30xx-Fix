# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import threading
from typing import Dict, Any

# Safe Import Gradio
try:
    import gradio as gr
except ImportError:
    gr = None

# Secure import of Wan2GP plugin base class
try:
    from shared.utils.plugins import WAN2GPPlugin
except ImportError:
    class WAN2GPPlugin:
        def __init__(self):
            self.tabs = {}
            self.name = ""
            self.version = "1.0.0"
            self.description = ""
            self._data_hooks = {}
            self._component_requests = []
            self._global_requests = []
        def register_data_hook(self, name, cb): pass
        def request_component(self, name): pass
        def request_global(self, name): pass
        def add_tab(self, *a, **k): pass
        def post_ui_setup(self, components): pass

# Safe import winsound for Windows
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(PLUGIN_DIR, "settings.json")

DEFAULT_SETTINGS = {
    "enabled": True,
    "sound_type": "chord",         # chord, double_beep, single_beep, asterisk, exclamation, custom_wav
    "beep_freq": 1000,             # Frequency in Hz (37..32767)
    "beep_duration": 300,          # Duration in ms
    "repeat_count": 1,             # Number of repetitions
    "repeat_delay": 150            # Pause between repeats in ms
}


def load_settings() -> dict:
    settings = DEFAULT_SETTINGS.copy()
    if os.path.isfile(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                settings.update(saved)
        except Exception as e:
            print(f"[BeepEndPlugin] Read error settings.json: {e}")
    return settings


def save_settings(settings: dict) -> bool:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[BeepEndPlugin] Write error settings.json: {e}")
        return False


class BeepEndPlugin(WAN2GPPlugin):
    """
    Sound notification plugin for Windows when video generation is completed.
Uses native winsound (no external dependencies) and plays a sound
in the background stream, without blocking the interface.
    """
    def __init__(self):
        super().__init__()
        self.name = "Beep on End (Windows)"
        self.version = "1.0.0"
        self.description = "Plays a beep in Windows when the generation is complete."
        self.settings = load_settings()
        self.last_beep_time = 0.0
        self._lock = threading.Lock()

    def setup_ui(self):
        # 1. Hook interception before saving video metadata (works immediately when video file is ready)
        self.register_data_hook("before_metadata_save", self.on_before_metadata_save)

        # 2. Request output_trigger to track completion via Gradio interface
        self.request_component("output_trigger")

        # 3. Plugin Settings Tab
        self.add_tab(
            tab_id="beep_end_tab",
            label="🔔beep end",
            component_constructor=self.create_settings_ui,
            position=-1
        )

    def post_ui_setup(self, components: Dict[str, Any]):
        output_trigger = components.get("output_trigger")
        if output_trigger is not None:
            try:
                output_trigger.change(
                    fn=self.on_ui_output_trigger,
                    inputs=[],
                    outputs=[]
                )
            except Exception as e:
                print(f"[BeepEndPlugin] Failed to bind output_trigger.change: {e}")

    def on_before_metadata_save(self, *args, **kwargs):
        """Triggers when the generation of a single video file is completed."""
        self.trigger_sound("metadata_save")
        # Be sure to return the original configs back to the Wan2GP system
        return kwargs.get("configs", args[0] if args else None)

    def on_ui_output_trigger(self):
        """Triggers when output is updated in the Gradio UI."""
        self.trigger_sound("ui_output_trigger")

    def trigger_sound(self, source: str = "unknown"):
        """Flow-safe signal call with 2.5 sec debouncing (excludes double trigger)."""
        if not self.settings.get("enabled", True):
            return

        with self._lock:
            now = time.time()
            if now - self.last_beep_time < 2.5:
                return
            self.last_beep_time = now

        threading.Thread(
            target=self._play_sound_worker,
            args=(self.settings.copy(),),
            daemon=True
        ).start()

    def _play_sound_worker(self, current_settings: dict):
        """Background Sound Player for Windows."""
        if not HAS_WINSOUND:
            # Fallback for console if winsound is unavailable
            sys.stdout.write("\a")
            sys.stdout.flush()
            return

        sound_type = current_settings.get("sound_type", "chord")
        freq = int(current_settings.get("beep_freq", 1000))
        duration = int(current_settings.get("beep_duration", 300))
        repeats = max(1, min(5, int(current_settings.get("repeat_count", 1))))
        delay_sec = max(0.05, float(current_settings.get("repeat_delay", 150)) / 1000.0)

        custom_wav_path = os.path.join(PLUGIN_DIR, "custom.wav")

        for i in range(repeats):
            try:
                if sound_type == "chord":
                    # Beautiful major chord: C5 (523) -> E5 (659) -> G5 (784) -> C6 (1046)
                    notes = [(523, 110), (659, 110), (784, 110), (1046, 260)]
                    for note_f, note_d in notes:
                        winsound.Beep(note_f, note_d)
                        time.sleep(0.03)

                elif sound_type == "double_beep":
                    # Nice double signal
                    winsound.Beep(880, 160)
                    time.sleep(0.08)
                    winsound.Beep(1175, 240)

                elif sound_type == "single_beep":
                    # Single customizable beep
                    clamped_freq = max(37, min(32767, freq))
                    clamped_dur = max(50, min(5000, duration))
                    winsound.Beep(clamped_freq, clamped_dur)

                elif sound_type == "asterisk":
                    # Windows Asterisk System Sound (Notification Bell)
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)

                elif sound_type == "exclamation":
                    # Windows Exclamation System Sound
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)

                elif sound_type == "custom_wav":
                    if os.path.isfile(custom_wav_path):
                        winsound.PlaySound(custom_wav_path, winsound.SND_FILENAME)
                    else:
                        # Fallback to system sound if file not found
                        winsound.MessageBeep(winsound.MB_ICONASTERISK)

                else:
                    winsound.MessageBeep(winsound.MB_OK)

            except Exception as e:
                print(f"[BeepEndPlugin] Sound playback error: {e}")

            if i < repeats - 1:
                time.sleep(delay_sec)

    def create_settings_ui(self):
        """Interface of the plugin settings tab in Gradio."""
        if gr is None:
            return

        with gr.Blocks() as demo:
            gr.Markdown(
                "## 🔔 Sound signal for end of generation (Windows)\n"
                "This plugin provides an audio signal via the built-in Windows tools immediately after the video is generated and saved."
            )

            with gr.Row():
                enabled_chk = gr.Checkbox(
                    label="Enable sound signal",
                    value=self.settings.get("enabled", True),
                    scale=1
                )

            sound_choices = [
                    ("The winning chord (C-E-G-C)", "chord"),
                    ("Double beep (880 Hz + 1175 Hz)", "double_beep"),
                    ("Single beep (customizable)", "single_beep"),
                    ("System bell (Windows Asterisk)", "asterisk"),
                    ("System Warning (Windows Exclamation)", "exclamation"),
                    ("File custom.wav (put custom.wav in the plugin folder)", "custom_wav"),
            ]

            with gr.Group():
                sound_type_dropdown = gr.Dropdown(
                    label="Type of sound signal",
                    choices=sound_choices,
                    value=self.settings.get("sound_type", "chord")
                )

                with gr.Row():
                    beep_freq_slider = gr.Slider(
                        label="Beep frequency (Hz) [for single beep]",
                        minimum=200,
                        maximum=4000,
                        step=50,
                        value=self.settings.get("beep_freq", 1000)
                    )
                    beep_dur_slider = gr.Slider(
                        label="Duration (ms) [for single beep]",
                        minimum=50,
                        maximum=1500,
                        step=50,
                        value=self.settings.get("beep_duration", 300)
                    )

                with gr.Row():
                    repeats_slider = gr.Slider(
                        label="Number of repetitions",
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=self.settings.get("repeat_count", 1)
                    )
                    delay_slider = gr.Slider(
                        label="Pause between repeats (ms)",
                        minimum=50,
                        maximum=1000,
                        step=50,
                        value=self.settings.get("repeat_delay", 150)
                    )

            with gr.Row():
                test_btn = gr.Button("🔊Check sound now", variant="secondary")
                save_btn = gr.Button("💾Save Settings", variant="primary")

            status_msg = gr.Markdown()

            # UI Event Handlers
            def handle_test(sound_type, freq, dur, repeats, delay):
                test_cfg = {
                    "sound_type": sound_type,
                    "beep_freq": freq,
                    "beep_duration": dur,
                    "repeat_count": repeats,
                    "repeat_delay": delay
                }
                threading.Thread(target=self._play_sound_worker, args=(test_cfg,), daemon=True).start()
                return "🔊 Playing test signal..."

            def handle_save(enabled, sound_type, freq, dur, repeats, delay):
                new_settings = {
                    "enabled": bool(enabled),
                    "sound_type": sound_type,
                    "beep_freq": int(freq),
                    "beep_duration": int(dur),
                    "repeat_count": int(repeats),
                    "repeat_delay": int(delay)
                }
                self.settings.update(new_settings)
                if save_settings(self.settings):
                    gr.Info("Plugin settings saved!")
                    return "✅ Configuration successfully saved to`settings.json`"
                else:
                    return "❌ Failed to save settings."

            test_btn.click(
                fn=handle_test,
                inputs=[sound_type_dropdown, beep_freq_slider, beep_dur_slider, repeats_slider, delay_slider],
                outputs=[status_msg]
            )

            save_btn.click(
                fn=handle_save,
                inputs=[enabled_chk, sound_type_dropdown, beep_freq_slider, beep_dur_slider, repeats_slider, delay_slider],
                outputs=[status_msg]
            )

        return demo
