# Beep on End (Windows) — Audio beep plugin for Wan2GP

A plugin for **Wan2GP** that plays a native Windows sound signal (beep, melody chord or system bell) immediately after the video generation is completed.

## Features
- **100% native Windows sound**: uses the standard Python library `winsound`. No additional packages (`pygame`, `sounddevice`, etc.) are required.
- **Does not slow down the system**: the sound is played in a separate asynchronous stream (`daemon thread`).
- **Double tracking mechanism**: intercepts metadata saving (`before_metadata_save`) + output event in Gradio (`output_trigger`) with built-in double-trigger protection (2.5 sec debounce).
- **Various sound presets**:
1. *The winning chord (C-E-G-C)* – a beautiful ascending major treble.
2. *Double beep (880 Hz + 1175 Hz)*.
3. *Single beep* with adjustable frequency (Hz) and duration (ms).
4. *System bell (Windows Asterisk)*.
5. *Windows Exclamation*.
6. *User WAV-file* (`custom.wav` in the plugin folder).
- ** Convenient tab in the interface **: the ability to test the sound with one button and save settings.

---

## Installation in Wan2GP

1. Copy the `beep_end` folder (or `wan2gp-beep-end`) to the Wan2GP plugin folder:
`
\Wan2GP\plugins\beep_end
`
2. Start or restart **Wan2GP**.
3. Go to the **Plugins** tab in the Wan2GP interface.
4. Find the plugin `Beep on End (Windows)` (or `beep_end`) in the list, check it and click **Save Settings**.
5. Restart Wan2GP.
6. In the top menu, you will see the tab **🔔 End Sound**, where you can choose a sound and check it with the button.
