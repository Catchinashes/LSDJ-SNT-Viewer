import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox, ttk
import pyaudio
import numpy as np
import math
import time

# Constants for waveform data and display
MAX_VALUE = 15  # 4-bit maximum (0–F)
FRAME_SIZE = 32  # 32 samples per waveform cycle
TOTAL_FRAMES = 16
SAMPLE_RATE = 44100
WAVEFORMS = [
    "Silent", "Sine", "Triangle", "Ramp Up", "Ramp Down", "Square", "Pulse",
    "Rounded Pulse", "Circular Pulse", "Triangular Pulse", "Ramp Pulse",
    "Sine Cubed", "Flame", "Semicircle"
]

# Constants for tone generation
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BASE_FREQUENCY = 16.35  # Frequency for C0


def get_frequency(note, octave):
    """Calculate the frequency for a given note and octave."""
    note_index = NOTES.index(note)
    # Each semitone is 2^(1/12); each octave doubles the frequency.
    return BASE_FREQUENCY * (2 ** (note_index / 12)) * (2 ** octave)


class LSDJWaveformEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("LSDJ Waveform Editor")

        # Waveform storage: 16 frames x 32 samples per frame
        self.waveform = [0] * (FRAME_SIZE * TOTAL_FRAMES)
        self.current_frame = 0
        self.wave1 = []
        self.wave2 = []

        # Editing settings
        self.wrap_mode = False  # This flag will be updated via wrap_mode_var
        self.draw_mode = "precision"  # "precision" or "free"
        self.edit_enabled = True
        self.selected_column = None

        # Generator panel visibility and draw mode variable
        self.gen_visible = tk.BooleanVar(value=False)
        self.draw_mode_var = tk.StringVar(value="precision")

        # New variable for wrap mode (do not reuse gen_visible)
        self.wrap_mode_var = tk.BooleanVar(value=False)

        # Mirror options for combining waves
        self.mirror_14 = tk.BooleanVar(value=False)
        self.mirror_12 = tk.BooleanVar(value=False)

        # Audio setup (single PyAudio instance)
        self.p = pyaudio.PyAudio()
        self.stream = None

        self.setup_ui()
        self.update_display()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top hexadecimal display
        self.top_container = tk.Canvas(main_frame, width=700, height=40, bg="black")
        self.top_container.pack()

        # Main waveform canvas
        self.canvas = tk.Canvas(main_frame, width=700, height=190, bg="black")
        self.canvas.pack()
        self.canvas.bind("<Configure>", lambda event: self.update_display())
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)

        # Bottom hexadecimal display
        self.bottom_container = tk.Canvas(main_frame, width=700, height=40, bg="black")
        self.bottom_container.pack()

        # Menu bar: File and Draw Mode options
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        menubar.add_cascade(label="File", menu=file_menu)
        draw_menu = tk.Menu(menubar, tearoff=0)
        draw_menu.add_radiobutton(label="Precision Drawing", variable=self.draw_mode_var,
                                  value="precision", command=self.set_precision_mode)
        draw_menu.add_radiobutton(label="Free Drawing", variable=self.draw_mode_var,
                                  value="free", command=self.set_free_mode)
        menubar.add_cascade(label="Draw Mode", menu=draw_menu)
        self.root.config(menu=menubar)

        # Navigation row: Prev, Animate Frames, Next
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(pady=5)
        ttk.Button(nav_frame, text="◀ Prev", command=self.prev_frame).grid(row=0, column=0)
        ttk.Button(nav_frame, text="Animate Frames", command=self.animate_frames).grid(row=0, column=1)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_frame).grid(row=0, column=2)
        self.frame_label = ttk.Label(nav_frame, text="Frame: 0/0F")
        self.frame_label.grid(row=1, column=0, columnspan=3)

        # Extra controls row: Edit Mode and Show Generate Panel
        extra_frame = ttk.Frame(main_frame)
        extra_frame.pack(pady=5)
        self.edit_toggle = tk.BooleanVar(value=True)
        ttk.Checkbutton(extra_frame, text="Edit Mode", variable=self.edit_toggle,
                        command=self.toggle_edit_mode).grid(row=0, column=0, padx=5)
        ttk.Checkbutton(extra_frame, text="Show Generate Panel", variable=self.gen_visible,
                        command=self.toggle_gen_panel).grid(row=0, column=1, padx=5)

        # Tone controls row: Note, Octave, Play Tone, Play Frame
        tone_frame = ttk.Frame(main_frame)
        tone_frame.pack(pady=5)
        ttk.Label(tone_frame, text="Note:").grid(row=0, column=0, padx=5)
        self.note_var = tk.StringVar(value="C")
        self.note_dropdown = ttk.Combobox(tone_frame, textvariable=self.note_var,
                                          values=NOTES, state="readonly", width=5)
        self.note_dropdown.grid(row=0, column=1, padx=5)
        ttk.Label(tone_frame, text="Octave:").grid(row=0, column=2, padx=5)
        self.octave_var = tk.IntVar(value=4)
        self.octave_dropdown = ttk.Combobox(tone_frame, textvariable=self.octave_var,
                                            values=list(range(8)), state="readonly", width=5)
        self.octave_dropdown.grid(row=0, column=3, padx=5)
        ttk.Button(tone_frame, text="Play Tone", command=self.play_tone_ui).grid(row=0, column=4, padx=5)
        # "Play Frame" uses the note/octave to compute a frequency and plays only the current frame.
        ttk.Button(tone_frame, text="Play Frame",
                   command=lambda: self.play_frame(get_frequency(self.note_var.get(), self.octave_var.get()))
                   ).grid(row=0, column=5, padx=5)

        # Waveform Generation Panel (hidden by default)
        self.gen_frame = ttk.LabelFrame(main_frame, text="Waveform Generation")
        if self.gen_visible.get():
            self.gen_frame.pack(pady=5, fill=tk.X)

        # --- Wave 1 Controls ---
        wave1_frame = ttk.Frame(self.gen_frame)
        wave1_frame.grid(row=0, column=0, padx=10, pady=5)
        self.wave1_canvas = tk.Canvas(wave1_frame, width=200, height=50, bg="black")
        self.wave1_canvas.pack()
        control_grid1 = ttk.Frame(wave1_frame)
        control_grid1.pack(pady=5)
        ttk.Label(control_grid1, text="Type:").grid(row=0, column=0)
        self.wave1_type = ttk.Combobox(control_grid1, values=WAVEFORMS, state="readonly", width=12)
        self.wave1_type.current(0)
        self.wave1_type.grid(row=0, column=1)
        ttk.Label(control_grid1, text="Vol (0-F):").grid(row=1, column=0)
        self.wave1_vol_label = ttk.Label(control_grid1, text="F")
        self.wave1_vol_label.grid(row=1, column=2)
        self.wave1_volume = ttk.Scale(control_grid1, from_=0, to=1.0, length=100,
                                      command=lambda e: self.update_slider_display("wave1_vol"))
        self.wave1_volume.set(1.0)
        self.wave1_volume.grid(row=1, column=1)
        ttk.Label(control_grid1, text="Freq (1-8):").grid(row=2, column=0)
        self.wave1_freq_label = ttk.Label(control_grid1, text="1")
        self.wave1_freq_label.grid(row=2, column=2)
        self.wave1_freq = ttk.Scale(control_grid1, from_=1, to=8, length=100,
                                    command=lambda e: self.update_slider_display("wave1_freq"))
        self.wave1_freq.set(1)
        self.wave1_freq.grid(row=2, column=1)
        ttk.Button(wave1_frame, text="Generate", command=self.generate_wave1).pack()
        self.wave1_hex = tk.Text(wave1_frame, width=32, height=2, bg="black", fg="white")
        self.wave1_hex.pack()
        self.wave1_hex.insert(tk.END, "Waiting for generation...")
        self.wave1_hex.config(state=tk.DISABLED)

        # --- Wave 2 Controls ---
        wave2_frame = ttk.Frame(self.gen_frame)
        wave2_frame.grid(row=0, column=1, padx=10, pady=5)
        self.wave2_canvas = tk.Canvas(wave2_frame, width=200, height=50, bg="black")
        self.wave2_canvas.pack()
        control_grid2 = ttk.Frame(wave2_frame)
        control_grid2.pack(pady=5)
        ttk.Label(control_grid2, text="Type:").grid(row=0, column=0)
        self.wave2_type = ttk.Combobox(control_grid2, values=WAVEFORMS, state="readonly", width=12)
        self.wave2_type.current(0)
        self.wave2_type.grid(row=0, column=1)
        ttk.Label(control_grid2, text="Vol (0-F):").grid(row=1, column=0)
        self.wave2_vol_label = ttk.Label(control_grid2, text="F")
        self.wave2_vol_label.grid(row=1, column=2)
        self.wave2_volume = ttk.Scale(control_grid2, from_=0, to=1.0, length=100,
                                      command=lambda e: self.update_slider_display("wave2_vol"))
        self.wave2_volume.set(1.0)
        self.wave2_volume.grid(row=1, column=1)
        ttk.Label(control_grid2, text="Freq (1-8):").grid(row=2, column=0)
        self.wave2_freq_label = ttk.Label(control_grid2, text="1")
        self.wave2_freq_label.grid(row=2, column=2)
        self.wave2_freq = ttk.Scale(control_grid2, from_=1, to=8, length=100,
                                    command=lambda e: self.update_slider_display("wave2_freq"))
        self.wave2_freq.set(1)
        self.wave2_freq.grid(row=2, column=1)
        ttk.Button(wave2_frame, text="Generate", command=self.generate_wave2).pack()
        self.wave2_hex = tk.Text(wave2_frame, width=32, height=2, bg="black", fg="white")
        self.wave2_hex.pack()
        self.wave2_hex.insert(tk.END, "Waiting for generation...")
        self.wave2_hex.config(state=tk.DISABLED)

        # --- Combination Controls (inside Gen Panel) ---
        comb_frame = ttk.Frame(self.gen_frame)
        comb_frame.grid(row=0, column=2, padx=10, pady=5)
        # Note: Use the new wrap_mode_var for the "Wrap Mode" checkbox.
        ttk.Checkbutton(comb_frame, text="Wrap Mode", variable=self.wrap_mode_var,
                        command=self.toggle_wrap_mode).pack(pady=2)
        ttk.Checkbutton(comb_frame, text="1/4 Mirror", variable=self.mirror_14).pack(pady=2)
        ttk.Checkbutton(comb_frame, text="1/2 Mirror", variable=self.mirror_12).pack(pady=2)
        ttk.Button(comb_frame, text="Combine Waves", command=self.combine_waves).pack(pady=2)
        ttk.Button(comb_frame, text="Tween Frames", command=self.tween_frames_prompt).pack(pady=2)

    def toggle_gen_panel(self):
        if self.gen_visible.get():
            self.gen_frame.pack(pady=5, fill=tk.X)
        else:
            self.gen_frame.forget()

    def toggle_wrap_mode(self):
        # Update self.wrap_mode from the dedicated wrap_mode_var.
        self.wrap_mode = self.wrap_mode_var.get()

    def update_slider_display(self, slider_type):
        if slider_type == "wave1_vol":
            value = int(self.wave1_volume.get() * MAX_VALUE)
            self.wave1_vol_label.config(text=f"{value:X}")
        elif slider_type == "wave1_freq":
            value = int(self.wave1_freq.get())
            self.wave1_freq_label.config(text=f"{value}")
        elif slider_type == "wave2_vol":
            value = int(self.wave2_volume.get() * MAX_VALUE)
            self.wave2_vol_label.config(text=f"{value:X}")
        elif slider_type == "wave2_freq":
            value = int(self.wave2_freq.get())
            self.wave2_freq_label.config(text=f"{value}")

    def draw_waveform(self):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x_scale = 22
        label_spacing = 10
        # Draw vertical grid lines
        for i in range(33):
            x = i * x_scale
            self.canvas.create_line(x, 0, x, canvas_height, fill="gray", dash=(2, 2))
        # Draw horizontal grid lines
        for i in range(MAX_VALUE + 1):
            y = 20 + i * label_spacing
            self.canvas.create_line(0, y, canvas_width, y, fill="gray", dash=(2, 2))
        # Draw current frame samples as blue rectangles
        start = self.current_frame * FRAME_SIZE
        for i in range(FRAME_SIZE):
            x = i * x_scale
            value = self.waveform[start + i]
            y = 20 + (MAX_VALUE - value) * label_spacing
            self.canvas.create_rectangle(x, y, x + x_scale, y + label_spacing, fill="blue")

    def display_hex_values(self):
        self.top_container.delete("all")
        self.bottom_container.delete("all")
        start = self.current_frame * FRAME_SIZE
        x_scale = 22
        # Top row: first 16 samples
        for i in range(16):
            x = i * x_scale * 2
            self.top_container.create_text(x + x_scale, 10,
                                           text=f"{self.waveform[start + i]:X}", fill="white")
        # Bottom row: next 16 samples
        for i in range(16, FRAME_SIZE):
            x = (i % 16) * x_scale * 2
            self.bottom_container.create_text(x + x_scale, 10,
                                              text=f"{self.waveform[start + i]:X}", fill="white")

    def play_frame(self, frequency=None):
        """
        Play the current frame as a periodic waveform at a given frequency.
        If frequency is not provided, it defaults to the natural frequency of the 32-sample cycle,
        i.e. SAMPLE_RATE / FRAME_SIZE (~1378 Hz).

        This method resamples the 32-sample frame so that one cycle lasts exactly 1/f seconds,
        then repeats that cycle to fill one second of audio.
        """
        if frequency is None:
            frequency = SAMPLE_RATE / FRAME_SIZE
        # Compute number of output samples per cycle:
        period_samples = int(SAMPLE_RATE / frequency)
        # Extract current frame (32 samples)
        start = self.current_frame * FRAME_SIZE
        frame_data = np.array(self.waveform[start:start + FRAME_SIZE], dtype=np.float32)
        # Interpolate the 32 samples to resample to period_samples samples:
        orig_indices = np.linspace(0, FRAME_SIZE, num=FRAME_SIZE, endpoint=False)
        new_indices = np.linspace(0, FRAME_SIZE, num=period_samples, endpoint=False)
        resampled_cycle = np.interp(new_indices, orig_indices, frame_data)
        # Repeat the cycle enough times to fill one second:
        num_cycles = int(np.ceil(SAMPLE_RATE / period_samples))
        full_wave = np.tile(resampled_cycle, num_cycles)[:SAMPLE_RATE]
        # Normalize from [0, MAX_VALUE] to [-1, 1]
        normalized = (full_wave / MAX_VALUE) * 2 - 1
        audio_data = (normalized * 32767).astype(np.int16).tobytes()
        stream = self.p.open(format=pyaudio.paInt16, channels=1,
                             rate=SAMPLE_RATE, output=True)
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()

    def play_tone_ui(self):
        """Compute frequency from tone controls and play the tone using the entire waveform."""
        note = self.note_var.get()
        octave = self.octave_var.get()
        frequency = get_frequency(note, octave)
        self.play_tone(frequency)

    def play_tone(self, frequency):
        """
        Play a tone generated by modulating the entire waveform with a sine wave
        at the given frequency. This reproduces the behavior from your older project.
        """
        sample_rate = 44100
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        total_samples = len(self.waveform)
        audio_waveform = np.array([
            self.waveform[int(i * total_samples / len(t))]
            for i in range(len(t))
        ])
        audio_waveform = (audio_waveform / MAX_VALUE) * 2 - 1
        audio_waveform = np.sin(2 * np.pi * frequency * t) * audio_waveform
        audio_data = (audio_waveform * 32767).astype(np.int16).tobytes()
        stream = self.p.open(format=pyaudio.paInt16, channels=1,
                             rate=sample_rate, output=True)
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()

    def generate_wave1(self):
        volume = self.wave1_volume.get()
        freq = int(self.wave1_freq.get())
        self.wave1 = self.generate_waveform(self.wave1_type.get(), volume, freq)
        self.draw_mini_waveform(self.wave1_canvas, self.wave1)
        self.update_hex_display(self.wave1_hex, self.wave1)

    def generate_wave2(self):
        volume = self.wave2_volume.get()
        freq = int(self.wave2_freq.get())
        self.wave2 = self.generate_waveform(self.wave2_type.get(), volume, freq)
        self.draw_mini_waveform(self.wave2_canvas, self.wave2)
        self.update_hex_display(self.wave2_hex, self.wave2)

    def generate_waveform(self, wave_type, volume, freq):
        vol_int = int(volume * MAX_VALUE)
        if wave_type == "Silent":
            return [0] * FRAME_SIZE
        samples = []
        for i in range(FRAME_SIZE):
            phase = (i * freq) % FRAME_SIZE
            p = phase / FRAME_SIZE
            if wave_type == "Sine":
                val = 0.5 * (1 + math.sin(2 * math.pi * p - math.pi / 2))
            elif wave_type == "Triangle":
                val = 2 * p if p < 0.5 else 2 * (1 - p)
            elif wave_type == "Ramp Up":
                val = p
            elif wave_type == "Ramp Down":
                val = 1 - p
            elif wave_type in ("Square", "Pulse"):
                val = 1 if phase < (FRAME_SIZE // 2) else 0
            elif wave_type == "Rounded Pulse":
                if phase < (FRAME_SIZE // 2):
                    p_half = p / 0.5
                    val = 0.5 * (1 - math.cos(math.pi * p_half))
                else:
                    val = 0
            elif wave_type == "Circular Pulse":
                if phase < (FRAME_SIZE // 2):
                    p_half = p / 0.5
                    val = math.sqrt(max(0, 1 - (1 - p_half) ** 2))
                else:
                    val = 0
            elif wave_type == "Triangular Pulse":
                if phase < (FRAME_SIZE // 2):
                    if phase < (FRAME_SIZE / 4):
                        val = phase / (FRAME_SIZE / 4)
                    else:
                        val = 1 - ((phase - (FRAME_SIZE / 4)) / (FRAME_SIZE / 4))
                else:
                    val = 0
            elif wave_type == "Ramp Pulse":
                if phase < (FRAME_SIZE // 2):
                    val = phase / (FRAME_SIZE / 2)
                else:
                    val = 0
            elif wave_type == "Sine Cubed":
                s = 0.5 * (1 + math.sin(2 * math.pi * p - math.pi / 2))
                val = s ** 3
            elif wave_type == "Flame":
                if phase < (FRAME_SIZE // 2):
                    p_half = p / 0.5
                    val = math.exp(-((p_half - 0.5) ** 2) / 0.1)
                else:
                    val = 0
            elif wave_type == "Semicircle":
                val = math.sqrt(max(0, 1 - (2 * p - 1) ** 2))
            else:
                val = 0
            scaled_val = int(val * vol_int)
            samples.append(min(vol_int, max(0, scaled_val)))
        return samples

    def draw_mini_waveform(self, canvas, data):
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        x_scale = w / FRAME_SIZE
        y_scale = h / MAX_VALUE
        points = []
        for i, val in enumerate(data):
            x = i * x_scale
            y = h - val * y_scale
            points.extend([x, y])
        canvas.create_line(points, fill="#00FF00", width=1)

    def update_hex_display(self, widget, data):
        widget.config(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        hex_values = [f"{x:X}" for x in data]
        widget.insert(tk.END, " ".join(hex_values))
        widget.config(state=tk.DISABLED)

    def combine_waves(self):
        if not self.wave1 or not self.wave2:
            messagebox.showerror("Error", "Generate both waves first!")
            return
        combined = []
        for w1, w2 in zip(self.wave1, self.wave2):
            # If wrap mode is enabled, wrap the sum modulo (MAX_VALUE+1).
            if self.wrap_mode:
                combined.append((w1 + w2) % (MAX_VALUE + 1))
            else:
                # Otherwise, clamp the value to MAX_VALUE.
                combined.append(min(MAX_VALUE, w1 + w2))
        if self.mirror_14.get():
            quarter = combined[:FRAME_SIZE // 4]
            combined = quarter * 4
        elif self.mirror_12.get():
            half = combined[:FRAME_SIZE // 2]
            combined = half + half
        start = self.current_frame * FRAME_SIZE
        self.waveform[start:start + FRAME_SIZE] = combined
        self.update_display()

    def tween_frames_prompt(self):
        top = tk.Toplevel(self.root)
        top.title("Tween Frames")
        tk.Label(top, text="Start Frame (0-15):").grid(row=0, column=0, padx=5, pady=5)
        start_entry = tk.Entry(top)
        start_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(top, text="End Frame (0-15):").grid(row=1, column=0, padx=5, pady=5)
        end_entry = tk.Entry(top)
        end_entry.grid(row=1, column=1, padx=5, pady=5)

        def on_generate():
            try:
                start = int(start_entry.get())
                end = int(end_entry.get())
                if start < 0 or end >= TOTAL_FRAMES or start >= end:
                    messagebox.showerror("Error", "Invalid frame indices!")
                else:
                    self.tween_frames(start, end)
                    top.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid integer indices.")

        tk.Button(top, text="Generate Tween", command=on_generate).grid(row=2, column=0, columnspan=2, pady=10)
        top.transient(self.root)
        top.grab_set()
        self.root.wait_window(top)

    def tween_frames(self, start, end):
        start_index = start * FRAME_SIZE
        end_index = end * FRAME_SIZE
        start_frame = self.waveform[start_index: start_index + FRAME_SIZE]
        end_frame = self.waveform[end_index: end_index + FRAME_SIZE]
        num_steps = end - start
        for step in range(1, num_steps):
            t_val = step / num_steps
            tweened = []
            for i in range(FRAME_SIZE):
                value = round(start_frame[i] + (end_frame[i] - start_frame[i]) * t_val)
                tweened.append(value)
            frame_index = (start + step) * FRAME_SIZE
            self.waveform[frame_index:frame_index + FRAME_SIZE] = tweened
        self.update_display()

    def animate_frames(self):
        def next_frame_animation(frame):
            if frame < TOTAL_FRAMES:
                self.current_frame = frame
                self.update_display()
                self.root.after(250, next_frame_animation, frame + 1)

        next_frame_animation(0)

    def toggle_edit_mode(self):
        self.edit_enabled = self.edit_toggle.get()

    def on_canvas_click(self, event):
        if self.draw_mode == "precision":
            w = self.canvas.winfo_width()
            self.selected_column = round((event.x / w) * (FRAME_SIZE - 1))
            self.handle_precision_edit(self.selected_column, event.y)
        else:
            self.handle_canvas_edit(event.x, event.y)

    def on_canvas_drag(self, event):
        if self.draw_mode == "precision":
            if self.selected_column is not None:
                self.handle_precision_edit(self.selected_column, event.y)
        else:
            self.handle_canvas_edit(event.x, event.y)

    def handle_precision_edit(self, column, y):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        value = MAX_VALUE - round((y / h) * MAX_VALUE)
        index = self.current_frame * FRAME_SIZE + column
        if 0 <= index < len(self.waveform) and 0 <= value <= MAX_VALUE:
            self.waveform[index] = value
            self.update_display()

    def handle_canvas_edit(self, x, y):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        index_in_frame = round((x / w) * (FRAME_SIZE - 1))
        index = self.current_frame * FRAME_SIZE + index_in_frame
        value = MAX_VALUE - round((y / h) * MAX_VALUE)
        if 0 <= index < len(self.waveform) and 0 <= value <= MAX_VALUE:
            self.waveform[index] = value
            self.update_display()

    def next_frame(self):
        if self.current_frame < TOTAL_FRAMES - 1:
            self.current_frame += 1
            self.update_display()

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_display()

    def set_precision_mode(self):
        self.draw_mode = "precision"
        self.draw_mode_var.set("precision")
        messagebox.showinfo("Draw Mode", "Precision Drawing Mode Enabled")

    def set_free_mode(self):
        self.draw_mode = "free"
        self.draw_mode_var.set("free")
        messagebox.showinfo("Draw Mode", "Free Drawing Mode Enabled")

    def update_display(self):
        self.draw_waveform()
        self.display_hex_values()
        self.frame_label.config(text=f"Frame: {self.current_frame:X}/0F")

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("SNT files", "*.snt")])
        if path:
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                    self.waveform = []
                    for byte in data:
                        self.waveform.append((byte >> 4) & 0x0F)
                        self.waveform.append(byte & 0x0F)
                self.current_frame = 0
                self.update_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file:\n{str(e)}")

    def save_file(self):
        path = filedialog.asksaveasfilename(defaultextension=".snt",
                                            filetypes=[("SNT files", "*.snt")])
        if path:
            try:
                with open(path, 'wb') as f:
                    packed = bytearray()
                    for i in range(0, len(self.waveform), 2):
                        high = self.waveform[i] << 4
                        low = self.waveform[i + 1]
                        packed.append(high | low)
                    f.write(packed)
                messagebox.showinfo("Success", "File saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LSDJWaveformEditor(root)
    root.mainloop()
