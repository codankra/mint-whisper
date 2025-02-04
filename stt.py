import tkinter as tk
from tkinter import scrolledtext
import sounddevice as sd
import faster_whisper
import threading
import numpy as np
import array

# Load Faster-Whisper model
model = faster_whisper.WhisperModel("small", compute_type="int8")

# Audio settings
SAMPLE_RATE = 16000
BUFFER_DURATION = 1  # Seconds per recording chunk
recording = False

# Tkinter GUI
root = tk.Tk()
root.title("Speech to Text")
root.geometry("500x300")

text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10)
text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

def normalize_audio(audio_data):
    """Normalize audio to range [-1, 1] without NumPy."""
    max_val = max(abs(sample) for sample in audio_data) or 1  # Prevent division by zero
    return [sample / max_val for sample in audio_data]


def convert_audio(audio_bytes):
    """Convert raw audio data to a properly formatted NumPy float32 array for Faster-Whisper."""
    
    # Ensure audio_bytes is in raw format
    if isinstance(audio_bytes, np.ndarray):
        audio_bytes = audio_bytes.flatten().astype(np.int16).tobytes()
    elif not isinstance(audio_bytes, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or NumPy array, got {type(audio_bytes)}")

    # Convert to a 16-bit integer array
    audio_array = array.array("h", audio_bytes)  # 'h' = 16-bit signed integers
    
    # Convert to float32 NumPy array normalized to [-1,1]
    return np.array(audio_array, dtype=np.float32) / 32768.0

def transcribe_audio():
    """Record and transcribe speech in real-time."""
    global recording
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while recording:
            audio_bytes, _ = stream.read(int(SAMPLE_RATE * BUFFER_DURATION))  # Reduce BUFFER_DURATION if needed
            audio_data = convert_audio(audio_bytes)
            segments, _ = model.transcribe(audio_data, beam_size=5)
            for segment in segments:
                text_area.insert(tk.END, segment.text + " ")
                text_area.see(tk.END)

def start_recording():
    """Start speech recognition."""
    global recording
    if not recording:
        recording = True
        threading.Thread(target=transcribe_audio, daemon=True).start()
        start_button.config(text="Stop", bg="red")

def stop_recording():
    """Stop speech recognition."""
    global recording
    recording = False
    start_button.config(text="Start", bg="green")

def toggle_recording():
    """Toggle between start and stop."""
    if recording:
        stop_recording()
    else:
        start_recording()

def copy_text():
    """Copy text to clipboard."""
    root.clipboard_clear()
    root.clipboard_append(text_area.get("1.0", tk.END))
    root.update()

def clear_text():
    """Clear the text area."""
    text_area.delete("1.0", tk.END)

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

start_button = tk.Button(button_frame, text="Start", bg="green", fg="white", command=toggle_recording, width=10)
start_button.pack(side=tk.LEFT, padx=5)

copy_button = tk.Button(button_frame, text="Copy", command=copy_text, width=10)
copy_button.pack(side=tk.LEFT, padx=5)

clear_button = tk.Button(button_frame, text="Clear", command=clear_text, width=10)
clear_button.pack(side=tk.LEFT, padx=5)

root.mainloop()
