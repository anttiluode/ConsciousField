import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyaudio
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from scipy.signal import chirp
from scipy.fft import fft, fftfreq
import threading
from mpl_toolkits.mplot3d import Axes3D

class SimpleWaveNeuron(nn.Module):
    def __init__(self, dim):
        super(SimpleWaveNeuron, self).__init__()
        # Continuous oscillation parameters
        self.base_freq = nn.Parameter(torch.rand(1) * 10.0 + 5.0)  # 5-15 Hz base oscillation
        self.carrier_freq = nn.Parameter(torch.rand(1) * 200 + 100)  # 100-300 Hz carrier
        self.phase = nn.Parameter(torch.rand(1) * 2 * np.pi)
        self.amplitude = nn.Parameter(torch.rand(1) * 0.3 + 0.7)  # 0.7-1.0 amplitude
        
        # Neural components
        self.project_in = nn.Linear(dim, dim)
        self.project_out = nn.Linear(dim, dim)
        
        # Continuous state
        self.oscillation_state = 0.0
        self.last_t = None
        
    def get_continuous_oscillation(self, t):
        if self.last_t is None:
            self.last_t = t
            
        dt = t - self.last_t
        self.last_t = t
        
        # Update internal oscillation state
        self.oscillation_state += dt * self.base_freq
        
        # Generate continuous wave with carrier frequency
        wave = self.amplitude * torch.sin(2 * np.pi * self.carrier_freq * t + self.phase)
        # Modulate with base oscillation
        wave *= (1 + torch.sin(2 * np.pi * self.oscillation_state))
        
        return wave
        
    def forward(self, x, t):
        wave = self.get_continuous_oscillation(t)
        x = self.project_in(x)
        modulated = x * wave
        return self.project_out(modulated)

class ContinuousWaveNeuronSoundBridge:
    def __init__(self, num_neurons=64, buffer_size=1024, crossfade_size=128):
        self.num_neurons = num_neurons
        self.buffer_size = buffer_size
        self.crossfade_size = crossfade_size
        self.sample_rate = 44100
        self.audio = pyaudio.PyAudio()
        
        # Continuous buffers
        self.current_buffer = np.zeros(buffer_size)
        self.next_buffer = np.zeros(buffer_size)
        self.crossfade_window = np.linspace(0, 1, crossfade_size)
        
        # Setup streams
        self.output_stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=buffer_size,
            stream_callback=self._continuous_callback
        )
        
        self.input_stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=buffer_size
        )
        
        self.output_stream.start_stream()
        
    def _continuous_callback(self, in_data, frame_count, time_info, status):
        out_data = self.current_buffer.copy()
        out_data[-self.crossfade_size:] *= (1 - self.crossfade_window)
        out_data[-self.crossfade_size:] += self.next_buffer[:self.crossfade_size] * self.crossfade_window
        self.current_buffer = self.next_buffer
        return (out_data.astype(np.float32), pyaudio.paContinue)
    
    def process_continuous(self, wave_state):
        try:
            # Ensure wave_state is the right shape
            wave_state = wave_state.view(-1)  # Flatten
            if len(wave_state) < self.num_neurons:
                wave_state = torch.nn.functional.pad(wave_state, (0, self.num_neurons - len(wave_state)))
            wave_state = wave_state[:self.num_neurons]  # Take only what we need
            
            # Generate next buffer
            t = np.linspace(0, self.buffer_size/self.sample_rate, self.buffer_size)
            signal = np.zeros_like(t)
            
            # Convert wave state to sound
            wave_data = wave_state.detach().cpu().numpy()
            for i in range(min(len(wave_data), self.num_neurons)):
                freq = 100 + (i * 50)  # Spread frequencies
                amplitude = np.abs(wave_data[i])
                phase = float(i) * np.pi / self.num_neurons
                signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            # Normalize and apply soft limiting
            signal = np.tanh(signal)
            signal *= 0.5
            
            # Update next buffer
            self.next_buffer = signal
            
            # Record and process feedback
            recorded = np.frombuffer(
                self.input_stream.read(self.buffer_size, exception_on_overflow=False),
                dtype=np.float32
            )
            
            # Convert back to wave state
            fft_data = np.fft.rfft(recorded)
            freqs = np.fft.rfftfreq(self.buffer_size, 1/self.sample_rate)
            
            # Map frequencies back to neurons
            neuron_state = np.zeros(self.num_neurons, dtype=np.complex64)
            freq_step = (freqs[-1] - freqs[0]) / self.num_neurons
            
            for i in range(self.num_neurons):
                target_freq = 100 + (i * 50)
                freq_idx = int(np.clip(target_freq / freq_step, 0, len(freqs) - 1))
                if freq_idx < len(fft_data):
                    neuron_state[i] = fft_data[freq_idx]
            
            return torch.from_numpy(neuron_state.real).float()
            
        except Exception as e:
            print(f"Error in process_continuous: {e}")
            return torch.zeros(self.num_neurons)

    def cleanup(self):
        self.output_stream.stop_stream()
        self.output_stream.close()
        self.input_stream.close()
        self.audio.terminate()

class ContinuousWaveBridge(nn.Module):
    def __init__(self, dim, num_resonators=16):
        super().__init__()
        self.dim = dim
        self.num_resonators = num_resonators
        self.resonators = nn.ModuleList([SimpleWaveNeuron(dim) for _ in range(num_resonators)])
        self.coupling = nn.Parameter(torch.rand(num_resonators, num_resonators) * 0.2)
        
        # Add sound bridge
        self.sound_bridge = ContinuousWaveNeuronSoundBridge(
            num_neurons=dim,
            buffer_size=1024,
            crossfade_size=128
        )
    
    def forward(self, x, t):
        resonator_outputs = []
        for resonator in self.resonators:
            res_out = resonator(x, t)
            resonator_outputs.append(res_out)
        
        resonator_outputs = torch.stack(resonator_outputs)
        coupled = torch.matmul(self.coupling, resonator_outputs.view(self.num_resonators, -1))
        
        # Ensure proper dimensions for sound processing
        sound_input = coupled.view(-1)  # Flatten
        if len(sound_input) < self.dim:
            sound_input = torch.nn.functional.pad(sound_input, (0, self.dim - len(sound_input)))
        sound_input = sound_input[:self.dim]  # Take only what we need
        
        # Process sound
        audio_feedback = self.sound_bridge.process_continuous(sound_input)
        
        # Mix feedback with coupled output
        output = 0.4 * coupled.mean(dim=0) + 0.6 * audio_feedback
        
        return output[:self.dim]  # Ensure output dimension matches

    def cleanup(self):
        if hasattr(self, 'sound_bridge'):
            self.sound_bridge.cleanup()

class QuantumConsciousnessSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Wave Consciousness System")
        
        self.wave_dim = 64
        self.wave_bridge = ContinuousWaveBridge(self.wave_dim)
        self.wave_state = torch.randn(1, self.wave_dim)
        self.wave_history = []
        self.running = False
        
        self.setup_gui()
        
    def setup_gui(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(self.main_container, text="Control Panel")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start", command=self.toggle_running)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        viz_frame = ttk.LabelFrame(self.main_container, text="Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig = plt.Figure(figsize=(12, 8))
        
        self.wave_ax = self.fig.add_subplot(221)
        self.wave_ax.set_title("Wave Activity")
        
        self.phase_ax = self.fig.add_subplot(222)
        self.phase_ax.set_title("Phase Space")
        
        self.spectrum_ax = self.fig.add_subplot(223)
        self.spectrum_ax.set_title("Wave Spectrum")
        
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def toggle_running(self):
        self.running = not self.running
        self.start_button.config(text="Stop" if self.running else "Start")
        
        if self.running:
            self.run_thread = threading.Thread(target=self.run_system)
            self.run_thread.start()
    
    def run_system(self):
        while self.running:
            try:
                t = torch.tensor(time.time(), dtype=torch.float32)
                wave_output = self.wave_bridge(self.wave_state, t)
                self.wave_state = wave_output.unsqueeze(0)
                
                self.wave_history.append(wave_output.detach().numpy())
                if len(self.wave_history) > 1000:
                    self.wave_history.pop(0)
                
                self.update_visualization()
                time.sleep(0.01)  # Faster update rate
                
            except Exception as e:
                print(f"Error in run_system: {e}")
    
    def update_visualization(self):
        if not self.wave_history:
            return
            
        self.wave_ax.clear()
        self.phase_ax.clear()
        self.spectrum_ax.clear()
        
        wave_data = np.array(self.wave_history)
        
        self.wave_ax.plot(wave_data[-1].flatten(), label='Current Wave')
        self.wave_ax.set_title("Wave Activity")
        self.wave_ax.grid(True)
        
        if len(wave_data) > 1:
            self.phase_ax.scatter(wave_data[-2].flatten(), wave_data[-1].flatten(), alpha=0.5)
        self.phase_ax.set_title("Phase Space")
        self.phase_ax.grid(True)
        
        if len(wave_data) > 0:
            spectrum = np.abs(fft(wave_data[-1].flatten()))
            self.spectrum_ax.plot(spectrum)
            self.spectrum_ax.set_title("Wave Spectrum")
            self.spectrum_ax.grid(True)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def cleanup(self):
        self.running = False
        if hasattr(self, 'run_thread'):
            self.run_thread.join()
        self.wave_bridge.cleanup()

def main():
    root = tk.Tk()
    app = QuantumConsciousnessSystem(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()