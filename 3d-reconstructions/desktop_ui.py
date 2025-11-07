#!/usr/bin/env python3
"""
Desktop GUI (Tkinter) for the Video → 3D COLMAP Pipeline.

Run with:
  python desktop_ui.py

Features:
- Select video file via file dialog.
- Configure all pipeline parameters (parity with run_pipeline.py).
- Launch pipeline in a background thread; live log output in scrollable text.
- Summary panel updates after completion.
- Open outputs folder in Explorer.
- Dry-run mode prints intended commands only.

Requires Python packages already installed for the pipeline. Tkinter is part of standard Python on Windows.
"""
import os
import sys
import threading
import time
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_DIR = Path(__file__).parent
PIPELINE = APP_DIR / 'run_pipeline.py'
DEFAULT_WORKDIR = APP_DIR / 'work'
OUTPUTS_DIR = APP_DIR / 'outputs'

class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Video → 3D Reconstruction (COLMAP Desktop)')
        self.geometry('1000x700')
        self.configure(bg='#1e1e1e')
        # Define presets (must exist before creating widgets)
        self.PRESETS = {
            'None (leave unchanged)': {},
            'Fast Debug': {
                'mode': 'fast',
                'interval': 0.7,
                'max_frames': 40,
                'dense_size': 1500,
                'matcher': 'sequential',
                'single_camera': True,
                'reuse_frames': True,
                'skip_dense': True,
                'run_poisson': False,
                'convert_obj': False,
                'overwrite': True,
                'dry_run': False,
            },
            'Small Object': {
                'mode': 'full',
                'interval': 0.3,
                'max_frames': 80,
                'dense_size': 2000,
                'matcher': 'exhaustive',
                'single_camera': True,
                'reuse_frames': True,
                'skip_dense': False,
                'run_poisson': True,
                'convert_obj': True,
                'overwrite': False,
                'dry_run': False,
            },
            'Indoor Room': {
                'mode': 'custom',
                'interval': 0.5,
                'max_frames': 80,
                'dense_size': 2000,
                'matcher': 'sequential',
                'single_camera': True,
                'reuse_frames': True,
                'skip_dense': False,
                'run_poisson': True,
                'convert_obj': False,
                'overwrite': False,
                'dry_run': False,
            },
            'Terrain / Aerial': {
                'mode': 'fast',
                'interval': 1.0,
                'max_frames': 120,
                'dense_size': 1600,
                'matcher': 'sequential',
                'single_camera': False,
                'reuse_frames': True,
                'skip_dense': False,
                'run_poisson': False,
                'convert_obj': False,
                'overwrite': False,
                'dry_run': False,
            },
        }

        # Now create widgets that reference presets
        self.create_widgets()
        self.proc = None
        self.stop_requested = False

    def create_widgets(self):
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except Exception:
            pass

        container = ttk.Frame(self)
        container.pack(fill='both', expand=True, padx=8, pady=8)

        # Presets frame
        presets = ttk.LabelFrame(container, text='Preset Loader')
        presets.pack(fill='x', padx=4, pady=4)
        self.preset_var = tk.StringVar(value='None (leave unchanged)')
        ttk.Label(presets, text='Preset:').grid(row=0, column=0, sticky='w')
        ttk.Combobox(presets, textvariable=self.preset_var, values=list(self.PRESETS.keys()), width=30, state='readonly').grid(row=0, column=1, sticky='w', padx=4)
        ttk.Button(presets, text='Apply Preset', command=self.apply_preset).grid(row=0, column=2, padx=4)

        # Top config frame
        cfg = ttk.LabelFrame(container, text='Configuration')
        cfg.pack(fill='x', padx=4, pady=4)

        # Video selection
        self.video_var = tk.StringVar()
        ttk.Label(cfg, text='Video file:').grid(row=0, column=0, sticky='w')
        ttk.Entry(cfg, textvariable=self.video_var, width=60).grid(row=0, column=1, sticky='we', padx=4)
        ttk.Button(cfg, text='Browse...', command=self.browse_video).grid(row=0, column=2, padx=4)

        # Mode & interval
        self.mode_var = tk.StringVar(value='custom')
        ttk.Label(cfg, text='Mode:').grid(row=1, column=0, sticky='w')
        ttk.Combobox(cfg, textvariable=self.mode_var, values=['custom','fast','full'], width=10).grid(row=1, column=1, sticky='w', padx=4)
        self.interval_var = tk.DoubleVar(value=0.5)
        ttk.Label(cfg, text='Interval(s):').grid(row=1, column=2, sticky='e')
        ttk.Entry(cfg, textvariable=self.interval_var, width=8).grid(row=1, column=3, sticky='w', padx=4)

        # Max frames / dense size
        self.max_frames_var = tk.IntVar(value=60)
        ttk.Label(cfg, text='Max Frames:').grid(row=2, column=0, sticky='w')
        ttk.Entry(cfg, textvariable=self.max_frames_var, width=8).grid(row=2, column=1, sticky='w', padx=4)
        self.dense_size_var = tk.IntVar(value=2000)
        ttk.Label(cfg, text='Dense Size:').grid(row=2, column=2, sticky='e')
        ttk.Entry(cfg, textvariable=self.dense_size_var, width=8).grid(row=2, column=3, sticky='w', padx=4)

        # Matcher & vocab
        self.matcher_var = tk.StringVar(value='exhaustive')
        ttk.Label(cfg, text='Matcher:').grid(row=3, column=0, sticky='w')
        ttk.Combobox(cfg, textvariable=self.matcher_var, values=['exhaustive','sequential','vocab'], width=12).grid(row=3, column=1, sticky='w', padx=4)
        self.vocab_var = tk.StringVar()
        ttk.Label(cfg, text='Vocab Tree:').grid(row=3, column=2, sticky='e')
        ttk.Entry(cfg, textvariable=self.vocab_var, width=30).grid(row=3, column=3, sticky='w', padx=4)
        ttk.Button(cfg, text='Browse...', command=self.browse_vocab).grid(row=3, column=4, padx=4)

        # Camera model
        self.camera_var = tk.StringVar(value='PINHOLE')
        ttk.Label(cfg, text='Camera Model:').grid(row=4, column=0, sticky='w')
        ttk.Combobox(cfg, textvariable=self.camera_var, values=['PINHOLE','SIMPLE_PINHOLE','RADIAL','SIMPLE_RADIAL','OPENCV','OPENCV_FISHEYE'], width=20).grid(row=4, column=1, sticky='w', padx=4)

        # Path & GPU
        self.colmap_var = tk.StringVar()
        ttk.Label(cfg, text='COLMAP Path:').grid(row=4, column=2, sticky='e')
        ttk.Entry(cfg, textvariable=self.colmap_var, width=30).grid(row=4, column=3, sticky='w', padx=4)
        self.gpu_var = tk.StringVar()
        ttk.Label(cfg, text='GPU Index:').grid(row=4, column=4, sticky='e')
        ttk.Entry(cfg, textvariable=self.gpu_var, width=5).grid(row=4, column=5, sticky='w', padx=4)

        # Work dir
        self.workdir_var = tk.StringVar(value=str(DEFAULT_WORKDIR))
        ttk.Label(cfg, text='Work Dir:').grid(row=5, column=0, sticky='w')
        ttk.Entry(cfg, textvariable=self.workdir_var, width=40).grid(row=5, column=1, columnspan=2, sticky='we', padx=4)
        ttk.Button(cfg, text='Browse...', command=self.browse_workdir).grid(row=5, column=3, padx=4)

        # Flags
        flags_frame = ttk.Frame(cfg)
        flags_frame.grid(row=6, column=0, columnspan=6, sticky='w', pady=4)
        # BooleanVar must be constructed with master keyword in some Python versions.
        self.single_camera_var = tk.BooleanVar(master=self, value=False)
        self.reuse_frames_var = tk.BooleanVar(master=self, value=True)
        self.skip_dense_var = tk.BooleanVar(master=self, value=False)
        self.run_poisson_var = tk.BooleanVar(master=self, value=False)
        self.convert_obj_var = tk.BooleanVar(master=self, value=False)
        self.overwrite_var = tk.BooleanVar(master=self, value=False)
        self.dry_run_var = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(flags_frame, text='Single Camera', variable=self.single_camera_var).pack(side='left', padx=4)
        ttk.Checkbutton(flags_frame, text='Reuse Frames', variable=self.reuse_frames_var).pack(side='left', padx=4)
        ttk.Checkbutton(flags_frame, text='Skip Dense', variable=self.skip_dense_var).pack(side='left', padx=4)
        ttk.Checkbutton(flags_frame, text='Run Poisson', variable=self.run_poisson_var).pack(side='left', padx=4)
        ttk.Checkbutton(flags_frame, text='Convert OBJ', variable=self.convert_obj_var).pack(side='left', padx=4)
        ttk.Checkbutton(flags_frame, text='Overwrite', variable=self.overwrite_var).pack(side='left', padx=4)
        ttk.Checkbutton(flags_frame, text='Dry Run', variable=self.dry_run_var).pack(side='left', padx=4)

        # Action buttons
        actions = ttk.Frame(container)
        actions.pack(fill='x', pady=6)
        self.run_btn = ttk.Button(actions, text='Run Pipeline', command=self.on_run)
        self.run_btn.pack(side='left', padx=4)
        self.stop_btn = ttk.Button(actions, text='Stop', command=self.on_stop, state='disabled')
        self.stop_btn.pack(side='left', padx=4)
        self.open_outputs_btn = ttk.Button(actions, text='Open Outputs Folder', command=self.open_outputs)
        self.open_outputs_btn.pack(side='left', padx=4)

        # Log output
        log_frame = ttk.LabelFrame(container, text='Logs')
        log_frame.pack(fill='both', expand=True, padx=4, pady=4)
        self.log_text = tk.Text(log_frame, height=25, wrap='word', bg='#111', fg='#0f0')
        self.log_text.pack(fill='both', expand=True)
        self.log_text.config(state='disabled')
        self.log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text['yscrollcommand'] = self.log_scroll.set
        self.log_scroll.pack(side='right', fill='y')

        # Summary
        summary_frame = ttk.LabelFrame(container, text='Summary')
        summary_frame.pack(fill='x', padx=4, pady=4)
        self.summary_var = tk.StringVar(value='Idle.')
        ttk.Label(summary_frame, textvariable=self.summary_var, anchor='w').pack(fill='x')

    def apply_preset(self):
        name = self.preset_var.get()
        preset = self.PRESETS.get(name, {})
        if not preset:
            return
        # Set values (use .set for Tk variables)
        if 'mode' in preset: self.mode_var.set(preset['mode'])
        if 'interval' in preset: self.interval_var.set(float(preset['interval']))
        if 'max_frames' in preset: self.max_frames_var.set(int(preset['max_frames']))
        if 'dense_size' in preset: self.dense_size_var.set(int(preset['dense_size']))
        if 'matcher' in preset: self.matcher_var.set(preset['matcher'])
        if 'single_camera' in preset: self.single_camera_var.set(bool(preset['single_camera']))
        if 'reuse_frames' in preset: self.reuse_frames_var.set(bool(preset['reuse_frames']))
        if 'skip_dense' in preset: self.skip_dense_var.set(bool(preset['skip_dense']))
        if 'run_poisson' in preset: self.run_poisson_var.set(bool(preset['run_poisson']))
        if 'convert_obj' in preset: self.convert_obj_var.set(bool(preset['convert_obj']))
        if 'overwrite' in preset: self.overwrite_var.set(bool(preset['overwrite']))
        if 'dry_run' in preset: self.dry_run_var.set(bool(preset['dry_run']))
        # Optional fields you may add later: camera model, GPU index
        self.append_log(f"Applied preset: {name}\n")

    def browse_video(self):
        path = filedialog.askopenfilename(title='Select Video', filetypes=[('Video Files','*.mp4;*.mov;*.avi;*.mkv'),('All Files','*.*')])
        if path:
            self.video_var.set(path)

    def browse_vocab(self):
        path = filedialog.askopenfilename(title='Select Vocabulary Tree', filetypes=[('BIN Files','*.bin'),('All Files','*.*')])
        if path:
            self.vocab_var.set(path)

    def browse_workdir(self):
        path = filedialog.askdirectory(title='Select Work Directory')
        if path:
            self.workdir_var.set(path)

    def open_outputs(self):
        if OUTPUTS_DIR.exists():
            os.startfile(str(OUTPUTS_DIR))
        else:
            messagebox.showinfo('Outputs', 'Outputs directory not found.')

    def append_log(self, line: str):
        self.log_text.config(state='normal')
        self.log_text.insert('end', line)
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def build_command(self):
        video = self.video_var.get().strip()
        if not video:
            raise ValueError('Video file not specified')
        if not Path(video).exists():
            raise ValueError(f'Video file does not exist: {video}')
        cmd = [sys.executable, str(PIPELINE), '--video', video]
        mode = self.mode_var.get()
        if mode != 'custom':
            cmd += ['--mode', mode]
        else:
            cmd += ['--interval', str(self.interval_var.get())]
        cmd += [
            '--max_frames', str(self.max_frames_var.get()),
            '--dense_size', str(self.dense_size_var.get()),
            '--camera_model', self.camera_var.get(),
            '--matcher', self.matcher_var.get(),
        ]
        if self.matcher_var.get() == 'vocab' and self.vocab_var.get():
            cmd += ['--vocab_tree', self.vocab_var.get()]
        # Flags
        if self.single_camera_var.get():
            cmd += ['--single_camera']
        if self.reuse_frames_var.get():
            cmd += ['--reuse_frames']
        if self.skip_dense_var.get():
            cmd += ['--skip_dense']
        if self.run_poisson_var.get():
            cmd += ['--run_poisson']
        if self.convert_obj_var.get():
            cmd += ['--convert_obj']
        if self.overwrite_var.get():
            cmd += ['--overwrite']
        if self.dry_run_var.get():
            cmd += ['--dry_run']
        if self.colmap_var.get():
            cmd += ['--colmap_path', self.colmap_var.get()]
        if self.gpu_var.get():
            cmd += ['--gpu_index', self.gpu_var.get()]
        if self.workdir_var.get():
            cmd += ['--work_dir', self.workdir_var.get()]
        return cmd

    def on_run(self):
        try:
            cmd = self.build_command()
        except Exception as e:
            messagebox.showerror('Error', str(e))
            return
        self.log_text.config(state='normal')
        self.log_text.delete('1.0','end')
        self.log_text.config(state='disabled')
        self.append_log('Running: ' + ' '.join(cmd) + '\n')
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.summary_var.set('Running...')
        self.stop_requested = False
        thread = threading.Thread(target=self.execute, args=(cmd,), daemon=True)
        thread.start()

    def on_stop(self):
        if self.proc and self.proc.poll() is None:
            self.stop_requested = True
            self.proc.terminate()
            self.append_log('\n[STOP REQUESTED]\n')
            self.summary_var.set('Stopped by user.')
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

    def execute(self, cmd):
        start_time = time.time()
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(APP_DIR), env=env)
            for raw in iter(self.proc.stdout.readline, b''):
                if self.stop_requested:
                    break
                try:
                    line = raw.decode('utf-8', errors='ignore')
                except Exception:
                    line = str(raw)
                self.append_log(line)
            ret = self.proc.wait()
        except Exception as e:
            self.append_log(f"ERROR: {e}\n")
            ret = -1
        elapsed = time.time() - start_time
        if self.stop_requested:
            status = 'Stopped'
        elif ret == 0:
            status = f'Success ({elapsed:.1f}s)'
        else:
            status = f'Failed (code {ret}) after {elapsed:.1f}s'
        self.summary_var.set(status)
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        # Show outputs summary
        if OUTPUTS_DIR.exists():
            files = list(OUTPUTS_DIR.glob('*'))
            if files:
                self.append_log('\nOutputs:\n')
                for f in files:
                    if f.is_file():
                        self.append_log(f'  - {f.name}\n')


def main():
    app = PipelineGUI()
    app.mainloop()

if __name__ == '__main__':
    main()
