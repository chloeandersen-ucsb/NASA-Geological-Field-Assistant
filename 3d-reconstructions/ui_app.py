#!/usr/bin/env python3
"""
Streamlit UI for the Video-to-3D Reconstruction Pipeline.

Provides a simple web UI to:
- Select a video file (upload or path input)
- Configure pipeline options (mode, interval, max frames, matcher, etc.)
- Run the pipeline and stream logs in real-time
- Display generated outputs and download links

Launch:
  streamlit run ui_app.py
"""
import sys
import os
import io
import time
import shutil
import subprocess
from pathlib import Path
import streamlit as st

APP_DIR = Path(__file__).parent
PIPELINE = APP_DIR / 'run_pipeline.py'
DEFAULT_WORKDIR = APP_DIR / 'work'
DEFAULT_OUTPUTS = APP_DIR / 'outputs'

st.set_page_config(page_title='Video → 3D (COLMAP) UI', layout='wide')
st.title('Video → 3D Reconstruction (COLMAP)')
st.caption('Simple testing UI to drive the Python pipeline')

with st.sidebar:
    st.header('1) Input video')
    uploaded = st.file_uploader('Upload a video (optional)', type=['mp4','mov','avi','mkv'])
    video_path_text = st.text_input('...or local video path', value='')

    st.header('2) Presets')
    mode = st.selectbox('Mode', options=['custom','fast','full'], index=0)
    interval = st.number_input('Interval (s)', min_value=0.05, max_value=5.0, value=0.5, step=0.05, help='Used when mode=custom or when overriding preset')
    max_frames = st.number_input('Max frames (0=no cap)', min_value=0, max_value=2000, value=60, step=10)
    dense_size = st.number_input('Dense max image size', min_value=256, max_value=6000, value=2000, step=128)

    st.header('3) Matching & Reconstruction')
    matcher = st.selectbox('Matcher', options=['exhaustive','sequential','vocab'])
    vocab_tree = st.text_input('Vocab tree path (for vocab matcher)', value='')
    camera_model = st.selectbox('Camera model', options=['PINHOLE','SIMPLE_PINHOLE','RADIAL','SIMPLE_RADIAL','OPENCV','OPENCV_FISHEYE'], index=0)
    single_camera = st.checkbox('Single camera', value=False)

    st.header('4) Execution')
    colmap_path = st.text_input('COLMAP executable (blank = auto)', value='')
    gpu_index = st.text_input('GPU index (blank = env/0)', value='')
    work_dir = st.text_input('Work dir', value=str(DEFAULT_WORKDIR))
    reuse_frames = st.checkbox('Reuse frames', value=True)
    skip_dense = st.checkbox('Skip dense (export sparse only)', value=False)
    run_poisson = st.checkbox('Run Poisson meshing', value=False)
    convert_obj = st.checkbox('Convert PLY to OBJ', value=False)
    overwrite = st.checkbox('Overwrite work dir', value=False)
    dry_run = st.checkbox('Dry run (no execution)', value=False)

    run_btn = st.button('Run Pipeline', type='primary', use_container_width=True)

placeholder = st.empty()
log_area = st.empty()
outputs_area = st.container()


def save_uploaded_file(uploaded_file, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / uploaded_file.name
    with open(out_path, 'wb') as f:
        f.write(uploaded_file.read())
    return out_path


def build_command(video_path: Path):
    cmd = [sys.executable, str(PIPELINE), '--video', str(video_path)]

    # Mode preset
    if mode != 'custom':
        cmd += ['--mode', mode]
    else:
        cmd += ['--interval', str(interval)]

    cmd += [
        '--max_frames', str(max_frames),
        '--dense_size', str(dense_size),
        '--camera_model', camera_model,
    ]

    # Matcher
    cmd += ['--matcher', matcher]
    if matcher == 'vocab' and vocab_tree:
        cmd += ['--vocab_tree', vocab_tree]

    # Flags
    if single_camera:
        cmd += ['--single_camera']
    if reuse_frames:
        cmd += ['--reuse_frames']
    if skip_dense:
        cmd += ['--skip_dense']
    if run_poisson:
        cmd += ['--run_poisson']
    if convert_obj:
        cmd += ['--convert_obj']
    if overwrite:
        cmd += ['--overwrite']
    if dry_run:
        cmd += ['--dry_run']

    # Paths / env
    if colmap_path:
        cmd += ['--colmap_path', colmap_path]
    if gpu_index:
        cmd += ['--gpu_index', gpu_index]
    if work_dir:
        cmd += ['--work_dir', work_dir]

    return cmd


def run_pipeline(cmd):
    st.write('Running:')
    st.code(' '.join(cmd))
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(APP_DIR))
    log_lines = []
    with proc.stdout:
        for raw in iter(proc.stdout.readline, b''):
            try:
                line = raw.decode('utf-8', errors='ignore')
            except Exception:
                line = str(raw)
            log_lines.append(line)
            if len(log_lines) > 1000:
                log_lines = log_lines[-1000:]
            log_area.code(''.join(log_lines))
    ret = proc.wait()
    return ret


def list_outputs(outputs_dir: Path):
    outputs = []
    if outputs_dir.exists():
        for p in sorted(outputs_dir.glob('*')):
            if p.is_file():
                outputs.append(p)
    return outputs


if run_btn:
    # Prepare video path
    video_on_disk = None
    if uploaded is not None:
        uploads_dir = APP_DIR / 'uploads'
        video_on_disk = save_uploaded_file(uploaded, uploads_dir)
    elif video_path_text:
        path = Path(video_path_text)
        if not path.exists():
            st.error(f'Video not found: {path}')
        else:
            video_on_disk = path
    else:
        st.warning('Provide a video via upload or local path.')

    if video_on_disk:
        cmd = build_command(video_on_disk)
        placeholder.info('Pipeline started...')
        t0 = time.time()
        code = run_pipeline(cmd)
        elapsed = time.time() - t0
        if code == 0:
            placeholder.success(f'Pipeline finished in {elapsed:.1f}s')
        else:
            placeholder.error(f'Pipeline failed with exit code {code}')

        # Render outputs
        outputs_area.subheader('Outputs')
        out_files = list_outputs(DEFAULT_OUTPUTS)
        if out_files:
            for f in out_files:
                with open(f, 'rb') as fh:
                    st.download_button(label=f'Download {f.name}', data=fh, file_name=f.name)
        else:
            st.info('No outputs found yet. Check logs above for errors.')

# Helpful links
with st.expander('Show advanced info'):
    st.markdown(f"Work dir: `{DEFAULT_WORKDIR}`  |  Outputs: `{DEFAULT_OUTPUTS}`")
    if (DEFAULT_WORKDIR / 'logs').exists():
        st.write('Logs:')
        for lf in sorted((DEFAULT_WORKDIR / 'logs').glob('*.log')):
            with st.expander(lf.name):
                try:
                    st.code(lf.read_text(encoding='utf-8', errors='ignore'))
                except Exception:
                    st.write('(binary or unreadable)')
