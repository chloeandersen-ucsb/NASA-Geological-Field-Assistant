#!/usr/bin/env python3
"""
Simple Flask-based web viewer for 3D point clouds and meshes.
Serves a Three.js viewer on http://localhost:5000
"""
import os
import sys
from pathlib import Path
from flask import Flask, render_template_string, send_from_directory, jsonify

app = Flask(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Viewer</title>
    <style>
        body {
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #fff;
            overflow: hidden;
        }
        #container {
            display: flex;
            height: 100vh;
        }
        #sidebar {
            width: 300px;
            background: #2a2a2a;
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid #444;
        }
        #viewer {
            flex: 1;
            position: relative;
        }
        h1 {
            margin: 0 0 20px 0;
            font-size: 24px;
            color: #4CAF50;
        }
        .folder-group {
            margin-bottom: 20px;
        }
        .folder-name {
            font-weight: bold;
            color: #8BC34A;
            margin-bottom: 10px;
            padding: 8px;
            background: #333;
            border-radius: 4px;
        }
        .file-item {
            padding: 8px 12px;
            margin: 5px 0;
            background: #333;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .file-item:hover {
            background: #444;
        }
        .file-item.active {
            background: #4CAF50;
        }
        .controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(42, 42, 42, 0.9);
            padding: 15px;
            border-radius: 8px;
            z-index: 100;
        }
        .control-button {
            display: block;
            width: 100%;
            padding: 8px;
            margin: 5px 0;
            background: #4CAF50;
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
            font-size: 14px;
        }
        .control-button:hover {
            background: #45a049;
        }
        #info {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(42, 42, 42, 0.9);
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 12px;
            z-index: 100;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="sidebar">
            <h1>3D Viewer</h1>
            <div id="file-list"></div>
        </div>
        <div id="viewer">
            <div class="controls">
                <button class="control-button" onclick="resetCamera()">Reset Camera</button>
                <button class="control-button" onclick="toggleWireframe()">Toggle Wireframe</button>
                <button class="control-button" onclick="togglePoints()">Toggle Points</button>
            </div>
            <div id="info">
                <div>Mouse: Rotate | Right-click: Pan | Scroll: Zoom</div>
                <div id="model-info"></div>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/PLYLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>

    <script>
        let scene, camera, renderer, controls;
        let currentObject = null;
        let wireframeMode = false;
        let pointsMode = false;

        function init() {
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a1a);

            // Camera
            camera = new THREE.PerspectiveCamera(
                75,
                (window.innerWidth - 300) / window.innerHeight,
                0.1,
                1000
            );
            camera.position.set(0, 2, 5);

            // Renderer
            const viewerDiv = document.getElementById('viewer');
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth - 300, window.innerHeight);
            viewerDiv.appendChild(renderer.domElement);

            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 5, 5);
            scene.add(directionalLight);

            // Grid
            const gridHelper = new THREE.GridHelper(10, 10);
            scene.add(gridHelper);

            // Load file list
            loadFileList();

            // Animation loop
            animate();

            // Handle window resize
            window.addEventListener('resize', onWindowResize);
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        function onWindowResize() {
            camera.aspect = (window.innerWidth - 300) / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth - 300, window.innerHeight);
        }

        function loadFileList() {
            fetch('/api/files')
                .then(response => response.json())
                .then(data => {
                    const fileListDiv = document.getElementById('file-list');
                    fileListDiv.innerHTML = '';

                    for (const [folder, files] of Object.entries(data)) {
                        const folderDiv = document.createElement('div');
                        folderDiv.className = 'folder-group';

                        const folderName = document.createElement('div');
                        folderName.className = 'folder-name';
                        folderName.textContent = folder;
                        folderDiv.appendChild(folderName);

                        files.forEach(file => {
                            const fileItem = document.createElement('div');
                            fileItem.className = 'file-item';
                            fileItem.textContent = file.name;
                            fileItem.onclick = () => loadModel(folder, file.path);
                            folderDiv.appendChild(fileItem);
                        });

                        fileListDiv.appendChild(folderDiv);
                    }
                });
        }

        function loadModel(folder, filepath) {
            // Remove previous object
            if (currentObject) {
                scene.remove(currentObject);
                currentObject = null;
            }

            // Update active state
            document.querySelectorAll('.file-item').forEach(el => el.classList.remove('active'));
            event.target.classList.add('active');

            const url = '/outputs/' + filepath;
            const ext = filepath.split('.').pop().toLowerCase();

            if (ext === 'ply') {
                const loader = new THREE.PLYLoader();
                loader.load(url, geometry => {
                    geometry.computeVertexNormals();

                    const material = new THREE.PointsMaterial({
                        size: 0.02,
                        vertexColors: true
                    });

                    currentObject = new THREE.Points(geometry, material);
                    scene.add(currentObject);

                    // Center and scale
                    centerAndScaleObject();
                    updateModelInfo(geometry);
                });
            } else if (ext === 'obj') {
                const loader = new THREE.OBJLoader();
                loader.load(url, object => {
                    currentObject = object;
                    scene.add(currentObject);

                    // Center and scale
                    centerAndScaleObject();
                    updateModelInfo(object);
                });
            }
        }

        function centerAndScaleObject() {
            if (!currentObject) return;

            const box = new THREE.Box3().setFromObject(currentObject);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());

            currentObject.position.sub(center);

            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2 / maxDim;
            currentObject.scale.setScalar(scale);

            camera.position.set(0, 2, 5);
            controls.target.set(0, 0, 0);
            controls.update();
        }

        function updateModelInfo(geometry) {
            const infoDiv = document.getElementById('model-info');
            if (geometry.attributes) {
                const count = geometry.attributes.position.count;
                infoDiv.textContent = `Points: ${count.toLocaleString()}`;
            }
        }

        function resetCamera() {
            camera.position.set(0, 2, 5);
            controls.target.set(0, 0, 0);
            controls.update();
        }

        function toggleWireframe() {
            if (!currentObject) return;
            wireframeMode = !wireframeMode;

            currentObject.traverse(child => {
                if (child.isMesh) {
                    child.material.wireframe = wireframeMode;
                }
            });
        }

        function togglePoints() {
            if (!currentObject) return;
            pointsMode = !pointsMode;

            if (currentObject.isPoints) {
                currentObject.material.size = pointsMode ? 0.05 : 0.02;
            }
        }

        // Initialize on load
        init();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    response = app.make_response(render_template_string(HTML_TEMPLATE))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/api/files')
def list_files():
    """List all PLY and OBJ files in outputs directory"""
    files_by_folder = {}
    
    if not OUTPUTS_DIR.exists():
        return jsonify({})
    
    for folder in sorted(OUTPUTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        
        folder_files = []
        
        # Search recursively for PLY and OBJ files
        for file in folder.rglob('*'):
            if file.suffix.lower() in ['.ply', '.obj']:
                rel_path = file.relative_to(OUTPUTS_DIR)
                folder_files.append({
                    'name': file.name,
                    'path': str(rel_path).replace('\\', '/')
                })
        
        if folder_files:
            files_by_folder[folder.name] = sorted(folder_files, key=lambda x: x['name'])
    
    return jsonify(files_by_folder)


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve output files"""
    return send_from_directory(OUTPUTS_DIR, filename)


def main():
    print("=" * 60)
    print("3D Web Viewer")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Outputs directory: {OUTPUTS_DIR}")
    print()
    print("Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
