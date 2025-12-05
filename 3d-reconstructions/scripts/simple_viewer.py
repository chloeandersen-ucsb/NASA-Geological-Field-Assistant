#!/usr/bin/env python3
"""Simple 3D viewer that actually works"""
from flask import Flask, send_from_directory
from pathlib import Path
import json

app = Flask(__name__)
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

@app.route('/')
def index():
    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>3D Viewer</title>
    <style>
        body { margin: 0; font-family: Arial; background: #1a1a1a; color: #fff; }
        #container { display: flex; height: 100vh; }
        #sidebar { width: 300px; background: #2a2a2a; padding: 20px; overflow-y: auto; }
        #viewer { flex: 1; }
        h1 { color: #4CAF50; margin: 0 0 20px 0; }
        .folder { margin-bottom: 20px; }
        .folder-name { font-weight: bold; color: #8BC34A; margin-bottom: 10px; }
        .file { padding: 8px; margin: 5px 0; background: #333; border-radius: 4px; cursor: pointer; }
        .file:hover { background: #444; }
        .file.active { background: #4CAF50; }
    </style>
</head>
<body>
    <div id="container">
        <div id="sidebar">
            <h1>3D Viewer</h1>
            <div id="files"></div>
        </div>
        <div id="viewer"></div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/PLYLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);
        
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(2, 2, 2);
        
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth - 300, window.innerHeight);
        document.getElementById('viewer').appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        
        const light1 = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(light1);
        
        const light2 = new THREE.DirectionalLight(0xffffff, 0.8);
        light2.position.set(5, 10, 5);
        scene.add(light2);
        
        const grid = new THREE.GridHelper(10, 10);
        scene.add(grid);
        
        let currentObject = null;
        
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
        
        fetch('/api/files')
            .then(r => r.json())
            .then(data => {
                const container = document.getElementById('files');
                for (const [folder, files] of Object.entries(data)) {
                    const div = document.createElement('div');
                    div.className = 'folder';
                    div.innerHTML = `<div class="folder-name">${folder}</div>`;
                    files.forEach(f => {
                        const fdiv = document.createElement('div');
                        fdiv.className = 'file';
                        fdiv.textContent = f.name;
                        fdiv.onclick = () => loadFile(f.path, fdiv);
                        div.appendChild(fdiv);
                    });
                    container.appendChild(div);
                }
            });
        
        function loadFile(path, elem) {
            if (currentObject) {
                scene.remove(currentObject);
            }
            
            document.querySelectorAll('.file').forEach(e => e.classList.remove('active'));
            elem.classList.add('active');
            
            const ext = path.split('.').pop().toLowerCase();
            const url = '/outputs/' + path;
            
            if (ext === 'ply') {
                new THREE.PLYLoader().load(url, geo => {
                    geo.computeVertexNormals();
                    const mat = new THREE.PointsMaterial({ size: 0.02, vertexColors: true });
                    currentObject = new THREE.Points(geo, mat);
                    scene.add(currentObject);
                    centerObject();
                });
            } else if (ext === 'obj') {
                new THREE.OBJLoader().load(url, obj => {
                    currentObject = obj;
                    scene.add(currentObject);
                    centerObject();
                });
            }
        }
        
        function centerObject() {
            const box = new THREE.Box3().setFromObject(currentObject);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2 / maxDim;
            currentObject.position.sub(center);
            currentObject.scale.setScalar(scale);
        }
        
        window.addEventListener('resize', () => {
            camera.aspect = (window.innerWidth - 300) / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth - 300, window.innerHeight);
        });
    </script>
</body>
</html>'''
    return html, 200, {'Cache-Control': 'no-store'}

@app.route('/api/files')
def files():
    result = {}
    for folder in OUTPUTS_DIR.iterdir():
        if folder.is_dir():
            files = []
            for f in folder.rglob('*'):
                if f.suffix.lower() in ['.ply', '.obj']:
                    files.append({'name': f.name, 'path': str(f.relative_to(OUTPUTS_DIR)).replace('\\', '/')})
            if files:
                result[folder.name] = files
    return json.dumps(result), 200, {'Content-Type': 'application/json'}

@app.route('/outputs/<path:filename>')
def serve(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

if __name__ == '__main__':
    print(f"\n🌐 Open: http://localhost:5001\n")
    app.run(host='0.0.0.0', port=5001, debug=False)
