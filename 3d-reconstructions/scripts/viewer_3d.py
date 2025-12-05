#!/usr/bin/env python3
"""
Standalone 3D Mesh Viewer for TripoSR outputs.
Displays OBJ/GLB files with textures in a native window.

Usage:
    python scripts/viewer_3d.py [path/to/mesh.obj]
    
Controls:
    Mouse drag: Rotate
    Mouse wheel: Zoom
    Right drag: Pan 
    R: Reset view
    W: Toggle wireframe
    L: Toggle lighting
    ESC: Quit
"""

import sys
import argparse
from pathlib import Path
import numpy as np

try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from PIL import Image
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("\nInstall required packages:")
    print("  pip install pygame PyOpenGL PyOpenGL_accelerate pillow")
    sys.exit(1)


class OBJLoader:
    """Simple OBJ file loader with texture support."""
    
    def __init__(self, filename, decimate_ratio=0.3):
        self.vertices = []
        self.texcoords = []
        self.normals = []
        self.faces = []
        self.texture = None
        self.display_list = None
        
        self.load_obj(filename)
        self.decimate_mesh(decimate_ratio)
        self.load_texture(filename)
        self.compile_display_list()
    
    def load_obj(self, filename):
        """Parse OBJ file."""
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                values = line.split()
                if not values:
                    continue
                
                if values[0] == 'v':
                    self.vertices.append([float(x) for x in values[1:4]])
                elif values[0] == 'vt':
                    self.texcoords.append([float(x) for x in values[1:3]])
                elif values[0] == 'vn':
                    self.normals.append([float(x) for x in values[1:4]])
                elif values[0] == 'f':
                    face = []
                    for v in values[1:]:
                        w = v.split('/')
                        face.append([int(w[0]) if w[0] else 0,
                                   int(w[1]) if len(w) > 1 and w[1] else 0,
                                   int(w[2]) if len(w) > 2 and w[2] else 0])
                    self.faces.append(face)
    
    def load_texture(self, obj_path):
        """Try to load texture.png from same directory."""
        texture_path = Path(obj_path).parent / 'texture.png'
        if not texture_path.exists():
            return
        
        try:
            img = Image.open(texture_path)
            # Keep higher res texture for quality, resize only if very large
            max_size = 2048
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                print(f"[INFO] Resized texture to {img.width}x{img.height}")
            else:
                print(f"[INFO] Texture size: {img.width}x{img.height}")
            
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = img.convert("RGB").tobytes()
            
            self.texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 
                        0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            print(f"[OK] Loaded texture: {texture_path}")
        except Exception as e:
            print(f"[WARN] Could not load texture: {e}")
    
    def decimate_mesh(self, ratio=0.5):
        """Reduce polygon count for better performance."""
        if ratio >= 1.0 or not self.faces:
            return
        
        original_count = len(self.faces)
        target_count = int(original_count * ratio)
        
        # Keep mesh quality - only decimate if very high poly
        if original_count < 20000:
            print(f"[INFO] Mesh has {original_count} faces - skipping decimation (model is reasonable size)")
            return
        
        # Smart decimation: prioritize keeping faces with texture detail
        step = max(1, original_count // target_count)
        self.faces = self.faces[::step]
        
        print(f"[INFO] Decimated mesh: {original_count} -> {len(self.faces)} faces ({len(self.faces)/original_count*100:.0f}%)")
    
    def compile_display_list(self):
        """Compile mesh to OpenGL display list for massive performance boost."""
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        
        if self.texture:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture)
        
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            for vertex in face:
                if vertex[2] > 0 and self.normals:
                    glNormal3fv(self.normals[vertex[2] - 1])
                if vertex[1] > 0 and self.texcoords:
                    glTexCoord2fv(self.texcoords[vertex[1] - 1])
                if vertex[0] > 0:
                    glVertex3fv(self.vertices[vertex[0] - 1])
        glEnd()
        
        if self.texture:
            glDisable(GL_TEXTURE_2D)
        
        glEndList()
        print(f"[OK] Compiled display list")
    
    def render(self, wireframe=False):
        """Render the mesh using compiled display list."""
        if wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        if self.display_list:
            glCallList(self.display_list)
        
    def __del__(self):
        """Clean up OpenGL resources."""
        if self.display_list:
            glDeleteLists(self.display_list, 1)
        if self.texture:
            glDeleteTextures([self.texture])


class Viewer3D:
    """Interactive 3D mesh viewer."""
    
    def __init__(self, mesh_path, width=1280, height=720, decimate=0.3):
        self.width = width
        self.height = height
        self.mesh_path = mesh_path
        
        # Camera state
        self.distance = 3.0
        self.rotation_x = 30.0
        self.rotation_y = 45.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Mouse state
        self.mouse_down = False
        self.mouse_right_down = False
        self.last_mouse_pos = (0, 0)
        
        # Render state
        self.wireframe = False
        self.lighting = True
        self.show_fps = True
        
        # Initialize
        pygame.init()
        pygame.display.set_caption(f"TripoSR Viewer - {Path(mesh_path).name}")
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        
        self.setup_gl()
        print(f"[INFO] Loading mesh with {decimate*100:.0f}% decimation...")
        self.mesh = OBJLoader(mesh_path, decimate_ratio=decimate)
        
    def setup_gl(self):
        """Initialize OpenGL settings."""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Lighting setup
        glLightfv(GL_LIGHT0, GL_POSITION, [5, 10, 5, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        # Additional light
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, [-5, -5, 5, 1])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.4, 0.4, 0.4, 1])
        
        glClearColor(0.15, 0.15, 0.15, 1)
        
        # Perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
    
    def toggle_wireframe(self):
        """Toggle wireframe mode."""
        self.wireframe = not self.wireframe
        print(f"[INFO] Wireframe: {'ON' if self.wireframe else 'OFF'}")
    
    def toggle_lighting(self):
        """Toggle lighting."""
        self.lighting = not self.lighting
        if self.lighting:
            glEnable(GL_LIGHTING)
        else:
            glDisable(GL_LIGHTING)
        print(f"[INFO] Lighting: {'ON' if self.lighting else 'OFF'}")
    
    def reset_view(self):
        """Reset camera to default position."""
        self.distance = 3.0
        self.rotation_x = 30.0
        self.rotation_y = 45.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        print("[INFO] View reset")
    
    def handle_events(self):
        """Process input events."""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                elif event.key == K_w:
                    self.toggle_wireframe()
                elif event.key == K_l:
                    self.toggle_lighting()
                elif event.key == K_r:
                    self.reset_view()
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_down = True
                    self.last_mouse_pos = event.pos
                elif event.button == 3:  # Right click
                    self.mouse_right_down = True
                    self.last_mouse_pos = event.pos
                elif event.button == 4:  # Scroll up
                    self.distance = max(0.5, self.distance - 0.2)
                elif event.button == 5:  # Scroll down
                    self.distance = min(20.0, self.distance + 0.2)
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
                elif event.button == 3:
                    self.mouse_right_down = False
            
            elif event.type == MOUSEMOTION:
                if self.mouse_down:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.rotation_y += dx * 0.5
                    self.rotation_x += dy * 0.5
                    self.last_mouse_pos = event.pos
                elif self.mouse_right_down:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.pan_x += dx * 0.003
                    self.pan_y -= dy * 0.003
                    self.last_mouse_pos = event.pos
        
        return True
    
    def render(self):
        """Render the scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Camera transform
        glTranslatef(self.pan_x, self.pan_y, -self.distance)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        # Draw mesh
        self.mesh.render(self.wireframe)
        
        # Draw FPS counter if enabled
        if self.show_fps:
            self.draw_fps()
        
        pygame.display.flip()
    
    def draw_fps(self):
        """Draw FPS counter in corner."""
        fps = int(pygame.time.Clock().get_fps())
        font = pygame.font.Font(None, 36)
        text = font.render(f"FPS: {fps}", True, (255, 255, 255))
        
        # Switch to 2D mode
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Render text as textured quad
        text_data = pygame.image.tostring(text, "RGBA", True)
        glRasterPos2d(10, 30)
        glDrawPixels(text.get_width(), text.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        glEnable(GL_DEPTH_TEST)
        if self.lighting:
            glEnable(GL_LIGHTING)
        
        # Restore 3D mode
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def run(self):
        """Main loop."""
        clock = pygame.time.Clock()
        running = True
        
        print("\n=== TripoSR 3D Viewer ===")
        print("Controls:")
        print("  Mouse drag     : Rotate")
        print("  Right drag     : Pan")
        print("  Mouse wheel    : Zoom")
        print("  W              : Toggle wireframe")
        print("  L              : Toggle lighting")
        print("  R              : Reset view")
        print("  ESC            : Quit")
        print("========================\n")
        
        while running:
            running = self.handle_events()
            self.render()
            clock.tick(60)
        
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="3D mesh viewer for TripoSR outputs")
    parser.add_argument('mesh', nargs='?', help='Path to OBJ mesh file')
    parser.add_argument('--width', type=int, default=1280, help='Window width')
    parser.add_argument('--height', type=int, default=720, help='Window height')
    parser.add_argument('--decimate', type=float, default=0.5, help='Mesh decimation ratio (0.1-1.0, lower=faster)')
    args = parser.parse_args()
    
    if args.mesh:
        mesh_path = args.mesh
    else:
        # Try to find the most recent output
        outputs = list(Path('outputs').rglob('mesh.obj'))
        if not outputs:
            print("[ERROR] No mesh file specified and no outputs found.")
            print("\nUsage: python scripts/viewer_3d.py path/to/mesh.obj")
            return 1
        mesh_path = str(max(outputs, key=lambda p: p.stat().st_mtime))
        print(f"[INFO] Auto-loading most recent: {mesh_path}")
    
    if not Path(mesh_path).exists():
        print(f"[ERROR] Mesh file not found: {mesh_path}")
        return 2
    
    viewer = Viewer3D(mesh_path, args.width, args.height, args.decimate)
    viewer.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
