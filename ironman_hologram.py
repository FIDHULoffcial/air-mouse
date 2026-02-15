"""
Iron Man Holographic Interface System
Advanced hand tracking with gesture-based 3D object manipulation
Inspired by Tony Stark's holographic workspace from the Iron Man movies
"""

import cv2
import numpy as np
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame
from pygame.locals import *
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading


@dataclass
class GestureState:
    """Stores the current state of hand gestures"""
    one_hand_active: bool = False
    two_hands_active: bool = False
    pinch_distance: float = 0.0
    hand_center: Tuple[float, float] = (0, 0)
    rotation_angle: float = 0.0
    is_grabbing: bool = False
    is_swiping: bool = False
    swipe_direction: str = ""
    hand_velocity: Tuple[float, float] = (0, 0)


class HandTracker:
    """Advanced hand tracking using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture recognition variables
        self.prev_landmarks = None
        self.gesture_history = deque(maxlen=10)
        self.swipe_threshold = 0.15
        
    def process_frame(self, frame):
        """Process video frame and detect hands"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results
    
    def get_hand_landmarks(self, results, hand_idx=0):
        """Extract landmark positions for a specific hand"""
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > hand_idx:
            return results.multi_hand_landmarks[hand_idx]
        return None
    
    def calculate_pinch_distance(self, landmarks):
        """Calculate distance between thumb tip and index finger tip"""
        if landmarks is None:
            return 0.0
        
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        
        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        return distance
    
    def get_hand_center(self, landmarks):
        """Calculate center point of hand (palm)"""
        if landmarks is None:
            return (0, 0)
        
        # Use wrist and middle finger MCP as reference
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]
        
        center_x = (wrist.x + middle_mcp.x) / 2
        center_y = (wrist.y + middle_mcp.y) / 2
        
        return (center_x, center_y)
    
    def calculate_hand_rotation(self, landmarks):
        """Calculate hand rotation angle"""
        if landmarks is None:
            return 0.0
        
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]
        
        angle = math.atan2(
            middle_mcp.y - wrist.y,
            middle_mcp.x - wrist.x
        )
        return math.degrees(angle)
    
    def detect_grab_gesture(self, landmarks):
        """Detect closed fist (grab) gesture"""
        if landmarks is None:
            return False
        
        # Check if all fingertips are close to palm
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        palm = landmarks.landmark[0]
        
        distances = []
        for tip_idx in fingertips:
            tip = landmarks.landmark[tip_idx]
            dist = math.sqrt(
                (tip.x - palm.x)**2 + 
                (tip.y - palm.y)**2
            )
            distances.append(dist)
        
        # If average distance is small, hand is closed (grabbing)
        avg_distance = sum(distances) / len(distances)
        return avg_distance < 0.15
    
    def detect_swipe(self, landmarks):
        """Detect swipe gestures"""
        if landmarks is None or self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return False, ""
        
        # Compare current and previous index finger position
        curr_index = landmarks.landmark[8]
        prev_index = self.prev_landmarks.landmark[8]
        
        delta_x = curr_index.x - prev_index.x
        delta_y = curr_index.y - prev_index.y
        
        self.prev_landmarks = landmarks
        
        # Determine swipe direction
        if abs(delta_x) > self.swipe_threshold:
            if delta_x > 0:
                return True, "RIGHT"
            else:
                return True, "LEFT"
        elif abs(delta_y) > self.swipe_threshold:
            if delta_y > 0:
                return True, "DOWN"
            else:
                return True, "UP"
        
        return False, ""
    
    def calculate_two_hand_distance(self, landmarks1, landmarks2):
        """Calculate distance between two hands for scaling"""
        center1 = self.get_hand_center(landmarks1)
        center2 = self.get_hand_center(landmarks2)
        
        distance = math.sqrt(
            (center1[0] - center2[0])**2 + 
            (center1[1] - center2[1])**2
        )
        return distance


class HolographicObject:
    """3D object with holographic rendering properties"""
    
    def __init__(self, obj_type="cube"):
        self.obj_type = obj_type
        self.position = [0.0, 0.0, -5.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.scale = 1.0
        self.color = [0.0, 0.8, 1.0, 0.7]  # Cyan holographic color
        self.wireframe = True
        self.particles = []
        
    def draw(self):
        """Render the 3D object with holographic effects"""
        glPushMatrix()
        
        # Apply transformations
        glTranslatef(*self.position)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(self.scale, self.scale, self.scale)
        
        # Set holographic material properties
        glColor4f(*self.color)
        
        # Enable wireframe for holographic effect
        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        
        # Draw object based on type
        if self.obj_type == "cube":
            self.draw_cube()
        elif self.obj_type == "sphere":
            self.draw_sphere()
        elif self.obj_type == "torus":
            self.draw_torus()
        elif self.obj_type == "arc_reactor":
            self.draw_arc_reactor()
        
        # Reset polygon mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glPopMatrix()
        
        # Draw holographic particles
        self.draw_particles()
    
    def draw_cube(self):
        """Draw a cube"""
        glutWireCube(2.0)
    
    def draw_sphere(self):
        """Draw a sphere"""
        glutWireSphere(1.0, 20, 20)
    
    def draw_torus(self):
        """Draw a torus"""
        glutWireTorus(0.5, 1.0, 20, 30)
    
    def draw_arc_reactor(self):
        """Draw an Iron Man arc reactor style object"""
        # Outer ring
        glutWireTorus(0.1, 1.5, 20, 30)
        
        # Inner rings
        glPushMatrix()
        for i in range(3):
            scale = 1.0 - (i * 0.25)
            glScalef(scale, scale, 1.0)
            glutWireTorus(0.05, 1.5, 15, 25)
        glPopMatrix()
        
        # Center core
        glPushMatrix()
        glScalef(0.3, 0.3, 0.3)
        glutWireSphere(1.0, 15, 15)
        glPopMatrix()
    
    def draw_particles(self):
        """Draw holographic particle effects"""
        glPointSize(2.0)
        glBegin(GL_POINTS)
        
        for particle in self.particles:
            alpha = particle['life']
            glColor4f(0.0, 0.8, 1.0, alpha * 0.5)
            glVertex3f(*particle['pos'])
        
        glEnd()
    
    def update_particles(self):
        """Update particle system"""
        # Remove dead particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        # Update existing particles
        for particle in self.particles:
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['pos'][2] += particle['vel'][2]
            particle['life'] -= 0.02
    
    def emit_particles(self, num_particles=5):
        """Emit new particles"""
        for _ in range(num_particles):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.01, 0.05)
            
            particle = {
                'pos': [self.position[0], self.position[1], self.position[2]],
                'vel': [
                    math.cos(angle) * speed,
                    math.sin(angle) * speed,
                    np.random.uniform(-0.02, 0.02)
                ],
                'life': 1.0
            }
            self.particles.append(particle)


class HUD:
    """Heads-Up Display overlay"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_scanlines(self, frame):
        """Draw holographic scanlines effect"""
        for i in range(0, self.height, 4):
            alpha = 0.1
            overlay = frame.copy()
            cv2.line(overlay, (0, i), (self.width, i), (0, 255, 255), 1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame
    
    def draw_corner_brackets(self, frame):
        """Draw corner brackets like Iron Man HUD"""
        color = (0, 255, 255)
        thickness = 2
        length = 50
        
        # Top-left
        cv2.line(frame, (20, 20), (20 + length, 20), color, thickness)
        cv2.line(frame, (20, 20), (20, 20 + length), color, thickness)
        
        # Top-right
        cv2.line(frame, (self.width - 20, 20), (self.width - 20 - length, 20), color, thickness)
        cv2.line(frame, (self.width - 20, 20), (self.width - 20, 20 + length), color, thickness)
        
        # Bottom-left
        cv2.line(frame, (20, self.height - 20), (20 + length, self.height - 20), color, thickness)
        cv2.line(frame, (20, self.height - 20), (20, self.height - 20 - length), color, thickness)
        
        # Bottom-right
        cv2.line(frame, (self.width - 20, self.height - 20), 
                (self.width - 20 - length, self.height - 20), color, thickness)
        cv2.line(frame, (self.width - 20, self.height - 20), 
                (self.width - 20, self.height - 20 - length), color, thickness)
        
        return frame
    
    def draw_reticle(self, frame, center):
        """Draw targeting reticle"""
        color = (0, 255, 255)
        radius = 30
        
        cv2.circle(frame, center, radius, color, 2)
        cv2.circle(frame, center, radius - 10, color, 1)
        
        # Crosshair
        cv2.line(frame, (center[0] - radius, center[1]), 
                (center[0] + radius, center[1]), color, 1)
        cv2.line(frame, (center[0], center[1] - radius), 
                (center[0], center[1] + radius), color, 1)
        
        return frame
    
    def draw_gesture_info(self, frame, gesture_state):
        """Display gesture information"""
        y_offset = 60
        color = (0, 255, 255)
        
        # Gesture status
        if gesture_state.one_hand_active:
            cv2.putText(frame, "MODE: ROTATION", (20, y_offset), 
                       self.font, 0.6, color, 2)
        elif gesture_state.two_hands_active:
            cv2.putText(frame, "MODE: TRANSLATION & SCALE", (20, y_offset), 
                       self.font, 0.6, color, 2)
        
        # Additional info
        y_offset += 30
        if gesture_state.is_grabbing:
            cv2.putText(frame, "STATUS: GRABBING", (20, y_offset), 
                       self.font, 0.5, (0, 255, 0), 2)
        
        y_offset += 25
        if gesture_state.is_swiping:
            cv2.putText(frame, f"SWIPE: {gesture_state.swipe_direction}", 
                       (20, y_offset), self.font, 0.5, (255, 255, 0), 2)
        
        return frame
    
    def draw_system_info(self, frame, fps):
        """Draw system information"""
        y_offset = self.height - 60
        color = (0, 255, 255)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), 
                   self.font, 0.5, color, 1)
        
        y_offset += 25
        cv2.putText(frame, "JARVIS v4.2 ONLINE", (20, y_offset), 
                   self.font, 0.5, color, 1)
        
        return frame


class IronManHolographicSystem:
    """Main holographic control system"""
    
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        
        # Initialize components
        self.hand_tracker = HandTracker()
        self.hud = HUD(width, height)
        self.holographic_object = HolographicObject("arc_reactor")
        self.gesture_state = GestureState()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # OpenGL/Pygame setup
        pygame.init()
        self.display = (width, height)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("IRON MAN - Holographic Interface System")
        
        # Initialize OpenGL
        self.init_opengl()
        
        # Gesture control variables
        self.prev_two_hand_distance = None
        self.base_scale = 1.0
        self.rotation_speed = 2.0
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Background holographic grid
        self.grid_alpha = 0.3
        
    def init_opengl(self):
        """Initialize OpenGL settings"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Lighting for holographic effect
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        light_pos = [5.0, 5.0, 5.0, 1.0]
        light_ambient = [0.2, 0.2, 0.4, 1.0]
        light_diffuse = [0.0, 0.8, 1.0, 1.0]
        
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    
    def draw_holographic_grid(self):
        """Draw background grid for holographic effect"""
        glDisable(GL_LIGHTING)
        glColor4f(0.0, 0.6, 0.8, self.grid_alpha)
        
        grid_size = 20
        grid_spacing = 1.0
        
        glBegin(GL_LINES)
        for i in range(-grid_size, grid_size + 1):
            # Horizontal lines
            glVertex3f(i * grid_spacing, -10, -20)
            glVertex3f(i * grid_spacing, 10, -20)
            
            # Vertical lines
            glVertex3f(-grid_size * grid_spacing, i * grid_spacing, -20)
            glVertex3f(grid_size * grid_spacing, i * grid_spacing, -20)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def process_gestures(self, results):
        """Process hand tracking results and update gesture state"""
        if not results.multi_hand_landmarks:
            self.gesture_state.one_hand_active = False
            self.gesture_state.two_hands_active = False
            return
        
        num_hands = len(results.multi_hand_landmarks)
        
        if num_hands == 1:
            # ONE HAND - ROTATION MODE
            self.gesture_state.one_hand_active = True
            self.gesture_state.two_hands_active = False
            
            landmarks = results.multi_hand_landmarks[0]
            
            # Get hand center and rotation
            center = self.hand_tracker.get_hand_center(landmarks)
            rotation = self.hand_tracker.calculate_hand_rotation(landmarks)
            
            # Update object rotation based on hand movement
            self.gesture_state.hand_center = center
            self.gesture_state.rotation_angle = rotation
            
            # Map hand position to rotation
            # X-axis rotation (up/down hand movement)
            self.holographic_object.rotation[0] = (center[1] - 0.5) * 180
            
            # Y-axis rotation (left/right hand movement)
            self.holographic_object.rotation[1] += (center[0] - 0.5) * self.rotation_speed
            
            # Z-axis rotation (hand rotation)
            self.holographic_object.rotation[2] = rotation
            
            # Check for grab gesture
            self.gesture_state.is_grabbing = self.hand_tracker.detect_grab_gesture(landmarks)
            
            # Check for swipe
            is_swipe, direction = self.hand_tracker.detect_swipe(landmarks)
            self.gesture_state.is_swiping = is_swipe
            self.gesture_state.swipe_direction = direction
            
            # Swipe actions
            if is_swipe:
                if direction == "LEFT":
                    self.holographic_object.rotation[1] -= 15
                elif direction == "RIGHT":
                    self.holographic_object.rotation[1] += 15
                
                # Emit particles on swipe
                self.holographic_object.emit_particles(10)
        
        elif num_hands == 2:
            # TWO HANDS - TRANSLATION AND SCALING MODE
            self.gesture_state.one_hand_active = False
            self.gesture_state.two_hands_active = True
            
            landmarks1 = results.multi_hand_landmarks[0]
            landmarks2 = results.multi_hand_landmarks[1]
            
            # Calculate centers of both hands
            center1 = self.hand_tracker.get_hand_center(landmarks1)
            center2 = self.hand_tracker.get_hand_center(landmarks2)
            
            # Average position for translation (left/right movement)
            avg_x = (center1[0] + center2[0]) / 2
            avg_y = (center1[1] + center2[1]) / 2
            
            # Translate object left/right based on hand position
            self.holographic_object.position[0] = (avg_x - 0.5) * 10
            self.holographic_object.position[1] = -(avg_y - 0.5) * 10
            
            # Calculate distance between hands for scaling
            current_distance = self.hand_tracker.calculate_two_hand_distance(
                landmarks1, landmarks2
            )
            
            if self.prev_two_hand_distance is not None:
                # Scale based on change in distance
                scale_factor = current_distance / self.prev_two_hand_distance
                self.holographic_object.scale *= scale_factor
                
                # Clamp scale
                self.holographic_object.scale = max(0.1, min(5.0, self.holographic_object.scale))
                
                # Emit particles when scaling
                if abs(scale_factor - 1.0) > 0.05:
                    self.holographic_object.emit_particles(5)
            
            self.prev_two_hand_distance = current_distance
            
            # Check both hands for pinch
            pinch1 = self.hand_tracker.calculate_pinch_distance(landmarks1)
            pinch2 = self.hand_tracker.calculate_pinch_distance(landmarks2)
            
            # If both hands are pinching, activate special mode
            if pinch1 < 0.05 and pinch2 < 0.05:
                self.gesture_state.is_grabbing = True
                # Could trigger special effects or mode
        else:
            self.prev_two_hand_distance = None
    
    def render_3d_scene(self):
        """Render the 3D holographic scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set background color (dark blue/black for space effect)
        glClearColor(0.0, 0.0, 0.1, 1.0)
        
        glLoadIdentity()
        
        # Camera position
        gluLookAt(0, 0, 0, 0, 0, -5, 0, 1, 0)
        
        # Draw holographic grid
        self.draw_holographic_grid()
        
        # Update and draw holographic object
        self.holographic_object.update_particles()
        self.holographic_object.draw()
    
    def capture_opengl_frame(self):
        """Capture the OpenGL rendered frame"""
        # Read pixels from OpenGL
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # Convert to numpy array and flip
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        image = np.flipud(image)
        
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def run(self):
        """Main loop"""
        print("=" * 60)
        print("IRON MAN HOLOGRAPHIC INTERFACE SYSTEM")
        print("=" * 60)
        print("\nGesture Controls:")
        print("  ONE HAND:")
        print("    - Move hand to rotate object")
        print("    - Close fist to grab")
        print("    - Swipe left/right to spin object")
        print("\n  TWO HANDS:")
        print("    - Move hands together left/right to translate")
        print("    - Move hands apart/together to scale (zoom)")
        print("    - Pinch both hands for special mode")
        print("\nKeyboard Controls:")
        print("  [1] Cube  [2] Sphere  [3] Torus  [4] Arc Reactor")
        print("  [W] Wireframe On/Off")
        print("  [R] Reset Object")
        print("  [Q] Quit")
        print("=" * 60)
        print("\nStarting system...")
        
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_1:
                        self.holographic_object.obj_type = "cube"
                    elif event.key == pygame.K_2:
                        self.holographic_object.obj_type = "sphere"
                    elif event.key == pygame.K_3:
                        self.holographic_object.obj_type = "torus"
                    elif event.key == pygame.K_4:
                        self.holographic_object.obj_type = "arc_reactor"
                    elif event.key == pygame.K_w:
                        self.holographic_object.wireframe = not self.holographic_object.wireframe
                    elif event.key == pygame.K_r:
                        # Reset object
                        self.holographic_object.position = [0.0, 0.0, -5.0]
                        self.holographic_object.rotation = [0.0, 0.0, 0.0]
                        self.holographic_object.scale = 1.0
            
            # Capture webcam frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Process hand tracking
            results = self.hand_tracker.process_frame(frame)
            
            # Update gesture state
            self.process_gestures(results)
            
            # Render 3D scene
            self.render_3d_scene()
            
            # Capture OpenGL frame
            gl_frame = self.capture_opengl_frame()
            
            # Blend webcam with 3D render
            alpha = 0.6
            blended = cv2.addWeighted(frame, 1 - alpha, gl_frame, alpha, 0)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.hand_tracker.mp_draw.draw_landmarks(
                        blended,
                        hand_landmarks,
                        self.hand_tracker.mp_hands.HAND_CONNECTIONS,
                        self.hand_tracker.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                        self.hand_tracker.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
            
            # Apply HUD overlays
            blended = self.hud.draw_scanlines(blended)
            blended = self.hud.draw_corner_brackets(blended)
            blended = self.hud.draw_gesture_info(blended, self.gesture_state)
            blended = self.hud.draw_system_info(blended, self.fps)
            
            # Show final composite
            cv2.imshow("IRON MAN - Holographic Interface", blended)
            
            # Update display
            pygame.display.flip()
            
            # Calculate FPS
            self.calculate_fps()
            
            # Control frame rate
            clock.tick(60)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    try:
        system = IronManHolographicSystem(width=1280, height=720)
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
