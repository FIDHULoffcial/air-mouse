"""
Iron Man Holographic Interface System - ENHANCED VERSION
Advanced features including:
- More Iron Man gesture types (flick, toss, expand, crumple)
- Enhanced particle systems
- Multi-object manipulation
- Virtual trash can
- Scene reconstruction mode
- Advanced visual effects
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
import json


@dataclass
class Hand:
    """Represents a tracked hand"""
    landmarks: any
    center: Tuple[float, float]
    velocity: Tuple[float, float]
    is_open: bool
    is_pinching: bool
    pinch_distance: float
    gesture: str


class AdvancedGestureRecognizer:
    """Advanced gesture recognition for Iron Man-style controls"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.prev_hands = []
        self.gesture_history = deque(maxlen=30)
        
        # Gesture thresholds
        self.flick_velocity_threshold = 0.25
        self.expand_threshold = 0.15
        self.toss_velocity_threshold = 0.3
        
    def process_frame(self, frame):
        """Process frame and extract hand data"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        current_hands = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand = self._analyze_hand(hand_landmarks, idx)
                current_hands.append(hand)
        
        # Detect complex gestures
        gestures = self._detect_advanced_gestures(current_hands)
        
        self.prev_hands = current_hands
        
        return results, current_hands, gestures
    
    def _analyze_hand(self, landmarks, idx):
        """Analyze a single hand"""
        # Calculate center
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]
        center = ((wrist.x + middle_mcp.x) / 2, (wrist.y + middle_mcp.y) / 2)
        
        # Calculate velocity
        velocity = (0, 0)
        if idx < len(self.prev_hands):
            prev_center = self.prev_hands[idx].center
            velocity = (center[0] - prev_center[0], center[1] - prev_center[1])
        
        # Check if hand is open (fingers extended)
        is_open = self._is_hand_open(landmarks)
        
        # Check pinch
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        pinch_distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        is_pinching = pinch_distance < 0.05
        
        # Determine gesture
        gesture = self._classify_hand_gesture(landmarks, velocity, is_open, is_pinching)
        
        return Hand(
            landmarks=landmarks,
            center=center,
            velocity=velocity,
            is_open=is_open,
            is_pinching=is_pinching,
            pinch_distance=pinch_distance,
            gesture=gesture
        )
    
    def _is_hand_open(self, landmarks):
        """Check if hand is open (fingers extended)"""
        # Check if fingertips are far from palm
        fingertips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        palm = landmarks.landmark[0]
        
        distances = []
        for tip_idx in fingertips:
            tip = landmarks.landmark[tip_idx]
            dist = math.sqrt((tip.x - palm.x)**2 + (tip.y - palm.y)**2)
            distances.append(dist)
        
        avg_distance = sum(distances) / len(distances)
        return avg_distance > 0.2
    
    def _classify_hand_gesture(self, landmarks, velocity, is_open, is_pinching):
        """Classify the hand gesture"""
        vel_magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2)
        
        if is_pinching:
            return "PINCH"
        elif not is_open and vel_magnitude < 0.05:
            return "FIST"
        elif is_open and vel_magnitude < 0.05:
            return "OPEN"
        elif is_open and vel_magnitude > self.flick_velocity_threshold:
            return "FLICK"
        elif not is_open and vel_magnitude > self.toss_velocity_threshold:
            return "TOSS"
        else:
            return "NEUTRAL"
    
    def _detect_advanced_gestures(self, current_hands):
        """Detect complex multi-hand gestures"""
        gestures = {
            'type': 'NONE',
            'params': {}
        }
        
        if len(current_hands) == 0:
            return gestures
        
        if len(current_hands) == 1:
            hand = current_hands[0]
            
            # Single hand gestures
            if hand.gesture == "FLICK":
                gestures['type'] = 'FLICK'
                gestures['params'] = {
                    'direction': self._get_velocity_direction(hand.velocity),
                    'velocity': hand.velocity
                }
            elif hand.gesture == "TOSS":
                gestures['type'] = 'TOSS'
                gestures['params'] = {
                    'direction': self._get_velocity_direction(hand.velocity),
                    'velocity': hand.velocity
                }
            elif hand.gesture == "PINCH":
                gestures['type'] = 'PINCH_DRAG'
                gestures['params'] = {'position': hand.center}
            elif hand.gesture == "OPEN":
                gestures['type'] = 'ROTATE'
                gestures['params'] = {'position': hand.center}
        
        elif len(current_hands) == 2:
            hand1, hand2 = current_hands[0], current_hands[1]
            
            # Two hand gestures
            if hand1.is_pinching and hand2.is_pinching:
                # Both hands pinching - could be for precise manipulation
                gestures['type'] = 'TWO_HAND_PINCH'
                gestures['params'] = {
                    'distance': self._calculate_hand_distance(hand1, hand2),
                    'center': self._calculate_center_point(hand1, hand2)
                }
            
            elif hand1.is_open and hand2.is_open:
                # Both hands open - expansion/contraction
                if len(self.prev_hands) == 2:
                    prev_distance = self._calculate_hand_distance(
                        self.prev_hands[0], self.prev_hands[1]
                    )
                    curr_distance = self._calculate_hand_distance(hand1, hand2)
                    
                    if curr_distance > prev_distance + self.expand_threshold:
                        gestures['type'] = 'EXPAND'
                        gestures['params'] = {'scale_factor': curr_distance / prev_distance}
                    elif curr_distance < prev_distance - self.expand_threshold:
                        gestures['type'] = 'CONTRACT'
                        gestures['params'] = {'scale_factor': curr_distance / prev_distance}
                    else:
                        gestures['type'] = 'TRANSLATE'
                        gestures['params'] = {
                            'position': self._calculate_center_point(hand1, hand2)
                        }
        
        self.gesture_history.append(gestures)
        return gestures
    
    def _get_velocity_direction(self, velocity):
        """Get direction from velocity vector"""
        angle = math.atan2(velocity[1], velocity[0])
        angle_deg = math.degrees(angle)
        
        if -45 <= angle_deg < 45:
            return "RIGHT"
        elif 45 <= angle_deg < 135:
            return "DOWN"
        elif angle_deg >= 135 or angle_deg < -135:
            return "LEFT"
        else:
            return "UP"
    
    def _calculate_hand_distance(self, hand1, hand2):
        """Calculate distance between two hands"""
        return math.sqrt(
            (hand1.center[0] - hand2.center[0])**2 + 
            (hand1.center[1] - hand2.center[1])**2
        )
    
    def _calculate_center_point(self, hand1, hand2):
        """Calculate center point between two hands"""
        return (
            (hand1.center[0] + hand2.center[0]) / 2,
            (hand1.center[1] + hand2.center[1]) / 2
        )


class HolographicObject3D:
    """Enhanced 3D object with advanced rendering"""
    
    def __init__(self, obj_type="cube", obj_id=0):
        self.obj_id = obj_id
        self.obj_type = obj_type
        self.position = [0.0, 0.0, -5.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.scale = 1.0
        self.color = [0.0, 0.8, 1.0, 0.7]
        self.wireframe = True
        self.active = True
        self.selected = False
        
        # Animation
        self.target_position = self.position.copy()
        self.target_rotation = self.rotation.copy()
        self.target_scale = self.scale
        
        # Particle system
        self.particles = []
        self.max_particles = 100
        
        # Glow effect
        self.glow_intensity = 0.0
        self.glow_pulse_speed = 2.0
        
    def update(self, dt):
        """Update object state"""
        # Smooth interpolation to targets
        lerp_factor = 0.15
        
        for i in range(3):
            self.position[i] += (self.target_position[i] - self.position[i]) * lerp_factor
            self.rotation[i] += (self.target_rotation[i] - self.rotation[i]) * lerp_factor
        
        self.scale += (self.target_scale - self.scale) * lerp_factor
        
        # Update glow
        self.glow_intensity = (math.sin(time.time() * self.glow_pulse_speed) + 1) / 2
        
        # Update particles
        self.update_particles()
    
    def draw(self):
        """Render the object"""
        if not self.active:
            return
        
        glPushMatrix()
        
        glTranslatef(*self.position)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(self.scale, self.scale, self.scale)
        
        # Apply glow effect if selected
        if self.selected:
            glow_color = [
                self.color[0],
                self.color[1] + self.glow_intensity * 0.2,
                self.color[2],
                self.color[3]
            ]
            glColor4f(*glow_color)
        else:
            glColor4f(*self.color)
        
        # Draw based on type
        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        
        if self.obj_type == "cube":
            glutWireCube(2.0)
        elif self.obj_type == "sphere":
            glutWireSphere(1.0, 20, 20)
        elif self.obj_type == "torus":
            glutWireTorus(0.5, 1.0, 20, 30)
        elif self.obj_type == "arc_reactor":
            self.draw_arc_reactor()
        elif self.obj_type == "iron_man_helmet":
            self.draw_iron_man_helmet()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPopMatrix()
        
        # Draw particles
        self.draw_particles()
    
    def draw_arc_reactor(self):
        """Draw arc reactor"""
        # Outer ring
        glutWireTorus(0.1, 1.5, 20, 30)
        
        # Multiple inner rings
        for i in range(4):
            glPushMatrix()
            scale = 1.0 - (i * 0.2)
            glScalef(scale, scale, 1.0)
            glutWireTorus(0.05, 1.5, 15, 25)
            glPopMatrix()
        
        # Center core with glow
        glPushMatrix()
        glScalef(0.3, 0.3, 0.3)
        glutWireSphere(1.0, 15, 15)
        glPopMatrix()
        
        # Energy beams
        glBegin(GL_LINES)
        for i in range(8):
            angle = (i / 8.0) * 2 * math.pi
            x = math.cos(angle) * 1.5
            y = math.sin(angle) * 1.5
            glVertex3f(0, 0, 0)
            glVertex3f(x, y, 0)
        glEnd()
    
    def draw_iron_man_helmet(self):
        """Draw simplified Iron Man helmet"""
        # Main head sphere
        glutWireSphere(1.0, 20, 20)
        
        # Eye slits
        glPushMatrix()
        glTranslatef(0.4, 0.3, 0.8)
        glScalef(0.3, 0.15, 0.1)
        glutWireCube(1.0)
        glPopMatrix()
        
        glPushMatrix()
        glTranslatef(-0.4, 0.3, 0.8)
        glScalef(0.3, 0.15, 0.1)
        glutWireCube(1.0)
        glPopMatrix()
        
        # Faceplate outline
        glPushMatrix()
        glTranslatef(0, 0, 0.8)
        glScalef(0.8, 1.0, 0.1)
        glutWireSphere(0.5, 15, 15)
        glPopMatrix()
    
    def update_particles(self):
        """Update particle system"""
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        for particle in self.particles:
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['pos'][2] += particle['vel'][2]
            particle['vel'][1] -= 0.001  # Gravity
            particle['life'] -= 0.015
    
    def draw_particles(self):
        """Draw particles"""
        glDisable(GL_LIGHTING)
        glPointSize(3.0)
        glBegin(GL_POINTS)
        
        for particle in self.particles:
            alpha = particle['life'] * 0.6
            glColor4f(0.0, 0.8, 1.0, alpha)
            glVertex3f(*particle['pos'])
        
        glEnd()
        glEnable(GL_LIGHTING)
    
    def emit_particles(self, num_particles=10):
        """Emit new particles"""
        if len(self.particles) + num_particles > self.max_particles:
            return
        
        for _ in range(num_particles):
            angle = np.random.uniform(0, 2 * np.pi)
            elevation = np.random.uniform(-np.pi/4, np.pi/4)
            speed = np.random.uniform(0.02, 0.08)
            
            particle = {
                'pos': self.position.copy(),
                'vel': [
                    math.cos(angle) * math.cos(elevation) * speed,
                    math.sin(elevation) * speed,
                    math.sin(angle) * math.cos(elevation) * speed
                ],
                'life': 1.0
            }
            self.particles.append(particle)
    
    def toss_animation(self, direction):
        """Animate object being tossed"""
        # Add velocity in toss direction
        scale_factor = 5.0
        
        if direction == "RIGHT":
            self.target_position[0] += scale_factor
        elif direction == "LEFT":
            self.target_position[0] -= scale_factor
        elif direction == "UP":
            self.target_position[1] += scale_factor
        elif direction == "DOWN":
            self.target_position[1] -= scale_factor
        
        # Add spinning
        self.target_rotation[1] += 360
        
        # Emit many particles
        self.emit_particles(30)
    
    def crumple_animation(self):
        """Animate object being crumpled/deleted"""
        self.target_scale = 0.01
        self.emit_particles(50)


class VirtualTrashCan:
    """Virtual trash can for disposing objects"""
    
    def __init__(self):
        self.position = [5.0, -3.0, -8.0]
        self.scale = 1.5
        self.color = [1.0, 0.3, 0.3, 0.5]
        self.active = True
        self.glow = 0.0
        
    def draw(self):
        """Draw trash can"""
        if not self.active:
            return
        
        glDisable(GL_LIGHTING)
        glPushMatrix()
        
        glTranslatef(*self.position)
        glScalef(self.scale, self.scale, self.scale)
        
        # Pulsing glow
        glow_color = [
            self.color[0] + self.glow * 0.3,
            self.color[1],
            self.color[2],
            self.color[3]
        ]
        glColor4f(*glow_color)
        
        # Draw trash can icon
        glutWireCube(1.0)
        
        # X mark
        glBegin(GL_LINES)
        glVertex3f(-0.3, -0.3, 0.6)
        glVertex3f(0.3, 0.3, 0.6)
        glVertex3f(0.3, -0.3, 0.6)
        glVertex3f(-0.3, 0.3, 0.6)
        glEnd()
        
        glPopMatrix()
        glEnable(GL_LIGHTING)
    
    def update(self, dt):
        """Update trash can"""
        self.glow = (math.sin(time.time() * 3) + 1) / 2
    
    def check_collision(self, obj):
        """Check if object is in trash can"""
        dist = math.sqrt(
            (obj.position[0] - self.position[0])**2 +
            (obj.position[1] - self.position[1])**2 +
            (obj.position[2] - self.position[2])**2
        )
        return dist < 2.0


class EnhancedHUD:
    """Advanced HUD with more information"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.notification_queue = deque(maxlen=5)
        self.notification_timer = 0
        
    def add_notification(self, message, duration=2.0):
        """Add notification to queue"""
        self.notification_queue.append({
            'message': message,
            'duration': duration,
            'start_time': time.time()
        })
    
    def draw(self, frame, gesture, hands, fps, object_count):
        """Draw complete HUD"""
        # Scanlines
        frame = self.draw_scanlines(frame)
        
        # Corner brackets
        frame = self.draw_corner_brackets(frame)
        
        # Main info panel
        frame = self.draw_info_panel(frame, gesture, hands, fps, object_count)
        
        # Gesture visual feedback
        frame = self.draw_gesture_feedback(frame, gesture, hands)
        
        # Notifications
        frame = self.draw_notifications(frame)
        
        # JARVIS status
        frame = self.draw_jarvis_status(frame)
        
        return frame
    
    def draw_scanlines(self, frame):
        """Holographic scanlines"""
        for i in range(0, self.height, 3):
            alpha = 0.05
            overlay = frame.copy()
            cv2.line(overlay, (0, i), (self.width, i), (0, 255, 255), 1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame
    
    def draw_corner_brackets(self, frame):
        """Corner frame brackets"""
        color = (0, 255, 255)
        thickness = 2
        length = 60
        
        corners = [
            [(20, 20), (20 + length, 20), (20, 20 + length)],
            [(self.width - 20, 20), (self.width - 20 - length, 20), (self.width - 20, 20 + length)],
            [(20, self.height - 20), (20 + length, self.height - 20), (20, self.height - 20 - length)],
            [(self.width - 20, self.height - 20), (self.width - 20 - length, self.height - 20), 
             (self.width - 20, self.height - 20 - length)]
        ]
        
        for corner in corners:
            cv2.line(frame, corner[0], corner[1], color, thickness)
            cv2.line(frame, corner[0], corner[2], color, thickness)
        
        return frame
    
    def draw_info_panel(self, frame, gesture, hands, fps, object_count):
        """Main information panel"""
        y = 70
        color = (0, 255, 255)
        
        # System status
        cv2.putText(frame, "STARK INDUSTRIES - HOLOGRAPHIC INTERFACE", 
                   (20, y), self.font, 0.5, color, 1)
        y += 25
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y), self.font, 0.4, color, 1)
        y += 20
        
        # Object count
        cv2.putText(frame, f"OBJECTS: {object_count}", (20, y), self.font, 0.4, color, 1)
        y += 20
        
        # Hands detected
        cv2.putText(frame, f"HANDS: {len(hands)}", (20, y), self.font, 0.4, color, 1)
        y += 25
        
        # Gesture type
        if gesture['type'] != 'NONE':
            cv2.putText(frame, f"GESTURE: {gesture['type']}", 
                       (20, y), self.font, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def draw_gesture_feedback(self, frame, gesture, hands):
        """Visual feedback for gestures"""
        if not hands:
            return frame
        
        for hand in hands:
            # Draw hand center
            center_x = int(hand.center[0] * self.width)
            center_y = int(hand.center[1] * self.height)
            
            # Different colors for different states
            if hand.is_pinching:
                color = (0, 255, 0)  # Green for pinch
                radius = 20
            elif hand.gesture == "FIST":
                color = (0, 0, 255)  # Red for fist
                radius = 25
            elif hand.gesture == "OPEN":
                color = (255, 255, 0)  # Yellow for open
                radius = 30
            else:
                color = (0, 255, 255)  # Cyan for neutral
                radius = 15
            
            cv2.circle(frame, (center_x, center_y), radius, color, 2)
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        # Draw gesture-specific feedback
        if gesture['type'] == 'FLICK' and 'direction' in gesture['params']:
            direction = gesture['params']['direction']
            cv2.putText(frame, f"FLICK {direction}", 
                       (self.width // 2 - 100, self.height // 2), 
                       self.font, 1.0, (255, 255, 0), 3)
        
        return frame
    
    def draw_notifications(self, frame):
        """Draw notification messages"""
        y = self.height - 100
        current_time = time.time()
        
        # Filter active notifications
        active_notifications = [
            n for n in self.notification_queue 
            if current_time - n['start_time'] < n['duration']
        ]
        
        for notification in active_notifications:
            elapsed = current_time - notification['start_time']
            alpha = 1.0 - (elapsed / notification['duration'])
            
            color = (int(255 * alpha), int(255 * alpha), 0)
            
            cv2.putText(frame, notification['message'], 
                       (self.width - 400, y), self.font, 0.5, color, 2)
            y -= 25
        
        return frame
    
    def draw_jarvis_status(self, frame):
        """Draw JARVIS system status"""
        y = self.height - 30
        color = (0, 255, 255)
        
        status_text = "J.A.R.V.I.S. v4.2 - ONLINE - ALL SYSTEMS OPERATIONAL"
        cv2.putText(frame, status_text, (20, y), self.font, 0.4, color, 1)
        
        return frame


class EnhancedIronManSystem:
    """Enhanced Iron Man holographic system"""
    
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        
        # Initialize components
        self.gesture_recognizer = AdvancedGestureRecognizer()
        self.hud = EnhancedHUD(width, height)
        
        # Objects management
        self.objects = []
        self.selected_object = None
        self.object_id_counter = 0
        
        # Add initial object
        self.add_object("arc_reactor")
        
        # Virtual trash can
        self.trash_can = VirtualTrashCan()
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # OpenGL/Pygame setup
        pygame.init()
        self.display = (width, height)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("IRON MAN - Enhanced Holographic System")
        
        self.init_opengl()
        
        # Performance
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.dt = 0
        self.last_frame_time = time.time()
        
    def init_opengl(self):
        """Initialize OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glEnable(GL_POINT_SMOOTH)
        
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.4, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.0, 0.8, 1.0, 1.0])
    
    def add_object(self, obj_type):
        """Add new holographic object"""
        obj = HolographicObject3D(obj_type, self.object_id_counter)
        self.object_id_counter += 1
        
        # Offset position for multiple objects
        offset = len(self.objects) * 3
        obj.position[0] = offset - 3.0
        obj.target_position = obj.position.copy()
        
        self.objects.append(obj)
        self.hud.add_notification(f"OBJECT CREATED: {obj_type.upper()}")
        
        return obj
    
    def remove_object(self, obj):
        """Remove object"""
        if obj in self.objects:
            self.objects.remove(obj)
            self.hud.add_notification("OBJECT DELETED")
            if self.selected_object == obj:
                self.selected_object = None
    
    def find_closest_object(self, position):
        """Find object closest to position"""
        if not self.objects:
            return None
        
        min_dist = float('inf')
        closest_obj = None
        
        for obj in self.objects:
            # Project position to object space
            dist = abs(obj.position[0] - (position[0] - 0.5) * 10) + \
                   abs(obj.position[1] - (0.5 - position[1]) * 10)
            
            if dist < min_dist:
                min_dist = dist
                closest_obj = obj
        
        return closest_obj if min_dist < 3.0 else None
    
    def process_gestures(self, gesture, hands):
        """Process detected gestures"""
        if gesture['type'] == 'NONE':
            if self.selected_object:
                self.selected_object.selected = False
                self.selected_object = None
            return
        
        if gesture['type'] == 'ROTATE' and len(hands) == 1:
            # Single hand rotation
            hand = hands[0]
            
            if self.selected_object is None:
                self.selected_object = self.find_closest_object(hand.center)
                if self.selected_object:
                    self.selected_object.selected = True
            
            if self.selected_object:
                # Rotate based on hand position
                self.selected_object.target_rotation[0] = (hand.center[1] - 0.5) * 180
                self.selected_object.target_rotation[1] += (hand.center[0] - 0.5) * 3
        
        elif gesture['type'] == 'PINCH_DRAG' and len(hands) == 1:
            # Pinch to grab and drag
            hand = hands[0]
            
            if self.selected_object is None:
                self.selected_object = self.find_closest_object(hand.center)
                if self.selected_object:
                    self.selected_object.selected = True
            
            if self.selected_object:
                # Move object
                self.selected_object.target_position[0] = (hand.center[0] - 0.5) * 10
                self.selected_object.target_position[1] = (0.5 - hand.center[1]) * 10
        
        elif gesture['type'] == 'FLICK':
            # Flick to spin object
            if self.selected_object:
                direction = gesture['params']['direction']
                
                if direction == "LEFT":
                    self.selected_object.target_rotation[1] -= 45
                elif direction == "RIGHT":
                    self.selected_object.target_rotation[1] += 45
                elif direction == "UP":
                    self.selected_object.target_rotation[0] -= 45
                elif direction == "DOWN":
                    self.selected_object.target_rotation[0] += 45
                
                self.selected_object.emit_particles(15)
        
        elif gesture['type'] == 'TOSS':
            # Toss object toward trash
            if self.selected_object:
                direction = gesture['params']['direction']
                self.selected_object.toss_animation(direction)
                self.hud.add_notification("OBJECT TOSSED")
        
        elif gesture['type'] == 'TRANSLATE' and len(hands) == 2:
            # Two hand translation
            position = gesture['params']['position']
            
            if self.selected_object is None:
                self.selected_object = self.find_closest_object(position)
                if self.selected_object:
                    self.selected_object.selected = True
            
            if self.selected_object:
                self.selected_object.target_position[0] = (position[0] - 0.5) * 10
                self.selected_object.target_position[1] = (0.5 - position[1]) * 10
        
        elif gesture['type'] in ['EXPAND', 'CONTRACT']:
            # Two hand scaling
            if self.selected_object:
                scale_factor = gesture['params']['scale_factor']
                self.selected_object.target_scale *= scale_factor
                self.selected_object.target_scale = max(0.1, min(5.0, self.selected_object.target_scale))
                
                if abs(scale_factor - 1.0) > 0.05:
                    self.selected_object.emit_particles(8)
    
    def update_scene(self):
        """Update all objects in scene"""
        # Update objects
        for obj in self.objects[:]:
            obj.update(self.dt)
            
            # Check if object is in trash
            if self.trash_can.check_collision(obj) and obj.scale < 0.5:
                self.remove_object(obj)
        
        # Update trash can
        self.trash_can.update(self.dt)
    
    def render_scene(self):
        """Render 3D scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.08, 1.0)
        
        glLoadIdentity()
        gluLookAt(0, 0, 0, 0, 0, -5, 0, 1, 0)
        
        # Draw grid
        self.draw_grid()
        
        # Draw objects
        for obj in self.objects:
            obj.draw()
        
        # Draw trash can
        self.trash_can.draw()
    
    def draw_grid(self):
        """Draw holographic grid"""
        glDisable(GL_LIGHTING)
        glColor4f(0.0, 0.5, 0.7, 0.2)
        
        grid_size = 20
        spacing = 1.0
        
        glBegin(GL_LINES)
        for i in range(-grid_size, grid_size + 1):
            glVertex3f(i * spacing, -10, -20)
            glVertex3f(i * spacing, 10, -20)
            glVertex3f(-grid_size * spacing, i * spacing, -20)
            glVertex3f(grid_size * spacing, i * spacing, -20)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def capture_gl_frame(self):
        """Capture OpenGL frame"""
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        image = np.flipud(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    def run(self):
        """Main loop"""
        print("="*70)
        print("IRON MAN ENHANCED HOLOGRAPHIC INTERFACE SYSTEM")
        print("="*70)
        print("\n🎯 ADVANCED GESTURE CONTROLS:")
        print("\n  ONE HAND:")
        print("    • OPEN HAND + MOVE → Rotate object")
        print("    • PINCH + DRAG → Move object precisely")
        print("    • FLICK → Quick spin in direction")
        print("    • FIST + FAST MOVE → Toss object")
        print("\n  TWO HANDS:")
        print("    • BOTH OPEN + MOVE → Translate object")
        print("    • BOTH OPEN + SPREAD → Expand/scale up")
        print("    • BOTH OPEN + CONTRACT → Shrink/scale down")
        print("    • BOTH PINCH → Precision mode")
        print("\n⌨️  KEYBOARD:")
        print("    [1-5] Add Cube/Sphere/Torus/ArcReactor/Helmet")
        print("    [W] Toggle wireframe")
        print("    [D] Delete selected object")
        print("    [R] Reset all")
        print("    [Q] Quit")
        print("="*70)
        
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Calculate delta time
            current_time = time.time()
            self.dt = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_1:
                        self.add_object("cube")
                    elif event.key == pygame.K_2:
                        self.add_object("sphere")
                    elif event.key == pygame.K_3:
                        self.add_object("torus")
                    elif event.key == pygame.K_4:
                        self.add_object("arc_reactor")
                    elif event.key == pygame.K_5:
                        self.add_object("iron_man_helmet")
                    elif event.key == pygame.K_w:
                        if self.selected_object:
                            self.selected_object.wireframe = not self.selected_object.wireframe
                    elif event.key == pygame.K_d:
                        if self.selected_object:
                            self.selected_object.crumple_animation()
                    elif event.key == pygame.K_r:
                        self.objects.clear()
                        self.add_object("arc_reactor")
                        self.hud.add_notification("SYSTEM RESET")
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process hand tracking
            results, hands, gesture = self.gesture_recognizer.process_frame(frame)
            
            # Process gestures
            self.process_gestures(gesture, hands)
            
            # Update scene
            self.update_scene()
            
            # Render 3D
            self.render_scene()
            
            # Capture GL frame
            gl_frame = self.capture_gl_frame()
            
            # Blend
            alpha = 0.65
            blended = cv2.addWeighted(frame, 1 - alpha, gl_frame, alpha, 0)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.gesture_recognizer.mp_draw.draw_landmarks(
                        blended, hand_landmarks,
                        self.gesture_recognizer.mp_hands.HAND_CONNECTIONS,
                        self.gesture_recognizer.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                        self.gesture_recognizer.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
            
            # Draw HUD
            blended = self.hud.draw(blended, gesture, hands, self.fps, len(self.objects))
            
            # Display
            cv2.imshow("IRON MAN - Enhanced Holographic Interface", blended)
            pygame.display.flip()
            
            # FPS
            self.frame_count += 1
            if time.time() - self.start_time > 1.0:
                self.fps = self.frame_count / (time.time() - self.start_time)
                self.frame_count = 0
                self.start_time = time.time()
            
            clock.tick(60)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    try:
        system = EnhancedIronManSystem(width=1280, height=720)
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
