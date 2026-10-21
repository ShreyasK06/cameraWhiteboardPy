import cv2
import mediapipe as mp
import numpy as np
import math

class FingerWhiteboard:
    def __init__(self, open_camera=True, size=(1280, 720), keyboard_hints=True):
        # open_camera=False lets another frontend (the Streamlit app) feed frames in
        # through process_frame() instead of reading from a local webcam.
        self.width, self.height = size
        self.keyboard_hints = keyboard_hints  # False in the browser, where there are no key shortcuts

        # Initialize MediaPipe hands with optimized settings for better performance
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,  # Lowered for faster detection
            min_tracking_confidence=0.5,   # Lowered for better tracking
            model_complexity=0              # Use lighter model for speed
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize camera with optimized settings
        self.cap = None
        if open_camera:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 60)              # Request higher FPS
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # Reduce buffer for lower latency
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG for better performance

        # Canvas setup - dark modern background with subtle grid
        self.reset_canvas()
        
        # Modern vibrant color palette
        self.colors = [
            (67, 56, 202),   # Modern Purple
            (34, 197, 94),   # Emerald Green
            (239, 68, 68),   # Bright Red
            (251, 191, 36),  # Amber Yellow
            (236, 72, 153),  # Pink
            (14, 165, 233),  # Sky Blue
            (255, 255, 255), # Pure White
            (156, 163, 175)  # Cool Gray
        ]
        self.color_names = ["Purple", "Emerald", "Red", "Amber", "Pink", "Sky", "White", "Gray"]
        self.current_color_index = 0
        self.draw_color = self.colors[0]
        self.brush_thickness = 15  # Much bolder brush
        self.eraser_thickness = 40
        
        # Previous position for smooth drawing
        self.prev_x, self.prev_y = 0, 0
        
        # Gesture detection stability - reduced for faster response
        self.gesture_count = 0
        self.min_gesture_count = 2
        
        # Performance optimization variables
        self.frame_skip = 0
        self.process_every_n_frames = 2  # Process every 2nd frame for hand detection
        self.last_hand_position = None
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = cv2.getTickCount()
        
        # Modern UI elements
        self.ui_background_color = (30, 30, 30)      # Dark UI background
        self.ui_accent_color = (67, 56, 202)         # Purple accent
        self.ui_text_color = (255, 255, 255)         # White text
        self.ui_secondary_text = (156, 163, 175)     # Gray text
        
        # Animation states
        self.button_hover_scale = 1.0
        self.color_button_animations = [1.0] * len(self.colors)
        
        # Screenshot notification
        self.screenshot_notification = {"show": False, "timestamp": 0, "message": ""}
        
        # Current gesture state for UI
        self.current_gesture = "READY"
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def detect_finger_gestures(self, landmarks):
        """
        Detect finger gestures:
        - Thumb + Pointer finger touching: DRAW
        - Pointer + Middle finger touching: ERASE
        """
        # Get finger tip positions
        thumb_tip = landmarks[4]      # Thumb tip
        pointer_tip = landmarks[8]    # Index finger tip (pointer)
        middle_tip = landmarks[12]    # Middle finger tip
        
        # Calculate distances between fingertips
        pointer_middle_dist = self.calculate_distance(
            [pointer_tip.x, pointer_tip.y], 
            [middle_tip.x, middle_tip.y]
        )
        
        pointer_thumb_dist = self.calculate_distance(
            [pointer_tip.x, pointer_tip.y], 
            [thumb_tip.x, thumb_tip.y]
        )
        
        # Drawing gesture: Thumb and pointer finger touching
        if pointer_thumb_dist < 0.05:
            return "DRAW"
        
        # Erasing gesture: Pointer and middle finger touching
        elif pointer_middle_dist < 0.04:
            return "ERASE"
        
        return "NONE"
    
    def calculate_fps(self):
        """Calculate and return current FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Calculate FPS every 30 frames
            current_time = cv2.getTickCount()
            time_diff = (current_time - self.fps_start_time) / cv2.getTickFrequency()
            fps = self.fps_counter / time_diff
            self.fps_counter = 0
            self.fps_start_time = current_time
            return fps
        return None
    
    def draw_glow_cursor(self, frame, x, y, color, mode):
        """Draw modern cursor with glow effect"""
        if mode == "draw":
            # Draw cursor with pulsing glow
            for radius in [20, 15, 10, 5]:
                alpha = 0.3 if radius == 20 else 0.7
                cv2.circle(frame, (x, y), radius, color, 2 if radius > 10 else -1)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
            
        elif mode == "erase":
            # Eraser cursor with dashed circle
            for i in range(0, 360, 30):
                angle1 = math.radians(i)
                angle2 = math.radians(i + 15)
                pt1 = (int(x + self.eraser_thickness * math.cos(angle1)), 
                       int(y + self.eraser_thickness * math.sin(angle1)))
                pt2 = (int(x + self.eraser_thickness * math.cos(angle2)), 
                       int(y + self.eraser_thickness * math.sin(angle2)))
                cv2.line(frame, pt1, pt2, color, 3)
            cv2.circle(frame, (x, y), 5, color, -1)
            
        else:  # idle
            cv2.circle(frame, (x, y), 12, color, 2)
            cv2.circle(frame, (x, y), 4, color, -1)
    
    def reset_canvas(self):
        """Blank the canvas (dark charcoal) and redraw the grid"""
        self.canvas = np.full((self.height, self.width, 3), 20, dtype=np.uint8)
        self.add_grid_pattern()

    def add_grid_pattern(self):
        """Add subtle grid pattern to canvas"""
        grid_color = (25, 25, 25)  # Slightly lighter than background
        grid_spacing = 50

        # Draw vertical lines
        for x in range(0, self.width, grid_spacing):
            cv2.line(self.canvas, (x, 0), (x, self.height), grid_color, 1)

        # Draw horizontal lines
        for y in range(0, self.height, grid_spacing):
            cv2.line(self.canvas, (0, y), (self.width, y), grid_color, 1)
    
    def draw_modern_ui(self, frame):
        """Draw modern UI elements with sleek design"""
        h, w = frame.shape[:2]
        
        # Draw modern header bar with better spacing
        header_height = 100
        cv2.rectangle(frame, (0, 0), (w, header_height), self.ui_background_color, -1)
        
        # Gradient effect for header
        for i in range(5):
            alpha = 0.2 - (i * 0.03)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, header_height - i), (w, header_height + 5 - i), self.ui_accent_color, -1)
            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        # App title with modern font - positioned better
        cv2.putText(frame, "AI WHITEBOARD", (20, 40), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, self.ui_text_color, 2)
        
        # Add subtitle with instructions
        subtitle = ("Thumb+Pointer: Draw | Pointer+Middle: Erase | 1-8: Colors | S: Screenshot | C: Clear | Q: Quit"
                    if self.keyboard_hints else "Thumb+Pointer: Draw | Pointer+Middle: Erase")
        cv2.putText(frame, subtitle,
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_secondary_text, 1)
        
        # Status indicator with better positioning
        status_color = (34, 197, 94) if hasattr(self, 'last_hand_position') and self.last_hand_position else (239, 68, 68)
        cv2.circle(frame, (w - 40, 35), 10, status_color, -1)
        cv2.circle(frame, (w - 40, 35), 12, self.ui_text_color, 2)
        
        # Status text
        status_text = "ONLINE" if hasattr(self, 'last_hand_position') and self.last_hand_position else "OFFLINE"
        cv2.putText(frame, status_text, (w - 100, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Draw color palette with modern cards - positioned below header
        self.draw_color_palette(frame)
        
        # Draw gesture status panel
        self.draw_status_panel(frame)
        
        # Show screenshot notification if active
        if self.screenshot_notification["show"]:
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - self.screenshot_notification["timestamp"] < 3.0:  # Show for 3 seconds
                # Draw notification
                msg = self.screenshot_notification["message"]
                text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
                x = (w - text_size[0]) // 2
                y = 150
                
                # Background for notification
                cv2.rectangle(frame, (x - 20, y - 30), (x + text_size[0] + 20, y + 10), (34, 197, 94), -1)
                cv2.rectangle(frame, (x - 20, y - 30), (x + text_size[0] + 20, y + 10), (255, 255, 255), 2)
                
                # Notification text
                cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            else:
                self.screenshot_notification["show"] = False
        
        return frame
    
    def show_screenshot_notification(self, message):
        """Show a temporary notification for screenshot"""
        self.screenshot_notification = {
            "show": True,
            "timestamp": cv2.getTickCount() / cv2.getTickFrequency(),
            "message": message
        }
    
    def draw_color_palette(self, frame):
        """Draw modern color palette"""
        h, w = frame.shape[:2]
        start_x = 200
        card_width = 50
        card_height = 50
        margin = 8
        
        for i, color in enumerate(self.colors):
            x = start_x + i * (card_width + margin)
            y = 110  # Position below the header
            
            # Animate selected color
            if i == self.current_color_index:
                self.color_button_animations[i] = min(1.2, self.color_button_animations[i] + 0.1)
            else:
                self.color_button_animations[i] = max(1.0, self.color_button_animations[i] - 0.05)
            
            scale = self.color_button_animations[i]
            scaled_width = int(card_width * scale)
            scaled_height = int(card_height * scale)
            
            # Draw shadow
            shadow_offset = 3
            cv2.rectangle(frame, 
                         (x + shadow_offset, y + shadow_offset), 
                         (x + scaled_width + shadow_offset, y + scaled_height + shadow_offset), 
                         (0, 0, 0), -1)
            
            # Draw color card
            cv2.rectangle(frame, (x, y), (x + scaled_width, y + scaled_height), color, -1)
            
            # Draw selection indicator
            if i == self.current_color_index:
                cv2.rectangle(frame, (x - 3, y - 3), (x + scaled_width + 3, y + scaled_height + 3), 
                             self.ui_text_color, 3)
                # Draw number indicator
                cv2.putText(frame, str(i + 1), (x + 20, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ui_text_color, 2)
            else:
                cv2.rectangle(frame, (x, y), (x + scaled_width, y + scaled_height), 
                             self.ui_secondary_text, 2)
    
    def draw_status_panel(self, frame):
        """Draw modern status panel"""
        h, w = frame.shape[:2]
        
        # Status panel background
        panel_height = 120
        panel_y = h - panel_height
        
        # Create gradient background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), self.ui_background_color, -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Current gesture status
        gesture_text = "🤚 Ready"
        gesture_color = self.ui_secondary_text
        
        if hasattr(self, 'current_gesture'):
            if self.current_gesture == "DRAW":
                gesture_text = "✏️ DRAWING"
                gesture_color = (34, 197, 94)
            elif self.current_gesture == "ERASE":
                gesture_text = "🗑️ ERASING"
                gesture_color = (239, 68, 68)
            elif self.current_gesture == "TRACKING":
                gesture_text = "👁️ TRACKING"
                gesture_color = (251, 191, 36)
            elif self.current_gesture == "NO_HAND":
                gesture_text = "❌ NO HAND"
                gesture_color = (239, 68, 68)
        
        cv2.putText(frame, gesture_text, (20, panel_y + 30), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, gesture_color, 2)
        
        # Current color and brush size
        cv2.putText(frame, f"Color: {self.color_names[self.current_color_index]}", 
                   (20, panel_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_text_color, 1)
        cv2.putText(frame, f"Brush: {self.brush_thickness}px", 
                   (20, panel_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_text_color, 1)
        
        # Instructions on the right
        instructions = [
            "👍 Thumb + Pointer: Draw",
            "✌️ Pointer + Middle: Erase", 
            "1-8: Colors • +/-: Brush • C: Clear" if self.keyboard_hints else "Colors and brush: use the sidebar"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (w - 400, panel_y + 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_secondary_text, 1)
    
    def process_frame(self, frame):
        """Track the hand in one BGR frame, update the canvas, and return the composited frame"""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Skip hand detection processing for some frames to improve FPS
        self.frame_skip += 1
        results = None
        
        if self.frame_skip >= self.process_every_n_frames:
            self.frame_skip = 0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process frame with MediaPipe
            results = self.hands.process(frame_rgb)
        
        # Draw modern UI
        frame = self.draw_modern_ui(frame)
        
        # Calculate and display FPS
        current_fps = self.calculate_fps()
        if current_fps is not None:
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        current_gesture = "NONE"
        
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw simplified hand landmarks for better performance
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
                )
                
                # Detect gesture
                gesture = self.detect_finger_gestures(hand_landmarks.landmark)
                current_gesture = gesture
                self.current_gesture = gesture  # Store for UI display
                
                # Get pointer finger tip position for drawing/erasing
                pointer_tip = hand_landmarks.landmark[8]
                x = int(pointer_tip.x * frame.shape[1])
                y = int(pointer_tip.y * frame.shape[0])
                
                # Store hand position for interpolation between frames
                self.last_hand_position = (x, y)
                
                # Modern gesture visualization with glow effects
                if gesture == "DRAW":
                    # Draw cursor with glow effect
                    self.draw_glow_cursor(frame, x, y, self.draw_color, "draw")
                    
                    # Draw on canvas with smooth, anti-aliased lines
                    if self.prev_x != 0 and self.prev_y != 0:
                        # Main line with anti-aliasing
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), 
                               self.draw_color, self.brush_thickness, cv2.LINE_AA)
                        # Soft glow effect
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), 
                               self.draw_color, self.brush_thickness + 4, cv2.LINE_AA)
                    else:
                        # Start with smooth circle
                        cv2.circle(self.canvas, (x, y), self.brush_thickness//2, self.draw_color, cv2.FILLED, cv2.LINE_AA)
                    self.prev_x, self.prev_y = x, y
                    
                elif gesture == "ERASE":
                    # Eraser cursor with modern styling
                    self.draw_glow_cursor(frame, x, y, (239, 68, 68), "erase")
                    
                    # Erase with smooth circular brush
                    cv2.circle(self.canvas, (x, y), self.eraser_thickness, (20, 20, 20), cv2.FILLED, cv2.LINE_AA)
                    self.prev_x, self.prev_y = 0, 0
                    
                else:
                    # Idle cursor
                    self.draw_glow_cursor(frame, x, y, (156, 163, 175), "idle")
                    self.prev_x, self.prev_y = 0, 0
                
                # Gesture status is now handled in the modern UI panel
        elif self.last_hand_position and not results:
            # Use last known hand position when skipping frames for better responsiveness
            x, y = self.last_hand_position
            self.draw_glow_cursor(frame, x, y, (128, 128, 128), "idle")
            self.current_gesture = "TRACKING"
        else:
            # No hand detected
            self.prev_x, self.prev_y = 0, 0
            self.last_hand_position = None
            self.current_gesture = "NO_HAND"
        
        # Combine camera feed with canvas using optimized blending
        # Make camera feed more transparent so drawing is more visible
        combined = cv2.addWeighted(frame, 0.4, self.canvas, 0.6, 0)
        return combined

    def run(self):
        """Main application loop"""
        print("Starting Finger Whiteboard...")
        print("Make sure your camera is working and your hand is visible.")
        print("\nControls:")
        print("- Touch thumb and pointer finger together: Draw")
        print("- Touch pointer and middle finger together: Erase")
        print("- Press 1-8 to change colors")
        print("- Press +/- to change brush size")
        print("- Press 'c' to clear canvas")
        print("- Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from camera")
                break

            combined = self.process_frame(frame)
            
            # Show the result with modern window title
            cv2.imshow('🎨 AI Whiteboard 2.0 - Modern Hand Gesture Control', combined)
            
            # Handle keyboard input with minimal wait time for better responsiveness
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting application...")
                break
            elif key == ord('c'):
                self.reset_canvas()
                print("Canvas cleared!")
            elif key == ord('s'):
                # Take screenshot of the entire application window
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_filename = f'whiteboard_screenshot_{timestamp}.png'
                canvas_filename = f'whiteboard_canvas_{timestamp}.png'
                
                # Save the full window (what the user sees)
                cv2.imwrite(screenshot_filename, combined)
                # Also save just the canvas/drawing
                cv2.imwrite(canvas_filename, self.canvas)
                
                print(f"Screenshot saved as '{screenshot_filename}'")
                print(f"Canvas saved as '{canvas_filename}'")
                
                # Show notification
                self.show_screenshot_notification("📸 Screenshot Saved!")
                
                # Brief flash effect
                flash_overlay = combined.copy()
                cv2.rectangle(flash_overlay, (0, 0), (combined.shape[1], combined.shape[0]), (255, 255, 255), 20)
                cv2.imshow('Modern AI Whiteboard', flash_overlay)
                cv2.waitKey(100)
            # Color selection with number keys 1-8
            elif key >= ord('1') and key <= ord('8'):
                color_index = key - ord('1')
                if color_index < len(self.colors):
                    self.current_color_index = color_index
                    self.draw_color = self.colors[color_index]
                    print(f"Color changed to: {self.color_names[color_index]}")
            elif key == ord('+') or key == ord('='):
                self.brush_thickness = min(self.brush_thickness + 2, 30)
                print(f"Brush size increased to: {self.brush_thickness}")
            elif key == ord('-'):
                self.brush_thickness = max(self.brush_thickness - 2, 5)
                print(f"Brush size decreased to: {self.brush_thickness}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Finger Whiteboard closed successfully!")

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "🎨 AI WHITEBOARD 2.0" + " " * 22 + "║")
    print("╠" + "═" * 58 + "╣")
    print("║ Modern hand gesture-controlled digital whiteboard      ║")
    print("║ Powered by MediaPipe & OpenCV                          ║")
    print("╠" + "═" * 58 + "╣")
    print("║ GESTURE CONTROLS:                                      ║")
    print("║ 👍 Thumb + Pointer finger → DRAW                      ║")
    print("║ ✌️  Pointer + Middle finger → ERASE                   ║")
    print("║                                                        ║")
    print("║ KEYBOARD SHORTCUTS:                                    ║")
    print("║ • 1-8: Colors (Purple, Emerald, Red, Amber...)        ║")
    print("║ • +/-: Brush size • C: Clear • S: Save • Q: Quit      ║")
    print("║                                                        ║")
    print("║ FEATURES:                                              ║")
    print("║ • 60 FPS optimized performance                         ║")
    print("║ • Modern dark theme with vibrant colors               ║")
    print("║ • Smooth anti-aliased drawing                          ║")
    print("║ • Real-time gesture recognition                        ║")
    print("╚" + "═" * 58 + "╝")
    
    try:
        whiteboard = FingerWhiteboard()
        whiteboard.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure your camera is connected and try again.")

if __name__ == "__main__":
    main()
