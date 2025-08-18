import sys
print(sys.executable)  # Prints the Python interpreter path being used
import asyncio
import logging
import math
import json
import cv2  # Import OpenCV for camera capture
import numpy as np  # Import numpy for array manipulation
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import VideoStreamTrack, AudioStreamTrack  # Import VideoStreamTrack and AudioStreamTrack
from av import VideoFrame, AudioFrame  # Import VideoFrame and AudioFrame for aiortc processing
import websockets  # Import the websockets library
import re  # Import regex for parsing candidate string
from ultralytics import YOLO
import time
import board
import busio
import adafruit_mlx90640
import pyaudio  # For audio capture
from fractions import Fraction  # Import Fraction for time_base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioStreamTrack(AudioStreamTrack):
    """Optimized audio stream track for capturing microphone audio"""
    
    def __init__(self, sample_rate=48000, channels=2):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = 960  # Reduced chunk size for lower latency (20ms at 48kHz)
        
        # PyAudio configuration
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        self._initialize_audio()
        
        # Frame timing
        self.samples_per_frame = 960  # 20ms frames at 48kHz
        self.frame_duration = self.samples_per_frame / self.sample_rate  # 0.02 seconds
        self.start_time = time.time()
        self.frame_count = 0
        
        # Timestamp tracking
        self._timestamp = 0
        self._time_base = Fraction(1, self.sample_rate)  # Time base as Fraction (e.g., 1/48000)
        
        logger.info(f"AudioStreamTrack initialized: {self.sample_rate}Hz, {self.channels} channels")

    def _initialize_audio(self):
        """Initialize audio capture with optimized parameters"""
        try:
            # Find the best audio input device
            default_device = self.audio.get_default_input_device_info()
            logger.info(f"Default audio device: {default_device['name']}")
            
            # Configure audio stream for low latency
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=None,
                input_device_index=default_device['index']
            )
            
            logger.info("Audio stream initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            raise

    async def next_timestamp(self):
        """Generate the next timestamp and time base for audio frames"""
        pts = self._timestamp
        self._timestamp += self.samples_per_frame  # Increment by samples per frame
        return pts, self._time_base

    async def recv(self):
        """Receive audio frames with precise timing"""
        try:
            # Calculate expected time for the next frame
            self.frame_count += 1
            expected_time = self.start_time + (self.frame_count * self.frame_duration)
            current_time = time.time()
            
            # Sleep to align with the expected frame time
            sleep_duration = expected_time - current_time
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            
            pts, time_base = await self.next_timestamp()
            
            # Read audio data
            try:
                audio_data = self.stream.read(self.samples_per_frame, exception_on_overflow=False)
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Ensure correct shape for packed layout (1, samples * channels)
                expected_length = self.samples_per_frame * self.channels
                if audio_array.size != expected_length:
                    logger.warning(f"Audio data size mismatch: got {audio_array.size}, expected {expected_length}")
                    # Pad or truncate to correct size
                    if audio_array.size < expected_length:
                        audio_array = np.pad(audio_array, (0, expected_length - audio_array.size), mode='constant')
                    else:
                        audio_array = audio_array[:expected_length]
                
                # Reshape to (1, samples * channels) for packed layout
                audio_array = audio_array.reshape(1, -1)
                
                # Create AudioFrame
                audio_frame = AudioFrame.from_ndarray(
                    audio_array,
                    format="s16",
                    layout="stereo" if self.channels == 2 else "mono"
                )
                audio_frame.sample_rate = self.sample_rate
                audio_frame.pts = pts
                audio_frame.time_base = time_base
                
                return audio_frame
                
            except Exception as e:
                logger.error(f"Error reading audio data: {e}")
                # Return last known good frame or silence if none available
                silence = np.zeros((1, self.samples_per_frame * self.channels), dtype=np.int16)
                audio_frame = AudioFrame.from_ndarray(
                    silence,
                    format="s16",
                    layout="stereo" if self.channels == 2 else "mono"
                )
                audio_frame.sample_rate = self.sample_rate
                audio_frame.pts = pts
                audio_frame.time_base = time_base
                return audio_frame
                
        except Exception as e:
            logger.error(f"Error in audio recv(): {e}")
            # Return silence frame with correct shape
            silence = np.zeros((1, self.samples_per_frame * self.channels), dtype=np.int16)
            audio_frame = AudioFrame.from_ndarray(
                silence,
                format="s16",
                layout="stereo" if self.channels == 2 else "mono"
            )
            audio_frame.sample_rate = self.sample_rate
            audio_frame.pts = pts
            audio_frame.time_base = time_base
            return audio_frame

    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        logger.info("Audio resources cleaned up")

    def __del__(self):
        self.cleanup()

class ThermalVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.width = 320
        self.height = 240
        self._initialize_thermal_camera()
        
        # Frame rate control
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 8  # 8 FPS for thermal camera
        
        logger.info("ThermalVideoStreamTrack initialized.")

    def _initialize_thermal_camera(self):
        """Initialize MLX90640 thermal camera"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Initialize I2C bus
                self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
                
                # Initialize MLX90640 camera
                self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
                logger.info(f"MLX90640 detected, serial number: {[hex(i) for i in self.mlx.serial_number]}")
                
                # Set refresh rate
                self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
                
                # Create a frame buffer to store thermal data
                self.thermal_frame = np.zeros((24 * 32,), dtype=float)
                
                logger.info(f"Thermal camera initialized successfully on attempt {attempt + 1}")
                return
            except Exception as e:
                logger.error(f"Thermal camera initialization error on attempt {attempt + 1}: {e}")
                time.sleep(0.5)
        
        raise IOError("Cannot initialize thermal camera after multiple attempts. Please ensure MLX90640 is connected.")

    def thermal_to_colormap(self, thermal_data):
        """Convert thermal data to displayable color image"""
        # Calculate dynamic range
        vmin = np.percentile(thermal_data, 5)
        vmax = np.percentile(thermal_data, 95)
        
        # Scale to 0-1 range
        scaled = np.clip((thermal_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
        
        # Convert to 8-bit
        thermal_8bit = (scaled * 255).astype(np.uint8)
        
        # Resize to desired output size
        thermal_resized = cv2.resize(thermal_8bit, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # Apply colormap
        thermal_color = cv2.applyColorMap(thermal_resized, cv2.COLORMAP_INFERNO)
        
        return thermal_color

    async def recv(self):
        """Receive and process thermal video frames"""
        try:
            # Frame rate control
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                await asyncio.sleep(self.frame_interval - (current_time - self.last_frame_time))
            
            self.last_frame_time = time.time()
            
            pts, time_base = await self.next_timestamp()

            # Read thermal frame
            try:
                self.mlx.getFrame(self.thermal_frame)
                thermal_data = np.array(self.thermal_frame).reshape((24, 32))
                
                # Convert to color image
                frame = self.thermal_to_colormap(thermal_data)
                
                # Add temperature info overlay
                avg_temp = np.mean(thermal_data)
                min_temp = np.min(thermal_data)
                max_temp = np.max(thermal_data)
                
                # Add "THERMAL" identifier in top-left corner
                cv2.putText(frame, "THERMAL", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Avg: {avg_temp:.1f}C", (10, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Min: {min_temp:.1f}C", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Max: {max_temp:.1f}C", (10, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Add thermal-specific visual marker (orange border)
                cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 165, 255), 3)
                
            except Exception as e:
                logger.error(f"Error reading thermal frame: {e}")
                # Return a black frame with error indicator
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
                cv2.putText(frame, "Thermal Error", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Convert BGR to RGB for WebRTC
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base

            return video_frame
            
        except Exception as e:
            logger.error(f"Error in thermal recv(): {e}")
            # Return a black frame with marker in case of error
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

class CameraVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.width = 320  # Reduced for lower latency
        self.height = 240  # Reduced for lower latency
        self._initialize_camera()
        
        # Load YOLOv8 model (consider exporting to NCNN for further optimization: model.export(format='ncnn'); then load the exported model)
        try:
            self.model = YOLO("yolov8n.pt")
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
        self.target_class_ids = {0: "person", 56: "chair", 60: "dining table"}
        
        # Motion tracking parameters (simplified)
        self.motion_threshold = 10  # Increased slightly to reduce false positives
        self.dead_time_seconds = 5
        self.human_tracker = {}
        self.prev_gray = None
        
        # Frame rate control
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 15  # Reduced to 15 FPS for lower latency
        
        # Detection skipping for optimization
        self.frame_counter = 0
        self.last_results = None  # Cache last YOLO results
        self.last_humans_detected = 0
        self.last_possibly_dead = 0
        
        logger.info("CameraVideoStreamTrack with YOLOv8 initialized.")

    def _initialize_camera(self):
        """Initialize camera with optimized GStreamer pipeline for Raspberry Pi Camera v2"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.cap:
                    self.cap.release()
                    time.sleep(0.5)  # Reduced delay
                    logger.info("Previous camera instance released")
                    
                # Optimized GStreamer pipeline for low latency
                pipeline = (
                    f"libcamerasrc ! "
                    f"video/x-raw, width={self.width}, height={self.height}, format=YUY2, framerate=15/1 ! "
                    f"queue max-size-buffers=1 leaky=downstream ! "
                    f"videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink sync=false drop=true max-buffers=1"
                )
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    logger.info(f"Camera initialized successfully on attempt {attempt + 1}")
                    return
                else:
                    logger.warning(f"Failed to open camera pipeline on attempt {attempt + 1}")
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Camera initialization error on attempt {attempt + 1}: {e}")
                time.sleep(0.5)
        
        raise IOError("Cannot open camera pipeline after multiple attempts. Please ensure camera is connected and libcamera is configured.")

    def cleanup(self):
        """Clean up camera resources"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            time.sleep(0.5)  # Reduced delay
            logger.info("Camera released.")
        self.cap = None

    def __del__(self):
        self.cleanup()
        
    async def recv(self):
        """Receive and process video frames"""
        try:
            # Frame rate control
            current_time = time.time()
            delta = current_time - self.last_frame_time
            if delta < self.frame_interval:
                await asyncio.sleep(self.frame_interval - delta)
            
            self.last_frame_time = current_time
            
            pts, time_base = await self.next_timestamp()
            timestamp = time.time()

            # Check if camera is still open
            if not self.cap or not self.cap.isOpened():
                logger.warning("Camera not available, reinitializing...")
                self._initialize_camera()

            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from camera, returning black frame with marker.")
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                # Draw a red square to distinguish from rendering issues
                cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
            else:
                frame = self._process_frame(frame, timestamp)

            # Convert BGR to RGB for WebRTC
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base

            return video_frame
            
        except Exception as e:
            logger.error(f"Error in recv(): {e}")
            # Return a black frame with marker in case of error
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

    def _process_frame(self, frame, timestamp):
        """Process frame with optimized YOLO detection and motion tracking"""
        # Draw a green square to confirm frame processing
        cv2.rectangle(frame, (50, 50), (100, 100), (0, 255, 0), 2)

        # Convert to grayscale for motion tracking (no blur for speed)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        humans_detected = self.last_humans_detected
        possibly_dead = self.last_possibly_dead
        new_tracker = self.human_tracker.copy()

        self.frame_counter += 1
        if self.frame_counter % 3 == 0:  # Run YOLO every 3 frames for lower latency
            # Run optimized YOLOv8 detection
            results = self.model(frame, verbose=False, imgsz=320, conf=0.5)[0]
            self.last_results = results
            humans_detected = 0
            possibly_dead = 0
            new_tracker = {}

            for box in results.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                class_id = int(cls)
                if class_id not in self.target_class_ids:
                    continue

                label_name = self.target_class_ids[class_id]
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                color = (255, 255, 255)

                if class_id == 0:  # Person: motion tracking
                    humans_detected += 1
                    roi_now = gray[y1:y2, x1:x2]
                    person_id = f"{x1}-{y1}-{x2}-{y2}"

                    moved = False
                    if self.prev_gray is not None and roi_now.shape == self.prev_gray[y1:y2, x1:x2].shape:
                        roi_prev = self.prev_gray[y1:y2, x1:x2]
                        moved = self.is_moving(roi_now, roi_prev)

                    last_move_time = self.human_tracker.get(person_id, 0)
                    if moved:
                        last_move_time = timestamp

                    new_tracker[person_id] = last_move_time
                    time_inactive = timestamp - last_move_time
                    is_dead = time_inactive > self.dead_time_seconds

                    label = "Dead" if is_dead else ("Alive" if moved else "No Motion")
                    color = (0, 0, 255) if is_dead else ((0, 255, 255) if not moved else (0, 255, 0))

                    if is_dead:
                        possibly_dead += 1
                else:
                    label = label_name
                    color = (0, 128, 255)

                # Draw detection and label on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            self.last_humans_detected = humans_detected
            self.last_possibly_dead = possibly_dead
        elif self.last_results:  # Reuse last detections for faster frames
            # Redraw cached detections (simplified, no motion update)
            for box in self.last_results.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                class_id = int(cls)
                if class_id not in self.target_class_ids:
                    continue

                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                label = self.target_class_ids[class_id]
                color = (0, 255, 0) if class_id == 0 else (0, 128, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display stats on frame
        cv2.putText(frame, "RGB + YOLO", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Humans: {humans_detected}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Possibly Dead: {possibly_dead}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 15
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add RGB-specific visual marker (green border)
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 0), 3)

        self.prev_gray = gray
        self.human_tracker = new_tracker
        
        return frame

    def is_moving(self, current_roi, previous_roi):
        diff = cv2.absdiff(current_roi, previous_roi)
        _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        return np.sum(motion_mask > 0) > 50  # Simplified check

async def run(offer, pc, camera_track, thermal_track, audio_track):
    @pc.on("track")
    def on_track(track):
        logger.info(f"Received track: {track.kind}")

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        logger.info(f"ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState in ["failed", "disconnected"]:
            logger.warning(f"ICE connection issue: {pc.iceConnectionState}")

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            logger.error("WebRTC connection failed")

    try:
        # Add RGB camera track first (will be track 0)
        if camera_track:
            pc.addTrack(camera_track)
            logger.info("RGB camera track added to peer connection (Track 0)")
            
        # Add thermal camera track second (will be track 1)
        if thermal_track:
            pc.addTrack(thermal_track)
            logger.info("Thermal camera track added to peer connection (Track 1)")
            
        # Add audio track third (will be track 2)
        if audio_track:
            pc.addTrack(audio_track)
            logger.info("Audio track added to peer connection (Track 2)")

        if not offer:
            logger.error("Invalid offer: Empty SDP")
            return None

        logger.info(f"Processing offer SDP: {offer}")
        if "m=video" not in offer:
            logger.error("Invalid offer SDP: Missing video media line")
            return None

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer, type="offer"))
        logger.info("Set remote description (offer)")
        
        answer = await pc.createAnswer()
        if not answer:
            logger.error("Failed to create answer")
            return None
        
        await pc.setLocalDescription(answer)
        logger.info(f"Set local description (answer)")
        logger.info(f"Answer SDP contains {answer.sdp.count('m=video')} video tracks and {answer.sdp.count('m=audio')} audio tracks")
        return pc.localDescription
    except Exception as e:
        logger.error(f"Error in run: {str(e)}")
        return None

def parse_ice_candidate_string(candidate_str):
    """
    Parses a raw ICE candidate string into its components required by RTCIceCandidate constructor.
    """
    if not candidate_str.startswith("candidate:"):
        raise ValueError(f"Invalid ICE candidate string format: does not start with 'candidate:': {candidate_str}")
    
    # Split the string after "candidate:"
    parts = candidate_str[len("candidate:"):].split()

    # Ensure enough core parts are present
    if len(parts) < 8:
        raise ValueError(f"Invalid ICE candidate string format: not enough core parts: {candidate_str}")

    foundation = parts[0]
    component = int(parts[1])
    protocol = parts[2]
    priority = int(parts[3])
    ip = parts[4]
    port = int(parts[5])
    
    # 'typ' is at index 6, the actual candidate type is at index 7
    candidate_type = parts[7]

    related_address = None
    related_port = None

    # Parse optional attributes like raddr and rport
    i = 8
    while i < len(parts):
        if parts[i] == "raddr" and i + 1 < len(parts):
            related_address = parts[i+1]
            i += 2
        elif parts[i] == "rport" and i + 1 < len(parts):
            related_port = int(parts[i+1])
            i += 2
        else:
            i += 1

    return {
        "foundation": foundation,
        "component": component,
        "protocol": protocol,
        "priority": priority,
        "ip": ip,
        "port": port,
        "type": candidate_type,
        "relatedAddress": related_address,
        "relatedPort": related_port
    }

async def main():
    """
    Main asynchronous function to set up and run the WebRTC streamer with audio.
    """
    uri = "wss://websockettest-eggy.onrender.com"
    peer_id = "python-peer"
    
    while True:  # Main reconnection loop
        websocket = None
        pc = None
        camera_track = None
        thermal_track = None
        audio_track = None
        
        try:
            # Create new peer connection for each attempt
            pc = RTCPeerConnection()
            
            # Initialize camera tracks
            try:
                camera_track = CameraVideoStreamTrack()
                logger.info("RGB camera track initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RGB camera track: {e}")
                continue
                
            try:
                thermal_track = ThermalVideoStreamTrack()
                logger.info("Thermal camera track initialized")
            except Exception as e:
                logger.error(f"Failed to initialize thermal camera track: {e}")
                continue
                
            # Initialize audio track
            try:
                audio_track = AudioStreamTrack()
                logger.info("Audio track initialized")
            except Exception as e:
                logger.error(f"Failed to initialize audio track: {e}")
                continue

            @pc.on("datachannel")
            def on_datachannel(channel):
                logger.info(f"Data channel '{channel.label}' received from remote peer.")

                @channel.on("message")
                def on_message(message):
                    logger.info(f"Data channel message received: {message}")

                @channel.on("open")
                def on_open():
                    logger.info(f"Data channel '{channel.label}' opened.")

                @channel.on("close")
                def on_close():
                    logger.info(f"Data channel '{channel.label}' closed.")

            @pc.on("icecandidate")
            async def on_ice_candidate(candidate):
                if candidate and websocket and not websocket.closed:
                    logger.info(f"Generated ICE candidate: {candidate.candidate}")
                    candidate_message_data = {
                        "SdpMid": candidate.sdpMid,
                        "SdpMLineIndex": candidate.sdpMLineIndex,
                        "Candidate": candidate.candidate
                    }
                    full_candidate_message = "CANDIDATE!" + json.dumps(candidate_message_data)
                    try:
                        await websocket.send(full_candidate_message)
                        logger.info("Sent ICE candidate to signaling server")
                    except Exception as e:
                        logger.error(f"Failed to send ICE candidate: {e}")

            async with websockets.connect(uri) as ws_conn:
                websocket = ws_conn
                logger.info(f"Connected to signaling server at {uri}")
                
                register_message = json.dumps({"type": "register", "peer_id": peer_id})
                await websocket.send(register_message)
                logger.info(f"Sent registration: {register_message}")
                
                # Send a notification to web clients that python-peer is connected
                notification_message = json.dumps({"type": "peer_connected", "peer_id": peer_id})
                await websocket.send(notification_message)
                logger.info("Sent peer connection notification to signaling server")

                while True:
                    try:
                        message = await websocket.recv()
                        message = message.decode('utf-8')
                        logger.info(f"Raw message received: {message}")
                        
                        if message.startswith("OFFER!"):
                            json_str = message[len("OFFER!"):]
                            try:
                                data = json.loads(json_str)
                                logger.info("Received offer from client")
                                offer_sdp = data["Sdp"]
                                
                                answer = await run(offer_sdp, pc, camera_track, thermal_track, audio_track)
                                answer_message_data = {
                                    "SessionType": answer.type.capitalize(),
                                    "Sdp": answer.sdp
                                }
                                full_answer_message = "ANSWER!" + json.dumps(answer_message_data)
                                await websocket.send(full_answer_message)
                                logger.info("Sent answer to client")
                                
                            except json.JSONDecodeError as e:
                                logger.error(f"Malformed OFFER JSON: {e}")
                            except KeyError as e:
                                logger.error(f"Missing key in OFFER: {e}")
                            except Exception as e:
                                logger.error(f"Error processing offer: {e}")
                                
                        elif message.startswith("CANDIDATE!"):
                            json_str = message[len("CANDIDATE!"):]
                            try:
                                data = json.loads(json_str)
                                logger.info("Received ICE candidate from client")
                                parsed_candidate_data = parse_ice_candidate_string(data["Candidate"])
                                
                                ice_candidate = RTCIceCandidate(
                                    foundation=parsed_candidate_data["foundation"],
                                    component=parsed_candidate_data["component"],
                                    protocol=parsed_candidate_data["protocol"],
                                    priority=parsed_candidate_data["priority"],
                                    ip=parsed_candidate_data["ip"],
                                    port=parsed_candidate_data["port"],
                                    type=parsed_candidate_data["type"],
                                    sdpMid=data["SdpMid"],
                                    sdpMLineIndex=data["SdpMLineIndex"],
                                    relatedAddress=parsed_candidate_data["relatedAddress"],
                                    relatedPort=parsed_candidate_data["relatedPort"]
                                )
                                await pc.addIceCandidate(ice_candidate)
                                logger.info("Added ICE candidate from client")
                                
                            except json.JSONDecodeError as e:
                                logger.error(f"Malformed CANDIDATE JSON: {e}")
                            except KeyError as e:
                                logger.error(f"Missing key in CANDIDATE: {e}")
                            except ValueError as e:
                                logger.error(f"Error parsing ICE candidate: {e}")
                            except Exception as e:
                                logger.error(f"Error processing candidate: {e}")
                                
                        elif message == "bye":
                            logger.info("Received 'bye', exiting")
                            return
                        else:
                            logger.info(f"Unhandled message: {message}")
                            
                    except websockets.ConnectionClosedOK:
                        logger.info("WebSocket connection closed gracefully.")
                        break
                    except websockets.ConnectionClosedError as e:
                        logger.error(f"WebSocket connection closed unexpectedly: {e}")
                        break
                    except Exception as e:
                        logger.exception("Error during WebSocket communication")
                        break

        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"Failed to connect to the signaling server: {e}")
        except Exception as e:
            logger.exception("An error occurred in main function")
        finally:
            # Clean up resources
            if camera_track:
                camera_track.cleanup()
            if audio_track:
                audio_track.cleanup()
            if pc and pc.connectionState != "closed":
                await pc.close()
            if websocket and websocket.state != websockets.protocol.State.CLOSED:
                await websocket.close()
            logger.info("Connections closed, attempting reconnection in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
    except Exception as e:
        logger.exception("An unhandled error occurred outside main:")
