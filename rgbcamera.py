import sys
print(sys.executable)  # Prints the Python interpreter path being used
import asyncio
import logging
import math
import json
import cv2  # Import OpenCV for camera capture
import numpy as np  # Import numpy for array manipulation
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import VideoStreamTrack  # Import VideoStreamTrack
from av import VideoFrame  # Import VideoFrame for aiortc video processing
import websockets  # Import the websockets library
import re  # Import regex for parsing candidate string
from ultralytics import YOLO
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.width = 640
        self.height = 480
        self._initialize_camera()
        
        # Load YOLOv8 model
        try:
            self.model = YOLO("yolov8n.pt")
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
        self.target_class_ids = {0: "person", 56: "chair", 60: "dining table"}
        
        # Motion tracking parameters
        self.motion_threshold = 7
        self.dead_time_seconds = 5
        self.human_tracker = {}
        self.prev_gray = None
        
        # Frame rate control
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30  # 30 FPS
        
        logger.info("CameraVideoStreamTrack with YOLOv8 initialized.")

    def _initialize_camera(self):
        """Initialize camera with GStreamer pipeline for Raspberry Pi Camera v2"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.cap:
                    self.cap.release()
                    time.sleep(1.0)  # Increased delay to ensure release
                    logger.info("Previous camera instance released")
                    
                # GStreamer pipeline for Raspberry Pi Camera v2 using libcamerasrc
                pipeline = (
                    f"libcamerasrc ! "
                    f"video/x-raw, width={self.width}, height={self.height}, format=YUY2, framerate=30/1 ! "
                    f"videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink drop=true max-buffers=1"
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
            time.sleep(1.0)  # Increased delay to ensure release
            logger.info("Camera released.")
        self.cap = None

    def __del__(self):
        self.cleanup()
        
    async def recv(self):
        """Receive and process video frames"""
        try:
            # Frame rate control
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                await asyncio.sleep(self.frame_interval - (current_time - self.last_frame_time))
            
            self.last_frame_time = time.time()
            
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
        """Process frame with YOLO detection and motion tracking"""
        # Uncomment the line below to test raw frames without YOLO processing
        # return frame
        
        # Draw a green square to confirm frame processing
        cv2.rectangle(frame, (50, 50), (100, 100), (0, 255, 0), 2)

        # Convert to grayscale for motion tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Run YOLOv8 detection
        results = self.model(frame, verbose=False)[0]
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
                if self.prev_gray is not None:
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

        # Display stats on frame
        cv2.putText(frame, f"Humans: {humans_detected}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Possibly Dead: {possibly_dead}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        self.prev_gray = gray.copy()
        self.human_tracker = new_tracker
        
        return frame

    def is_moving(self, current_roi, previous_roi):
        if current_roi.shape != previous_roi.shape:
            return False
        diff = cv2.absdiff(current_roi, previous_roi)
        _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        return np.sum(motion_mask > 0) > 50


class DummyAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=48000, amplitude=0.1):
        super().__init__()
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self._counter = 0

    async def recv(self):
        await asyncio.sleep(0.02)  # Simulate 20ms audio chunk
        samples = np.zeros(960, dtype=np.int16)  # 960 samples for 20ms at 48kHz
        t = np.arange(self._counter, self._counter + 960) / self.sample_rate
        samples = (self.amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        self._counter += 960
        return samples.tobytes()  # Convert to bytes for WebRTC


async def run(offer, pc, camera_track):
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
        if camera_track:
            pc.addTrack(camera_track)
            logger.info("Camera track added to peer connection")

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
        logger.info(f"Set local description (answer): {answer.sdp}")
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
    Main asynchronous function to set up and run the WebRTC streamer.
    """
    uri = "wss://websockettest-eggy.onrender.com"
    peer_id = "python-peer"
    
    while True:  # Main reconnection loop
        websocket = None
        pc = None
        camera_track = None
        
        try:
            # Create new peer connection for each attempt
            pc = RTCPeerConnection()
            
            # Initialize camera track
            try:
                camera_track = CameraVideoStreamTrack()
                logger.info("Camera track initialized")
            except Exception as e:
                logger.error(f"Failed to initialize camera track: {e}")
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

                while True:
                    try:
                        message = await websocket.recv()
                        message = message.decode('utf-8')
                        logger.info(f"Raw message received: {message}")
                        
                        if message.startswith("OFFER!"):
                            json_str = message[len("OFFER!"):]
                            try:
                                data = json.loads(json_str)
                                logger.info("Received offer from Unity")
                                offer_sdp = data["Sdp"]
                                
                                answer = await run(offer_sdp, pc, camera_track)
                                answer_message_data = {
                                    "SessionType": answer.type.capitalize(),
                                    "Sdp": answer.sdp
                                }
                                full_answer_message = "ANSWER!" + json.dumps(answer_message_data)
                                await websocket.send(full_answer_message)
                                logger.info("Sent answer to Unity")
                                
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
                                logger.info("Received ICE candidate from Unity")
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
                                logger.info("Added ICE candidate from Unity")
                                
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