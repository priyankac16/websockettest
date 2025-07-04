import sys
print(sys.executable) # Prints the Python interpreter path being used
import asyncio
import logging
import math
import json
import cv2 # Import OpenCV for camera capture
import numpy as np # Import numpy for array manipulation
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import VideoStreamTrack # Import VideoStreamTrack
from av import VideoFrame # Import VideoFrame for aiortc video processing
import websockets  # Import the websockets library
import re # Import regex for parsing candidate string
from ultralytics import YOLO
from av import VideoFrame
import time


# Configure logging
logging.basicConfig(level=logging.INFO) # Changed to INFO for less verbose default, will use debug where needed
logger = logging.getLogger(__name__)


class CameraVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam. Please ensure it's connected and not in use.")
        
        self.width = 640
        self.height = 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Load YOLOv8 model
        self.model = YOLO("yolov8n.pt")
        self.target_class_ids = {0: "person", 56: "chair", 60: "dining table"}
        
        # Motion tracking parameters
        self.motion_threshold = 7
        self.dead_time_seconds = 5
        self.human_tracker = {}
        self.prev_gray = None
        
        logger.info("CameraVideoStreamTrack with YOLOv8 initialized.")

    async def recv(self):
        logger.debug("CameraVideoStreamTrack.recv called.")
        pts, time_base = await self.next_timestamp()
        timestamp = time.time()

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera, returning black frame.")
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            logger.debug("Successfully read frame from camera.")

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

        # Convert BGR to RGB for WebRTC
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        self.prev_gray = gray.copy()
        self.human_tracker = new_tracker

        await asyncio.sleep(1 / 30)  # Control frame rate
        return video_frame

    def is_moving(self, current_roi, previous_roi):
        if current_roi.shape != previous_roi.shape:
            return False
        diff = cv2.absdiff(current_roi, previous_roi)
        _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        return np.sum(motion_mask > 0) > 50

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released.")


class DummyAudioTrack(MediaStreamTrack):
    """
    A dummy audio track that generates a sine wave.
    This track simulates audio data for the WebRTC stream.
    It implements the abstract 'recv' method required by MediaStreamTrack.
    """
    kind = "audio"

    def __init__(self, sample_rate=48000, amplitude=0.1):
        super().__init__()
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self._counter = 0

    async def recv(self):
        """
        Generates a chunk of sine wave audio samples.
        This method is called by aiortc to get media data.
        """
        await asyncio.sleep(0.02)  # Simulate audio chunk duration (e.g., 20ms for 48kHz audio)
        # Generate 160 samples of a 440 Hz sine wave
        # The sample calculation creates a single sample value, then converts to bytes
        # and repeats it 160 times to simulate a small audio buffer.
        t = self._counter / self.sample_rate
        sample = self.amplitude * math.sin(2 * math.pi * 440 * t)
        # Convert float sample to 16-bit signed integer bytes (little-endian)
        # and repeat for 160 samples to create a dummy audio chunk.
        samples = bytes(int(sample * 32767).to_bytes(2, byteorder='little', signed=True) * 160)
        self._counter += 160
        return samples

async def run(offer, pc):
    """
    Handles the WebRTC offer/answer exchange and track reception.
    """
    @pc.on("track")
    def on_track(track):
        """
        Callback when a remote track is received.
        (Not strictly needed for a sender, but good practice for completeness)
        """
        logger.info(f"Python received track {track.kind}")
        if track.kind == "audio":
            # In a receiving scenario, you would process the audio here.
            # For this sender script, it's just a log.
            pass
        elif track.kind == "video":
            logger.info(f"Python received video track from Unity. (Not processing incoming video)")
            # If you wanted to receive video from Unity, you'd process it here.

    # Set the remote description (the offer received from Unity)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer, type="offer"))

    # Create and set the local description (the answer to Unity's offer)
    await pc.setLocalDescription(await pc.createAnswer())

    # Return the local description (answer) to be sent back via signaling
    return pc.localDescription

def parse_ice_candidate_string(candidate_str):
    """
    Parses a raw ICE candidate string into its components required by RTCIceCandidate constructor.
    Example candidate string: "candidate:459923758 1 udp 2122129151 192.168.0.24 64807 typ host generation 0 ufrag WSyX network-id 1 network-cost 10"
    """
    if not candidate_str.startswith("candidate:"):
        raise ValueError(f"Invalid ICE candidate string format: does not start with 'candidate:': {candidate_str}")
    
    # Split the string after "candidate:"
    parts = candidate_str[len("candidate:"):].split()

    # Ensure enough core parts are present
    # foundation, component, protocol, priority, ip, port, typ, type
    if len(parts) < 8:
        raise ValueError(f"Invalid ICE candidate string format: not enough core parts (expected at least 8): {candidate_str}")

    foundation = parts[0]
    component = int(parts[1])
    protocol = parts[2]
    priority = int(parts[3])
    ip = parts[4]
    port = int(parts[5])
    
    # 'typ' is at index 6, the actual candidate type (e.g., 'host', 'srflx') is at index 7
    candidate_type = parts[7] 

    related_address = None
    related_port = None

    # Parse optional attributes like raddr and rport
    i = 8 # Start checking from after 'typ host'
    while i < len(parts):
        if parts[i] == "raddr" and i + 1 < len(parts):
            related_address = parts[i+1]
            i += 2
        elif parts[i] == "rport" and i + 1 < len(parts):
            related_port = int(parts[i+1])
            i += 2
        else:
            i += 1 # Move to next part if not a recognized keyword

    parsed_data = {
        "foundation": foundation,
        "component": component,
        "protocol": protocol,
        "priority": priority,
        "ip": ip,
        "port": port,
        "type": candidate_type,
        "relatedAddress": related_address, # Changed to camelCase
        "relatedPort": related_port       # Changed to camelCase
    }
    logger.debug(f"Parsed candidate data: {parsed_data}") # Debug print
    return parsed_data


def start_camera_capture(camera_index=0):
    """
    Opens the camera and starts capturing frames.

    Args:
        camera_index (int): The index of the camera to use (default: 0, the default camera).
                           If you have multiple cameras, try 1, 2, etc.
    Returns:
        cv2.VideoCapture: The VideoCapture object if the camera opens successfully,
                        None otherwise.
    """

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return None  # Indicate failure to open camera

    return cap

def read_and_display_camera(capture_object):
    """
    Reads frames from the camera capture and displays them in a window.

    Args:
        capture_object (cv2.VideoCapture): The VideoCapture object returned by start_camera_capture().
    """

    if capture_object is None:
        print("Error: No camera capture object provided.")
        return

    while True:
        ret, frame = capture_object.read()

        if not ret:
            print("Error: Could not read frame from camera.")
            break

        #cv2.imshow("Camera Feed", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy windows *after* the loop
    capture_object.release()
    cv2.destroyAllWindows()



async def main():
    """
    Main asynchronous function to set up and run the WebRTC streamer.
    """
    uri = "wss://websockettest-eggy.onrender.com" # Use wss:// for secure WebSocket
    peer_id = "python-peer"
    websocket = None
    pc = RTCPeerConnection()
   # camera_track = CameraVideoStreamTrack()
    # --- TEMPORARILY REMOVED DummyAudioTrack to resolve ValueError ---
    # pc.addTrack(DummyAudioTrack()) 

    # Callback for when a data channel is received from the remote peer
    @pc.on("datachannel")
    def on_datachannel(channel):
        #pc.addTrack(camera_track) 
        #pc.addTrack(DummyAudioTrack()) 
        print(f"Data channel '{channel.label}' received from remote peer.")
        try:
            camera_track = CameraVideoStreamTrack()
            pc.addTrack(camera_track)
            print("Camera track added.")
        except Exception as e:
            print(f"Failed to initialize camera track: {e}")      

        @channel.on("message")
        def on_message(message):
            print(f"Data channel message received: {message}")
            # You can send a response back on the data channel here
            # channel.send(f"Echo: {message}")
        @channel.on("open")
        def on_open():
            #start_camera_capture(0)
            print(f"Data channel '{channel.label}' opened.")
        @channel.on("close")
        def on_on_close(): # Corrected function name from on_close to on_on_close
            print(f"Data channel '{channel.label}' closed.")

    # Callback for when ICE candidates are generated by aiortc
    @pc.on("icecandidate")
    async def on_ice_candidate(candidate):
        if candidate:
            print(f"Generated ICE candidate: {candidate.candidate}")
            # Format candidate message to match Unity's expected JSON structure
            candidate_message_data = {
                "SdpMid": candidate.sdpMid,
                "SdpMLineIndex": candidate.sdpMLineIndex,
                "Candidate": candidate.candidate
            }
            # Prepend "CANDIDATE!" as expected by Unity
            full_candidate_message = "CANDIDATE!" + json.dumps(candidate_message_data)
            # Ensure websocket is not None and is open before sending
            if websocket and not websocket.closed:
                await websocket.send(full_candidate_message)
                print(f"Sent ICE candidate to signaling server:\n{full_candidate_message}")
            else:
                print("WebSocket not open, cannot send ICE candidate.")


    try:
        # Establish a WebSocket connection to the signaling server
        async with websockets.connect(uri) as ws_conn: # Renamed 'websocket' to 'ws_conn' for clarity
            websocket = ws_conn # Assign to the outer 'websocket' variable
            print(f"Connected to signaling server at {uri}")

            # Send registration message to the signaling server
            register_message = json.dumps({"type": "register", "peer_id": peer_id})
            await websocket.send(register_message)
            print(f"Sent registration: {register_message}")

            # Main loop for receiving signaling messages
            while True:
                try:
                    # Decode the received bytes message to a string
                    message = (await websocket.recv()).decode('utf-8') 
                    if message is None: # Connection closed
                        break
                    
                    # Handle different types of signaling messages from Unity
                    if message.startswith("OFFER!"):
                        # Extract JSON part after "OFFER!"
                        json_str = message[len("OFFER!"):]
                        try:
                            data = json.loads(json_str)
                            # Unity's SessionDescription uses "SessionType" and "Sdp"
                            offer_sdp = data["Sdp"]
                            print("Received offer:\n", offer_sdp)
                            
                            # Process the offer and create an answer
                            answer = await run(offer_sdp, pc)
                            
                            # Format answer message to match Unity's expected JSON structure
                            answer_message_data = {
                                "SessionType": answer.type.capitalize(), # "answer" -> "Answer"
                                "Sdp": answer.sdp
                            }
                            # Prepend "ANSWER!" as expected by Unity
                            full_answer_message = "ANSWER!" + json.dumps(answer_message_data)
                            await websocket.send(full_answer_message)
                            print("Sent answer to signaling server:\n", full_answer_message)

                        except json.JSONDecodeError:
                            print(f"Received malformed OFFER JSON: {json_str}")
                        except KeyError as e:
                            print(f"Missing key in OFFER message: {e} in {json_str}")

                    elif message.startswith("CANDIDATE!"):
                        # Extract JSON part after "CANDIDATE!"
                        json_str = message[len("CANDIDATE!"):]
                        try:
                            data = json.loads(json_str)
                            
                            # Parse the raw candidate string from Unity
                            parsed_candidate_data = parse_ice_candidate_string(data["Candidate"])
                            
                            # Instantiate RTCIceCandidate with all required positional arguments
                            # Now using camelCase for relatedAddress and relatedPort
                            ice_candidate_args = {
                                "foundation": parsed_candidate_data["foundation"],
                                "component": parsed_candidate_data["component"],
                                "protocol": parsed_candidate_data["protocol"],
                                "priority": parsed_candidate_data["priority"],
                                "ip": parsed_candidate_data["ip"],
                                "port": parsed_candidate_data["port"],
                                "type": parsed_candidate_data["type"],
                                "sdpMid": data["SdpMid"],
                                "sdpMLineIndex": data["SdpMLineIndex"]
                            }
                            if parsed_candidate_data["relatedAddress"] is not None:
                                ice_candidate_args["relatedAddress"] = parsed_candidate_data["relatedAddress"]
                            if parsed_candidate_data["relatedPort"] is not None:
                                ice_candidate_args["relatedPort"] = parsed_candidate_data["relatedPort"]

                            ice_candidate = RTCIceCandidate(**ice_candidate_args) # Use ** to unpack dictionary as kwargs

                            await pc.addIceCandidate(ice_candidate)
                            print("Received and added ICE candidate from Unity.")
                        except json.JSONDecodeError:
                            print(f"Received malformed CANDIDATE JSON: {json_str}")
                        except KeyError as e:
                            print(f"Missing key in CANDIDATE message: {e} in {json_str}")
                        except ValueError as e: # Catch parsing errors from parse_ice_candidate_string
                            print(f"Error parsing ICE candidate string: {e}")
                        except Exception as e: # Catch any other errors during candidate processing
                            logger.exception(f"Unexpected error processing CANDIDATE: {json_str}")
                    
                    elif message == "bye": # Simple 'bye' message (no JSON prefix)
                        print("Received 'bye', exiting")
                        break
                    else:
                        print(f"Received unhandled message: {message}")

                except websockets.ConnectionClosedOK:
                    print("WebSocket connection closed gracefully.")
                    break
                except websockets.ConnectionClosedError as e:
                    print(f"WebSocket connection closed unexpectedly: {e}")
                    break
                except Exception as e:
                    logger.exception("Error during WebSocket communication:")
                    break

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Failed to connect to the signaling server: {e}")
    except Exception as e:
        logger.exception("An error occurred in main function:")
    finally:
        # Ensure peer connection and WebSocket are closed on exit
        if pc.connectionState != "closed":
            await pc.close()
        # Check if websocket object exists and is not already closed
        if websocket and websocket.state != websockets.protocol.State.CLOSED:
            await websocket.close()
        print("Connections closed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
    except Exception as e:
        logger.exception("An unhandled error occurred outside main:")