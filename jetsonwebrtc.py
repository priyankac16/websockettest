import sys
import asyncio
import logging
import json
import cv2
import numpy as np
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import VideoStreamTrack
from av import VideoFrame
import websockets
from ultralytics import YOLO
import time
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.width = 640
        self.height = 480

        # Load YOLOv8 model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = YOLO("yolov8n.pt").to(self.device)
            logger.info("YOLOv8 model loaded on %s", self.device)
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

        self.target_class_ids = {0: "person", 56: "chair", 60: "dining table"}
        self.motion_threshold = 7
        self.dead_time_seconds = 5
        self.human_tracker = {}
        self.prev_gray = None
        self.last_good_frame = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10

        self._initialize_camera_jetson()

    def _initialize_camera_jetson(self):
        logger.info("Initializing camera using V4L2 for Jetson...")
        for index in range(5):
            self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    logger.info(f"Camera opened at index {index}")
                    return
                else:
                    self.cap.release()
        raise IOError("Could not open any camera on Jetson Nano.")

        # Optional: Uncomment this for CSI camera
        # gst_pipeline = (
        #     "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, "
        #     "format=NV12, framerate=30/1 ! nvvidconv ! "
        #     "video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
        # )
        # self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    def cleanup(self):
        if self.cap:
            self.cap.release()
            logger.info("Camera released.")

    def __del__(self):
        self.cleanup()

    async def recv(self):
        try:
            pts, time_base = await self.next_timestamp()
            ret, frame = self.cap.read()
            if not ret:
                self.consecutive_failures += 1
                logger.warning(f"Camera read failed #{self.consecutive_failures}")
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(frame, "NO SIGNAL", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if self.consecutive_failures > self.max_consecutive_failures:
                    self.stop()
                    return None
            else:
                self.consecutive_failures = 0
                frame = self._process_frame(frame, time.time())

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame
        except Exception as e:
            logger.error(f"Error in recv: {e}")
            self.stop()
            return None

    def _process_frame(self, frame, timestamp):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        results = self.model(frame, verbose=False, device=self.device)[0]

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

            if class_id == 0:
                humans_detected += 1
                roi_now = gray[y1:y2, x1:x2]
                person_id = f"{x1}-{y1}-{x2}-{y2}"
                moved = False
                if self.prev_gray is not None and roi_now.size > 0:
                    roi_prev = self.prev_gray[y1:y2, x1:x2]
                    moved = self.is_moving(roi_now, roi_prev)
                last_move_time = self.human_tracker.get(person_id, 0)
                if moved:
                    last_move_time = timestamp
                new_tracker[person_id] = last_move_time
                is_dead = (timestamp - last_move_time) > self.dead_time_seconds
                label = "Dead" if is_dead else "Alive" if moved else "No Motion"
                color = (0, 0, 255) if is_dead else ((0, 255, 0) if moved else (0, 255, 255))
                if is_dead:
                    possibly_dead += 1
            else:
                label = label_name
                color = (0, 128, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Humans: {humans_detected}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Possibly Dead: {possibly_dead}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        self.prev_gray = gray.copy()
        self.human_tracker = new_tracker
        return frame

    def is_moving(self, current_roi, previous_roi):
        if current_roi.shape != previous_roi.shape:
            return False
        diff = cv2.absdiff(current_roi, previous_roi)
        _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        return np.sum(motion_mask > 0) > 50

# WebRTC handler functions
async def run(offer, pc, camera_track):
    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer, type="offer"))
    if camera_track:
        pc.addTrack(camera_track)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return pc.localDescription

def parse_ice_candidate_string(candidate_str):
    if not candidate_str.startswith("candidate:"):
        raise ValueError("Invalid ICE candidate format")
    parts = candidate_str[len("candidate:"):].split()
    return {
        "foundation": parts[0], "component": int(parts[1]),
        "protocol": parts[2], "priority": int(parts[3]),
        "ip": parts[4], "port": int(parts[5]), "type": parts[7],
        "relatedAddress": parts[9] if "raddr" in parts else None,
        "relatedPort": int(parts[11]) if "rport" in parts else None
    }

async def main():
    uri = "wss://websockettest-eggy.onrender.com"
    peer_id = "jetson-peer"

    while True:
        pc = RTCPeerConnection()
        camera_track = None
        websocket = None
        try:
            camera_track = CameraVideoStreamTrack()

            @pc.on("icecandidate")
            async def on_ice_candidate(candidate):
                if candidate and websocket and not websocket.closed:
                    await websocket.send("CANDIDATE!" + json.dumps({
                        "SdpMid": candidate.sdpMid,
                        "SdpMLineIndex": candidate.sdpMLineIndex,
                        "Candidate": candidate.candidate
                    }))

            async with websockets.connect(uri) as ws_conn:
                websocket = ws_conn
                await websocket.send(json.dumps({"type": "register", "peer_id": peer_id}))

                while True:
                    message_raw = await websocket.recv()
                    message = message_raw.decode('utf-8') if isinstance(message_raw, bytes) else message_raw

                    if message.startswith("OFFER!"):
                        data = json.loads(message[len("OFFER!"):])
                        answer = await run(data["Sdp"], pc, camera_track)
                        await websocket.send("ANSWER!" + json.dumps({
                            "SessionType": answer.type.capitalize(),
                            "Sdp": answer.sdp
                        }))
                    elif message.startswith("CANDIDATE!"):
                        data = json.loads(message[len("CANDIDATE!"):])
                        parsed = parse_ice_candidate_string(data["Candidate"])
                        await pc.addIceCandidate(RTCIceCandidate(
                            foundation=parsed["foundation"], component=parsed["component"],
                            protocol=parsed["protocol"], priority=parsed["priority"],
                            ip=parsed["ip"], port=parsed["port"], type=parsed["type"],
                            sdpMid=data["SdpMid"], sdpMLineIndex=data["SdpMLineIndex"],
                            relatedAddress=parsed["relatedAddress"], relatedPort=parsed["relatedPort"]
                        ))
                    elif message == "bye":
                        break
        except Exception as e:
            logger.error("Error in main loop", exc_info=True)
        finally:
            if camera_track: camera_track.cleanup()
            if pc and pc.connectionState != "closed":
                await pc.close()
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
