import sys
print(sys.executable)
import asyncio
import logging
import math
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.signaling import TcpSocketSignaling  # Changed import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyAudioTrack(MediaStreamTrack):
    """
    A dummy audio track that generates a sine wave.
    """
    kind = "audio"

    def __init__(self, sample_rate=48000, amplitude=0.1):
        super().__init__()
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self._counter = 0

    async def recv(self):
        await asyncio.sleep(0.02)  # Simulate audio chunk duration
        t = self._counter / self.sample_rate
        sample = self.amplitude * math.sin(2 * math.pi * 440 * t)  # 440 Hz sine wave
        samples = bytes(int(sample * 32767).to_bytes(2, byteorder='little', signed=True) * 160) # Example: 160 samples
        self._counter += 160
        return samples

async def run(offer, pc):
    @pc.on("track")
    def on_track(track):
        logger.info(f"Python received track {track.kind}")
        if track.kind == "audio":
            # You would typically process the received audio here if this were the receiving end
            pass

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer, type="offer"))
    await pc.setLocalDescription(await pc.createAnswer())
    return pc.localDescription

async def main():
    # Replace 'python-peer' with a unique ID for your Python streamer
    #signaling_obj = TcpSocketSignaling(uri="ws://localhost:8080", peer_id="python-peer") # Changed instantiation
    #signaling_obj = TcpSocketSignaling("localhost", 8080, peer_id="python-peer")
    signaling_obj = TcpSocketSignaling("localhost", 8080)
    pc = RTCPeerConnection()

    pc.addTrack(DummyAudioTrack())

    await signaling_obj.connect()

    print("Waiting for offer from Unity...")
    offer_sdp = await signaling_obj.receive()
    print("Received offer:\n", offer_sdp)

    answer = await run(offer_sdp, pc)
    print("Sending answer:\n", answer.sdp)
    await signaling_obj.send(answer)

    print("Waiting for remote ICE candidates...")
    while True:
        candidate = await signaling_obj.receive()
        if candidate.type == "candidate":
            try:
                ice_candidate = RTCIceCandidate.from_json(candidate.sdp)
                await pc.addIceCandidate(ice_candidate)
            except Exception as e:
                logger.error(f"Error adding ICE candidate: {e}")
        elif candidate.type == "bye":
            print("Received 'bye', exiting")
            break

    try:
        await asyncio.sleep(3600)  # Keep running for an hour
    finally:
        await pc.close()
        await signaling_obj.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
