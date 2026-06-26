# src/camera/base_camera.py
from abc import ABC, abstractmethod

class BaseCamera(ABC):
    def __init__(self):
        self.target_fps: float = 20.0

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """Initializes hardware interfaces, camera configurations, and AI models."""
        pass

    @abstractmethod
    def update_frame(self) -> bool:
        """Reads the latest frame from the buffer and runs local AI processing."""
        pass

    @abstractmethod
    def get_ai_metadata(self) -> list:
        """Returns a standardized list of detected objects tracking classes and coordinates."""
        pass

    @abstractmethod
    def get_latest_frame(self):
        """Returns the latest frame from the camera."""
        pass

    @abstractmethod
    def write_pre_buffer_to_incident(self, incident_clip_path: str) -> None:
        """Writes the pre-buffered frames to an incident clip."""
        pass

    @abstractmethod
    def start_recording(self, output_path: str) -> None:
        """Starts recording the camera output to a file."""
        pass

    @abstractmethod
    def stop_recording(self) -> None:
        """Stops recording the camera output."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the camera."""
        pass