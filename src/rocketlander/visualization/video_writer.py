"""Write rendered frames to an MP4 video."""

from pathlib import Path
from types import TracebackType

import imageio
import numpy as np


class VideoWriter:
    """Streams frames to an MP4 file through imageio and ffmpeg, so long videos don't fill memory.

    Use as ``with VideoWriter(path, fps) as video: video.add(frame) ...``.
    """

    def __init__(self, path: str | Path, fps: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frames = 0
        self._writer = imageio.get_writer(self.path, fps=fps)
        self._last: np.ndarray | None = None

    def add(self, frame: np.ndarray) -> None:
        self._writer.append_data(frame)
        self._last = frame
        self.frames += 1

    def hold(self, seconds: float) -> None:
        """Repeat the last frame, e.g. so the outcome of an episode stays readable."""
        if self._last is not None:
            for _ in range(round(seconds * self.fps)):
                self.add(self._last)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
