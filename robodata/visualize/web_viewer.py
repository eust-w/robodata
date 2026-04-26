"""Web-based dataset visualizer.

Zero-dependency viewer (stdlib only) that serves frames over HTTP
and renders everything in the browser. Supports segment overlay
with phase labels.

Usage:
    robodata visualize ./dataset --backend web
    robodata visualize pusht --backend web --segment
"""

from __future__ import annotations

import io
import json
import logging
import socket
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import numpy as np

from robodata.core.models import Episode, Frame
from robodata.formats.registry import FormatRegistry

logger = logging.getLogger(__name__)


def _encode_jpeg(img_rgb: np.ndarray) -> bytes:
    """Encode RGB uint8 array to JPEG bytes.

    Tries cv2 first, then PIL, then falls back to BMP.
    """
    # Try OpenCV
    try:
        import cv2
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            return buf.tobytes()
    except ImportError:
        pass

    # Try PIL
    try:
        from PIL import Image
        pil_img = Image.fromarray(img_rgb)
        bio = io.BytesIO()
        pil_img.save(bio, format="JPEG", quality=85)
        return bio.getvalue()
    except ImportError:
        pass

    # Fallback: BMP (no dependencies, larger)
    h, w = img_rgb.shape[:2]
    row_size = (w * 3 + 3) & ~3
    padding = row_size - w * 3
    pixel_size = row_size * h
    file_size = 54 + pixel_size

    header = bytearray(54)
    header[0:2] = b"BM"
    header[2:6] = file_size.to_bytes(4, "little")
    header[10:14] = (54).to_bytes(4, "little")
    header[14:18] = (40).to_bytes(4, "little")
    header[18:22] = w.to_bytes(4, "little")
    header[22:26] = h.to_bytes(4, "little")
    header[26:28] = (1).to_bytes(2, "little")
    header[28:30] = (24).to_bytes(2, "little")
    header[34:38] = pixel_size.to_bytes(4, "little")

    pixels = bytearray()
    pad_bytes = b"\x00" * padding
    for y_coord in range(h - 1, -1, -1):
        row = img_rgb[y_coord]
        for x_coord in range(w):
            red, green, blue = row[x_coord]
            pixels.extend([blue, green, red])
        pixels.extend(pad_bytes)

    return bytes(header) + bytes(pixels)


class WebBackend:
    """Loads dataset data and serves it to the web frontend.

    Only extracts lightweight metadata (actions, states, camera keys)
    on init. Frame images are loaded lazily on demand.
    """

    def __init__(self, dataset_path: Path, max_episodes: int = 50):
        self.dataset_path = Path(dataset_path)
        self.format_name = FormatRegistry.detect_format(dataset_path)
        self.reader = FormatRegistry.get_reader(self.format_name)
        self.info = self.reader.inspect(dataset_path)

        self._episode_objects: list[Episode] = []
        self._episode_lengths: list[int] = []
        self._camera_keys: list[str] = []
        self._episode_actions: list[np.ndarray | None] = []
        self._episode_states: list[np.ndarray | None] = []
        self._segmentations: list[dict | None] = []

        # Cache for materialized frames (populated lazily)
        self._frame_cache: dict[int, list[Frame]] = {}

        print(f"Loading episodes from {self.format_name} dataset...")

        for episode_idx, episode in enumerate(self.reader.read_episodes(dataset_path)):
            if episode_idx >= max_episodes:
                break
            if (episode_idx + 1) % 10 == 0:
                print(f"  Loaded {episode_idx + 1} episodes...")

            self._episode_objects.append(episode)

            # Extract only actions/states — iterate frames without storing them
            actions = []
            states = []
            frame_count = 0
            for frame in episode.frames():
                frame_count += 1
                if frame.action is not None:
                    actions.append(frame.action)
                if frame.state is not None:
                    states.append(frame.state)
                # Detect cameras from first frame of first episode
                if i == 0 and frame_count == 1:
                    for cam_name in frame.images.keys():
                        self._camera_keys.append(cam_name)

            self._episode_lengths.append(frame_count)
            self._episode_actions.append(np.array(actions) if actions else None)
            self._episode_states.append(np.array(states) if states else None)
            self._segmentations.append(None)

        print(f"Loaded {len(self._episode_objects)} episodes")

    def set_segmentation(self, episode_idx: int, segments_data: list[dict]) -> None:
        """Set segmentation data for an episode."""
        if episode_idx < len(self._segmentations):
            self._segmentations[episode_idx] = segments_data

    def _get_frames(self, episode_idx: int) -> list[Frame]:
        """Lazily materialize and cache frames for an episode."""
        if episode_idx not in self._frame_cache:
            episode = self._episode_objects[episode_idx]
            self._frame_cache[episode_idx] = list(episode.frames())
        return self._frame_cache[episode_idx]

    def get_num_episodes(self) -> int:
        return len(self._episode_objects)

    def get_episode_length(self, episode_idx: int) -> int:
        if episode_idx >= len(self._episode_lengths):
            return 0
        return self._episode_lengths[episode_idx]

    def get_frame_image(self, episode_idx: int, frame_idx: int, camera_key: str) -> np.ndarray | None:
        if episode_idx >= len(self._episode_objects):
            return None
        frames = self._get_frames(episode_idx)
        if frame_idx >= len(frames):
            return None
        frame = frames[frame_idx]
        if camera_key not in frame.images:
            return None
        img = frame.images[camera_key].load()
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def get_episode_actions(self, episode_idx: int) -> np.ndarray | None:
        if episode_idx >= len(self._episode_actions):
            return None
        return self._episode_actions[episode_idx]

    def get_episode_states(self, episode_idx: int) -> np.ndarray | None:
        if episode_idx >= len(self._episode_states):
            return None
        return self._episode_states[episode_idx]

    def get_segmentation(self, episode_idx: int) -> list[dict] | None:
        if episode_idx >= len(self._segmentations):
            return None
        return self._segmentations[episode_idx]

    def get_camera_keys(self) -> list[str]:
        return self._camera_keys

    def get_fps(self) -> float:
        return self.info.inferred_fps or 30.0

    def get_name(self) -> str:
        return f"{self.dataset_path.name} ({self.format_name})"

    def get_info_dict(self) -> dict:
        return {
            "name": self.get_name(),
            "format": self.format_name,
            "num_episodes": self.get_num_episodes(),
            "cameras": self.get_camera_keys(),
            "fps": self.get_fps(),
            "total_frames": sum(
                self.get_episode_length(i) for i in range(self.get_num_episodes())
            ),
        }

    def get_episode_dict(self, episode_idx: int) -> dict:
        num_frames = self.get_episode_length(episode_idx)

        actions = self.get_episode_actions(episode_idx)
        states = self.get_episode_states(episode_idx)

        result: dict[str, Any] = {
            "num_frames": num_frames,
            "actions": None,
            "states": None,
            "segments": self.get_segmentation(episode_idx),
        }

        if actions is not None and actions.size > 0:
            # Limit to 8 dims, round for compact JSON
            actions_subset = actions[:, :8] if actions.ndim > 1 and actions.shape[1] > 8 else actions
            result["actions"] = np.round(actions_subset, 4).tolist()

        if states is not None and states.size > 0:
            states_subset = states[:, :8] if states.ndim > 1 and states.shape[1] > 8 else states
            result["states"] = np.round(states_subset, 4).tolist()

        return result


def _make_handler(backend: WebBackend):
    """Create an HTTP request handler class with access to the backend."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress request logs

        def _send_json(self, data: Any) -> None:
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_image(self, img_bytes: bytes, content_type: str = "image/jpeg") -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(img_bytes)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(img_bytes)

        def _send_404(self) -> None:
            self.send_response(404)
            self.end_headers()

        def do_GET(self) -> None:
            try:
                self._route()
            except BrokenPipeError:
                pass
            except ConnectionResetError:
                pass

        def _route(self) -> None:
            path = self.path

            if path == "/":
                self._send_html(HTML_TEMPLATE)

            elif path == "/api/info":
                self._send_json(backend.get_info_dict())

            elif path.startswith("/api/episode/"):
                try:
                    idx = int(path.split("/")[3])
                    self._send_json(backend.get_episode_dict(idx))
                except (IndexError, ValueError):
                    self._send_404()

            elif path.startswith("/frame/"):
                parts = path.split("/")
                # /frame/<episode>/<frame>/<camera...>
                if len(parts) >= 5:
                    try:
                        ep = int(parts[2])
                        fr = int(parts[3])
                        cam = "/".join(parts[4:])  # Camera key may contain /
                        img = backend.get_frame_image(ep, fr, cam)
                        if img is not None:
                            img_bytes = _encode_jpeg(img)
                            ct = "image/jpeg"
                            if img_bytes[:2] == b"BM":
                                ct = "image/bmp"
                            self._send_image(img_bytes, ct)
                        else:
                            self._send_404()
                    except (ValueError, IndexError):
                        self._send_404()
                else:
                    self._send_404()
            else:
                self._send_404()

    return Handler


class WebViewer:
    """Web-based dataset viewer.

    Starts a local HTTP server and opens the browser.

    Example:
        >>> viewer = WebViewer("path/to/dataset")
        >>> viewer.show()
    """

    def __init__(
        self,
        dataset_path: str | Path,
        max_episodes: int = 50,
        port: int = 0,
        segment: bool = False,
    ):
        self.backend = WebBackend(Path(dataset_path), max_episodes)
        self.port = port

        if segment:
            self._run_segmentation()

    def _run_segmentation(self) -> None:
        """Run PELT segmentation with labels using already-extracted data."""
        try:
            from robodata.segment import SegmentAnalyzer, SegmentConfig
            from robodata.segment.labeler import PhaseLabeler
            from robodata.segment.plot import PHASE_COLORS
        except ImportError:
            print("Segmentation requires ruptures: pip install robodata[segment]")
            return

        print("Running segmentation with phase labels...")
        # Use l2 cost model for speed (rbf is ~20x slower on long episodes)
        config = SegmentConfig(label_phases=True, cost_model="l2")
        analyzer = SegmentAnalyzer(config=config)

        for episode_idx in range(self.backend.get_num_episodes()):
            try:
                # Use already-extracted states (avoids re-iterating frames)
                states = self.backend.get_episode_states(episode_idx)
                if states is None or states.size == 0:
                    continue

                segments = analyzer.analyze(states)
                segments_data = [
                    {
                        "start_frame": segment.start_frame,
                        "end_frame": segment.end_frame,
                        "label": segment.label,
                        "color": color,
                        "duration_frames": segment.duration_frames,
                    }
                    for segment in segments
                ]
                self.backend.set_segmentation(episode_idx, segments_data)
            except Exception as e:
                logger.warning("Segmentation failed for episode %d: %s", episode_idx, e)

        print("Segmentation complete.")

    def show(self) -> None:
        """Start server and open browser."""
        handler = _make_handler(self.backend)

        # Find a free port
        if self.port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                self.port = s.getsockname()[1]

        server = HTTPServer(("127.0.0.1", self.port), handler)
        url = f"http://127.0.0.1:{self.port}"

        print(f"\nRobodata Viewer running at: {url}")
        print("Press Ctrl+C to stop.\n")

        # Open browser after a short delay
        def open_browser():
            import time
            time.sleep(0.3)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down viewer...")
            server.shutdown()


def web_visualize(
    dataset_path: str | Path,
    max_episodes: int = 50,
    segment: bool = False,
    port: int = 0,
) -> None:
    """Visualize a dataset in the browser.

    Args:
        dataset_path: Path to dataset (any supported format).
        max_episodes: Maximum episodes to load.
        segment: Run segmentation and show phase overlay.
        port: Server port (0 = auto).
    """
    viewer = WebViewer(dataset_path, max_episodes=max_episodes, segment=segment, port=port)
    viewer.show()


# ============================================================
# HTML Template
# ============================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robodata Viewer</title>
<style>
:root {
  --bg: #0f1117;
  --bg2: #1a1d2e;
  --bg3: #252838;
  --text: #e4e4e7;
  --text-dim: #8b8fa3;
  --accent: #6c9fff;
  --green: #66bb6a;
  --red: #ef5350;
  --border: #2a2d3e;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  overflow-x: hidden;
}

header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1.5rem;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
}

header h1 {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--accent);
}

header .meta {
  font-size: 0.8rem;
  color: var(--text-dim);
  font-family: 'SF Mono', 'Menlo', monospace;
}

.badge {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 600;
  background: var(--accent);
  color: var(--bg);
  margin-left: 0.5rem;
}

main {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 52px);
  overflow-y: auto;
  overflow-x: hidden;
}

/* Camera grid */
#cameras {
  display: flex;
  gap: 4px;
  padding: 8px;
  background: var(--bg);
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

#cameras img {
  border-radius: 4px;
  background: #000;
  image-rendering: auto;
}

/* Timeline + controls bar */
#controls-bar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem 1rem;
  background: var(--bg2);
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
  flex-shrink: 0;
}

#controls-bar select,
#controls-bar button {
  background: var(--bg3);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.3rem 0.5rem;
  font-size: 0.8rem;
  cursor: pointer;
}

#controls-bar button:hover { background: var(--accent); color: var(--bg); }

#play-btn {
  width: 60px;
  font-weight: 600;
}

#frame-counter {
  font-family: 'SF Mono', 'Menlo', monospace;
  font-size: 0.8rem;
  color: var(--text-dim);
  min-width: 120px;
}

#speed-select { width: 65px; }

/* Timeline */
#timeline-wrap {
  flex: 1;
  min-width: 200px;
  position: relative;
  height: 32px;
}

#scrubber {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 6px;
  background: var(--bg3);
  border-radius: 3px;
  outline: none;
  position: relative;
  z-index: 2;
  margin-top: 13px;
}

#scrubber::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
  border: 2px solid var(--bg);
}

#segment-overlay {
  position: absolute;
  top: 10px;
  left: 0;
  right: 0;
  height: 12px;
  border-radius: 3px;
  overflow: hidden;
  z-index: 1;
  pointer-events: none;
}

#segment-overlay .seg {
  position: absolute;
  height: 100%;
  opacity: 0.7;
  transition: opacity 0.15s;
}

/* Segment legend */
#segment-legend {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  font-size: 0.7rem;
  color: var(--text-dim);
}

#segment-legend .leg-item {
  display: flex;
  align-items: center;
  gap: 0.3rem;
}

#segment-legend .leg-dot {
  width: 10px;
  height: 10px;
  border-radius: 2px;
}

/* Current segment label */
#current-phase {
  font-size: 0.8rem;
  font-weight: 600;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  min-width: 100px;
  text-align: center;
}

/* Charts */
#charts {
  display: flex;
  gap: 2px;
  padding: 4px;
  background: var(--bg);
  height: 150px;
  flex-shrink: 0;
}

.chart-container {
  flex: 1;
  position: relative;
  min-width: 0;
}

.chart-container canvas {
  width: 100%;
  height: 100%;
  background: var(--bg2);
  border-radius: 4px;
}

.marker-canvas {
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none;
  background: transparent !important;
}

/* Shortcuts hint */
#shortcuts {
  position: fixed;
  bottom: 8px;
  right: 12px;
  font-size: 0.65rem;
  color: var(--text-dim);
  opacity: 0.5;
}

/* Loading overlay */
#loading {
  position: fixed;
  inset: 0;
  background: var(--bg);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  color: var(--accent);
  z-index: 100;
}

#loading.hidden { display: none; }
</style>
</head>
<body>

<div id="loading">Loading dataset...</div>

<header>
  <div>
    <h1 id="dataset-name">Robodata Viewer</h1>
    <span class="badge" id="format-badge"></span>
  </div>
  <span class="meta" id="dataset-meta"></span>
</header>

<main>
  <div id="cameras"></div>

  <div id="controls-bar">
    <select id="episode-select" title="Episode"></select>
    <button id="play-btn" title="Space">Play</button>
    <select id="speed-select" title="Playback speed">
      <option value="0.25">0.25x</option>
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
    <div id="timeline-wrap">
      <div id="segment-overlay"></div>
      <input type="range" id="scrubber" min="0" max="1" value="0">
    </div>
    <span id="frame-counter">0 / 0</span>
    <span id="current-phase"></span>
    <div id="segment-legend"></div>
  </div>

  <div id="charts">
    <div class="chart-container">
      <canvas id="action-chart"></canvas>
      <canvas id="action-marker" class="marker-canvas"></canvas>
    </div>
    <div class="chart-container">
      <canvas id="state-chart"></canvas>
      <canvas id="state-marker" class="marker-canvas"></canvas>
    </div>
  </div>
</main>

<div id="shortcuts">Space: Play/Pause &nbsp; ←→: Frame &nbsp; ↑↓: Episode &nbsp; [/]: Speed</div>

<script>
const state = {
  ep: 0,
  frame: 0,
  numFrames: 0,
  playing: false,
  speed: 1,
  fps: 30,
  cameras: [],
  epData: null,
  info: null,
  lastFrameTime: 0,
};

const $ = id => document.getElementById(id);

// ── Init ──
async function init() {
  state.info = await (await fetch('/api/info')).json();
  state.fps = state.info.fps || 30;
  state.cameras = state.info.cameras || [];

  // Header
  $('dataset-name').textContent = state.info.name;
  $('format-badge').textContent = state.info.format;
  $('dataset-meta').textContent =
    `${state.info.num_episodes} episodes · ${state.info.total_frames} frames · ${state.fps} fps`;

  // Camera grid
  const camEl = $('cameras');
  camEl.innerHTML = '';
  if (state.cameras.length === 0) {
    camEl.innerHTML = '<div style="color:var(--text-dim);padding:2rem;text-align:center">No camera data</div>';
  } else {
    // Load first frame to detect native resolution, then size all images
    const firstCam = state.cameras[0];
    const probe = new Image();
    probe.onload = () => {
      const natW = probe.naturalWidth;
      const natH = probe.naturalHeight;
      // Target: scale small images up so they're at least 360px tall,
      // but cap at 50% of viewport height. Keep aspect ratio.
      const maxH = Math.floor(window.innerHeight * 0.5);
      const targetH = Math.max(Math.min(natH * Math.ceil(360 / Math.max(natH, 1)), maxH), natH);
      const scale = targetH / natH;
      const targetW = Math.round(natW * scale);

      for (const cam of state.cameras) {
        const img = document.createElement('img');
        img.id = `cam-${cam.replace(/\//g, '-')}`;
        img.alt = cam;
        img.draggable = false;
        img.width = targetW;
        img.height = targetH;
        // Use crisp rendering only for very small images (< 200px)
        if (natH < 200) img.style.imageRendering = 'pixelated';
        camEl.appendChild(img);
      }
      updateFrame();
    };
    probe.src = `/frame/0/0/${firstCam}`;
  }

  // Episode selector
  const sel = $('episode-select');
  sel.innerHTML = '';
  for (let i = 0; i < state.info.num_episodes; i++) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `Episode ${i}`;
    sel.appendChild(opt);
  }
  sel.onchange = () => loadEpisode(parseInt(sel.value));

  // Controls
  $('play-btn').onclick = togglePlay;
  $('speed-select').onchange = e => { state.speed = parseFloat(e.target.value); };
  $('scrubber').oninput = e => {
    state.frame = parseInt(e.target.value);
    updateFrame();
  };

  // Keyboard
  document.addEventListener('keydown', onKey);

  await loadEpisode(0);
  $('loading').classList.add('hidden');
}

async function loadEpisode(idx) {
  state.ep = idx;
  state.frame = 0;
  state.playing = false;
  $('play-btn').textContent = 'Play';
  $('episode-select').value = idx;

  state.epData = await (await fetch(`/api/episode/${idx}`)).json();
  state.numFrames = state.epData.num_frames;

  const scrub = $('scrubber');
  scrub.max = Math.max(0, state.numFrames - 1);
  scrub.value = 0;

  renderSegmentOverlay();
  drawCharts();
  updateFrame();
}

function updateFrame() {
  // Frame counter
  $('frame-counter').textContent = `${state.frame + 1} / ${state.numFrames}`;

  // Camera images
  for (const cam of state.cameras) {
    const img = document.getElementById(`cam-${cam.replace(/\//g, '-')}`);
    if (img) {
      const src = `/frame/${state.ep}/${state.frame}/${cam}`;
      if (img.src !== location.origin + src) {
        img.src = src;
      }
    }
  }

  // Scrubber
  $('scrubber').value = state.frame;

  // Update chart markers
  drawChartMarkers();

  // Update current phase label
  updatePhaseLabel();
}

// ── Segment overlay ──
function renderSegmentOverlay() {
  const overlay = $('segment-overlay');
  const legend = $('segment-legend');
  overlay.innerHTML = '';
  legend.innerHTML = '';

  const segs = state.epData?.segments;
  if (!segs || segs.length === 0) {
    $('current-phase').textContent = '';
    return;
  }

  const nf = state.numFrames;
  const seen = new Map();

  for (const seg of segs) {
    const div = document.createElement('div');
    div.className = 'seg';
    div.style.left = `${(seg.start / nf) * 100}%`;
    div.style.width = `${((seg.end - seg.start) / nf) * 100}%`;
    div.style.background = seg.color;
    div.title = `${seg.label} [${seg.start}:${seg.end}]`;
    overlay.appendChild(div);
    seen.set(seg.label, seg.color);
  }

  // Legend
  for (const [label, color] of seen) {
    const item = document.createElement('span');
    item.className = 'leg-item';
    item.innerHTML = `<span class="leg-dot" style="background:${color}"></span>${label}`;
    legend.appendChild(item);
  }
}

function updatePhaseLabel() {
  const el = $('current-phase');
  const segs = state.epData?.segments;
  if (!segs) { el.textContent = ''; return; }

  for (const seg of segs) {
    if (state.frame >= seg.start && state.frame < seg.end) {
      el.textContent = seg.label;
      el.style.background = seg.color;
      el.style.color = '#000';
      return;
    }
  }
  el.textContent = '';
  el.style.background = 'transparent';
}

// ── Charts ──
let actionChartData = null;
let stateChartData = null;

function drawCharts() {
  actionChartData = state.epData?.actions;
  stateChartData = state.epData?.states;
  drawChart($('action-chart'), actionChartData, 'Actions');
  drawChart($('state-chart'), stateChartData, 'States');
}

function drawChart(canvas, data, title) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  ctx.fillStyle = '#1a1d2e';
  ctx.fillRect(0, 0, w, h);

  if (!data || data.length === 0) {
    ctx.fillStyle = '#8b8fa3';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`No ${title.toLowerCase()} data`, w / 2, h / 2);
    return;
  }

  // Title
  ctx.fillStyle = '#8b8fa3';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(title, 8, 14);

  const padL = 8, padR = 8, padT = 22, padB = 8;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const nFrames = data.length;
  const nDims = Array.isArray(data[0]) ? Math.min(data[0].length, 8) : 1;

  // Find global min/max
  let gMin = Infinity, gMax = -Infinity;
  for (const row of data) {
    if (Array.isArray(row)) {
      for (let d = 0; d < nDims; d++) {
        if (row[d] < gMin) gMin = row[d];
        if (row[d] > gMax) gMax = row[d];
      }
    } else {
      if (row < gMin) gMin = row;
      if (row > gMax) gMax = row;
    }
  }
  const range = gMax - gMin || 1;

  // Segment backgrounds
  const segs = state.epData?.segments;
  if (segs) {
    for (const seg of segs) {
      const x1 = padL + (seg.start / nFrames) * plotW;
      const x2 = padL + (seg.end / nFrames) * plotW;
      ctx.fillStyle = seg.color + '18';
      ctx.fillRect(x1, padT, x2 - x1, plotH);
    }
  }

  // Draw lines
  const colors = ['#6c9fff', '#66bb6a', '#ef5350', '#ff9800', '#ab47bc', '#fdd835', '#29b6f6', '#e0e0e0'];
  for (let d = 0; d < nDims; d++) {
    ctx.strokeStyle = colors[d % colors.length];
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    for (let i = 0; i < nFrames; i++) {
      const val = Array.isArray(data[i]) ? data[i][d] : data[i];
      const x = padL + (i / Math.max(nFrames - 1, 1)) * plotW;
      const y = padT + plotH - ((val - gMin) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // Store chart params for marker drawing
  canvas._chartParams = { padL, padR, padT, padB, plotW, plotH, nFrames, w, h };
}

function drawChartMarkers() {
  drawChartMarker($('action-chart'), $('action-marker'));
  drawChartMarker($('state-chart'), $('state-marker'));
}

function drawChartMarker(chartCanvas, markerCanvas) {
  if (!chartCanvas._chartParams) return;

  const { padL, padT, plotW, plotH, nFrames } = chartCanvas._chartParams;
  const dpr = window.devicePixelRatio || 1;
  const rect = chartCanvas.getBoundingClientRect();

  // Size marker canvas to match chart canvas
  markerCanvas.width = rect.width * dpr;
  markerCanvas.height = rect.height * dpr;
  markerCanvas.style.width = rect.width + 'px';
  markerCanvas.style.height = rect.height + 'px';

  const ctx = markerCanvas.getContext('2d');
  ctx.clearRect(0, 0, markerCanvas.width, markerCanvas.height);
  ctx.scale(dpr, dpr);

  // Frame marker line
  const x = padL + (state.frame / Math.max(nFrames - 1, 1)) * plotW;
  ctx.strokeStyle = '#ef5350';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(x, padT);
  ctx.lineTo(x, padT + plotH);
  ctx.stroke();
}

// ── Playback ──
function togglePlay() {
  state.playing = !state.playing;
  $('play-btn').textContent = state.playing ? 'Pause' : 'Play';
  if (state.playing) {
    state.lastFrameTime = performance.now();
    requestAnimationFrame(playLoop);
  }
}

function playLoop(ts) {
  if (!state.playing) return;

  const interval = 1000 / (state.fps * state.speed);
  const elapsed = ts - state.lastFrameTime;

  if (elapsed >= interval) {
    state.lastFrameTime = ts - (elapsed % interval);
    if (state.frame < state.numFrames - 1) {
      state.frame++;
      updateFrame();
    } else {
      state.playing = false;
      $('play-btn').textContent = 'Play';
      return;
    }
  }
  requestAnimationFrame(playLoop);
}

// ── Keyboard ──
function onKey(e) {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;

  switch (e.key) {
    case ' ':
      e.preventDefault();
      togglePlay();
      break;
    case 'ArrowRight':
      e.preventDefault();
      state.playing = false;
      $('play-btn').textContent = 'Play';
      state.frame = Math.min(state.frame + 1, state.numFrames - 1);
      updateFrame();
      break;
    case 'ArrowLeft':
      e.preventDefault();
      state.playing = false;
      $('play-btn').textContent = 'Play';
      state.frame = Math.max(state.frame - 1, 0);
      updateFrame();
      break;
    case 'ArrowUp':
      e.preventDefault();
      if (state.ep > 0) loadEpisode(state.ep - 1);
      break;
    case 'ArrowDown':
      e.preventDefault();
      if (state.ep < state.info.num_episodes - 1) loadEpisode(state.ep + 1);
      break;
    case ']':
      $('speed-select').selectedIndex = Math.min(
        $('speed-select').selectedIndex + 1,
        $('speed-select').options.length - 1
      );
      state.speed = parseFloat($('speed-select').value);
      break;
    case '[':
      $('speed-select').selectedIndex = Math.max($('speed-select').selectedIndex - 1, 0);
      state.speed = parseFloat($('speed-select').value);
      break;
  }
}

// Resize charts on window resize
window.addEventListener('resize', () => { drawCharts(); drawChartMarkers(); });

init();
</script>
</body>
</html>
"""
