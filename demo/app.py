"""
HybridMatt Web Demo — real-time video matting in the browser.

Captures webcam server-side with OpenCV, runs the model, and streams
MJPEG directly to the browser — no WebSocket per-frame overhead.

Usage (from project root):
  python -m demo.app --checkpoint checkpoints/stage2/best.pth
  python -m demo.app --checkpoint checkpoints/stage2/best.pth --downsample-ratio 0.5 --port 5000

Then open http://localhost:5000 in your browser.
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template, jsonify
from torchvision.transforms.functional import to_tensor

# Add project root to path so we can import model/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import HybridMattingNetwork

# ── Args ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='HybridMatt web demo')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--downsample-ratio', type=float, default=0.5)
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--port', type=int, default=5000)
parser.add_argument('--camera', type=int, default=0)
args = parser.parse_args()

# ── Model ───────────────────────────────────────────────────────────────────

device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f'Device: {device}')

model = HybridMattingNetwork(variant='mobilenetv3', pretrained_backbone=False)
state = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(state, strict=False)
model.eval()
model.to(device)
print(f'Loaded: {args.checkpoint}')

# ── Shared state (written by inference thread, read by MJPEG streams) ──────

lock = threading.Lock()
latest = {
    'input': None,     # JPEG bytes
    'alpha': None,     # JPEG bytes
    'composite': None, # JPEG bytes
    'fps': 0.0,
    'running': False,
}

# ── Inference loop (runs in a background thread) ───────────────────────────

def inference_loop():
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f'ERROR: Cannot open camera {args.camera}')
        return

    r1 = r2 = r3 = r4 = None
    latest['running'] = True
    print('Camera opened — inference running.')

    while latest['running']:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        t0 = time.time()

        # Convert BGR -> RGB -> tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        src = to_tensor(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            fgr, pha, r1, r2, r3, r4 = model(src, r1, r2, r3, r4,
                                               downsample_ratio=args.downsample_ratio)

        # Alpha matte
        pha_np = (pha[0, 0].cpu().numpy() * 255).astype(np.uint8)
        alpha_bgr = cv2.cvtColor(pha_np, cv2.COLOR_GRAY2BGR)

        # Green-screen composite
        fgr_np = (fgr[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        alpha_3ch = np.stack([pha[0, 0].cpu().numpy()] * 3, axis=2).astype(np.float32)
        green = np.zeros_like(fgr_np)
        green[:, :, 1] = 177
        composite = (fgr_np.astype(np.float32) * alpha_3ch +
                     green.astype(np.float32) * (1 - alpha_3ch)).astype(np.uint8)
        composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

        # Encode to JPEG
        _, inp_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        _, alp_buf = cv2.imencode('.jpg', alpha_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        _, com_buf = cv2.imencode('.jpg', composite_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])

        fps = 1.0 / max(time.time() - t0, 1e-6)

        with lock:
            latest['input'] = inp_buf.tobytes()
            latest['alpha'] = alp_buf.tobytes()
            latest['composite'] = com_buf.tobytes()
            latest['fps'] = round(fps, 1)

    cap.release()
    print('Camera released.')


# ── Flask app ───────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))


@app.route('/')
def index():
    return render_template('demo.html')


def mjpeg_stream(key):
    """Generator that yields MJPEG frames for a given stream key."""
    while True:
        with lock:
            frame_bytes = latest[key]
        if frame_bytes is None:
            time.sleep(0.03)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)  # ~100 FPS cap to avoid busy-loop


@app.route('/stream/input')
def stream_input():
    return Response(mjpeg_stream('input'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream/alpha')
def stream_alpha():
    return Response(mjpeg_stream('alpha'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream/composite')
def stream_composite():
    return Response(mjpeg_stream('composite'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    with lock:
        return jsonify({'fps': latest['fps'], 'running': latest['running']})


if __name__ == '__main__':
    # Start inference in background thread
    thread = threading.Thread(target=inference_loop, daemon=True)
    thread.start()

    print(f'\n  Open http://localhost:{args.port} in your browser\n')
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
