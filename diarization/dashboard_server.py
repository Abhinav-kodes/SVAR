import os
import sys
import json
import time
import http.server
import socketserver
import urllib.parse
from datetime import datetime
from denoising.audio_loader import load_audio
from denoising.pipeline import DenoiserPipeline
from diarization.pipeline import DiarizationPipeline

PORT = 8050
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_CALLS_DIR = os.path.join(REPO_ROOT, "data", "sample_calls")
DASHBOARD_DIR = os.path.join(REPO_ROOT, "diarization", "dashboard")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


log("Loading DenoiserPipeline...")
t0 = time.time()
denoiser_pipeline = DenoiserPipeline()
log(f"DenoiserPipeline ready in {time.time()-t0:.1f}s")

log("Loading DiarizationPipeline (pyannote, this may take a while on first load)...")
t0 = time.time()
diarizer_pipeline = DiarizationPipeline()
log(f"DiarizationPipeline object created in {time.time()-t0:.1f}s (model loads lazily on first call)")


class DashboardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if path == "/" or path == "/index.html":
            log("GET / → serving dashboard HTML")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(os.path.join(DASHBOARD_DIR, "index.html"), "rb") as f:
                self.wfile.write(f.read())
            return

        elif path == "/api/sample_calls":
            audio_files = sorted([
                f for f in os.listdir(SAMPLE_CALLS_DIR)
                if f.endswith(('.wav', '.mp3', '.opus')) and not f.endswith('_denoised.wav')
            ])
            log(f"GET /api/sample_calls → {len(audio_files)} files: {audio_files}")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(audio_files).encode('utf-8'))
            return

        elif path.startswith("/audio/"):
            filename = urllib.parse.unquote(path[len("/audio/"):])
            filepath = os.path.join(SAMPLE_CALLS_DIR, filename)

            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                log(f"GET /audio/{filename} → serving {size_mb:.1f}MB")
                self.send_response(200)
                if filename.endswith(".mp3"):
                    content_type = "audio/mpeg"
                elif filename.endswith(".opus"):
                    content_type = "audio/ogg"
                else:
                    content_type = "audio/wav"

                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(os.path.getsize(filepath)))
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                log(f"GET /audio/{filename} → 404 NOT FOUND")
                self.send_error(404, "Audio file not found")
                return

        super().do_GET()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == "/api/diarize":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                payload = json.loads(body.decode("utf-8"))
                filename = payload.get("filename", "")
                filepath = os.path.join(SAMPLE_CALLS_DIR, filename)

                if not os.path.exists(filepath):
                    log(f"POST /api/diarize → 404: {filename} not found")
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"File {filename} not found"}).encode("utf-8"))
                    return

                log(f"POST /api/diarize → processing '{filename}'")
                pipeline_start = time.time()

                # Step 1: Load audio
                t0 = time.time()
                audio, sr = load_audio(filepath, target_sr=16000)
                dur = len(audio) / sr
                log(f"  [1/3] Audio loaded: {dur:.1f}s, {sr}Hz, {len(audio)} samples ({time.time()-t0:.2f}s)")

                # Step 2: Denoise
                t0 = time.time()
                clean_audio, denoise_metrics = denoiser_pipeline.process(audio, sr)
                log(f"  [2/3] Denoising complete: SNR {denoise_metrics['snr_before_db']:.1f}→{denoise_metrics['snr_after_db']:.1f}dB, grade={denoise_metrics['audio_quality_grade']} ({time.time()-t0:.2f}s)")

                # Step 3: Diarize (pyannote)
                t0 = time.time()
                log(f"  [3/3] Running pyannote speaker diarization (first call loads model ~30-60s on CPU)...")
                diar_res = diarizer_pipeline.process(clean_audio, sr)
                log(f"  [3/3] Diarization complete: {len(diar_res['segments'])} segments ({time.time()-t0:.1f}s)")

                diar_res["duration_s"] = round(float(len(audio) / sr), 2)
                diar_res["denoise_metrics"] = denoise_metrics

                total_time = time.time() - pipeline_start
                agent_s = diar_res["talk_ratio"]["agent_duration_s"]
                cust_s = diar_res["talk_ratio"]["customer_duration_s"]
                log(f"  Pipeline done in {total_time:.1f}s — agent={agent_s}s, customer={cust_s}s, {len(diar_res['segments'])} segments")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(diar_res).encode("utf-8"))
            except Exception as e:
                import traceback
                log(f"  ERROR: {e}")
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Error processing diarization: {str(e)}"}).encode("utf-8"))
            return


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main():
    log(f"Starting SVAR Diarization Dashboard at http://localhost:{PORT}")
    log(f"Sample calls: {SAMPLE_CALLS_DIR}")
    server = ReusableTCPServer(("", PORT), DashboardHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
