import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

LOG_FILE = "training_log.json"
CHECKPOINT_DIR = "checkpoints"

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/log":
            self.serve_log()
        elif self.path == "/checkpoints":
            self.serve_checkpoints()
        else:
            self.send_response(404)
            self.end_headers()

    def serve_log(self):
        if not os.path.exists(LOG_FILE):
            self.send_json([])
            return
        entries = []
        with open(LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        self.send_json(entries)

    def serve_checkpoints(self):
        if not os.path.exists(CHECKPOINT_DIR):
            self.send_json([])
            return
        files = sorted([
            f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")
        ])
        self.send_json(files)

    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")  # Allow browser fetch
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # Silence request logs

if __name__ == "__main__":
    server = HTTPServer(("localhost", 8765), Handler)
    print("Dashboard server running at http://localhost:8765")
    server.serve_forever()
