import os
import json
import torch
import glob
from http.server import HTTPServer, BaseHTTPRequestHandler
from model import ConnectFourNet

CHECKPOINT_DIR = "checkpoints"

def load_best_model():
    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt")))
    if not files:
        print("No checkpoints found — model will play randomly")
        return None
    path = files[-1]
    print(f"Loaded: {path}")
    model = ConnectFourNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

model = load_best_model()

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/move":
            self.send_response(404); self.end_headers(); return

        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))

        col = get_model_move(body["board"], body["valid_cols"])
        self.send_json({"col": col})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

def get_model_move(state, valid_cols):
    import random
    if model is None or not valid_cols:
        return random.choice(valid_cols)

    import torch, numpy as np
    x = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        scores = model(x).numpy()

    # Mask invalid cols
    for c in range(7):
        if c not in valid_cols:
            scores[c] = -1e9

    return int(np.argmax(scores))

if __name__ == "__main__":
    server = HTTPServer(("localhost", 8766), Handler)
    print("Play server running at http://localhost:8766")
    server.serve_forever()
