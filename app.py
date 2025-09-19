import cv2
import numpy as np
import io
import requests
import json
import base64
import os
from flask import Flask, request, jsonify

class GeeTestIdentifier:
    def __init__(self, background, puzzle_piece, debugger=False):
        self.background = self._read_image(background)
        self.puzzle_piece = self._read_image(puzzle_piece)
        self.debugger = debugger
    
    @staticmethod
    def _read_image(image_source):
        return cv2.imdecode(np.frombuffer(image_source, np.uint8), cv2.IMREAD_ANYCOLOR)

    def find_puzzle_piece_position(self):
        edge_puzzle_piece = cv2.Canny(self.puzzle_piece, 100, 200)
        edge_background = cv2.Canny(self.background, 100, 200)
        edge_puzzle_piece_rgb = cv2.cvtColor(edge_puzzle_piece, cv2.COLOR_GRAY2RGB)
        edge_background_rgb = cv2.cvtColor(edge_background, cv2.COLOR_GRAY2RGB)
        res = cv2.matchTemplate(edge_background_rgb, edge_puzzle_piece_rgb, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        h, w = edge_puzzle_piece.shape[:2]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center_x = top_left[0] + w // 2
        center_y = top_left[1] + h // 2
        position_from_left = center_x
        position_from_bottom = self.background.shape[0] - center_y
        if self.debugger:
            cv2.rectangle(self.background, top_left, bottom_right, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.png', self.background)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return {
            "output_image_data": f"data:image/png;base64,{encoded_image}"
        }

app = Flask(__name__)

@app.route('/', methods=['GET'])
def process_images():
    bg_url = request.args.get('bg_url')
    puzzle_url = request.args.get('puzzle_url')
    
    if not bg_url or not puzzle_url:
        return jsonify({"error": "Missing 'bg_url' or 'puzzle_url' query parameters. Please use the /?bg_url=<URL>&puzzle_url=<URL> format."}), 400
    
    try:
        bg_response = requests.get(bg_url, timeout=10)
        puzzle_response = requests.get(puzzle_url, timeout=10)
        bg_response.raise_for_status()
        puzzle_response.raise_for_status()
        identifier = GeeTestIdentifier(background=bg_response.content, puzzle_piece=puzzle_response.content, debugger=True)
        result = identifier.find_puzzle_piece_position()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
