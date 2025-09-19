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
            cv2.line(self.background, (0, center_y), (self.background.shape[1], center_y), (0, 0, 255), 2)
            cv2.line(self.background, (center_x, 0), (center_x, self.background.shape[0]), (0, 0, 255), 2)
        _, buffer = cv2.imencode('.png', self.background)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return {
            "output_image_data": f"data:image/png;base64,{encoded_image}"
        }

app = Flask(__name__)

@app.route('/', methods=['POST'])
def process_images():
    data = request.get_json()
    if not data or 'bg_image' not in data or 'puzzle_image' not in data:
        return jsonify({"error": "Invalid JSON payload. Missing 'bg_image' or 'puzzle_image'."}), 400
    
    try:
        bg_image_data = base64.b64decode(data['bg_image'])
        puzzle_image_data = base64.b64decode(data['puzzle_image'])
        
        identifier = GeeTestIdentifier(background=bg_image_data, puzzle_piece=puzzle_image_data, debugger=True)
        result = identifier.find_puzzle_piece_position()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
