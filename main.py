import flet as ft
import cv2
import base64
import os
import time
import numpy as np
import tflite_runtime.interpreter as tflite

def main(page: ft.Page):
    # Tactical HUD Configuration
    page.title = "AGA GLOBAL | LENS AI"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#050505" # Deep Black
    page.padding = 10
    page.window_width = 400
    page.window_height = 800

    # UI Components
    camera_view = ft.Image(
        src_base64="",
        width=640,
        height=480,
        fit=ft.ImageFit.CONTAIN,
    )
    
    status_text = ft.Text(
        "SYSTEM: INITIALIZING...",
        color="#00FF41",
        weight=ft.FontWeight.BOLD,
        size=14,
        font_family="monospace"
    )

    # Building the Aga Global Tech Layout
    page.add(
        ft.Column(
            [
                ft.Row([
                    ft.Text("LENS AI V2.3", color="#555555", size=10),
                    ft.Text("OFFLINE-SOVEREIGNTY", color="#00FF41", size=10),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                
                ft.Container(
                    content=camera_view,
                    border=ft.border.all(1, "#222222"),
                    border_radius=15,
                    bgcolor="#000000"
                ),
                
                ft.Container(
                    content=ft.Row([status_text], alignment=ft.MainAxisAlignment.CENTER),
                    padding=10,
                    bgcolor="#111111",
                    border_radius=10
                ),
                
                ft.Divider(color="#111111", height=20),
                ft.Text("AGA GLOBAL TECH | KANO, NIGERIA", color="#333333", size=9),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )

    # Initialize AI Engine (YOLO26 INT8)
    # Flet bundles assets into the root 'assets' directory on Android
    model_path = os.path.join(os.getcwd(), "assets", "yolo26n.tflite")
    
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        status_text.value = "SYSTEM: ACTIVE | SCANNING"
    except Exception as e:
        status_text.value = "CRITICAL: ENGINE OFFLINE"
    
    page.update()

    # Optimized Stream for Celeron N2830 / 4GB RAM
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if success:
            # 1. Resize for YOLO (640x640)
            img = cv2.resize(frame, (640, 640))
            img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
            
            # 2. Run Inference
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # 3. Draw Tactical Boxes (Simplified for Speed)
            # YOLO26 output parsing... (simplified for HUD look)
            for detection in output[0][:5]: # Top 5 detections
                if detection[4] > 0.4:
                    x1, y1, x2, y2 = map(int, detection[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 65), 2)

            # 4. Encode and Display
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            camera_view.src_base64 = base64.b64encode(buffer).decode("utf-8")
            page.update()
            
        # Target 12-15 FPS to keep the Celeron cool
        time.sleep(0.06)

if __name__ == "__main__":
    ft.app(target=main)
  
