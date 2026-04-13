import flet as ft
import cv2
import base64
import os
import sys
import numpy as np
import tflite_runtime.interpreter as tflite
import time

def main(page: ft.Page):
    page.title = "AGA GLOBAL | LENS AI"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#050505"
    page.padding = 10

    # UI Components
    camera_view = ft.Image(src_base64="", width=640, height=480, fit=ft.ImageFit.CONTAIN)
    status_text = ft.Text("SYSTEM: SCANNING", color="#00FF41", weight="bold", font_family="monospace")
    
    # Branding
    header = ft.Row([
        ft.Text("SOVEREIGNTY V2.3", size=10, color="#555555"),
        ft.Text("AGA GLOBAL TECH", size=10, color="#00FF41")
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

    page.add(
        header,
        ft.Container(content=camera_view, border=ft.border.all(1, "#333333"), border_radius=15),
        ft.Row([status_text], alignment=ft.MainAxisAlignment.CENTER),
    )

    # Asset Path Logic: Android unzips assets to a specific internal path
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_path, "assets", "yolo26n.tflite")

    # Initialize Engine
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        status_text.value = "ENGINE: ONLINE | SECURE"
    except:
        status_text.value = "ENGINE: OFFLINE (ASSET ERROR)"
    
    page.update()

    # Camera Loop optimized for Celeron/4GB RAM
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if success:
            # HUD Visuals
            h, w, _ = frame.shape
            cv2.rectangle(frame, (int(w*0.3), int(h*0.3)), (int(w*0.7), int(h*0.7)), (0, 255, 65), 1)
            
            # Encode for HUD display
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            camera_view.src_base64 = base64.b64encode(buffer).decode("utf-8")
            page.update()
        time.sleep(0.06) # Target 15 FPS to stay cool

if __name__ == "__main__":
    ft.app(target=main)
        
