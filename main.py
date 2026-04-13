import flet as ft
import cv2
import base64
import os
import sys
import numpy as np
import threading
import time

def main(page: ft.Page):
    # Tactical HUD Design
    page.title = "AGA GLOBAL | LENS AI"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#050505"
    page.padding = 10

    # UI Elements
    camera_view = ft.Image(src_base64="", width=640, height=480, fit=ft.ImageFit.CONTAIN)
    status_text = ft.Text("SYSTEM: SCANNING", color="#00FF41", weight="bold", font_family="monospace")
    
    page.add(
        ft.Row([ft.Text("LENS AI V2.3 | ONNX CORE", size=10, color="#555555"), 
                ft.Text("AGA GLOBAL TECH", size=10, color="#00FF41")], 
               alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ft.Container(content=camera_view, border=ft.border.all(1, "#333333"), border_radius=15),
        ft.Row([status_text], alignment=ft.MainAxisAlignment.CENTER),
    )

    # Path Logic for Android
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_path, "assets", "yolo26n.onnx")

    # Load ONNX Engine via OpenCV DNN
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        status_text.value = "ENGINE: ONNX ONLINE"
    except:
        status_text.value = "ENGINE: ASSET ERROR"

    page.update()

    # Shared variable for the vision thread
    cap = cv2.VideoCapture(0)

    def vision_thread():
        while True:
            success, frame = cap.read()
            if success:
                # 1. AI Inference (640x640)
                # We blob it for the ONNX engine
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
                net.setInput(blob)
                
                # Note: On a Celeron, we only run inference when the CPU is ready
                # For now, we update the HUD every frame
                h, w, _ = frame.shape
                # Drawing a tactical reticle
                cv2.line(frame, (int(w/2)-20, int(h/2)), (int(w/2)+20, int(h/2)), (0, 255, 65), 1)
                cv2.line(frame, (int(w/2), int(h/2)-20), (int(w/2), int(h/2)+20), (0, 255, 65), 1)
                
                # 2. Update UI
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                camera_view.src_base64 = base64.b64encode(buffer).decode("utf-8")
                page.update()
            
            # 15 FPS throttle to prevent Celeron overheating
            time.sleep(0.06)

    # Start vision in a separate thread so UI stays responsive
    threading.Thread(target=vision_thread, daemon=True).start()

ft.app(target=main)
    
