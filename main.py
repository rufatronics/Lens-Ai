import flet as ft
import cv2
import base64
import os
import sys
import threading

def main(page: ft.Page):
    page.title = "AGA GLOBAL | LENS AI"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#050505"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    # --- UI ELEMENTS ---
    status_label = ft.Text("SYSTEM READY", color="#00FF41", weight="bold")
    log_window = ft.Text("Waiting for initialization...", size=10, color="#555555")
    camera_view = ft.Image(visible=False, width=640, height=480, fit=ft.ImageFit.CONTAIN)
    
    start_button = ft.ElevatedButton(
        "INITIALIZE SYSTEM", 
        icon=ft.icons.PLAY_ARROW,
        color="#00FF41",
        on_click=lambda _: start_vision()
    )

    # --- DIAGNOSTIC LOGIC ---
    def log(msg):
        log_window.value = f"> {msg}"
        page.update()

    def start_vision():
        # 1. Request Camera Permission (THE FIX)
        log("Requesting Camera Access...")
        # Flet handles the Android system handshake here
        page.permission_request(ft.PermissionType.CAMERA)
        
        # 2. Verify Model Path
        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(sys.executable)
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
            
        model_path = os.path.join(base_path, "assets", "yolo26n.onnx")
        
        if not os.path.exists(model_path):
            log(f"CRITICAL ERROR: Model not found at {model_path}")
            return

        log("Model Found. Opening Camera...")
        start_button.visible = False
        camera_view.visible = True
        page.update()

        # 3. Start the actual vision thread
        threading.Thread(target=vision_loop, args=(model_path,), daemon=True).start()

    def vision_loop(model_path):
        try:
            net = cv2.dnn.readNetFromONNX(model_path)
            cap = cv2.VideoCapture(0)
            
            while True:
                success, frame = cap.read()
                if success:
                    # Simple drawing to confirm it works
                    cv2.putText(frame, "LENS ACTIVE", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 65), 2)
                    
                    _, buffer = cv2.imencode(".jpg", frame)
                    camera_view.src_base64 = base64.b64encode(buffer).decode("utf-8")
                    page.update()
                else:
                    log("ERROR: Camera stream lost.")
                    break
        except Exception as e:
            log(f"ENGINE ERROR: {str(e)}")

    # Initial Layout
    page.add(
        ft.Column([
            ft.Text("AGA GLOBAL TECH", size=20, weight="bold", color="#00FF41"),
            status_label,
            ft.Divider(color="#222222"),
            camera_view,
            start_button,
            log_window
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    )

ft.app(target=main)
                
