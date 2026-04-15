import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import pyautogui

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

mp_draw = vision.drawing_utils
hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS


def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        for device_id in range(3):
            cap = cv2.VideoCapture(device_id, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, _ = cap.read()
            if ok:
                print(f'Opened camera device {device_id} with backend {backend}')
                return cap
            cap.release()
    return None


# tao hand landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

# dieu khien
cap = open_camera()
pTime = 0

left_click_active = False
right_click_active = False
scroll_mode = False
scroll_prev_y = 0
last_action_time = 0
action_cooldown = 0.8  
click_threshold = 45
volume_threshold = 35
scroll_threshold = 50
prev_mouse_x = None
prev_mouse_y = None
smoothing = 0.25
margin_ratio = 0.15

print("Hệ thống đang khởi động... Nhấn 'q' để thoát.")

if cap is None:
    print("Không tìm thấy camera hoạt động. Hãy kiểm tra:")
    print(" - Camera đã được bật chưa?")
    print(" - Camera có đang được dùng bởi ứng dụng khác không?")
    print(" - Quyền truy cập camera đã cấp cho Python chưa?")
    print(" - Thử chỉ số camera khác với CV2")
    sys.exit(1)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Không thể đọc khung hình từ camera. Có thể camera đang bận hoặc tín hiệu yếu.")
        break

    # lat anh
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    timestamp = int(time.time() * 1000) #nhan dien tay
    results = detector.detect_for_video(mp_image, timestamp)

    status = "RELAXING"
    color = (0, 255, 0)
    gesture_text = ""

    
    if results.hand_landmarks:
        for handLms in results.hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, hand_connections)

            
            thumb = handLms[4]
            index = handLms[8]
            middle = handLms[12]
            ring = handLms[16]
            pinky = handLms[20]

            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)
            mx, my = int(middle.x * w), int(middle.y * h)
            rx, ry = int(ring.x * w), int(ring.y * h)
            px, py = int(pinky.x * w), int(pinky.y * h)

            # Vẽ hiệu ứng các đầu ngón
            cv2.circle(img, (tx, ty), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (ix, iy), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (mx, my), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (tx, ty), (ix, iy), (255, 0, 255), 2)
            cv2.line(img, (tx, ty), (mx, my), (255, 0, 255), 2)
            cv2.line(img, (ix, iy), (mx, my), (255, 0, 255), 1)

            left_distance = math.hypot(ix - tx, iy - ty)
            right_distance = math.hypot(mx - tx, my - ty)
            ring_distance = math.hypot(rx - tx, ry - ty)
            pinky_distance = math.hypot(px - tx, py - ty)
            scroll_distance = math.hypot(mx - ix, my - iy)

            if left_distance < click_threshold:
                scroll_mode = False
                right_click_active = False
                if not left_click_active and time.time() - last_action_time > action_cooldown:
                    left_click_active = True
                    last_action_time = time.time()
                    pyautogui.click()
                    gesture_text = "LEFT CLICK"
                status = "LEFT CLICK READY"
                color = (0, 0, 255)
            elif right_distance < click_threshold:
                scroll_mode = False
                left_click_active = False
                if not right_click_active and time.time() - last_action_time > action_cooldown:
                    right_click_active = True
                    last_action_time = time.time()
                    pyautogui.click(button='right')
                    gesture_text = "RIGHT CLICK"
                status = "RIGHT CLICK READY"
                color = (0, 0, 255)
            elif ring_distance < volume_threshold:
                scroll_mode = False
                left_click_active = False
                right_click_active = False
                if time.time() - last_action_time > action_cooldown:
                    last_action_time = time.time()
                    pyautogui.press('volumeup')
                    gesture_text = "VOLUME UP"
                status = "VOLUME UP"
                color = (0, 128, 255)
            elif pinky_distance < volume_threshold:
                scroll_mode = False
                left_click_active = False
                right_click_active = False
                if time.time() - last_action_time > action_cooldown:
                    last_action_time = time.time()
                    pyautogui.press('volumedown')
                    gesture_text = "VOLUME DOWN"
                status = "VOLUME DOWN"
                color = (0, 128, 255)
            elif scroll_distance < scroll_threshold:
                left_click_active = False
                right_click_active = False
                if not scroll_mode:
                    scroll_mode = True
                    scroll_prev_y = int((index.y + middle.y) / 2 * screen_h)
                    gesture_text = "SCROLL MODE"
                else:
                    current_scroll_y = int((index.y + middle.y) / 2 * screen_h)
                    delta = scroll_prev_y - current_scroll_y
                    if abs(delta) > 5:
                        pyautogui.scroll(delta * 3)
                        gesture_text = f"SCROLL {'UP' if delta > 0 else 'DOWN'}"
                        scroll_prev_y = current_scroll_y
                    else:
                        gesture_text = "SCROLL MODE"
                status = "SCROLLING"
                color = (255, 165, 0)
            else:
                left_click_active = False
                right_click_active = False
                scroll_mode = False
                margin_w = int(w * margin_ratio) #anh xa toa do
                margin_h = int(h * margin_ratio)
                target_x = min(max(index.x * w, margin_w), w - margin_w)
                target_y = min(max(index.y * h, margin_h), h - margin_h)
                norm_x = (target_x - margin_w) / (w - 2 * margin_w)
                norm_y = (target_y - margin_h) / (h - 2 * margin_h)
                screen_x = int(norm_x * screen_w)
                screen_y = int(norm_y * screen_h)

                if prev_mouse_x is None:
                    prev_mouse_x = screen_x
                    prev_mouse_y = screen_y

                smooth_x = int(prev_mouse_x + (screen_x - prev_mouse_x) * smoothing)
                smooth_y = int(prev_mouse_y + (screen_y - prev_mouse_y) * smoothing)
                prev_mouse_x = smooth_x
                prev_mouse_y = smooth_y

                pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)
                status = "MOVE CURSOR"
                color = (0, 255, 0)
                gesture_text = "MOVING CURSOR"


    
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.rectangle(img, (0, h-80), (w, h), (50, 50, 50), cv2.FILLED)
    cv2.putText(img, f"SYSTEM STATUS: {status}", (20, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(img, gesture_text, (20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.namedWindow("IoT Hand Gesture Control", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("IoT Hand Gesture Control", cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow("IoT Hand Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()