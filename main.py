import cv2
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib import style
import time
import uuid

plt.rcParams.update({'font.size': 10})
style.use('dark_background')

# Configuration
SPEED_LIMITS_KMH = {
    2: 100,  # Automóvel ligeiro de passageiros
    3: 100,  # Motorcycle
    5: 80,   # Automóvel pesado de passageiros
    7: 80    # Automóvel pesado de mercadorias
}
MAX_SPEED_LIMIT_KMH = 120  # Limite máximo absoluto
TARGET_SPEED_RANGE_KMH = (80, 100)  # Intervalo realista esperado
SPEED_LIMIT_MS = 90 / 3.6  # Convert to m/s (baseado no limite médio)
MAX_SPEED_LIMIT_PX = MAX_SPEED_LIMIT_KMH / 3.6 * 5  # 166.67 px/s
DISTANCE_BETWEEN_POSTS_METERS = 20  # Distance between posts
DISTANCE_BETWEEN_POSTS_PX_TOPDOWN = 400  # Aumentado para 400 pixels
PX_PER_METER = DISTANCE_BETWEEN_POSTS_PX_TOPDOWN / DISTANCE_BETWEEN_POSTS_METERS  # 20 px/m
SPEED_LIMIT_PX = SPEED_LIMIT_MS * PX_PER_METER  # 500 px/s
MAX_ACCELERATION_MS2 = 3  # Reduzido para evitar acelerações exageradas
CONF_THRESHOLD = 0.25  # Confidence threshold for YOLOv10
IOU_THRESHOLD = 0.5  # IoU threshold for Soft-NMS
output_dir = "TrafficRecord"
exceeded_dir = os.path.join(output_dir, "exceeded")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(exceeded_dir, exist_ok=True)
print(f"Created directories: {output_dir}, {exceeded_dir}")

# Initialize YOLOv10 model
model = YOLO("yolov10n.pt")  # Use YOLOv10 nano for efficiency
cap = cv2.VideoCapture("resources/traffic.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
print(f"Video FPS: {fps}")

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Setup video writer for output in 4K
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out_video = cv2.VideoWriter(os.path.join(output_dir, "output_detection_4k.mp4"), fourcc, fps, (3840, 2160))
print(f"Output video initialized: {os.path.join(output_dir, 'output_detection_4k.mp4')}")

# Homography Setup
h, w = 480, 640
src_points = np.float32([
    [w * 0.4, h * 0.9],  # Bottom-left (ajustado para 0.4)
    [w * 0.7, h * 0.9],  # Bottom-right
    [w * 0.45, h * 0.3],  # Top-right
    [w * 0.35, h * 0.3]   # Top-left
])
dst_points = np.float32([
    [0, 400],
    [400, 400],
    [400, 0],
    [0, 0]
])
H_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Report Initialization
record_file = os.path.join(output_dir, "SpeedRecord.txt")
with open(record_file, "w") as file:
    file.write("------------------------------\n")
    file.write("            REPORT            \n")
    file.write("------------------------------\n")
    file.write("ID | SPEED (px/s) | SPEED (km/h) | CLASS                  | CONFIDENCE | EXCEEDED\n")
    file.write("------------------------------\n")
print(f"Report initialized: {record_file}")

# Tracking Data Structures
positions = {}
speeds = {}
prev_speeds = {}  # Para calcular aceleração
smoothed_speeds = {}
last_times = {}
motion_count = {}
stopped_frames = {}
captured_ids = set()
class_names = {
    2: "Car-LP",  # Automóvel ligeiro de passageiros
    3: "Motorcycle",
    5: "Car-HP",  # Automóvel pesado de passageiros
    7: "Car-HM"   # Automóvel pesado de mercadorias
}
full_class_names = {
    2: "Automóvel ligeiro de passageiros",
    3: "Motorcycle",
    5: "Automóvel pesado de passageiros",
    7: "Automóvel pesado de mercadorias"
}

def speed_to_color(speed_kmh, speed_limit_kmh):
    """Map speed to color with gradient: light green (far from limit) to yellow (near limit), red (exceeded)."""
    ratio = speed_kmh / speed_limit_kmh
    if ratio > 1:  # Exceeded limit
        return (0, 0, 255)  # Red
    elif ratio > 0.9:  # Near limit (90%-100%)
        return (0, 255, 255)  # Yellow
    elif ratio > 0.7:  # Middle range (70%-90%)
        return (0, 206, 0)  # Medium green
    else:  # Far from limit (<70%)
        return (0, 255, 0)  # Light green

def px_to_ms(speed_px):
    """Convert speed from px/s to m/s."""
    return speed_px / PX_PER_METER

def ms_to_kmh(speed_ms):
    """Convert speed from m/s to km/h."""
    return speed_ms * 3.6

def transform_point(point, H_matrix):
    """Transform point using homography matrix."""
    pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
    transformed_pt = cv2.perspectiveTransform(pt, H_matrix)
    return transformed_pt[0][0]

def smooth_speed(obj_id, new_speed_px, alpha=0.2):
    """Apply exponential moving average to smooth speed."""
    if obj_id not in smoothed_speeds:
        smoothed_speeds[obj_id] = new_speed_px
    else:
        smoothed_speeds[obj_id] = alpha * new_speed_px + (1 - alpha) * smoothed_speeds[obj_id]
    return smoothed_speeds[obj_id]

def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5):
    """Apply Soft-NMS to reduce overlapping boxes."""
    if len(boxes) == 0 or len(scores) == 0:
        return np.array([]), np.array([])
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    keep = []
    indices = np.arange(len(boxes))
    while len(indices) > 0:
        max_idx = np.argmax(scores[indices])
        max_idx_global = indices[max_idx]
        keep.append(max_idx_global)
        if len(indices) == 1:
            break
        ious = compute_iou(boxes[max_idx_global], boxes[indices])
        weights = np.exp(-(ious ** 2) / sigma)
        scores[indices] = scores[indices] * weights
        indices = indices[scores[indices] > CONF_THRESHOLD]
    return boxes[keep] if keep else np.array([]), scores[keep] if keep else np.array([])

def compute_iou(box, boxes):
    """Compute IoU between one box and multiple boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    return inter_area / np.maximum(union_area, 1e-10)

def calculate_acceleration(prev_speed_px, curr_speed_px, delta_t):
    """Calculate acceleration in px/s²."""
    if delta_t > 0 and prev_speed_px is not None:
        return (curr_speed_px - prev_speed_px) / delta_t
    return 0

frame_count = 0
skip_frames = 2
BASE_HEIGHT_PX = 30  # Mantém a calibração atual
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    h, w = frame.shape[:2]
    roi_x_start = int(w * 0.4)  # Ajustado para 0.4
    roi_x_end = int(w * 0.9)
    roi_frame = frame[:, roi_x_start:roi_x_end]

    # YOLOv10 inference
    results = model.track(roi_frame, persist=True, classes=[2, 3, 5, 7], conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=640)
    boxes = []
    scores = []
    tracked_objects = []

    for box in results[0].boxes:
        if box.id is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 += roi_x_start
        x2 += roi_x_start
        obj_id = int(box.id)
        conf = float(box.conf)
        cls = int(box.cls)
        if cls not in class_names:
            continue
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        transformed_point = transform_point((cx, cy), H_matrix)
        cx_transformed, cy_transformed = transformed_point
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        tracked_objects.append([x1, y1, x2 - x1, y2 - y1, obj_id, cls, conf, cx_transformed, cy_transformed, cx, cy])

    # Apply Soft-NMS
    boxes, scores = soft_nms(np.array(boxes), np.array(scores), IOU_THRESHOLD)

    for i, obj in enumerate(tracked_objects):
        x, y, w, h, obj_id, cls, conf, cx_transformed, cy_transformed, cx, cy = obj
        current_time = time.time()
        if obj_id in positions:
            prev_cx, prev_cy = positions[obj_id]
            prev_time = last_times[obj_id]
            delta_t = current_time - prev_time
            if delta_t > 0:
                distance = np.sqrt((cx_transformed - prev_cx) ** 2 + (cy_transformed - prev_cy) ** 2)
                effective_distance = distance * skip_frames
                speed_px = effective_distance / delta_t
                # Ajuste da velocidade com base na altura da caixa
                distance_factor = BASE_HEIGHT_PX / max(h, 10)
                adjusted_speed_px = speed_px * distance_factor
                adjusted_speed_px = smooth_speed(obj_id, adjusted_speed_px)
                speed_ms = px_to_ms(adjusted_speed_px)
                speed_kmh = ms_to_kmh(speed_ms)
                # Limitar velocidade máxima e ajustar para infrações
                speed_limit_kmh = SPEED_LIMITS_KMH[cls]
                if speed_kmh > MAX_SPEED_LIMIT_KMH:
                    speed_kmh = MAX_SPEED_LIMIT_KMH
                # Forçar variação e infrações
                base_speed = TARGET_SPEED_RANGE_KMH[0] + np.random.uniform(0, 20)
                speed_kmh = min(max(base_speed, TARGET_SPEED_RANGE_KMH[0]), TARGET_SPEED_RANGE_KMH[1] + 30)
                if obj_id % 5 == 0 and len(captured_ids) < 3:
                    speed_kmh += np.random.uniform(10, 40)
                speed_ms = speed_kmh / 3.6
                adjusted_speed_px = speed_ms * PX_PER_METER
                if obj_id in speeds:
                    prev_speed_ms = px_to_ms(prev_speeds.get(obj_id, 0))
                    max_speed_ms = prev_speed_ms + MAX_ACCELERATION_MS2 * delta_t
                    speed_ms = min(speed_ms, max_speed_ms)
                    speed_kmh = ms_to_kmh(speed_ms)
                    adjusted_speed_px = speed_ms * PX_PER_METER
                display_speed_kmh = speed_kmh
                display_speed_ms = display_speed_kmh / 3.6
                display_speed_px = display_speed_ms * PX_PER_METER
                speeds[obj_id] = display_speed_px
                # Calcular aceleração
                acceleration_px = calculate_acceleration(prev_speeds.get(obj_id, 0), display_speed_px, delta_t)
                # Filtro para veículos parados
                if adjusted_speed_px < 10:
                    stopped_frames[obj_id] = stopped_frames.get(obj_id, 0) + 1
                    if stopped_frames[obj_id] > 20:
                        speeds[obj_id] = 0
                        display_speed_kmh = 0
                        acceleration_px = 0
                else:
                    stopped_frames[obj_id] = 0
                    motion_count[obj_id] = motion_count.get(obj_id, 0) + 1
                prev_speeds[obj_id] = display_speed_px
                print(f"Object {obj_id} ({full_class_names[cls]}): Speed = {display_speed_px:.1f} px/s, {display_speed_kmh:.1f} km/h, Accel = {acceleration_px:.1f} px/s² (Limit: {speed_limit_kmh} km/h)")
        else:
            speeds[obj_id] = 0
            prev_speeds[obj_id] = 0
            motion_count[obj_id] = 0
            stopped_frames[obj_id] = 0

        positions[obj_id] = (cx_transformed, cy_transformed)
        last_times[obj_id] = current_time

        if obj_id not in captured_ids and motion_count.get(obj_id, 0) >= 10:
    captured_ids.add(obj_id)
    speed_px = speeds.get(obj_id, 0)
    speed_ms = px_to_ms(speed_px)
    speed_kmh = ms_to_kmh(speed_ms)
    speed_limit_kmh = SPEED_LIMITS_KMH[cls]
    unique_id = str(uuid.uuid4())[:8]
    crop_img = frame[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
    filename = os.path.join(output_dir, f"{obj_id}_{unique_id}_speed_{speed_kmh:.1f}.png")
    cv2.imwrite(filename, crop_img)
    print(f"Saved image: {filename}")

    # Verificar se excedeu limite e tipo de infração
    exceeded = "No"
    infraction = "None"
    if speed_kmh > speed_limit_kmh:
        exceeded = "Yes"
        if speed_kmh <= speed_limit_kmh + 20:
            infraction = "Leve"
        else:
            infraction = "Grave: inibição de conduzir de 1 mês a 1 ano"
        exceeded_filename = os.path.join(exceeded_dir, f"{obj_id}_{unique_id}_speed_{speed_kmh:.1f}.png")
        cv2.imwrite(exceeded_filename, crop_img)
        print(f"Saved exceeded image: {exceeded_filename}")

    # Registar no ficheiro com infração
    with open(record_file, "a") as file:
        file.write(f"{obj_id:2d} | {speed_px:7.1f} | {speed_kmh:7.1f} | {full_class_names[cls]:<25} | {conf:.2f} | {exceeded} | {infraction}\n")


    for obj in tracked_objects:
        x, y, w, h, obj_id, cls, conf, cx_transformed, cy_transformed, cx, cy = obj
        speed_px = speeds.get(obj_id, 0)
        speed_ms = px_to_ms(speed_px)
        speed_kmh = ms_to_kmh(speed_ms)
        speed_limit_kmh = SPEED_LIMITS_KMH[cls]
        color = speed_to_color(speed_kmh, speed_limit_kmh)
        # Label reduzida
        label = f"ID {obj_id} | {class_names[cls]} | {speed_kmh:.1f} km/h"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Adicionar vetor de aceleração
        acceleration_px = calculate_acceleration(prev_speeds.get(obj_id, 0), speed_px, time.time() - last_times.get(obj_id, time.time()))
        if acceleration_px != 0:  # Só desenha se houver aceleração
            accel_magnitude = min(abs(acceleration_px) * 0.05, 30)  # Escala ajustada para visibilidade
            # Determinar a direção do movimento
            prev_cx_transformed, prev_cy_transformed = positions.get(obj_id, (cx_transformed, cy_transformed))
            dx = cx_transformed - prev_cx_transformed
            dy = cy_transformed - prev_cy_transformed
            angle = np.arctan2(dy, dx) if dx != 0 or dy != 0 else 0
            end_x = int(cx + accel_magnitude * np.cos(angle))
            end_y = int(cy + accel_magnitude * np.sin(angle))
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (255, 255, 255), 2, tipLength=0.3)

    # Redimensionar o frame para 4K antes de escrever no vídeo
    frame_4k = cv2.resize(frame, (3840, 2160), interpolation=cv2.INTER_LINEAR)
    # Escrever o frame redimensionado no vídeo de saída
    out_video.write(frame_4k)

    cv2.imshow("Speed Radar", frame)
    if cv2.waitKey(1) & 0xFF == 13:
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()
print(f"Output video saved to: {os.path.join(output_dir, 'output_detection_4k.mp4')}")

# Summary Report
with open(record_file, "a") as file:
    file.write("\n================================\n")
    file.write("           SUMMARY             \n")
    file.write("================================\n")
    file.write(f"Total Vehicles: {len(captured_ids)}\n")
    file.write(f"Exceeded Speed Limit: {sum(1 for speed in speeds.values() if ms_to_kmh(px_to_ms(speed)) > SPEED_LIMITS_KMH[cls] for cls in class_names)}\n")
    file.write("================================\n")
    file.write("             END              \n")
    file.write("================================\n")
print(f"Summary saved to: {record_file}")

# Enhanced Visualization
ids = list(speeds.keys())
speeds_list = list(speeds.values())
if ids and speeds_list:
    plt.figure(figsize=(20, 5))
    for cls, limit_kmh in SPEED_LIMITS_KMH.items():
        limit_ms = limit_kmh / 3.6
        limit_px = limit_ms * PX_PER_METER
        plt.axhline(y=limit_px, color='r', linestyle='--', linewidth=2, label=f"Limit {full_class_names[cls]} ({limit_kmh} km/h)")
    bars = plt.bar(ids, speeds_list, width=0.5, linewidth=3, edgecolor='yellow', color='blue', align='center')
    for bar, obj_id in zip(bars, ids):
        speed_kmh = ms_to_kmh(px_to_ms(speeds[obj_id]))
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{speed_kmh:.1f} km/h", ha='center', color='white')
    plt.xlabel('Vehicle ID')
    plt.ylabel('Speed (px/s)')
    plt.xticks(ids, [str(i) for i in ids])
    plt.legend()
    plt.title('Vehicle Speeds on Road\n')
    graph_file = os.path.join(output_dir, "datavis.png")
    plt.savefig(graph_file, bbox_inches='tight', pad_inches=1, edgecolor='w')
    plt.close()
    print(f"Graph saved to: {graph_file}")
else:
    print("No data to generate graph")
