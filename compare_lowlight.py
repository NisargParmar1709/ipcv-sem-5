import argparse
import cv2
import time
import sys
import numpy as np
from detector import Detector
from enhance import apply_mode
import matplotlib.pyplot as plt

def process_video(video_path, model_path, mode):
    detector = Detector(
        model_path=model_path,
        input_shape=(192, 192),
        score_th=0.4,
        nms_th=0.5,
        providers=['CPUExecutionProvider']
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return None
        
    total_persons = 0
    infer_times = []
    fps_list = []
    
    last_tick = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (640, 360))
        enhanced, _ = apply_mode(frame, mode)
        
        start_time = time.perf_counter()
        bboxes, _, _ = detector.inference(enhanced)
        infer_ms = (time.perf_counter() - start_time) * 1000
        infer_times.append(infer_ms)
        
        total_persons += len(bboxes)
        
        now = time.time()
        dt = now - last_tick
        if dt > 0:
            fps_list.append(1.0 / dt)
        last_tick = now
        
    cap.release()
    
    return {
        'total_persons': total_persons,
        'avg_infer_ms': np.mean(infer_times) if infer_times else 0,
        'avg_fps': np.mean(fps_list) if fps_list else 0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default='model/model.onnx')
    args = parser.parse_args()
    
    print(f"Processing video: {args.video} with mode 'none'...")
    res_none = process_video(args.video, args.model, 'none')
    if not res_none:
        sys.exit(1)
        
    print(f"Processing video: {args.video} with mode 'auto'...")
    res_auto = process_video(args.video, args.model, 'auto')
    
    print("\n--- Comparison Results ---")
    print("Mode  | Total Persons | Avg Infer ms | Avg FPS")
    print(f"None  | {res_none['total_persons']:<13} | {res_none['avg_infer_ms']:<12.1f} | {res_none['avg_fps']:.1f}")
    print(f"Auto  | {res_auto['total_persons']:<13} | {res_auto['avg_infer_ms']:<12.1f} | {res_auto['avg_fps']:.1f}")
    
    # Generate bar chart
    labels = ['Mode None', 'Mode Auto']
    persons = [res_none['total_persons'], res_auto['total_persons']]
    infer = [res_none['avg_infer_ms'], res_auto['avg_infer_ms']]
    fps = [res_none['avg_fps'], res_auto['avg_fps']]
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    axs[0].bar(labels, persons, color=['gray', 'blue'])
    axs[0].set_title('Total Persons Detected')
    
    axs[1].bar(labels, infer, color=['orange', 'red'])
    axs[1].set_title('Avg Infer ms')
    
    axs[2].bar(labels, fps, color=['green', 'lightgreen'])
    axs[2].set_title('Avg FPS')
    
    plt.tight_layout()
    plt.savefig('comparison_report.png')
    plt.close()
    
    print("\nSaved chart to comparison_report.png")

if __name__ == "__main__":
    main()
