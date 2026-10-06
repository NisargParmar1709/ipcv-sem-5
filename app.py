import argparse
import cv2
import time
import sys
from detector import Detector
from enhance import apply_mode, mean_brightness, sobel_edges, canny_edges, lbp_image
from monitor import ResourceMonitor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--model', type=str, default='model/model.onnx')
    parser.add_argument('--size', type=str, default='192,192')
    parser.add_argument('--score', type=float, default=0.4)
    parser.add_argument('--nms', type=float, default=0.5)
    parser.add_argument('--mode', type=str, default='auto', choices=['none', 'gamma', 'clahe', 'hist', 'auto'])
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    args = parser.parse_args()

    input_shape = tuple(map(int, args.size.split(',')))
    detector = Detector(
        model_path=args.model,
        input_shape=input_shape,
        score_th=args.score,
        nms_th=args.nms,
        providers=['CPUExecutionProvider']
    )

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    modes = ['none', 'gamma', 'clahe', 'hist', 'auto']
    mode_idx = modes.index(args.mode)
    
    analysis_modes = ['off', 'sobel', 'canny', 'lbp']
    analysis_idx = 0
    
    show_graph = True

    monitor = ResourceMonitor()

    print("Controls:")
    print("  'e' - Cycle enhancement mode")
    print("  'v' - Cycle analysis view")
    print("  'g' - Toggle graph panel")
    print("  'ESC' - Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (args.width, args.height))
        
        current_mode = modes[mode_idx]
        enhanced, was_fixed = apply_mode(frame, current_mode)

        start_time = time.perf_counter()
        bboxes, scores, class_ids = detector.inference(enhanced)
        infer_ms = (time.perf_counter() - start_time) * 1000

        monitor.tick(infer_ms)

        display_frame = frame.copy()
        for bbox, score in zip(bboxes, scores):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"person {score:.2f}", (x1, max(y1-10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        b = mean_brightness(frame)
        mode_str = f"Mode: {current_mode}"
        if was_fixed:
            mode_str += "*"
        
        cv2.putText(display_frame, mode_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Brightness: {b:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Persons: {len(bboxes)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if show_graph:
            panel = monitor.draw_panel(args.width, 150)
            output = cv2.vconcat([display_frame, panel])
        else:
            output = display_frame

        cv2.imshow("Person Detection", output)

        analysis_mode = analysis_modes[analysis_idx]
        if analysis_mode != 'off':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if analysis_mode == 'sobel':
                view = sobel_edges(gray)
            elif analysis_mode == 'canny':
                view = canny_edges(gray)
            elif analysis_mode == 'lbp':
                view = lbp_image(gray)
                
            cv2.imshow("Analysis View", view)
        else:
            if cv2.getWindowProperty("Analysis View", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Analysis View")

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('e'):
            mode_idx = (mode_idx + 1) % len(modes)
        elif key == ord('v'):
            analysis_idx = (analysis_idx + 1) % len(analysis_modes)
        elif key == ord('g'):
            show_graph = not show_graph

    cap.release()
    cv2.destroyAllWindows()

    monitor.save_report("usage_report")
    print("\n" + monitor.summary())

if __name__ == "__main__":
    main()
