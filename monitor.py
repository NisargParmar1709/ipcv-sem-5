import psutil
import time
import cv2
import numpy as np
import csv
from collections import deque

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.cpu_count = psutil.cpu_count()
        # Prime the CPU measurement
        self.process.cpu_percent()
        
        self.fps_smoothed = 0.0
        self.last_tick_time = time.time()
        self.last_record_time = time.time()
        self.start_time = time.time()
        
        # Histories for plotting (150 samples)
        self.history_fps = deque(maxlen=150)
        self.history_infer = deque(maxlen=150)
        self.history_cpu = deque(maxlen=150)
        self.history_ram = deque(maxlen=150)
        
        # All rows for CSV
        self.rows = []
        
    def tick(self, infer_ms):
        now = time.time()
        dt = now - self.last_tick_time
        self.last_tick_time = now
        
        current_fps = 1.0 / dt if dt > 0 else 0
        if self.fps_smoothed == 0:
            self.fps_smoothed = current_fps
        else:
            self.fps_smoothed = 0.1 * current_fps + 0.9 * self.fps_smoothed
            
        # Record every 0.5s
        if now - self.last_record_time >= 0.5:
            self.last_record_time = now
            cpu_percent = self.process.cpu_percent() / self.cpu_count
            ram_mb = self.process.memory_info().rss / (1024 * 1024)
            
            self.history_fps.append(self.fps_smoothed)
            self.history_infer.append(infer_ms)
            self.history_cpu.append(cpu_percent)
            self.history_ram.append(ram_mb)
            
            self.rows.append([now - self.start_time, self.fps_smoothed, infer_ms, cpu_percent, ram_mb])
            
    def _draw_graph(self, img, data, x, y, w, h, title, color):
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)
        
        if not data:
            cv2.putText(img, f"{title}: N/A", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return
            
        current_val = data[-1]
        cv2.putText(img, f"{title}: {current_val:.1f}", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if len(data) > 1:
            max_val = max(data)
            min_val = min(data)
            val_range = max_val - min_val if max_val != min_val else 1
            
            pts = []
            for i, val in enumerate(data):
                px = x + int(i * (w / 150.0))
                py = y + h - 5 - int((val - min_val) / val_range * (h - 30))
                pts.append((px, py))
                
            for i in range(1, len(pts)):
                cv2.line(img, pts[i-1], pts[i], color, 2)
                
    def draw_panel(self, width, height=150):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        w = width // 4
        self._draw_graph(panel, list(self.history_fps), 0, 0, w, height, "FPS", (0, 255, 0))
        self._draw_graph(panel, list(self.history_infer), w, 0, w, height, "Infer ms", (0, 165, 255))
        self._draw_graph(panel, list(self.history_cpu), w*2, 0, w, height, "CPU %", (255, 0, 0))
        self._draw_graph(panel, list(self.history_ram), w*3, 0, width - w*3, height, "RAM MB", (255, 255, 0))
        return panel

    def summary(self):
        if not self.rows:
            return "No data collected."
            
        fps = [r[1] for r in self.rows]
        infer = [r[2] for r in self.rows]
        cpu = [r[3] for r in self.rows]
        ram = [r[4] for r in self.rows]
        
        return (f"--- Resource Summary ---\n"
                f"FPS:      Avg {np.mean(fps):.1f} | Peak {np.max(fps):.1f}\n"
                f"Infer ms: Avg {np.mean(infer):.1f} | Peak {np.max(infer):.1f}\n"
                f"CPU %:    Avg {np.mean(cpu):.1f} | Peak {np.max(cpu):.1f}\n"
                f"RAM MB:   Avg {np.mean(ram):.1f} | Peak {np.max(ram):.1f}")
                
    def save_report(self, prefix):
        if not self.rows:
            return
            
        csv_file = f"{prefix}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_s', 'FPS', 'Infer_ms', 'CPU_percent', 'RAM_MB'])
            writer.writerows(self.rows)
            
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            times = [r[0] for r in self.rows]
            fps = [r[1] for r in self.rows]
            infer = [r[2] for r in self.rows]
            cpu = [r[3] for r in self.rows]
            ram = [r[4] for r in self.rows]
            
            fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            axs[0].plot(times, fps, 'g')
            axs[0].set_ylabel('FPS')
            axs[1].plot(times, infer, 'orange')
            axs[1].set_ylabel('Infer ms')
            axs[2].plot(times, cpu, 'b')
            axs[2].set_ylabel('CPU %')
            axs[3].plot(times, ram, 'y')
            axs[3].set_ylabel('RAM MB')
            axs[3].set_xlabel('Time (s)')
            
            plt.tight_layout()
            plt.savefig(f"{prefix}.png")
            plt.close()
        except ImportError:
            print("matplotlib not found, skipping PNG report generation.")

if __name__ == "__main__":
    mon = ResourceMonitor()
    mon.tick(10.5)
    time.sleep(0.6)
    mon.tick(12.1)
    panel = mon.draw_panel(640, 150)
    assert panel.shape == (150, 640, 3)
    print("monitor.py tests passed!")
    print(mon.summary())
