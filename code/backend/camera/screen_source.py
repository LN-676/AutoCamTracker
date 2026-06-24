import ctypes
import tkinter as tk
import mss
import numpy as np
import cv2


def enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class ScreenSource:
    def __init__(self):
        self.sct = mss.mss()
        self.region = None

    def select_region(self, root, status_label):
        selector = tk.Toplevel(root)
        selector.attributes("-fullscreen", True)
        selector.attributes("-alpha", 0.25)
        selector.attributes("-topmost", True)
        selector.config(bg="black")
        selector.overrideredirect(True)

        canvas = tk.Canvas(selector, cursor="cross", bg="gray", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        start_x = 0
        start_y = 0
        rect_id = None

        def on_mouse_down(event):
            nonlocal start_x, start_y, rect_id
            start_x = event.x_root
            start_y = event.y_root
            rect_id = canvas.create_rectangle(
                start_x,
                start_y,
                start_x,
                start_y,
                outline="blue",
                width=3
            )

        def on_mouse_drag(event):
            if rect_id is not None:
                canvas.coords(rect_id, start_x, start_y, event.x_root, event.y_root)

        def on_mouse_up(event):
            end_x = event.x_root
            end_y = event.y_root

            left = min(start_x, end_x)
            top = min(start_y, end_y)
            width = abs(end_x - start_x)
            height = abs(end_y - start_y)

            selector.destroy()

            if width < 10 or height < 10:
                status_label.config(text="狀態：選取範圍太小")
                return

            self.region = {
                "left": int(left),
                "top": int(top),
                "width": int(width),
                "height": int(height)
            }

            status_label.config(text=f"狀態：已選取區域 {width}x{height}")

        def cancel_select(event):
            selector.destroy()
            status_label.config(text="狀態：已取消選取")

        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_drag)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)
        selector.bind("<Escape>", cancel_select)

    def get_frame(self):
        if self.region is None:
            return None

        img = np.array(self.sct.grab(self.region))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return frame