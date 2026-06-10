from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "management" / "AutoCamTracker_Development" / "0610"
OUT_PDF = OUT_DIR / "AutoCamTracker_V1_AI_Development_Spec.pdf"

FONT_CANDIDATES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]

PAGE_W = 1200
PAGE_H = 1600
MARGIN_X = 76
TOP = 70
BOTTOM = 1500


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


FONT_TITLE = load_font(50)
FONT_H1 = load_font(36)
FONT_H2 = load_font(27)
FONT_BODY = load_font(21)
FONT_SMALL = load_font(18)
FONT_CODE = load_font(18)
FONT_TINY = load_font(14)


def new_page() -> Image.Image:
    return Image.new("RGB", (PAGE_W, PAGE_H), "#F8F8F4")


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        current = ""
        for char in paragraph:
            candidate = current + char
            if not current or text_size(draw, candidate, font)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
        if paragraph == "":
            lines.append("")
    return lines


def header(draw: ImageDraw.ImageDraw, title: str, page_no: int) -> None:
    draw.text((MARGIN_X, 42), title, font=FONT_H1, fill="#172026")
    draw.line((MARGIN_X, 105, PAGE_W - MARGIN_X, 105), fill="#D5DDE0", width=3)
    draw.text((PAGE_W - 132, 1530), f"Page {page_no}", font=FONT_TINY, fill="#718087")
    draw.text((MARGIN_X, 1530), "AutoCamTracker V1 AI Development Spec", font=FONT_TINY, fill="#718087")


def rounded_rect(draw: ImageDraw.ImageDraw, box, fill, outline, width=3, radius=14) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, color="#2F4858", width=5) -> None:
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        pts = [(ex, ey), (ex - direction * 18, ey - 10), (ex - direction * 18, ey + 10)]
    else:
        direction = 1 if ey > sy else -1
        pts = [(ex, ey), (ex - 10, ey - direction * 18), (ex + 10, ey - direction * 18)]
    draw.polygon(pts, fill=color)


class TextDoc:
    def __init__(self) -> None:
        self.pages: list[Image.Image] = []
        self.page_no = 0
        self.page = None
        self.draw = None
        self.y = TOP

    def add_page(self, title: str) -> None:
        self.page_no += 1
        self.page = new_page()
        self.draw = ImageDraw.Draw(self.page)
        header(self.draw, title, self.page_no)
        self.y = 135
        self.pages.append(self.page)

    def ensure(self, height: int, title: str) -> None:
        if self.y + height > BOTTOM:
            self.add_page(title)

    def h2(self, text: str, title: str = "規格內容") -> None:
        self.ensure(60, title)
        self.draw.text((MARGIN_X, self.y), text, font=FONT_H2, fill="#172026")
        self.y += 46

    def para(self, text: str, title: str = "規格內容", font=FONT_BODY, indent=0, gap=9) -> None:
        max_width = PAGE_W - MARGIN_X * 2 - indent
        lines = wrap_text(self.draw, text, font, max_width)
        needed = max(1, len(lines)) * 31 + 8
        self.ensure(needed, title)
        for line in lines:
            self.draw.text((MARGIN_X + indent, self.y), line, font=font, fill="#33464E")
            self.y += 31
        self.y += gap

    def bullets(self, items: list[str], title: str = "規格內容") -> None:
        for item in items:
            self.para(f"- {item}", title=title, font=FONT_BODY, indent=18, gap=3)
        self.y += 8

    def code(self, text: str, title: str = "規格內容") -> None:
        lines = text.strip("\n").split("\n")
        needed = len(lines) * 27 + 42
        self.ensure(needed, title)
        x1, y1 = MARGIN_X, self.y
        x2, y2 = PAGE_W - MARGIN_X, self.y + needed - 8
        rounded_rect(self.draw, (x1, y1, x2, y2), "#FFFFFF", "#CDD7DC", width=2, radius=10)
        y = y1 + 18
        for line in lines:
            self.draw.text((x1 + 24, y), line, font=FONT_CODE, fill="#25343A")
            y += 27
        self.y = y2 + 22


def cover_page() -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    draw.text((76, 100), "AutoCamTracker V1", font=FONT_TITLE, fill="#172026")
    draw.text((80, 170), "AI Development Specification / AI 開發規格書", font=FONT_H1, fill="#52666F")
    draw.line((80, 235, 1120, 235), fill="#D5DDE0", width=4)
    summary = (
        "本文件可直接交給 AI 開發工具使用。目標是用 Python、OpenCV、Tkinter、"
        "Ultralytics YOLO Track Mode 建立 V1：支援 MacBook webcam、本機影片、"
        "螢幕區域輸入；使用 BoT-SORT 與 Deep OC-SORT 追蹤車輛；提供 before / after "
        "同步畫面、遠景 / 中景 / 特寫 reframe、錄影與評估紀錄。"
    )
    y = 310
    for line in wrap_text(draw, summary, FONT_BODY, 980):
        draw.text((86, y), line, font=FONT_BODY, fill="#33464E")
        y += 36
    cards = [
        ("Language / 語言", "Python"),
        ("UI / 介面", "Tkinter"),
        ("Vision / 影像處理", "OpenCV"),
        ("Detector / 偵測", "Ultralytics YOLO"),
        ("Trackers / 追蹤器", "BoT-SORT, Deep OC-SORT"),
        ("Webcam / 鏡頭", "MacBook camera via cv2.VideoCapture(0)"),
    ]
    colors = ["#E7F0F2", "#EDE8F5", "#F4E8E0", "#EAF0E3", "#F3EDDD", "#E9EEF0"]
    outlines = ["#4F8A9A", "#8167A9", "#B67352", "#6E8C51", "#A9823C", "#60717A"]
    y = 610
    for i, (k, v) in enumerate(cards):
        x = 90 + (i % 2) * 520
        if i and i % 2 == 0:
            y += 170
        rounded_rect(draw, (x, y, x + 470, y + 125), colors[i], outlines[i], width=3)
        draw.text((x + 26, y + 24), k, font=FONT_H2, fill="#172026")
        draw.text((x + 26, y + 73), v, font=FONT_SMALL, fill="#40525B")
    draw.text((80, 1488), "Reference: https://docs.ultralytics.com/modes/track", font=FONT_TINY, fill="#718087")
    return page


def data_flow_page(page_no: int) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    header(draw, "資料流動圖 / Data Flow Diagram", page_no)
    nodes = [
        ((80, 190, 410, 380), "#E7F0F2", "#4F8A9A", "1. Input Video\n輸入影像", "webcam / video file / screen region\nraw_frame"),
        ((500, 190, 830, 380), "#E7F0F2", "#4F8A9A", "YOLO Track\nYOLO 追蹤模式", "model.track(frame, persist=True)\ntracker=botsort/deepocsort"),
        ((790, 505, 1120, 715), "#EDE8F5", "#8167A9", "2. YOLO Data\n辨識資料管理", "tracked_detections[]\nvehicle_tracks{}\ntrack_id as identity"),
        ((500, 850, 830, 1070), "#F4E8E0", "#B67352", "3. Target Tracking\n目標追蹤狀態", "selected_track_ids[]\ntarget lost / reacquire / reset"),
        ((80, 850, 410, 1070), "#EAF0E3", "#6E8C51", "4. Reframe\n重新構圖", "union bbox\ncrop + resize\nsmooth + dead zone"),
        ((500, 1225, 830, 1440), "#F3EDDD", "#A9823C", "5. Tkinter UI\n介面與輸出", "before / after\ncontrols / recording / logs"),
    ]
    for box, fill, outline, title, body in nodes:
        rounded_rect(draw, box, fill, outline, width=4)
        y = box[1] + 22
        for line in wrap_text(draw, title, FONT_H2, box[2] - box[0] - 44):
            draw.text((box[0] + 22, y), line, font=FONT_H2, fill="#172026")
            y += 34
        y += 8
        for line in wrap_text(draw, body, FONT_SMALL, box[2] - box[0] - 44):
            draw.text((box[0] + 22, y), line, font=FONT_SMALL, fill="#40525B")
            y += 27
    draw_arrow(draw, (410, 285), (500, 285))
    draw_arrow(draw, (830, 330), (955, 505))
    draw_arrow(draw, (790, 615), (665, 850))
    draw_arrow(draw, (500, 960), (410, 960))
    draw_arrow(draw, (245, 1070), (500, 1320))
    draw_arrow(draw, (665, 1070), (665, 1225))
    rounded_rect(draw, (80, 440, 660, 655), "#FFFFFF", "#CAD4D9", width=2)
    note = (
        "重要規則：使用者點選 bbox 時，實際選到的是 track_id。"
        "後續單選、多選、auto select、target lost、reframe 都必須以 track_id 為核心，"
        "不要只依賴單一幀的 bbox index。"
    )
    y = 470
    for line in wrap_text(draw, note, FONT_BODY, 520):
        draw.text((110, y), line, font=FONT_BODY, fill="#33464E")
        y += 33
    return page


def build_text_pages(start_page_no: int) -> list[Image.Image]:
    doc = TextDoc()
    doc.page_no = start_page_no - 1
    doc.add_page("開發總規格 / Overall Requirements")

    doc.h2("專案目標 / Project Goal")
    doc.bullets([
        "建立 Python 桌面應用程式，用於車輛偵測、追蹤、自動 reframe 與 demo recording。",
        "支援 MacBook webcam、本機影片檔、螢幕區域擷取三種 input source。",
        "使用 Ultralytics YOLO Track Mode，並支援 BoT-SORT 與 Deep OC-SORT tracker。",
        "UI 使用 Tkinter，需同步顯示 before view 與 after view。",
        "追蹤目標可單選或多選，reframe 支援遠景、中景、特寫。",
    ])

    doc.h2("技術條件 / Technical Requirements")
    doc.bullets([
        "Programming Language / 程式語言：Python。",
        "UI Framework / 介面框架：Tkinter。",
        "Computer Vision / 影像處理：OpenCV。",
        "Detection and Tracking / 偵測與追蹤：Ultralytics YOLO Python API。",
        "Default Tracker / 預設追蹤器：BoT-SORT，tracker config 使用 botsort.yaml。",
        "Optional Tracker / 可選追蹤器：Deep OC-SORT，tracker config 使用 deepocsort.yaml。",
        "MacBook Webcam / MacBook 鏡頭：預設使用 cv2.VideoCapture(0)，並允許調整 camera index。",
        "Target Identity / 目標身份：必須使用 Ultralytics tracking result 的 track_id 作為車輛身份。",
    ])

    doc.h2("Ultralytics 使用規則 / Ultralytics Usage Rules")
    doc.code("""
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # 或 path/to/best.pt
results = model.track(
    frame,
    persist=True,
    tracker="botsort.yaml",      # or "deepocsort.yaml"
    conf=0.25,
    iou=0.7,
)
""")
    doc.bullets([
        "persist=True 必須啟用，讓 tracker 在連續 frame 之間維持 track_id。",
        "track_id 來源應從 result.boxes.id 取得；若 result.boxes.is_track 為 False，該幀不可更新 selected target。",
        "BoT-SORT 適合一般與移動鏡頭場景，作為 V1 預設。",
        "Deep OC-SORT 適合擁擠或容易 ID swap 的 moving-camera 場景，作為可切換選項。",
    ])

    modules = [
        (
            "1. Input Video + YOLO Detection / 輸入影片與 YOLO 影像辨識",
            "負責取得影像來源，載入 YOLO 模型，對每一幀執行 model.track()，輸出 raw_frame 與 tracked_detections。",
            [
                "Webcam Input / 網路攝影機輸入：MacBook camera，預設 index=0。",
                "Local Video File Input / 本機影片檔輸入：支援 mp4、mov 等 OpenCV 可讀格式。",
                "Screen Region Capture / 螢幕區域擷取：使用者可框選螢幕區域，偵測網路影片或播放器畫面。",
                "YOLO Model Loading / 載入 YOLO 模型：支援官方模型與自訂 best.pt。",
                "Tracker Selection / 追蹤器選擇：botsort.yaml 或 deepocsort.yaml。",
                "Vehicle Classes / 車輛類別：car、truck、bus、motorcycle，person 可選。",
            ],
            "輸入：source_type、camera_index、video_path、screen_region、model_path、tracker_name、conf、iou。\n輸出：raw_frame、tracked_detections[]、annotated_detection_frame。",
            "不負責 target selection、target lost 決策、reframe、UI layout、recording file writing。",
        ),
        (
            "2. YOLO Data / 辨識資料管理",
            "負責把 tracked_detections 整理成穩定的車輛資料，並以 track_id 建立 vehicle_tracks。",
            [
                "Detection Data Store / 偵測資料儲存：保存目前幀的 tracked detections。",
                "Vehicle Tracks / 車輛軌跡資料：以 track_id 為 key 保存最新 bbox、confidence、center history。",
                "Detection History / 偵測歷史：保留最近 N 幀資料，支援 reacquire 與 debug。",
                "Candidate Ranking / 候選排序：提供 auto select one / multiple 使用。",
                "Filtering / 過濾：依 class、confidence threshold 篩選車輛。",
            ],
            "輸入：tracked_detections[]、frame_index、timestamp。\n輸出：current_vehicle_candidates[]、vehicle_tracks{}、ranked_candidates[]。",
            "不負責使用者選取、不負責重設追蹤、不負責畫面裁切。",
        ),
        (
            "3. Target Tracking / 目標追蹤狀態管理",
            "負責管理 selected_track_ids、單選、多選、auto select、target lost、簡化 reacquire 與 reset。",
            [
                "Manual Select / 手動選取：使用者點 bbox label 後選取對應 track_id。",
                "Single Select / 單選：一次只追蹤一台車。",
                "Multi Select / 多選：可追蹤多個 track_id。",
                "Auto Select One / 自動選一台：預設選最大 bbox 或最靠近中心車輛。",
                "Auto Select Multiple / 自動選多台：選前 N 個 ranked candidates。",
                "Target Lost Handling / 目標遺失處理：超過 max_lost_frames 後提醒失敗、清除選取、回到可選狀態。",
                "Reacquire / 重新捕捉：短暫消失時，用 tracker 與 detection history 嘗試找回。",
            ],
            "輸入：current_vehicle_candidates[]、vehicle_tracks{}、user_action、tracking_config。\n輸出：selected_targets[]、selected_track_ids[]、tracking_status、lost_alert。",
            "不負責 YOLO 推論、不負責 OpenCV crop、不負責 Tkinter widget layout。",
        ),
        (
            "4. Reframe / Digital Zoom / Output Frame / 重新構圖、數位變焦與輸出畫面",
            "負責根據 selected targets 產生 after view：crop + resize、smooth、dead zone、遠景 / 中景 / 特寫。",
            [
                "Single Target Framing / 單目標構圖：用 selected track 的 latest bbox 計算 crop center。",
                "Multi-target Group Framing / 多目標群組構圖：合併所有 selected bbox 成 union bbox。",
                "Framing Presets / 構圖預設：Wide Shot 遠景、Medium Shot 中景、Close-up Shot 特寫。",
                "Smooth Movement / 平滑移動：crop window 逐步靠近目標中心。",
                "Dead Zone / 死區：小幅偏移不移動 crop，減少抖動。",
                "Boundary Clamp / 邊界限制：crop window 不可超出 frame。",
            ],
            "輸入：raw_frame、selected_targets[]、framing_config。\n輸出：tracking_output_frame、before_after_comparison_frame、framing_status。",
            "不負責偵測、不負責選取、不負責錄影寫檔。",
        ),
        (
            "5. Tkinter UI + Recording + Debug Log / Tkinter 介面、錄影與除錯紀錄",
            "負責整合各區塊、顯示 before / after、提供按鈕、錄影與輸出 evaluation log。",
            [
                "Before View / 追蹤前畫面：顯示 raw 或 detection frame、bbox、label、track_id、confidence。",
                "After View / 追蹤後畫面：顯示 reframe 後的 output frame。",
                "Input Controls / 輸入控制：webcam、video file、screen region。",
                "Tracker Controls / 追蹤器控制：BoT-SORT / Deep OC-SORT 切換。",
                "Selection Controls / 選取控制：single、multi、clear、auto one、auto multiple。",
                "Framing Controls / 構圖控制：遠景、中景、特寫。",
                "Status Labels / 狀態標籤：FPS、tracker、selected ids、tracking status、lost count。",
                "Recording / 錄影：raw、detection、tracking output、before/after comparison。",
                "Debug Log / 除錯紀錄：FPS、track_id、confidence、crop position、lost events、framing mode。",
            ],
            "輸入：raw_frame、tracked_detections、selected_targets、tracking_output_frame、status objects、user actions。\n輸出：Tkinter display、video files、debug_log.csv 或 debug_log.json。",
            "UI 不應放置核心 tracking 演算法；只負責呼叫 Target Tracking 模組並呈現結果。",
        ),
    ]

    for title, purpose, features, io, boundary in modules:
        doc.add_page(title[:28])
        doc.h2(title)
        doc.para(f"功能與範圍：{purpose}")
        doc.h2("必要功能 / Required Features")
        doc.bullets(features)
        doc.h2("資料介面 / Data Interface")
        doc.code(io)
        doc.h2("不負責範圍 / Out of Scope")
        doc.para(boundary)

    doc.add_page("流程與驗收 / Flow and Acceptance")
    doc.h2("每一幀處理流程 / Per-frame Pipeline")
    doc.code("""
raw_frame = input.read()
tracked_detections = yolo.track(raw_frame, tracker, persist=True)
vehicle_tracks = detection_store.update(tracked_detections)
selected_targets = target_tracker.update(vehicle_tracks, user_actions)
tracking_output_frame = reframer.render(raw_frame, selected_targets)
ui.update(before_frame, tracking_output_frame, status)
recorder.write_if_enabled(...)
logger.write_frame_metrics(...)
""")
    doc.h2("Target Lost 流程 / Target Lost Flow")
    doc.bullets([
        "若 selected track_id 沒有出現在目前 frame，lost_frame_count + 1。",
        "若 lost_frame_count 未超過 max_lost_frames，維持上一個 crop 或嘗試 reacquire。",
        "若超過 max_lost_frames，tracking_status = failed。",
        "UI 顯示追蹤失敗提醒，清除 selected_track_ids，回到可重新選取狀態。",
    ])
    doc.h2("V1 驗收條件 / Acceptance Criteria")
    doc.bullets([
        "MacBook webcam 可透過 OpenCV index 0 啟動並顯示 live frame。",
        "本機影片檔可載入、播放、暫停、停止。",
        "螢幕區域擷取可作為 input source。",
        "YOLO tracking 能在 before view 顯示 bbox、track_id、confidence。",
        "BoT-SORT 與 Deep OC-SORT 可在 UI 或 config 切換。",
        "使用者可點選 bbox 追蹤單台或多台車。",
        "Auto select one / multiple 可根據 ranked candidates 選目標。",
        "After view 可根據 selected targets 自動 reframe。",
        "遠景 / 中景 / 特寫三種構圖模式可切換。",
        "Target lost 時 UI 會提醒、重置選取，並回到可重新選取狀態。",
        "可輸出 raw、detection、tracking output、before/after comparison recording。",
        "可輸出 debug / evaluation log。",
    ])
    doc.h2("建議檔案結構 / Suggested Files")
    doc.code("""
video_detector.py      # Input Video + YOLO Detection
detection_store.py     # YOLO Data / track_id data store
target_tracker.py      # Target Tracking state management
reframer.py            # Reframe / Digital Zoom / Output Frame
app.py                 # Tkinter UI + Recording + Debug Log
config.py              # model path, tracker, thresholds, UI sizes
""")

    return doc.pages


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = [cover_page(), data_flow_page(2)]
    pages.extend(build_text_pages(3))
    pages[0].save(OUT_PDF, save_all=True, append_images=pages[1:], resolution=150.0)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
