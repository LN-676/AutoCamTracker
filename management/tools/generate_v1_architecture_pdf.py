from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "management" / "AutoCamTracker_Development" / "0610"
OUT_PDF = OUT_DIR / "AutoCamTracker_V1_Architecture_Data_Flow.pdf"

FONT_CANDIDATES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for font_path in FONT_CANDIDATES:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = load_font(54)
FONT_H1 = load_font(38)
FONT_H2 = load_font(28)
FONT_BODY = load_font(22)
FONT_SMALL = load_font(18)
FONT_TINY = load_font(15)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        current = ""
        for char in paragraph:
            candidate = current + char
            if text_size(draw, candidate, font)[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += text_size(draw, line, font)[1] + line_gap
    return y


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str,
    width: int = 3,
    radius: int = 18,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = "#2F4858",
    width: int = 5,
) -> None:
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        points = [(ex, ey), (ex - direction * 18, ey - 10), (ex - direction * 18, ey + 10)]
    else:
        direction = 1 if ey > sy else -1
        points = [(ex, ey), (ex - 10, ey - direction * 18), (ex + 10, ey - direction * 18)]
    draw.polygon(points, fill=color)


def draw_page_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str = "") -> None:
    draw.text((70, 48), title, font=FONT_H1, fill="#172026")
    if subtitle:
        draw.text((72, 96), subtitle, font=FONT_BODY, fill="#60717A")
    draw.line([(70, 132), (1120, 132)], fill="#D7DEE2", width=3)


def create_page() -> Image.Image:
    return Image.new("RGB", (1200, 1600), "#F7F8F5")


def page_overview() -> Image.Image:
    image = create_page()
    draw = ImageDraw.Draw(image)
    draw.text((70, 72), "AutoCamTracker V1 架構與資料流", font=FONT_TITLE, fill="#162127")
    draw.text((74, 148), "Architecture and Data Flow / 軟體架構與資料流動", font=FONT_H2, fill="#52666F")

    summary = (
        "V1 採用 5 個區塊：Input Video + YOLO Detection、YOLO Data、Target Tracking、"
        "Reframe / Digital Zoom / Output Frame、Tkinter UI + Recording + Debug Log。"
        "資料從輸入影像開始，經過 YOLO 偵測與資料整理，再由追蹤狀態決定要鎖定的車輛，"
        "最後重新構圖輸出到 UI，並同步錄影與記錄評估資料。"
    )
    draw_wrapped(draw, (74, 230), summary, FONT_BODY, "#273940", 1010, 12)

    cards = [
        ("1", "Input Video + YOLO Detection\n輸入影片與 YOLO 影像辨識", "Webcam / Video File / Screen Region\nYOLO model loading\nVehicle detection results"),
        ("2", "YOLO Data\n辨識資料管理", "Detected vehicle list\nDetection history\nCandidate ranking"),
        ("3", "Target Tracking\n目標追蹤狀態管理", "Single / multi select\nTarget lost handling\nSimple reacquire / reset"),
        ("4", "Reframe + Digital Zoom\n重新構圖與數位變焦", "Crop + resize\nSmooth + dead zone\nWide / medium / close"),
        ("5", "Tkinter UI + Recording + Log\n介面、錄影與紀錄", "Before / after view\nControls + status\nRecording + evaluation log"),
    ]
    colors = ["#E7F0F2", "#EDE8F5", "#F4E8E0", "#EAF0E3", "#F3EDDD"]
    outlines = ["#4F8A9A", "#8167A9", "#B67352", "#6E8C51", "#A9823C"]
    y = 430
    for idx, (num, title, body) in enumerate(cards):
        x = 90 + (idx % 2) * 520
        if idx == 4:
            x = 350
        if idx and idx % 2 == 0:
            y += 260
        box = (x, y, x + 470, y + 215)
        rounded_rect(draw, box, colors[idx], outlines[idx], width=4)
        draw.ellipse((x + 24, y + 24, x + 76, y + 76), fill=outlines[idx])
        tw, th = text_size(draw, num, FONT_H2)
        draw.text((x + 50 - tw / 2, y + 50 - th / 2 - 2), num, font=FONT_H2, fill="white")
        draw_wrapped(draw, (x + 94, y + 24), title, FONT_H2, "#172026", 340, 5)
        draw_wrapped(draw, (x + 32, y + 108), body, FONT_SMALL, "#40525B", 405, 8)

    draw.text((72, 1518), f"Output: {OUT_PDF.name}", font=FONT_TINY, fill="#718087")
    return image


def page_data_flow() -> Image.Image:
    image = create_page()
    draw = ImageDraw.Draw(image)
    draw_page_header(draw, "資料怎麼流動", "Data Flow Diagram / 從輸入影像到 UI、錄影與評估紀錄")

    nodes = {
        "input": ((80, 210, 410, 390), "#E7F0F2", "#4F8A9A", "1. Input Video\n輸入影像", "Webcam / Video File / Screen Region\n網路攝影機 / 影片檔 / 螢幕區域"),
        "detect": ((500, 210, 830, 390), "#E7F0F2", "#4F8A9A", "YOLO Detection\nYOLO 影像辨識", "bbox / class / confidence / center\n邊界框 / 類別 / 信心分數 / 中心點"),
        "data": ((790, 510, 1120, 710), "#EDE8F5", "#8167A9", "2. YOLO Data\n辨識資料管理", "Detection history\nDetected vehicle list\nCandidate ranking"),
        "tracking": ((500, 850, 830, 1060), "#F4E8E0", "#B67352", "3. Target Tracking\n目標追蹤", "Selected targets\nTarget lost\nReacquire / reset"),
        "reframe": ((80, 850, 410, 1060), "#EAF0E3", "#6E8C51", "4. Reframe\n重新構圖", "Crop + resize\nSmooth / dead zone\n遠景 / 中景 / 特寫"),
        "ui": ((500, 1220, 830, 1440), "#F3EDDD", "#A9823C", "5. Tkinter UI\n介面顯示", "Before / After views\nControls / Status\nRecording / Log"),
    }

    for box, fill, outline, title, body in nodes.values():
        rounded_rect(draw, box, fill, outline, width=4)
        draw_wrapped(draw, (box[0] + 24, box[1] + 24), title, FONT_H2, "#172026", box[2] - box[0] - 48, 6)
        draw_wrapped(draw, (box[0] + 24, box[1] + 106), body, FONT_SMALL, "#40525B", box[2] - box[0] - 48, 7)

    draw_arrow(draw, (410, 300), (500, 300))
    draw_arrow(draw, (830, 330), (955, 510))
    draw_arrow(draw, (790, 620), (650, 850))
    draw_arrow(draw, (500, 955), (410, 955))
    draw_arrow(draw, (245, 1060), (500, 1305))
    draw_arrow(draw, (665, 1060), (665, 1220))
    draw_arrow(draw, (830, 1320), (980, 1320), "#7A6430")

    rounded_rect(draw, (955, 1215, 1120, 1425), "#FFF8E8", "#A9823C", width=3, radius=14)
    draw_wrapped(draw, (975, 1240), "Recording + Debug Log\n錄影與除錯紀錄", FONT_SMALL, "#172026", 125, 8)
    draw_wrapped(draw, (975, 1320), "Raw / Detection / Tracking / Before-After\nFPS / lost count / crop position", FONT_TINY, "#40525B", 125, 6)

    callouts = [
        ((80, 455, 480, 620), "UI 點選 bbox 或按 Auto Select 後，Target Tracking 會更新 selected targets / 已選目標。"),
        ((80, 650, 480, 780), "Target lost 時：提醒追蹤失敗，清除 selected target，回到可重新選取狀態。"),
        ((500, 455, 760, 620), "YOLO Data 會保存最近幾幀資料，支援 candidate ranking 與簡化 reacquire。"),
    ]
    for box, text in callouts:
        rounded_rect(draw, box, "#FFFFFF", "#CAD4D9", width=2, radius=12)
        draw_wrapped(draw, (box[0] + 20, box[1] + 22), text, FONT_SMALL, "#33464E", box[2] - box[0] - 40, 8)

    return image


def page_details() -> Image.Image:
    image = create_page()
    draw = ImageDraw.Draw(image)
    draw_page_header(draw, "區塊內容整理", "Module Contents / 每個區塊負責的功能")

    sections = [
        ("1. Input Video + YOLO Detection / 輸入影片與 YOLO 影像辨識", [
            "Webcam Input / 網路攝影機輸入",
            "Local Video File Input / 本機影片檔輸入",
            "Screen Region Capture / 螢幕區域擷取",
            "YOLO Model Loading / 載入 YOLO 模型",
            "Vehicle Detection / 車輛偵測",
            "Raw Frame Output / 原始畫面輸出",
        ]),
        ("2. YOLO Data / 辨識資料管理", [
            "Detected Vehicle List / 已辨識車輛列表",
            "Detection History / 偵測歷史",
            "Vehicle Candidate Ranking / 車輛候選排序",
            "Selected Target Reference / 已選目標參照",
        ]),
        ("3. Target Tracking / 目標追蹤狀態管理", [
            "Single Select / 單選車輛",
            "Multi Select / 多選車輛",
            "Auto Select One or Multiple / 自動選一台或多台",
            "Target Lost Handling / 目標遺失處理",
            "Reacquire / 重新捕捉目標",
            "Tracking Reset / 追蹤重置",
        ]),
        ("4. Reframe / Digital Zoom / Output Frame / 重新構圖、數位變焦與輸出畫面", [
            "Crop + Resize / 裁切與縮放",
            "Smooth Movement / 平滑移動",
            "Dead Zone / 死區",
            "Wide / Medium / Close-up / 遠景 / 中景 / 特寫",
            "Single Target and Multi-target Group Framing / 單目標與多目標群組構圖",
        ]),
        ("5. Tkinter UI + Recording + Debug Log / Tkinter 介面、錄影與除錯紀錄", [
            "Before / After Synchronized Display / 追蹤前後同步顯示",
            "FPS / Status / Controls / 每秒幀數、狀態與控制",
            "Raw, Detection, Tracking, Before-After Recording / 四種錄影輸出",
            "Evaluation Log / 評估紀錄",
        ]),
    ]

    y = 165
    for title, bullets in sections:
        draw.text((78, y), title, font=FONT_H2, fill="#172026")
        y += 44
        for bullet in bullets:
            draw.text((105, y), f"- {bullet}", font=FONT_BODY, fill="#40525B")
            y += 36
        y += 22

    return image


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = [page_overview(), page_data_flow(), page_details()]
    pages[0].save(OUT_PDF, save_all=True, append_images=pages[1:], resolution=150.0)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
