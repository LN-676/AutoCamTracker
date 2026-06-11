from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = ROOT / "code" / "V1" / "video_detector.py"
OUT_DIR = ROOT / "management" / "AutoCamTracker_Development" / "0611"
OUT_PDF = OUT_DIR / "video_detector_technical_report_zh.pdf"

PAGE_W = 1200
PAGE_H = 1600
MARGIN_X = 76
TOP_Y = 132
BOTTOM_Y = 1492

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


FONT_TITLE = load_font(48)
FONT_H1 = load_font(34)
FONT_H2 = load_font(26)
FONT_BODY = load_font(20)
FONT_SMALL = load_font(17)
FONT_CODE = load_font(16)
FONT_TINY = load_font(13)


def new_page() -> Image.Image:
    return Image.new("RGB", (PAGE_W, PAGE_H), "#F8F8F4")


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if paragraph == "":
            lines.append("")
            continue
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
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    color: str,
    max_width: int,
    line_gap: int = 7,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=color)
        y += text_size(draw, line, font)[1] + line_gap
    return y


def rounded_rect(draw: ImageDraw.ImageDraw, box, fill, outline, width=2, radius=12) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


class PdfDoc:
    def __init__(self) -> None:
        self.pages: list[Image.Image] = []
        self.page: Image.Image | None = None
        self.draw: ImageDraw.ImageDraw | None = None
        self.page_no = 0
        self.y = TOP_Y

    def add_page(self, title: str) -> None:
        self.page_no += 1
        self.page = new_page()
        self.draw = ImageDraw.Draw(self.page)
        self.draw.text((MARGIN_X, 42), title, font=FONT_H1, fill="#172026")
        self.draw.line((MARGIN_X, 102, PAGE_W - MARGIN_X, 102), fill="#D5DDE0", width=3)
        self.draw.text((MARGIN_X, 1530), "AutoCamTracker V1 video_detector.py 技術報告", font=FONT_TINY, fill="#718087")
        self.draw.text((PAGE_W - 132, 1530), f"第 {self.page_no} 頁", font=FONT_TINY, fill="#718087")
        self.y = TOP_Y
        self.pages.append(self.page)

    def ensure(self, height: int, title: str) -> None:
        if self.y + height > BOTTOM_Y:
            self.add_page(title)

    def h2(self, text: str, title: str = "技術報告") -> None:
        assert self.draw is not None
        self.ensure(54, title)
        self.draw.text((MARGIN_X, self.y), text, font=FONT_H2, fill="#172026")
        self.y += 42

    def para(self, text: str, title: str = "技術報告", font=FONT_BODY, indent: int = 0, gap: int = 10) -> None:
        assert self.draw is not None
        lines = wrap_text(self.draw, text, font, PAGE_W - MARGIN_X * 2 - indent)
        self.ensure(max(1, len(lines)) * 29 + gap, title)
        for line in lines:
            self.draw.text((MARGIN_X + indent, self.y), line, font=font, fill="#33464E")
            self.y += 29
        self.y += gap

    def bullets(self, items: list[str], title: str = "技術報告") -> None:
        for item in items:
            self.para(f"- {item}", title=title, font=FONT_BODY, indent=18, gap=2)
        self.y += 8

    def code_block(self, text: str, title: str = "技術報告") -> None:
        assert self.draw is not None
        lines = text.strip("\n").split("\n")
        height = len(lines) * 25 + 34
        self.ensure(height, title)
        x1, y1 = MARGIN_X, self.y
        x2, y2 = PAGE_W - MARGIN_X, self.y + height
        rounded_rect(self.draw, (x1, y1, x2, y2), "#FFFFFF", "#CAD4D9", width=2, radius=10)
        y = y1 + 16
        for line in lines:
            self.draw.text((x1 + 20, y), line, font=FONT_CODE, fill="#25343A")
            y += 25
        self.y = y2 + 18

    def line_item(self, line_no: str, code: str, explanation: str, title: str = "逐行說明") -> None:
        assert self.draw is not None
        code_lines = wrap_text(self.draw, code, FONT_CODE, 430)
        explain_lines = wrap_text(self.draw, explanation, FONT_SMALL, 520)
        height = max(len(code_lines) * 23, len(explain_lines) * 24, 28) + 20
        self.ensure(height, title)
        x = MARGIN_X
        y = self.y
        rounded_rect(self.draw, (x, y, PAGE_W - MARGIN_X, y + height - 6), "#FFFFFF", "#E2E8EA", width=1, radius=8)
        self.draw.text((x + 12, y + 10), line_no, font=FONT_SMALL, fill="#7A6430")
        cy = y + 10
        for code_line in code_lines:
            self.draw.text((x + 78, cy), code_line, font=FONT_CODE, fill="#172026")
            cy += 23
        ey = y + 10
        for explain_line in explain_lines:
            self.draw.text((x + 535, ey), explain_line, font=FONT_SMALL, fill="#40525B")
            ey += 24
        self.y += height


def cover_page() -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    draw.text((76, 88), "video_detector.py 技術報告", font=FONT_TITLE, fill="#172026")
    draw.text((80, 156), "逐行中文說明與 Debug 地圖", font=FONT_H1, fill="#52666F")
    draw.line((80, 222, 1120, 222), fill="#D5DDE0", width=4)
    summary = (
        "這份 PDF 說明 AutoCamTracker V1 第一個模組 video_detector.py。"
        "內容包含模組責任、資料流、重要資料結構、常見 debug 點，以及每一行程式碼的中文功能說明。"
        "英文術語已盡量轉成中文；必要技術名詞保留原字並附中文解釋。"
    )
    y = draw_wrapped(draw, (88, 296), summary, FONT_BODY, "#33464E", 990, 12)
    rounded_rect(draw, (86, y + 45, 1114, y + 295), "#E7F0F2", "#4F8A9A", width=3, radius=18)
    points = [
        "模組定位：輸入影片與 YOLO 影像辨識",
        "輸入來源：MacBook 鏡頭、本機影片檔、螢幕區域擷取",
        "追蹤方式：Ultralytics YOLO track 模式，支援 BoT-SORT 與 Deep OC-SORT",
        "輸出資料：原始畫面與 TrackedDetection 追蹤偵測資料",
        "不負責：目標選取、UI 排版、重新構圖、錄影寫檔",
    ]
    yy = y + 75
    for point in points:
        draw.text((122, yy), f"- {point}", font=FONT_BODY, fill="#263A42")
        yy += 38
    draw.text((80, 1488), f"來源檔案：{SOURCE_FILE}", font=FONT_TINY, fill="#718087")
    return page


def line_explanations() -> list[tuple[str, str, str]]:
    source_lines = SOURCE_FILE.read_text(encoding="utf-8").splitlines()
    explanations = {
        1: "開始檔案說明文字，標明這是輸入影片與 YOLO 偵測模組。",
        2: "空行，讓說明文字分段。",
        3: "說明接下來列出這個模組的責任。",
        4: "責任一：開啟網路攝影機、本機影片檔或螢幕區域來源。",
        5: "責任二：載入 Ultralytics YOLO 模型。",
        6: "責任三：使用 BoT-SORT 或 Deep OC-SORT 做 YOLO 追蹤。",
        7: "責任四：回傳原始畫面與追蹤偵測結果。",
        8: "空行，分隔責任列表與邊界說明。",
        9: "說明這個模組不管理目標選取、使用者介面、重新構圖等功能。",
        10: "延續上一行，說明也不負責錄影檔案輸出。",
        11: "結束檔案說明文字。",
        13: "啟用延後型別註解解析，避免型別在執行時太早被解析。",
        15: "匯入 dataclass，用來快速建立設定與資料物件。",
        16: "匯入目前時間函式，用來給偵測結果加時間戳記。",
        17: "匯入型別工具：任意型別、可迭代資料、固定字串型別。",
        20: "定義輸入來源型別，只允許鏡頭、影片檔、螢幕區域三種。",
        21: "定義追蹤器名稱型別，只允許 BoT-SORT 與 Deep OC-SORT。",
        24: "建立追蹤器設定對照表，key 是程式內名稱，value 是 Ultralytics 設定檔。",
        25: "BoT-SORT 對應 Ultralytics 的 botsort.yaml。",
        26: "Deep OC-SORT 對應 Ultralytics 的 deepocsort.yaml。",
        27: "結束追蹤器設定對照表。",
        29: "定義 V1 要保留的車輛類別名稱集合。",
        32: "宣告下一個 class 是資料類別，Python 會自動產生初始化方法。",
        33: "定義輸入與模型設定物件。",
        34: "設定輸入來源，預設為網路攝影機。",
        35: "設定攝影機編號，MacBook 內建鏡頭通常是 0。",
        36: "設定本機影片路徑，只有影片檔模式需要。",
        37: "設定螢幕擷取區域，格式是 x、y、寬、高。",
        38: "設定 YOLO 模型路徑，預設使用 yolo11n.pt。",
        39: "設定追蹤器，預設使用 BoT-SORT。",
        40: "設定信心分數門檻，低於此數值的偵測結果會被過濾。",
        41: "設定 IoU 門檻，用於 YOLO 推論與追蹤匹配。",
        42: "設定是否只保留車輛類別。",
        45: "宣告下一個 class 是資料類別。",
        46: "定義單一追蹤偵測結果的資料格式。",
        47: "追蹤 ID，代表同一台車跨幀的身份；可能為空。",
        48: "邊界框座標，格式是左上與右下座標。",
        49: "YOLO 類別編號。",
        50: "YOLO 類別名稱，例如 car 或 truck。",
        51: "偵測信心分數。",
        52: "邊界框中心點座標。",
        53: "這筆資料來自第幾幀。",
        54: "這筆資料建立時的時間戳記。",
        55: "這筆資料使用的追蹤器名稱。",
        58: "定義 VideoDetector 類別，負責讀取畫面與執行 YOLO 追蹤。",
        59: "類別說明文字：讀取畫面並執行 YOLO 追蹤模式。",
        61: "初始化 VideoDetector，需要傳入 InputConfig。",
        62: "保存設定物件，後續所有方法都會讀取它。",
        63: "YOLO 模型物件，初始化時尚未載入。",
        64: "OpenCV 影像來源物件，初始化時尚未開啟。",
        65: "螢幕擷取物件，初始化時尚未開啟。",
        66: "目前讀到第幾幀，從 0 開始。",
        67: "保留 OpenCV 模組參照，目前主要是除錯與延伸用途。",
        69: "定義載入模型的方法。",
        70: "在方法內匯入 YOLO，避免只 import 檔案時就需要 ultralytics 套件。",
        72: "用設定中的模型路徑載入 YOLO 模型。",
        74: "定義開啟輸入來源的方法。",
        75: "如果來源是攝影機或影片檔，就走 OpenCV 流程。",
        76: "匯入 OpenCV。",
        78: "保存 OpenCV 模組參照。",
        79: "宣告 source 變數可能是整數或字串。",
        80: "判斷是否為攝影機模式。",
        81: "攝影機模式使用 camera_index 作為來源。",
        82: "否則就是影片檔模式。",
        83: "影片檔模式檢查是否有影片路徑。",
        84: "沒有影片路徑就丟出錯誤。",
        85: "影片檔模式使用 video_path 作為來源。",
        87: "用 OpenCV 開啟攝影機或影片檔。",
        88: "檢查來源是否成功開啟。",
        89: "開啟失敗就丟出執行時錯誤。",
        91: "如果來源是螢幕區域，就走螢幕擷取流程。",
        92: "檢查是否有指定螢幕區域。",
        93: "沒有螢幕區域就丟出錯誤。",
        94: "匯入 mss 螢幕擷取套件。",
        96: "建立 mss 螢幕擷取物件。",
        98: "如果來源類型不在支援範圍內。",
        99: "丟出不支援來源類型的錯誤。",
        101: "定義讀取單張畫面的方法。",
        102: "攝影機與影片檔都用 OpenCV 讀取。",
        103: "確認影像來源已經開啟。",
        104: "未開啟就丟出錯誤。",
        105: "從 OpenCV 來源讀取一幀，回傳成功旗標與畫面。",
        106: "如果讀取失敗。",
        107: "回傳空值，代表影片結束或來源中斷。",
        108: "成功讀取後，幀編號加一。",
        109: "回傳原始畫面。",
        111: "螢幕區域模式使用另一套讀取流程。",
        112: "確認螢幕擷取物件已經建立。",
        113: "未建立就丟出錯誤。",
        114: "匯入 OpenCV，用於色彩格式轉換。",
        115: "匯入 NumPy，用於把螢幕截圖轉成陣列。",
        117: "讀取螢幕區域座標；若沒有則使用零區域保底。",
        118: "呼叫 mss 擷取螢幕畫面。",
        119: "傳入左上角座標與寬高。",
        120: "結束 grab 參數。",
        121: "把擷取結果轉成 NumPy 陣列。",
        122: "把 BGRA 色彩格式轉為 OpenCV 常用的 BGR。",
        123: "成功讀取後，幀編號加一。",
        124: "回傳螢幕擷取畫面。",
        126: "如果來源類型沒有匹配任何流程，回傳空值。",
        128: "定義對單張畫面執行追蹤的方法。",
        129: "檢查 YOLO 模型是否已載入。",
        130: "未載入模型就丟出錯誤。",
        132: "根據設定選擇追蹤器 yaml 檔名。",
        133: "呼叫 YOLO 的追蹤模式。",
        134: "傳入當前畫面。",
        135: "啟用跨幀保留狀態，這是取得穩定 track_id 的關鍵。",
        136: "指定追蹤器設定檔。",
        137: "傳入信心分數門檻。",
        138: "傳入 IoU 門檻。",
        139: "關閉 Ultralytics 詳細輸出。",
        140: "結束 model.track 呼叫。",
        141: "把 YOLO 原始結果解析成 TrackedDetection 清單。",
        143: "定義一次完成讀取畫面與追蹤的方法。",
        144: "先讀取一張畫面。",
        145: "如果沒有讀到畫面。",
        146: "回傳空畫面與空偵測結果。",
        147: "對畫面執行追蹤。",
        148: "回傳原始畫面與追蹤偵測結果。",
        150: "定義關閉資源的方法。",
        151: "如果 OpenCV 來源存在。",
        152: "釋放 OpenCV 影像來源。",
        153: "如果螢幕擷取物件存在。",
        154: "關閉螢幕擷取物件。",
        156: "定義解析 YOLO 結果的內部方法。",
        157: "建立解析後結果清單。",
        158: "建立目前時間戳記，給本批結果共用。",
        160: "逐一處理 YOLO 回傳的 result。",
        161: "取得類別編號到名稱的對照表。",
        162: "取得 boxes 物件，裡面包含 bbox、類別、信心分數、追蹤 ID。",
        163: "如果沒有 boxes。",
        164: "跳過這個 result。",
        166: "取得 bbox 座標並轉成 Python list。",
        167: "取得類別編號並轉成 Python list。",
        168: "取得信心分數並轉成 Python list。",
        169: "取得追蹤 ID 並轉成 Python list。",
        171: "逐一處理每個 bbox。",
        172: "取得目前 bbox 的類別編號；若資料不足就用 -1。",
        173: "用類別編號查類別名稱；查不到就用編號字串。",
        174: "取得信心分數；若資料不足就用 0。",
        176: "如果設定只保留車輛，且類別不在車輛清單內。",
        177: "跳過非車輛目標。",
        178: "如果信心分數低於門檻。",
        179: "跳過低信心分數目標。",
        181: "將 bbox 四個座標轉為浮點數。",
        182: "取得追蹤 ID；如果沒有 ID，設定為空值。",
        183: "把解析好的資料加入結果清單。",
        184: "建立 TrackedDetection 物件。",
        185: "寫入追蹤 ID。",
        186: "寫入邊界框。",
        187: "寫入類別編號。",
        188: "寫入類別名稱。",
        189: "寫入信心分數。",
        190: "計算並寫入中心點。",
        191: "寫入目前幀編號。",
        192: "寫入時間戳記。",
        193: "寫入追蹤器名稱。",
        194: "結束 TrackedDetection 建立。",
        195: "結束 append 呼叫。",
        197: "回傳所有解析後的追蹤偵測結果。",
        199: "宣告下一個方法是靜態方法，不需要使用 self。",
        200: "定義轉換成 Python list 的工具方法。",
        201: "如果傳入值是空值。",
        202: "回傳空清單。",
        203: "如果物件有 cpu 方法，通常代表是 PyTorch 張量。",
        204: "先把張量移到 CPU。",
        205: "如果物件有 numpy 方法。",
        206: "轉成 NumPy 陣列。",
        207: "如果物件有 tolist 方法。",
        208: "轉成 Python list。",
        209: "最後備援：用 list 建構子轉成清單。",
    }
    return [
        (str(index), line, explanations.get(index, "空行或結構分隔，用來提升程式可讀性。"))
        for index, line in enumerate(source_lines, start=1)
    ]


def build_pdf() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = PdfDoc()
    doc.pages.append(cover_page())
    doc.page_no = 1

    doc.add_page("模組總覽")
    doc.h2("模組責任")
    doc.bullets([
        "開啟 MacBook 鏡頭、本機影片檔或螢幕區域擷取來源。",
        "載入 Ultralytics YOLO 模型。",
        "使用 YOLO 追蹤模式，搭配 BoT-SORT 或 Deep OC-SORT。",
        "輸出原始畫面與追蹤偵測資料。",
    ])
    doc.h2("不負責範圍")
    doc.bullets([
        "不負責使用者選取哪台車。",
        "不負責單選、多選、目標遺失重置。",
        "不負責重新構圖、數位變焦或畫面裁切。",
        "不負責 Tkinter 介面排版與錄影檔案寫入。",
    ])
    doc.h2("資料流")
    doc.code_block("""
InputConfig 設定
-> VideoDetector.open_source() 開啟來源
-> VideoDetector.read_frame() 讀取單張畫面
-> VideoDetector.track_frame() 執行 YOLO 追蹤
-> VideoDetector._parse_results() 解析結果
-> list[TrackedDetection] 追蹤偵測資料
""")
    doc.h2("Debug 優先檢查點")
    doc.bullets([
        "鏡頭打不開：檢查 open_source() 第 87-89 行。",
        "模型載入失敗：檢查 load_model() 第 69-72 行。",
        "有畫面但沒有框：檢查 _parse_results() 第 162-164 行。",
        "有偵測但被過濾：檢查車輛類別與信心分數，第 176-179 行。",
        "後續無法追蹤：檢查 track_id 是否為空，第 182 行。",
    ])

    doc.add_page("資料結構說明")
    doc.h2("InputConfig：輸入與模型設定")
    doc.bullets([
        "source_type：輸入來源，包含鏡頭、影片檔、螢幕區域。",
        "camera_index：攝影機編號，MacBook 內建鏡頭通常是 0。",
        "video_path：本機影片檔路徑。",
        "screen_region：螢幕擷取區域，格式是 x、y、寬、高。",
        "model_path：YOLO 模型檔案路徑。",
        "tracker_name：追蹤器名稱，BoT-SORT 或 Deep OC-SORT。",
        "confidence_threshold：信心分數門檻。",
        "iou_threshold：交集聯集比門檻。",
        "vehicle_classes_only：是否只保留車輛類別。",
    ])
    doc.h2("TrackedDetection：追蹤偵測輸出")
    doc.bullets([
        "track_id：追蹤 ID，同一台車跨畫面幀的身份。",
        "bbox：邊界框座標。",
        "class_id / class_name：類別編號與類別名稱。",
        "confidence：信心分數。",
        "center：中心點。",
        "frame_index：畫面幀編號。",
        "timestamp：時間戳記。",
        "tracker_name：使用的追蹤器名稱。",
    ])

    doc.add_page("逐行說明")
    for line_no, code, explanation in line_explanations():
        doc.line_item(line_no, code if code else "空行", explanation)

    doc.pages[0].save(OUT_PDF, save_all=True, append_images=doc.pages[1:], resolution=150.0)
    print(OUT_PDF)


if __name__ == "__main__":
    build_pdf()
