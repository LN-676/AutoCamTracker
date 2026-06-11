# AutoCamTracker V1.1

AutoCamTracker 是一個以影片、螢幕區域或 webcam 作為輸入的車輛偵測與追蹤工具。V1.1 版本重點是先完成可操作的桌面原型：讀取畫面、用 YOLO 偵測車輛、選擇單一目標、並用數位裁切輸出追蹤畫面。

## 功能簡述

- 支援 webcam、影片檔、螢幕區域三種輸入來源。
- 支援 YOLO 模型偵測車輛，並顯示 bbox、track id、confidence。
- 可在 Before 畫面直接點選 bbox，或使用 Auto Track 選擇追蹤車輛。
- After 畫面會依追蹤目標做數位變焦與置中構圖。
- 支援影片播放速度調整與時間軸拖曳。
- 支援自訂 Before / After 畫面尺寸，並可跟隨視窗伸縮。

## 區塊分工

- `code/V1/app.py`：Tkinter UI、控制列、Before / After 顯示、時間軸、使用者互動。
- `code/V1/video_detector.py`：影片/webcam/螢幕來源讀取、YOLO 模型載入、偵測與追蹤器串接。
- `code/V1/detection_store.py`：保存目前偵測結果、track 歷史與候選車輛排序。
- `code/V1/target_tracker.py`：單一目標選取、lost 狀態與追蹤狀態管理。
- `code/V1/reframer.py`：依目標 bbox 建立 crop window，輸出追蹤構圖畫面。
- `code/V1/tracker_adapter.py`：外部追蹤器 adapter。
- `code/model/`：YOLO 模型與外部 tracker 程式資料。
- `management/AutoCamTracker_Development/`：開發規格、技術報告與版本變更紀錄。

## 使用方式

建議從專案根目錄執行：

```bash
.venv/bin/python run_v1_app.py
```

基本流程：

1. 在 Source 區選擇 `webcam`、`video_file` 或 `screen_region`。
2. 若使用影片，按 `Browse Video` 選擇檔案。
3. 在 Tracking 區選擇模型、tracker 與 framing 模式。
4. 按 `Start` 開始偵測。
5. 在 Before 畫面點選車輛 bbox，或按 `Auto Track`。
6. 在 After 畫面確認追蹤構圖結果。

## V1.1 注意事項

- V1.1 仍以單一車輛追蹤為主。
- 目前 tracker id 屬於短期連續追蹤 id；車輛離開畫面後再回來，要維持同一全域 id，後續需要加入 Vehicle ReID 特徵比對。
- 大型 `.pt` 模型建議使用 Git LFS 管理。

---

# AutoCamTracker V1.1 English

AutoCamTracker is a vehicle detection and tracking desktop tool that can use a video file, a selected screen region, or a webcam as the input source. The V1.1 release focuses on a usable desktop prototype: loading frames, detecting vehicles with YOLO, selecting one target vehicle, and producing a digitally reframed tracking output.

## Feature Overview

- Supports three input sources: webcam, video file, and screen region.
- Uses YOLO models to detect vehicles and display bbox, track id, and confidence.
- Allows target selection by clicking a bbox in the Before view or by using Auto Track.
- The After view digitally zooms and centers the frame based on the selected target.
- Supports video playback speed control and timeline seeking.
- Supports custom Before / After display sizes that adapt to the application window.

## Project Structure

- `code/V1/app.py`: Tkinter UI, control bar, Before / After display, timeline, and user interaction.
- `code/V1/video_detector.py`: Video, webcam, and screen input handling, YOLO model loading, detection, and tracker integration.
- `code/V1/detection_store.py`: Stores current detections, track history, and candidate vehicle ranking.
- `code/V1/target_tracker.py`: Single target selection, lost state handling, and tracking state management.
- `code/V1/reframer.py`: Builds the crop window from the target bbox and produces the tracking output frame.
- `code/V1/tracker_adapter.py`: Adapter for external trackers.
- `code/model/`: YOLO model and external tracker resources.
- `management/AutoCamTracker_Development/`: Development specs, technical reports, and version change logs.

## How To Run

Run from the project root:

```bash
.venv/bin/python run_v1_app.py
```

Basic workflow:

1. Select `webcam`, `video_file`, or `screen_region` in the Source section.
2. If using a video file, click `Browse Video` and choose a file.
3. Select the model, tracker, and framing mode in the Tracking section.
4. Click `Start` to begin detection.
5. Click a vehicle bbox in the Before view, or click `Auto Track`.
6. Check the reframed tracking result in the After view.

## V1.1 Notes

- V1.1 focuses on single-vehicle tracking.
- Current tracker ids are short-term continuous tracking ids. To keep the same global id after a vehicle leaves and returns to the frame, a future version should add Vehicle ReID feature matching.
- Large `.pt` model files should be managed with Git LFS.
