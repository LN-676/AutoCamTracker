# AutoCamTracker V1.21

AutoCamTracker 是一個以影片、螢幕區域或 webcam 作為輸入的車輛偵測與追蹤工具。V1.21 將 ReID 改為手動控管的 Master Feature Gallery：YOLO 只做偵測，tracker 預設使用普通 BoT-SORT，Identity DB 只保存 GID、最後 bbox 與基本 metadata。

## 功能簡述

- 支援 webcam、影片檔、螢幕區域三種輸入來源。
- 支援 YOLO 模型偵測車輛，並顯示 bbox、track id、confidence。
- 預設 detection / tracking 模型為 `yolo26s.pt`。
- 預設 Identity ReID 模型為 `yolo26s-reid.onnx`。
- 可在 Before 畫面直接點選 bbox，或使用 Auto Track 選擇追蹤車輛。
- Identity DB 會記錄車輛全域 ID、最後 bbox、最後 local track 與基本 metadata。
- Feature Gallery 提供 Master / Pending / Candidate 結構；正式 ReID 身份來源只使用 Master。
- `Add Feature` 會手動把目前可見 bbox 的 ReID snapshot 加入 Master，每台車最多 500 張。
- `Find GID` 會手動使用 Master Gallery Top-K matching 找回目前畫面中的同一台車。
- `Link BBox` 可在 feature 不足時，先選 GID 再手動把目前 bbox 綁到同一台車。
- 支援雙擊 GID 欄位自訂車輛 ID 顯示名稱。
- After 畫面會依追蹤目標做數位變焦與置中構圖。
- 支援影片播放速度調整與時間軸拖曳。
- Before / After 畫面會跟隨視窗等比縮放。

## 區塊分工

- `code/V1/app.py`：Tkinter UI、控制列、Before / After 顯示、時間軸、使用者互動。
- `code/V1/video_detector.py`：影片/webcam/螢幕來源讀取、YOLO 模型載入、偵測與追蹤器串接。
- `code/V1/detection_store.py`：保存目前偵測結果、track 歷史與候選車輛排序。
- `code/V1/target_tracker.py`：單一目標選取、lost 狀態與追蹤狀態管理。
- `code/V1/reframer.py`：依目標 bbox 建立 crop window，輸出追蹤構圖畫面。
- `code/V1/tracker_adapter.py`：外部追蹤器 adapter。
- `code/V1/vehicle_identity_store.py`：SQLite Identity DB，只保存 GID、last bbox 與 metadata。
- `code/V1/feature_gallery.py`：Master / Pending / Candidate feature gallery、crop quality filter、duplicate rejection、Top-K matching，以及 FAISS / Qdrant / Milvus 預留介面。
- `code/V1/reid_embedding.py`：Ultralytics ReID encoder 包裝，用於抽取車輛 appearance embedding。
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
6. 若要建立正式 ReID feature，先選 GID，確認該車 bbox 仍可見，按 `Add Feature` 加入 Master。
7. 若 feature 不足或跨攝影機需要補綁定，先選 GID，按 `Link BBox`，再點 Before 畫面的 bbox。
8. 若要用 GID 找回車輛，選 GID 後按 `Find GID`。
9. 在 After 畫面確認追蹤構圖結果。

## V1.21 注意事項

- V1.21 仍以單一車輛追蹤與互動式重新辨識為主。
- `LID` 是 YOLO / tracker 的短期本地 ID；`GID` 是 AutoCamTracker 的長期車輛身份。
- ReID 模型只會在 `Add Feature` 或手動 `Find GID` 時執行。
- 點 bbox、每幀追蹤、UI refresh 都不會自動寫 feature，避免髒資料進入 Master Gallery。
- Master feature 會經過 crop quality filter 與 duplicate rejection。
- 預設 feature backend 是 SQLite；FAISS / Qdrant / Milvus 目前保留介面，尚未啟用。

---

# AutoCamTracker V1.21 English

AutoCamTracker is a vehicle detection and tracking desktop tool that can use a video file, a selected screen region, or a webcam as the input source. V1.21 moves ReID into a manually curated Master Feature Gallery: YOLO only detects, the tracker defaults to plain BoT-SORT, and the Identity DB stores only GID, last bbox, and basic metadata.

## Feature Overview

- Supports three input sources: webcam, video file, and screen region.
- Uses YOLO models to detect vehicles and display bbox, track id, and confidence.
- Defaults to `yolo26s.pt` for detection / tracking.
- Defaults to `yolo26s-reid.onnx` for Identity ReID.
- Allows target selection by clicking a bbox in the Before view or by using Auto Track.
- Records global vehicle IDs, last bbox, last local track, and basic metadata in the Identity DB.
- Adds a Master / Pending / Candidate Feature Gallery; only Master is the official ReID identity source.
- `Add Feature` manually adds the current visible bbox snapshot to Master, capped at 500 features per vehicle.
- `Find GID` manually uses Master Gallery Top-K matching to recover the selected vehicle in the current frame.
- `Link BBox` lets the user manually bind a visible bbox to an existing GID when feature coverage is not enough.
- Allows editing the displayed GID label by double-clicking the GID column.
- The After view digitally zooms and centers the frame based on the selected target.
- Supports video playback speed control and timeline seeking.
- The Before / After views scale proportionally with the application window.

## Project Structure

- `code/V1/app.py`: Tkinter UI, control bar, Before / After display, timeline, and user interaction.
- `code/V1/video_detector.py`: Video, webcam, and screen input handling, YOLO model loading, detection, and tracker integration.
- `code/V1/detection_store.py`: Stores current detections, track history, and candidate vehicle ranking.
- `code/V1/target_tracker.py`: Single target selection, lost state handling, and tracking state management.
- `code/V1/reframer.py`: Builds the crop window from the target bbox and produces the tracking output frame.
- `code/V1/tracker_adapter.py`: Adapter for external trackers.
- `code/V1/vehicle_identity_store.py`: SQLite Identity DB for GID, last bbox, and metadata only.
- `code/V1/feature_gallery.py`: Master / Pending / Candidate feature gallery, crop quality filter, duplicate rejection, Top-K matching, and reserved FAISS / Qdrant / Milvus interfaces.
- `code/V1/reid_embedding.py`: Wrapper around the Ultralytics ReID encoder.
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
6. To create an official ReID feature, select a GID, make sure its bbox is visible, and click `Add Feature`.
7. When feature coverage is not enough, select a GID, click `Link BBox`, then click the bbox in the Before view.
8. To recover a vehicle by GID, select the GID and click `Find GID`.
9. Check the reframed tracking result in the After view.

## V1.21 Notes

- V1.21 still focuses on single-vehicle tracking and interactive re-identification.
- `LID` is the short-term tracker ID; `GID` is AutoCamTracker's long-lived vehicle identity.
- The ReID model runs only during `Add Feature` or manual `Find GID`.
- Clicking a bbox, per-frame tracking, and UI refresh never auto-write features.
- Master features pass crop quality filtering and duplicate rejection.
- SQLite is the default feature backend; FAISS / Qdrant / Milvus are reserved interfaces for future use.
