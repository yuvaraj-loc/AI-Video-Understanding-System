"""
SmartVision AI - Real-Time Video Intelligence Platform

Run UI:
  streamlit run main.py

Run API:
  python main.py --mode api --host 0.0.0.0 --port 8000

Core dependencies:
  pip install streamlit opencv-python

Optional (features unlock):
  pip install ultralytics torch openai-whisper fastapi uvicorn
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import importlib
import os
import tempfile
import textwrap
import threading
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def try_import(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


cv2 = try_import("cv2")


def require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install opencv-python.")


def select_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred
    torch = try_import("torch")
    if torch is not None and getattr(torch, "cuda", None) is not None:
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "00:00"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


@dataclass
class AlertEvent:
    timestamp: float
    frame_index: int
    event_type: str
    message: str
    severity: str = "info"


@dataclass
class AnalysisConfig:
    sample_rate_fps: float = 4.0
    max_frames: int = 0
    enable_object_detection: bool = True
    enable_action_recognition: bool = True
    enable_transcription: bool = True
    enable_summarization: bool = True
    min_confidence: float = 0.35
    motion_threshold: float = 12.0
    yolo_model: str = "yolov8n.pt"
    whisper_model: str = "base"
    device: str = "auto"
    alert_classes: Iterable[str] = dataclasses.field(
        default_factory=lambda: [
            "person",
            "knife",
            "gun",
            "fire",
            "smoke",
            "car",
            "truck",
        ]
    )
    alert_keywords: Iterable[str] = dataclasses.field(
        default_factory=lambda: ["help", "intruder", "gun", "fire", "emergency"]
    )

    def normalized_alert_classes(self) -> List[str]:
        return [item.strip().lower() for item in self.alert_classes if item.strip()]

    def normalized_alert_keywords(self) -> List[str]:
        return [item.strip().lower() for item in self.alert_keywords if item.strip()]


@dataclass
class AnalysisResult:
    video_path: str
    fps: float
    duration_sec: float
    frames_processed: int
    object_counts: Dict[str, int]
    events: List[AlertEvent]
    motion_scores: List[float]
    transcript: str
    summary: str
    insights: List[str]
    captions: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "fps": self.fps,
            "duration_sec": self.duration_sec,
            "frames_processed": self.frames_processed,
            "object_counts": self.object_counts,
            "events": [dataclasses.asdict(event) for event in self.events],
            "motion_scores": self.motion_scores,
            "transcript": self.transcript,
            "summary": self.summary,
            "insights": self.insights,
            "captions": self.captions,
            "warnings": self.warnings,
        }


class ModelRegistry:
    def __init__(self) -> None:
        self._yolo_models: Dict[str, Any] = {}
        self._whisper_models: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_yolo(self, model_name: str) -> Optional[Any]:
        ultralytics = try_import("ultralytics")
        if ultralytics is None:
            return None
        with self._lock:
            if model_name not in self._yolo_models:
                self._yolo_models[model_name] = ultralytics.YOLO(model_name)
            return self._yolo_models[model_name]

    def get_whisper(self, model_size: str) -> Optional[Any]:
        whisper = try_import("whisper")
        if whisper is None:
            return None
        with self._lock:
            if model_size not in self._whisper_models:
                self._whisper_models[model_size] = whisper.load_model(model_size)
            return self._whisper_models[model_size]


MODEL_REGISTRY = ModelRegistry()


class VideoAnalyzer:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.device = select_device(config.device)
        self.alert_classes = set(config.normalized_alert_classes())
        self.alert_keywords = set(config.normalized_alert_keywords())
        self.warnings: List[str] = []

    def analyze_video(
        self,
        video_path: str,
        progress_cb: Optional[Callable[[float, str, Optional[AlertEvent]], None]] = None,
    ) -> AnalysisResult:
        require_cv2()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_interval = max(int(round(fps / max(self.config.sample_rate_fps, 1e-3))), 1)

        yolo = None
        yolo_failed = False
        if self.config.enable_object_detection:
            yolo = MODEL_REGISTRY.get_yolo(self.config.yolo_model)
            if yolo is None:
                self.warnings.append("YOLO not available. Install ultralytics to enable.")

        object_counts: Counter[str] = Counter()
        events: List[AlertEvent] = []
        motion_scores: List[float] = []
        captions: List[str] = []

        prev_gray = None
        frames_processed = 0
        last_timestamp = 0.0
        frame_index = -1
        progress_total = total_frames if total_frames > 0 else None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % frame_interval != 0:
                continue

            frames_processed += 1
            timestamp = frame_index / fps if fps > 0 else 0.0
            last_timestamp = timestamp

            if self.config.enable_action_recognition:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    score = float(diff.mean())
                    motion_scores.append(score)
                    if score >= self.config.motion_threshold:
                        event = AlertEvent(
                            timestamp=timestamp,
                            frame_index=frame_index,
                            event_type="motion",
                            message=f"High motion detected (score {score:.1f})",
                            severity="warning",
                        )
                        events.append(event)
                        if progress_cb:
                            progress_cb(0.0, "Motion alert", event)
                prev_gray = gray

            if yolo is not None and self.config.enable_object_detection and not yolo_failed:
                try:
                    results = yolo.predict(
                        frame, verbose=False, conf=self.config.min_confidence, device=self.device
                    )
                except Exception as exc:
                    self.warnings.append(f"YOLO inference failed: {exc}")
                    yolo_failed = True
                    results = []

                for result in results:
                    names = getattr(result, "names", {}) or {}
                    boxes = getattr(result, "boxes", []) or []
                    for box in boxes:
                        class_id = int(box.cls[0]) if hasattr(box, "cls") else -1
                        name = names.get(class_id, str(class_id))
                        confidence = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                        if confidence < self.config.min_confidence:
                            continue
                        object_counts[name] += 1
                        if name.lower() in self.alert_classes:
                            event = AlertEvent(
                                timestamp=timestamp,
                                frame_index=frame_index,
                                event_type="object",
                                message=f"Detected {name} ({confidence:.2f})",
                                severity="warning",
                            )
                            events.append(event)
                            if progress_cb:
                                progress_cb(0.0, "Object alert", event)

                if object_counts:
                    caption = build_caption(object_counts)
                    if caption and (not captions or captions[-1] != caption):
                        captions.append(caption)

            if progress_cb:
                if progress_total:
                    progress = clamp(frame_index / progress_total, 0.0, 1.0)
                elif self.config.max_frames > 0:
                    progress = clamp(frames_processed / self.config.max_frames, 0.0, 1.0)
                else:
                    progress = 0.0
                progress_cb(progress, f"Processed {frames_processed} frames", None)

            if self.config.max_frames > 0 and frames_processed >= self.config.max_frames:
                break

        cap.release()

        transcript = ""
        if self.config.enable_transcription:
            transcript = self.transcribe_audio(video_path)

        if transcript:
            transcript_lower = transcript.lower()
            for keyword in self.alert_keywords:
                if keyword in transcript_lower:
                    event = AlertEvent(
                        timestamp=0.0,
                        frame_index=0,
                        event_type="keyword",
                        message=f"Keyword detected: {keyword}",
                        severity="warning",
                    )
                    events.append(event)

        duration_sec = 0.0
        if total_frames > 0 and fps > 0:
            duration_sec = total_frames / fps
        elif last_timestamp:
            duration_sec = last_timestamp

        summary, insights = build_summary(
            duration_sec=duration_sec,
            fps=fps,
            frames_processed=frames_processed,
            object_counts=object_counts,
            events=events,
            motion_scores=motion_scores,
            transcript=transcript,
        )
        if not self.config.enable_summarization:
            summary = ""
            insights = []

        return AnalysisResult(
            video_path=video_path,
            fps=fps,
            duration_sec=duration_sec,
            frames_processed=frames_processed,
            object_counts=dict(object_counts),
            events=events,
            motion_scores=motion_scores,
            transcript=transcript,
            summary=summary,
            insights=insights,
            captions=captions,
            warnings=self.warnings,
        )

    def transcribe_audio(self, video_path: str) -> str:
        whisper_model = MODEL_REGISTRY.get_whisper(self.config.whisper_model)
        if whisper_model is None:
            self.warnings.append("Whisper not available. Install openai-whisper to enable.")
            return ""
        try:
            result = whisper_model.transcribe(video_path)
            return result.get("text", "").strip()
        except Exception as exc:
            self.warnings.append(f"Whisper transcription failed: {exc}")
            return ""


def build_caption(object_counts: Counter[str]) -> str:
    if not object_counts:
        return ""
    top_objects = [name for name, _ in object_counts.most_common(3)]
    objects_text = ", ".join(top_objects)
    return f"Scene likely contains {objects_text}."


def build_summary(
    duration_sec: float,
    fps: float,
    frames_processed: int,
    object_counts: Counter[str],
    events: List[AlertEvent],
    motion_scores: List[float],
    transcript: str,
) -> Tuple[str, List[str]]:
    insights: List[str] = []
    top_objects = object_counts.most_common(5)

    if top_objects:
        insights.append(
            "Top objects: " + ", ".join(f"{name} ({count})" for name, count in top_objects)
        )
    if events:
        insights.append(f"Alerts triggered: {len(events)}")
    if motion_scores:
        avg_motion = sum(motion_scores) / max(len(motion_scores), 1)
        insights.append(f"Average motion score: {avg_motion:.1f}")
    if transcript:
        insights.append(f"Transcript length: {len(transcript.split())} words")

    summary_lines = [
        f"Duration: {format_duration(duration_sec)}",
        f"FPS: {fps:.1f}",
        f"Frames analyzed: {frames_processed}",
    ]
    if top_objects:
        summary_lines.append(
            "Objects: " + ", ".join(f"{name} ({count})" for name, count in top_objects)
        )
    if events:
        summary_lines.append(f"Alerts: {len(events)}")
    if transcript:
        snippet = transcript[:240].strip()
        if len(transcript) > 240:
            snippet += "..."
        summary_lines.append(f"Transcript snippet: {snippet}")

    summary = "\n".join(summary_lines)
    return summary, insights


def build_video_qa_response(question: str, result: AnalysisResult) -> str:
    question_lower = question.lower().strip()
    if not question_lower:
        return "Ask a question to query the analyzed video."

    for obj, count in result.object_counts.items():
        if obj.lower() in question_lower and "how many" in question_lower:
            return f"I detected about {count} instances of {obj}."

    if "alert" in question_lower:
        return f"There were {len(result.events)} alerts." if result.events else "No alerts."

    if "duration" in question_lower:
        return f"Duration is {format_duration(result.duration_sec)}."

    if result.transcript:
        words = result.transcript.split()
        snippet = " ".join(words[:40])
        return f"Transcript start: {snippet}{'...' if len(words) > 40 else ''}"

    return "I do not have enough data yet. Try enabling transcription or object detection."


def parse_list_input(text: str) -> List[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def run_streamlit_app() -> None:
    st = try_import("streamlit")
    if st is None:
        raise RuntimeError("Streamlit is required. Install streamlit.")

    st.set_page_config(page_title="SmartVision AI", layout="wide")
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

        :root {
            --bg: #f7f3ea;
            --surface: #ffffff;
            --ink: #1c1c1c;
            --muted: #5c5c5c;
            --accent: #0f6b6d;
            --accent-2: #f4a261;
        }

        .stApp {
            background: radial-gradient(circle at top left, #fff7e6, var(--bg));
            color: var(--ink);
            font-family: 'IBM Plex Sans', sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
        }

        .hero {
            padding: 1.2rem 1.6rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #ffffff, #fff1da);
            box-shadow: 0 12px 40px rgba(15, 25, 25, 0.08);
            animation: floatIn 0.8s ease;
        }

        .card {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: var(--surface);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(15, 107, 109, 0.08);
        }

        .muted {
            color: var(--muted);
        }

        @keyframes floatIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>SmartVision AI</h1>
            <p class="muted">Real-Time Video Intelligence Platform</p>
            <p>Object detection, motion-aware action signals, speech-to-text, and AI summaries in one dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Analysis Controls")
    sample_rate_fps = st.sidebar.slider("Sample rate (fps)", 1.0, 12.0, 4.0, 0.5)
    max_frames = st.sidebar.number_input("Max frames (0 = all)", 0, 100000, 0, 50)
    min_confidence = st.sidebar.slider("Min confidence", 0.05, 0.95, 0.35, 0.05)
    motion_threshold = st.sidebar.slider("Motion threshold", 2.0, 40.0, 12.0, 1.0)
    yolo_model = st.sidebar.text_input("YOLO model", "yolov8n.pt")
    whisper_model = st.sidebar.text_input("Whisper model", "base")
    device = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"])

    enable_object_detection = st.sidebar.checkbox("Object detection", True)
    enable_action_recognition = st.sidebar.checkbox("Action recognition (motion)", True)
    enable_transcription = st.sidebar.checkbox("Speech to text", True)
    enable_summarization = st.sidebar.checkbox("Summarization", True)

    alert_classes_input = st.sidebar.text_area(
        "Alert classes (comma separated)",
        "person, knife, gun, fire, smoke, car, truck",
        height=80,
    )
    alert_keywords_input = st.sidebar.text_area(
        "Alert keywords (comma separated)",
        "help, intruder, gun, fire, emergency",
        height=80,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Optional models: ultralytics (YOLO), openai-whisper (STT)")

    config = AnalysisConfig(
        sample_rate_fps=sample_rate_fps,
        max_frames=int(max_frames),
        enable_object_detection=enable_object_detection,
        enable_action_recognition=enable_action_recognition,
        enable_transcription=enable_transcription,
        enable_summarization=enable_summarization,
        min_confidence=min_confidence,
        motion_threshold=motion_threshold,
        yolo_model=yolo_model,
        whisper_model=whisper_model,
        device=device,
        alert_classes=parse_list_input(alert_classes_input),
        alert_keywords=parse_list_input(alert_keywords_input),
    )

    tabs = st.tabs(["Dashboard", "Transcript", "Alerts", "Video Q/A", "API"])

    def render_result(result: AnalysisResult) -> None:
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Duration", format_duration(result.duration_sec))
        metrics_cols[1].metric("Frames", result.frames_processed)
        metrics_cols[2].metric("Alerts", len(result.events))
        top_object = "-"
        if result.object_counts:
            top_object = max(result.object_counts.items(), key=lambda item: item[1])[0]
        metrics_cols[3].metric("Top object", top_object)

        st.markdown("---")
        if result.summary:
            st.subheader("Summary")
            st.text(result.summary)

        if result.insights:
            st.subheader("Insights")
            for insight in result.insights:
                st.markdown(f"- {insight}")

        if result.captions:
            st.subheader("Auto captions")
            for caption in result.captions[:5]:
                st.markdown(f"- {caption}")

        if result.object_counts:
            st.subheader("Object counts")
            rows = [
                {"object": name, "count": count}
                for name, count in sorted(
                    result.object_counts.items(), key=lambda item: item[1], reverse=True
                )
            ]
            try:
                st.dataframe(rows, use_container_width=True)
            except Exception:
                st.json(rows)

        if result.warnings:
            st.subheader("Warnings")
            for warning in result.warnings:
                st.warning(warning)

    with tabs[0]:
        st.subheader("Upload and analyze")
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        run_button = st.button("Run analysis")

        if uploaded and run_button:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded.read())
                temp_path = temp_file.name

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            alerts_placeholder = st.container()

            def progress_cb(progress: float, message: str, event: Optional[AlertEvent]) -> None:
                if progress > 0:
                    progress_bar.progress(clamp(progress, 0.0, 1.0))
                if message:
                    status_text.text(message)
                if event is not None:
                    with alerts_placeholder:
                        st.warning(f"{format_duration(event.timestamp)} - {event.message}")
                    if hasattr(st, "toast"):
                        st.toast(event.message)

            analyzer = VideoAnalyzer(config)
            try:
                result = analyzer.analyze_video(temp_path, progress_cb=progress_cb)
            except Exception as exc:
                st.error(str(exc))
                return

            os.unlink(temp_path)

            st.success("Analysis complete")
            st.session_state["analysis_result"] = result

        result = st.session_state.get("analysis_result")
        if result is not None:
            render_result(result)

    with tabs[1]:
        st.subheader("Transcript")
        result = st.session_state.get("analysis_result")
        if result is None:
            st.write("Run an analysis to generate a transcript.")
        elif result.transcript:
            st.text_area("Transcript", result.transcript, height=240)
        else:
            st.write("Transcript is empty. Enable speech to text in the sidebar.")

    with tabs[2]:
        st.subheader("Alerts")
        result = st.session_state.get("analysis_result")
        if result is None:
            st.write("Alerts appear here after processing.")
        elif result.events:
            rows = [dataclasses.asdict(event) for event in result.events]
            try:
                st.dataframe(rows, use_container_width=True)
            except Exception:
                st.json(rows)
        else:
            st.write("No alerts detected for the current analysis.")

    with tabs[3]:
        st.subheader("Video Q/A")
        question = st.text_input("Ask a question about the video")
        if st.button("Answer"):
            result = st.session_state.get("analysis_result")
            if result is None:
                st.info("Run analysis first to enable Q/A.")
            else:
                st.success(build_video_qa_response(question, result))

    with tabs[4]:
        st.subheader("API")
        st.code(
            textwrap.dedent(
                """
                POST /analyze
                - file: video file
                - sample_rate_fps: float
                - max_frames: int
                - enable_object_detection: bool
                - enable_action_recognition: bool
                - enable_transcription: bool
                - enable_summarization: bool
                - min_confidence: float
                - motion_threshold: float
                - yolo_model: str
                - whisper_model: str
                - device: auto|cpu|cuda
                - alert_classes: comma separated
                - alert_keywords: comma separated
                """
            ).strip(),
            language="text",
        )


FASTAPI_AVAILABLE = False
FastAPI = None
UploadFile = None
File = None
Form = None
JSONResponse = None
try:
    from fastapi import FastAPI as _FastAPI
    from fastapi import UploadFile as _UploadFile
    from fastapi import File as _File
    from fastapi import Form as _Form
    from fastapi.responses import JSONResponse as _JSONResponse

    FastAPI = _FastAPI
    UploadFile = _UploadFile
    File = _File
    Form = _Form
    JSONResponse = _JSONResponse
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False


app = None
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="SmartVision AI",
        description="Real-time video intelligence API",
        version="0.1.0",
    )

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/analyze")
    async def analyze_api(
        file: UploadFile = File(...),
        sample_rate_fps: float = Form(4.0),
        max_frames: int = Form(0),
        enable_object_detection: bool = Form(True),
        enable_action_recognition: bool = Form(True),
        enable_transcription: bool = Form(True),
        enable_summarization: bool = Form(True),
        min_confidence: float = Form(0.35),
        motion_threshold: float = Form(12.0),
        yolo_model: str = Form("yolov8n.pt"),
        whisper_model: str = Form("base"),
        device: str = Form("auto"),
        alert_classes: str = Form("person, knife, gun, fire, smoke, car, truck"),
        alert_keywords: str = Form("help, intruder, gun, fire, emergency"),
    ) -> Any:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        config = AnalysisConfig(
            sample_rate_fps=sample_rate_fps,
            max_frames=max_frames,
            enable_object_detection=enable_object_detection,
            enable_action_recognition=enable_action_recognition,
            enable_transcription=enable_transcription,
            enable_summarization=enable_summarization,
            min_confidence=min_confidence,
            motion_threshold=motion_threshold,
            yolo_model=yolo_model,
            whisper_model=whisper_model,
            device=device,
            alert_classes=parse_list_input(alert_classes),
            alert_keywords=parse_list_input(alert_keywords),
        )

        analyzer = VideoAnalyzer(config)
        try:
            result = analyzer.analyze_video(temp_path, progress_cb=None)
        except Exception as exc:
            os.unlink(temp_path)
            return JSONResponse(status_code=400, content={"error": str(exc)})
        os.unlink(temp_path)

        return result.to_dict()


def is_streamlit_running() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_api_server(host: str, port: int) -> None:
    if not FASTAPI_AVAILABLE or app is None:
        raise RuntimeError("FastAPI is not available. Install fastapi and uvicorn.")
    uvicorn = try_import("uvicorn")
    if uvicorn is None:
        raise RuntimeError("Uvicorn is required. Install uvicorn.")
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartVision AI entrypoint")
    parser.add_argument("--mode", choices=["ui", "api"], default="ui")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if is_streamlit_running():
        run_streamlit_app()
        return

    if args.mode == "api":
        run_api_server(args.host, args.port)
        return

    print("Run the UI with: streamlit run main.py")


if __name__ == "__main__":
    main()