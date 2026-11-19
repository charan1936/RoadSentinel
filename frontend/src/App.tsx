import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, ArrowDownToLine, ArrowUpRight, AlertCircle, Camera,
  CheckCircle2, ChevronLeft, ChevronRight, Clock3,
  FileVideo, Filter, Gauge, LayoutDashboard, LoaderCircle, RefreshCw,
  Search, ShieldCheck, Sparkles, TrafficCone, Wifi, X
} from "lucide-react";
import RoadDirectionCalibrator from "./components/RoadDirectionCalibrator";
import VideoJobReport from "./components/VideoJobReport";
import ParkingZoneCalibrator from "./components/ParkingZoneCalibrator";

const API = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
type EventRow = {
  event_id: string;
  event_type: string;
  track_id?: number;
  class_name?: string;
  timestamp?: number;
  timestamp_seconds?: number;
  frame_number?: number;
  start_seconds?: number;
  end_seconds?: number;
  duration_seconds?: number;
  confidence?: number;
  helmet_detector_confidence?: number;
  helmet_model_track_id?: number;
  track_source?: string;
  participant_count?: number;
  related_track_ids?: number[];
  vehicle_track_id?: number;
  vehicle_class_name?: string;
  image_space_separation_pixels?: number;
  person_displacement_pixels?: number;
  person_displacement_heights?: number;
  evidence_frame?: number;
  status: string;
  notes?: string;
  evidence_url?: string | null;
  job_id?: string;
  reviewer?: string;
  review_note?: string;
  reviewed_at?: string;
  review_history?: { previous_status: string; status: string; reviewer: string; note: string; changed_at: string }[];
};
type VideoJob = {
  job_id: string;
  status: string;
  stage?: string;
  progress?: number;
  original_filename?: string;
  error?: string;
  interpretation?: string;
  candidate_event_status?: string;
  candidate_event_error?: string;
  candidate_events?: EventRow[];
  parking_events?: EventRow[];
  frames_processed?: number;
  detection_rows?: number;
  unique_track_ids?: number;
  device?: string;
  device_name?: string;
  model_checkpoint?: string;
  model_name?: string;
  model_version?: string;
  model_class_names?: Record<string, string>;
  torch_version?: string;
  ultralytics_version?: string;
  track_confidence?: number;
  inference_imgsz?: number;
  annotated_video_url?: string;
  detections_csv_url?: string;
  summary_url?: string;
  candidate_events_csv_url?: string;
  candidate_events_json_url?: string;
  directional_report_url?: string;
  directional_candidate_count?: number;
  parking_report_url?: string;
  parking_candidate_count?: number;
};
type Summary = {
  total_events: number;
  event_types: Record<string, number>;
  review_statuses: Record<string, number>;
  interpretation?: string;
};
type EventsResponse = { total: number; limit: number; offset: number; events: EventRow[] };
type VideoJobsResponse = { jobs: VideoJob[] };
type QaCitation = { event_id: string; reason: string; timestamp_seconds?: number; status?: string };
type QaMessage = { role: "user" | "assistant"; content: string; citations?: QaCitation[]; insufficient_data?: boolean; source_event_count?: number; included_event_count?: number; created_at?: string };

function fmtTime(value?: number) {
  if (value == null || !Number.isFinite(value)) return "—";
  const mins = Math.floor(value / 60);
  const secs = Math.floor(value % 60);
  const tenths = Math.floor((value % 1) * 10);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${tenths}`;
}
function pretty(value?: string) {
  return (value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (m) => m.toUpperCase());
}
function statusClass(status: string) {
  if (status === "needs_human_review" || status === "candidate_violation") return "review";
  if (status === "confirmed_by_reviewer") return "confirmed";
  return "neutral";
}

// Tracking-only observations and parking candidates belong in separate review areas.
function isMainReviewEvent(event: EventRow) {
  return !["movement_observation", "parking_zone_candidate"].includes(event.event_type);
}

export default function App() {
  const [health, setHealth] = useState<"checking" | "online" | "offline">("checking");
  const [modelAvailable, setModelAvailable] = useState<boolean | null>(null);
  const [uploadReady, setUploadReady] = useState(false);
  const [analysisConfigError, setAnalysisConfigError] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [events, setEvents] = useState<EventRow[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [eventType, setEventType] = useState("all");
  const [status, setStatus] = useState("all");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<EventRow | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [activeJob, setActiveJob] = useState<VideoJob | null>(null);
  const [showDirectionCalibrator, setShowDirectionCalibrator] = useState(false);
  const [reportJobId, setReportJobId] = useState<string | null>(null);
  const [directionAnalysisLoading, setDirectionAnalysisLoading] = useState(false);
  const [directionAnalysisError, setDirectionAnalysisError] = useState("");
  const [directionEvents, setDirectionEvents] = useState<EventRow[]>([]);
  const [showParkingCalibrator, setShowParkingCalibrator] = useState(false);
  const [parkingAnalysisLoading, setParkingAnalysisLoading] = useState(false);
  const [parkingAnalysisError, setParkingAnalysisError] = useState("");
  const [parkingEvents, setParkingEvents] = useState<EventRow[]>([]);
  const [videoJobs, setVideoJobs] = useState<VideoJob[]>([]);
  const [pollingJob, setPollingJob] = useState(false);
  const [reviewerName, setReviewerName] = useState("Local reviewer");
  const [reviewNote, setReviewNote] = useState("");
  const [reviewSaving, setReviewSaving] = useState(false);
  const [reviewError, setReviewError] = useState("");
  const [aiConfigured, setAiConfigured] = useState(false);
  const [aiSummary, setAiSummary] = useState<any>(null);
  const [aiSummaryLoading, setAiSummaryLoading] = useState(false);
  const [aiSummaryError, setAiSummaryError] = useState("");
  const [qaQuestion, setQaQuestion] = useState("");
  const [qaMessages, setQaMessages] = useState<QaMessage[]>([]);
  const [qaSessionId, setQaSessionId] = useState("");
  const [qaLoading, setQaLoading] = useState(false);
  const [qaError, setQaError] = useState("");
  const annotatedVideoRef = useRef<HTMLVideoElement>(null);
  const [aiEventExplanation, setAiEventExplanation] = useState<any>(null);
  const [aiEventExplanationLoading, setAiEventExplanationLoading] = useState(false);
  const [aiEventExplanationError, setAiEventExplanationError] = useState<{ event_id: string; message: string } | null>(null);
  const [activeSection, setActiveSection] = useState("overview");
  const eventSearchRef = useRef<HTMLInputElement>(null);
  const pageSize = 10;

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [h, s, e, j] = await Promise.all([
        fetch(`${API}/health`),
        fetch(`${API}/api/v1/summary`),
        fetch(`${API}/api/v1/events?limit=${pageSize}&offset=${offset}`),
        fetch(`${API}/api/v1/videos/jobs`),
      ]);
      if (!h.ok || !s.ok || !e.ok || !j.ok) throw new Error("API returned an error. Check the FastAPI terminal.");
      const [healthData, summaryData, eventData, jobsData] = await Promise.all([h.json(), s.json(), e.json(), j.json()]);
      setHealth(healthData.status === "ok" ? "online" : "offline");
      setModelAvailable(Boolean(healthData.model_available));
      setUploadReady(Boolean(healthData.upload_enabled));
      setAnalysisConfigError(healthData.configuration_error || "");
      setAiConfigured(Boolean(healthData.ai_summary_available));
      setSummary(summaryData);
      const eventPayload = eventData as EventsResponse;
      const jobsPayload = jobsData as VideoJobsResponse;
      setEvents(eventPayload.events || []);
      setTotal(eventPayload.total || 0);
      setVideoJobs(jobsPayload.jobs || []);
    } catch (err) {
      setHealth("offline");
      setError(err instanceof Error ? err.message : "Could not connect to RoadSentinel API.");
    } finally {
      setLoading(false);
    }
  }, [offset]);

  useEffect(() => { void refresh(); }, [refresh]);

  useEffect(() => {
    const sections = Array.from(document.querySelectorAll<HTMLElement>("#overview, #events, #pipeline"));
    const observer = new IntersectionObserver((entries) => {
      const current = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (current) setActiveSection(current.target.id);
    }, { rootMargin: "-18% 0px -58% 0px", threshold: [0, 0.2, 0.45] });
    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && selected) { setSelected(null); return; }
      const target = event.target as HTMLElement | null;
      const typing = target && (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable);
      if ((event.key === "/" && !typing) || ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k")) {
        event.preventDefault();
        document.getElementById("events")?.scrollIntoView({ behavior: "smooth", block: "start" });
        window.setTimeout(() => eventSearchRef.current?.focus(), 300);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selected]);

  const uploadVideo = async () => {
    if (!uploadFile) { setUploadError("Choose a video file first."); return; }
    setUploading(true); setUploadError(""); setActiveJob(null); setParkingEvents([]); setDirectionEvents([]); setAiSummary(null); setAiSummaryError(""); setQaMessages([]); setQaSessionId(""); setQaError("");
    try {
      const form = new FormData(); form.append("file", uploadFile);
      const response = await fetch(`${API}/api/v1/videos/upload`, { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Video upload failed.");
      setActiveJob(data); setVideoJobs((current) => [data, ...current.filter((job) => job.job_id !== data.job_id)]); setPollingJob(true); setUploadFile(null);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Could not upload video.");
    } finally { setUploading(false); }
  };

  useEffect(() => {
    if (!activeJob?.job_id || !pollingJob) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const response = await fetch(`${API}/api/v1/videos/jobs/${activeJob.job_id}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not read processing status.");
        if (data.status === "completed" && data.candidate_event_status === "completed") {
          const eventsResponse = await fetch(`${API}/api/v1/videos/jobs/${data.job_id}/events`);
          const eventsData = await eventsResponse.json();
          if (!eventsResponse.ok) throw new Error(eventsData.detail || "Could not load candidate events.");
          const allEvents = (eventsData.events || []) as EventRow[];
          data.candidate_events = allEvents.filter(isMainReviewEvent);
          data.parking_events = allEvents.filter((event) => event.event_type === "parking_zone_candidate");
          data.direction_events = allEvents.filter((event) => event.event_type === "wrong_way_movement_candidate");
        }
        if (!cancelled) {
          setActiveJob(data);
          setParkingEvents(data.parking_events || []);
          setDirectionEvents(data.direction_events || []);
          setVideoJobs((current) => current.map((job) => job.job_id === data.job_id ? data : job));
          if (data.status === "completed" || data.status === "failed") setPollingJob(false);
        }
      } catch (err) {
        if (!cancelled) { setUploadError(err instanceof Error ? err.message : "Could not check processing status."); setPollingJob(false); }
      }
    };
    void poll();
    const timer = window.setInterval(() => { void poll(); }, 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [activeJob?.job_id, pollingJob]);

  useEffect(() => {
    if (!activeJob?.job_id || activeJob.status !== "completed" || activeJob.candidate_event_status !== "completed") return;
    const key = `roadsentinel-qa-session-${activeJob.job_id}`;
    const savedSession = window.localStorage.getItem(key);
    if (!savedSession) return;
    let cancelled = false;
    setQaSessionId(savedSession);
    fetch(`${API}/api/v1/videos/jobs/${activeJob.job_id}/questions?session_id=${encodeURIComponent(savedSession)}`)
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not restore the saved Q&A.");
        if (!cancelled) setQaMessages(data.messages || []);
      })
      .catch((err) => { if (!cancelled) setQaError(err instanceof Error ? err.message : "Could not restore the saved Q&A."); });
    return () => { cancelled = true; };
  }, [activeJob?.job_id, activeJob?.status, activeJob?.candidate_event_status]);

  const startAnotherUpload = () => { setActiveJob(null); setParkingEvents([]); setDirectionEvents([]); setUploadError(""); setPollingJob(false); setAiSummary(null); setAiSummaryError(""); setQaMessages([]); setQaSessionId(""); setQaQuestion(""); setQaError(""); };

  const loadVideoJob = async (jobId: string) => {
    try {
      const response = await fetch(`${API}/api/v1/videos/jobs/${jobId}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not load this video job.");
      if (data.status === "completed" && data.candidate_event_status === "completed") {
        const eventResponse = await fetch(`${API}/api/v1/videos/jobs/${jobId}/events`);
        const eventData = await eventResponse.json();
        if (eventResponse.ok) {
          const allEvents = (eventData.events || []) as EventRow[];
          data.candidate_events = allEvents.filter(isMainReviewEvent);
          data.parking_events = allEvents.filter((event) => event.event_type === "parking_zone_candidate");
          data.direction_events = allEvents.filter((event) => event.event_type === "wrong_way_movement_candidate");
        }
      }
      setActiveJob(data);
      setParkingEvents(data.parking_events || []);
      setDirectionEvents(data.direction_events || []);
      setPollingJob(data.status === "queued" || data.status === "processing");
      setAiSummary(null); setAiSummaryError(""); setQaMessages([]); setQaSessionId(""); setQaError("");
      setUploadError("");
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Could not load this video job.");
    }
  };

  const generateAiSummary = async () => {
    if (!activeJob?.job_id) return;
    setAiSummaryLoading(true); setAiSummaryError("");
    try {
      const response = await fetch(`${API}/api/v1/videos/jobs/${activeJob.job_id}/ai-summary`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not generate the AI summary.");
      setAiSummary(data);
    } catch (err) {
      setAiSummaryError(err instanceof Error ? err.message : "Could not generate the AI summary.");
    } finally {
      setAiSummaryLoading(false);
    }
  };

  const analyzeCalibratedDirection = async (calibration: { roi_polygon: { x: number; y: number }[]; direction_start: { x: number; y: number }; direction_end: { x: number; y: number } }) => {
    if (!activeJob?.job_id) return;
    setDirectionAnalysisLoading(true); setDirectionAnalysisError("");
    try {
      const response = await fetch(`${API}/api/v1/videos/jobs/${activeJob.job_id}/direction-calibration`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(calibration) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not analyze direction candidates.");
      const allEvents = (data.events || []) as EventRow[];
      const candidates = allEvents.filter((event) => event.event_type === "wrong_way_movement_candidate");
      setDirectionEvents(candidates);
      setActiveJob((current) => current ? {
        ...current,
        candidate_events: allEvents.filter(isMainReviewEvent),
        directional_report_url: data.report_url,
        directional_candidate_count: data.total,
      } : current);
      setShowDirectionCalibrator(false);
    } catch (err) { setDirectionAnalysisError(err instanceof Error ? err.message : "Could not analyze direction candidates."); }
    finally { setDirectionAnalysisLoading(false); }
  };
    const analyzeParkingZone = async (calibration: {
    roi_polygon: { x: number; y: number }[];
    min_stationary_seconds: number;
  }) => {
    if (!activeJob?.job_id) return;

    setParkingAnalysisLoading(true);
    setParkingAnalysisError("");

    try {
      const response = await fetch(
        `${API}/api/v1/videos/jobs/${activeJob.job_id}/parking-analysis`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(calibration),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not analyze parking candidates.");
      }

      const parkingOnly = ((data.events || []) as EventRow[]).filter(
        (event) => event.event_type === "parking_zone_candidate"
      );
      setParkingEvents(parkingOnly);
      setActiveJob((current) => current ? ({
        ...current,
        parking_candidate_count: parkingOnly.length,
        parking_report_url: data.report_url,
        candidate_events: ((data.events || []) as EventRow[]).filter(isMainReviewEvent),
      }) : current);

      setShowParkingCalibrator(false);
    } catch (err) {
      setParkingAnalysisError(
        err instanceof Error
          ? err.message
          : "Could not analyze parking candidates."
      );
    } finally {
      setParkingAnalysisLoading(false);
    }
  };
  const askVideoQuestion = async (event: FormEvent) => {
    event.preventDefault();
    if (!activeJob?.job_id || !qaQuestion.trim()) return;
    setQaLoading(true); setQaError("");
    try {
      const response = await fetch(`${API}/api/v1/videos/jobs/${activeJob.job_id}/questions`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: qaQuestion.trim(), session_id: qaSessionId || undefined }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not answer this question.");
      setQaSessionId(data.session_id);
      window.localStorage.setItem(`roadsentinel-qa-session-${activeJob.job_id}`, data.session_id);
      setQaMessages((current) => [...current, { role: "user", content: data.question, created_at: data.created_at }, { role: "assistant", content: data.answer, citations: data.citations, insufficient_data: data.insufficient_data, source_event_count: data.source_event_count, included_event_count: data.included_event_count, created_at: data.created_at }]);
      setQaQuestion("");
    } catch (err) {
      setQaError(err instanceof Error ? err.message : "Could not answer this question.");
    } finally { setQaLoading(false); }
  };

  const openTimelineEvent = (event: EventRow) => {
    const seconds = Number(event.timestamp_seconds ?? event.timestamp ?? event.start_seconds);
    if (Number.isFinite(seconds) && annotatedVideoRef.current) {
      const video = annotatedVideoRef.current;
      const seekAndPlay = () => {
        video.currentTime = Math.max(0, seconds);
        void video.play().catch(() => undefined);
      };
      if (video.readyState === 0) video.addEventListener("loadedmetadata", seekAndPlay, { once: true });
      else seekAndPlay();
    }
    setSelected(event);
  };

  const explainSelectedEvent = async () => {
    if (!selected?.job_id) return;
    const event = selected;
    setAiEventExplanationLoading(true);
    setAiEventExplanationError(null);
    try {
      const response = await fetch(`${API}/api/v1/videos/jobs/${event.job_id}/events/${encodeURIComponent(event.event_id)}/ai-explanation`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not explain this event.");
      setAiEventExplanation(data);
    } catch (err) {
      setAiEventExplanationError({ event_id: event.event_id, message: err instanceof Error ? err.message : "Could not explain this event." });
    } finally { setAiEventExplanationLoading(false); }
  };

  const saveReviewDecision = async (nextStatus: string) => {
    if (!selected) return;
    setReviewSaving(true); setReviewError("");
    try {
      const response = await fetch(`${API}/api/v1/review-decisions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_id: selected.event_id,
          status: nextStatus,
          reviewer: reviewerName.trim(),
          note: reviewNote.trim(),
          job_id: selected.job_id,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not save reviewer decision.");
      const updated: EventRow = data.event;
      setSelected(updated);
      setReviewNote("");
      if (selected.job_id) {
        setActiveJob((current) => current ? {
          ...current,
          candidate_events: (current.candidate_events || []).map((event: EventRow) =>
            event.event_id === updated.event_id ? updated : event
          ),
        } : current);
        setDirectionEvents((current) => current.map((event) => event.event_id === updated.event_id ? updated : event));
        setParkingEvents((current) => current.map((event) => event.event_id === updated.event_id ? updated : event));
      } else {
        void refresh();
      }
    } catch (err) {
      setReviewError(err instanceof Error ? err.message : "Could not save reviewer decision.");
    } finally {
      setReviewSaving(false);
    }
  };

  const filteredEvents = useMemo(() => events.filter((event) => {
    const typeMatches = eventType === "all" || event.event_type === eventType;
    const statusMatches = status === "all" || event.status === status;
    const needle = search.trim().toLowerCase();
    const searchMatches = !needle || [
      event.event_id, event.class_name, event.event_type, String(event.track_id ?? "")
    ].some((field) => (field || "").toLowerCase().includes(needle));
    return typeMatches && statusMatches && searchMatches;
  }), [events, eventType, status, search]);

  const reviewCount = summary?.review_statuses?.needs_human_review || 0;
  const candidateCount = summary?.review_statuses?.candidate_violation || 0;
  const observationCount = summary?.review_statuses?.observation || 0;
  const confirmedCount = summary?.review_statuses?.confirmed_by_reviewer || 0;
  const pendingCount = reviewCount + candidateCount;
  const completedJobCount = videoJobs.filter((job) => job.status === "completed").length;

  const downloadCsv = () => {
    window.open(`${API}/api/v1/reports/events.csv`, "_blank", "noopener,noreferrer");
  };

  const calibrationTimeline = (title: string, events: EventRow[], analyzed: boolean, emptyMessage: string) => {
    const orderedEvents = [...events].sort((a, b) =>
      Number(a.start_seconds ?? a.timestamp_seconds ?? a.timestamp ?? 0)
      - Number(b.start_seconds ?? b.timestamp_seconds ?? b.timestamp ?? 0)
    );
    return <section className="calibration-results-section" aria-label={`${title} event timeline`}>
      <div className="panel-heading compact">
        <div>
          <div className="section-kicker">CALIBRATED EVENT TIMELINE</div>
          <h3>{title} <span className="count-chip">{orderedEvents.length}</span></h3>
        </div>
      </div>
      {orderedEvents.length === 0
        ? <p className="panel-subtitle">{analyzed ? emptyMessage : "Run this calibration to show its candidate timeline here."}</p>
        : <ol className="calibration-event-timeline">
          {orderedEvents.map((event) => {
            const eventTime = event.start_seconds ?? event.timestamp_seconds ?? event.timestamp;
            return <li key={event.event_id}>
              <button className="calibration-event-row" onClick={() => openTimelineEvent(event)}>
                {event.evidence_url
                  ? <img src={`${API}${event.evidence_url}`} alt={`Evidence for ${event.event_id}`} loading="lazy" />
                  : <span className="calibration-event-no-image"><Camera size={18} /></span>}
                <span className="calibration-event-details">
                  <strong>{fmtTime(eventTime)}{event.end_seconds != null ? `–${fmtTime(event.end_seconds)}` : ""} · {pretty(event.event_type)}</strong>
                  <span>{pretty(event.class_name || "Road user")} · Track #{event.track_id ?? "—"}{event.duration_seconds != null ? ` · ${Number(event.duration_seconds).toFixed(1)}s` : ""}</span>
                  <span className={`status-badge ${statusClass(event.status || "needs_human_review")}`}><span />{pretty(event.status || "needs_human_review")}</span>
                </span>
              </button>
            </li>;
          })}
        </ol>}
    </section>;
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><TrafficCone size={22} strokeWidth={1.8} /></div>
          <div><div className="brand-name">ROAD<span>SENTINEL</span></div><div className="brand-caption">TRAFFIC INTELLIGENCE</div></div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list">
          <a className={`nav-item ${activeSection === "overview" ? "active" : ""}`} href="#overview"><LayoutDashboard size={17} /> Overview {activeSection === "overview" && <span className="nav-pip" />}</a>
          <a className={`nav-item ${activeSection === "events" ? "active" : ""}`} href="#events"><Activity size={17} /> Events {activeSection === "events" && <span className="nav-pip" />}</a>
          <a className={`nav-item ${activeSection === "pipeline" ? "active" : ""}`} href="#pipeline"><FileVideo size={17} /> Video {activeSection === "pipeline" && <span className="nav-pip" />}</a>
        </nav>
        <div className="side-spacer" />
        <div className="side-status">
          <div className="side-status-top"><span className={`status-dot ${health}`} /> SYSTEM STATUS</div>
          <div className="side-status-main">{health === "online" ? (uploadReady ? "Analysis ready" : modelAvailable ? "Check settings" : "Model unavailable") : health === "checking" ? "Connecting to API…" : "API connection unavailable"}</div>
          <div className="side-status-sub">{health === "online" ? "Video analysis readiness" : "Local development environment"}</div>
        </div>
        <div className="sidebar-footer"><span className="mini-logo">RS</span><span>ROADSENTINEL <span className="muted">· v0.1</span></span></div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumbs"><span>Workspace</span><span className="crumb-slash">/</span><strong>Overview</strong></div>
          <div className="topbar-actions">
            <div className={`connection-pill ${health}`}><span className="status-dot" />{health === "online" ? "API CONNECTED" : health === "checking" ? "CONNECTING" : "API OFFLINE"}</div>
            <button className="icon-button" onClick={() => void refresh()} aria-label="Refresh data" title="Refresh data"><RefreshCw size={16} className={loading ? "spin" : ""} /></button>
            <div className="avatar">RS</div>
          </div>
        </header>

        <div className="page-content">
          <section className="hero-row" id="overview">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> TRAFFIC MONITORING / LIVE WORKSPACE</div>
              <h1>Traffic intelligence<span className="title-dot">.</span></h1>
              <p className="hero-copy">Upload footage, review events, and export your findings.</p>
            </div>
            <div className="hero-actions">
              <button className="button button-secondary" onClick={downloadCsv}><ArrowDownToLine size={16} /> Export report</button>
              <a className="button button-primary" href="#pipeline"><FileVideo size={16} /> Video pipeline <ArrowUpRight size={15} /></a>
            </div>
          </section>

          {error && <div className="error-banner"><AlertCircle size={18} /><div><strong>Could not load live data</strong><span>{error} Confirm the API is running and check the configured API URL.</span></div><button onClick={() => void refresh()}>Retry</button></div>}

          <section className="metric-grid">
            <Metric icon={<Activity size={17} />} label="Analysis records" value={summary?.total_events} foot="Global archive; may include tracking observations" accent="cyan" />
            <Metric icon={<Clock3 size={17} />} label="Pending review" value={pendingCount} foot="Candidates requiring human review" accent="amber" />
            <Metric icon={<ShieldCheck size={17} />} label="Reviewer-confirmed" value={confirmedCount} foot="Reviewer decisions in the global archive" accent="green" />
            <Metric icon={<FileVideo size={17} />} label="Completed video analyses" value={completedJobCount} foot={`${videoJobs.length} saved video job${videoJobs.length === 1 ? "" : "s"}`} accent="blue" />
          </section>

          <section className="content-grid">
            <div className="panel activity-panel">
              <div className="panel-heading">
              <div><div className="section-kicker">ANALYSIS OVERVIEW</div><h2>Review status</h2></div>
                <span className="live-tag"><span className="status-dot online" /> API DATA</span>
              </div>
              <div className="distribution">
                <div className="distribution-main">
                  <div className="donut" style={{ "--movement": `${summary?.total_events ? pendingCount / summary.total_events * 100 : 0}%`, "--stationary": `${summary?.total_events ? confirmedCount / summary.total_events * 100 : 0}%` } as React.CSSProperties}>
                    <div className="donut-center"><strong>{summary?.total_events ?? "—"}</strong><span>RECORDS</span></div>
                  </div>
                  <div className="distribution-legend">
                    <Legend color="amber" label="Pending review" value={pendingCount} />
                    <Legend color="green" label="Reviewer-confirmed" value={confirmedCount} />
                    <Legend color="blue" label="Other records" value={Math.max(0, (summary?.total_events || 0) - pendingCount - confirmedCount)} />
                  </div>
                </div>
                <div className="distribution-note"><div className="note-icon"><ShieldCheck size={18} /></div><div><strong>Evidence-first review</strong><p>Every event is presented with its current review status. Model detections alone are not proof of a violation.</p></div></div>
              </div>
              <div className="panel-divider" />
              <div className="panel-heading compact">
                <div><div className="section-kicker">REVIEW QUEUE</div><h2>Event status</h2></div>
              </div>
              <div className="status-rows">
                <StatusRow label="Raw observations · not violations" count={observationCount} total={summary?.total_events || 0} color="blue" />
                <StatusRow label="Candidate violation" count={candidateCount} total={summary?.total_events || 0} color="amber" />
                <StatusRow label="Needs human review" count={reviewCount} total={summary?.total_events || 0} color="amber" />
                <StatusRow label="Confirmed by reviewer" count={confirmedCount} total={summary?.total_events || 0} color="green" />
              </div>
            </div>

            <div className="panel pipeline-panel" id="pipeline">
              <div className="panel-heading">
                <div><div className="section-kicker">UPLOAD & ANALYZE</div><h2>Analyze a new video</h2></div>
                <div className="pipeline-icon"><FileVideo size={19} /></div>
              </div>
              {health === "online" && <div className={`model-readiness ${uploadReady ? "ready" : "unavailable"}`}><span className={`status-dot ${uploadReady ? "online" : "offline"}`} />{uploadReady ? "Model ready" : !modelAvailable ? "Model checkpoint not found. Set ROAD_SENTINEL_MODEL_PATH." : analysisConfigError || "Analysis settings are invalid."}</div>}
              <div style={{ border: "1px dashed var(--border, #334155)", borderRadius: 14, padding: 18, marginBottom: 16, background: "rgba(255,255,255,0.02)" }}>
                <label htmlFor="road-video-upload" style={{ display: "block", fontWeight: 650, marginBottom: 8 }}>Choose dashcam or CCTV footage</label>
                <input id="road-video-upload" type="file" accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm,.mp4,.avi,.mov,.mkv,.webm" onChange={(e) => { setUploadFile(e.target.files?.[0] || null); setUploadError(""); }} style={{ display: "block", width: "100%", marginBottom: 12 }} />
                <div style={{ color: "var(--muted, #94a3b8)", fontSize: 12, lineHeight: 1.6, marginBottom: 12 }}>MP4, AVI, MOV, MKV or WEBM · up to 1 GB · processed locally using your configured GPU/CPU.</div>
                <button className="button button-primary" onClick={() => void uploadVideo()} disabled={!uploadFile || uploading || pollingJob || !uploadReady} style={{ opacity: (!uploadFile || uploading || pollingJob || !uploadReady) ? 0.55 : 1, cursor: (!uploadFile || uploading || pollingJob || !uploadReady) ? "not-allowed" : "pointer" }}>
                  {uploading ? <LoaderCircle size={16} className="spin" /> : <FileVideo size={16} />}{uploading ? "Uploading…" : "Upload & analyze"}
                </button>
                {uploadError && <div className="error-banner" style={{ marginTop: 12 }}><AlertCircle size={17} /><div><strong>Upload/processing issue</strong><span>{uploadError}</span></div></div>}
              </div>
              {!!videoJobs.length && <details className="recent-video-jobs"><summary><span>Recent analyses</span><span className="count-chip">{videoJobs.length}</span></summary><div className="recent-video-job-list">{videoJobs.slice(0, 6).map((job) => <button className={`recent-video-job ${activeJob?.job_id === job.job_id ? "selected" : ""}`} key={job.job_id} onClick={() => void loadVideoJob(job.job_id)}><span className="recent-video-job-icon"><FileVideo size={15} /></span><span className="recent-video-job-name">{job.original_filename || "Uploaded video"}<small>{job.stage || pretty(job.status)}</small></span><span className={`status-badge ${job.status === "completed" ? "confirmed" : job.status === "failed" ? "review" : "neutral"}`}><span />{pretty(job.status)}</span></button>)}</div></details>}
              {activeJob && <div style={{ border: "1px solid var(--border, #334155)", borderRadius: 14, padding: 16, marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
                  <strong>{activeJob.original_filename || "Uploaded video"}</strong>
                  <span className={`status-badge ${activeJob.status === "completed" ? "confirmed" : activeJob.status === "failed" ? "review" : "neutral"}`}><span />{pretty(activeJob.status)}</span>
                </div>
                <div style={{ height: 7, background: "rgba(148,163,184,.2)", borderRadius: 99, overflow: "hidden", margin: "10px 0" }}><div style={{ width: `${Math.max(0, Math.min(100, activeJob.progress || 0))}%`, height: "100%", background: "#36c6d9", transition: "width .3s" }} /></div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, color: "var(--muted, #94a3b8)", fontSize: 12, flexWrap: "wrap" }}><span>{activeJob.stage || "Waiting…"}</span><strong>{activeJob.progress || 0}%</strong></div>
                {activeJob.error && <p style={{ color: "#fca5a5", marginTop: 12, overflowWrap: "anywhere" }}>{activeJob.error}</p>}
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
                  <button className="button button-secondary small" onClick={() => setReportJobId(activeJob.job_id)}><FileVideo size={15} /> View category report</button>
                </div>
                {activeJob.status === "completed" && <>
                  <div className="detail-metrics" style={{ marginTop: 16 }}>
                    <DetailMetric label="Frames processed" value={String(activeJob.frames_processed ?? "—")} />
                    <DetailMetric label="Unique track IDs" value={String(activeJob.unique_track_ids ?? 0)} />
                    <DetailMetric label="Model checkpoint" value={String(activeJob.model_name ? `${activeJob.model_name}/${activeJob.model_version || activeJob.model_checkpoint || ""}` : activeJob.model_checkpoint ?? "Configured model")} />
                    <DetailMetric label="Compute device" value={String(activeJob.device_name ?? activeJob.device ?? "configured device")} />
                  </div>
                  {activeJob.model_class_names && <details className="model-classes">
                    <summary>Detector classes ({Object.keys(activeJob.model_class_names).length})</summary>
                    <div className="model-class-list">{Object.entries(activeJob.model_class_names).map(([classId, className]) => <span key={classId}><b>{classId}</b>{className}</span>)}</div>
                    <p>These are object classes from the configured detector, not violation labels. {activeJob.inference_imgsz ? `Inference size: ${activeJob.inference_imgsz}px.` : ""}{activeJob.track_confidence != null ? ` Confidence threshold: ${activeJob.track_confidence}.` : ""}</p>
                    {(activeJob.torch_version || activeJob.ultralytics_version) && <small>PyTorch {activeJob.torch_version || "not recorded"} · Ultralytics {activeJob.ultralytics_version || "not recorded"}</small>}
                  </details>}
                  {activeJob.annotated_video_url && <div style={{ marginTop: 16 }}>
                    <strong style={{ display: "block", marginBottom: 8 }}>Annotated video</strong>
                    <video ref={annotatedVideoRef} controls preload="metadata" style={{ width: "100%", borderRadius: 10, background: "#000", maxHeight: 360 }} src={`${API}${activeJob.annotated_video_url}`} />
                    <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 12 }}>
                      <a className="button button-secondary small" href={`${API}${activeJob.annotated_video_url}`} download><ArrowDownToLine size={15} /> Download annotated video</a>
                      <a className="button button-secondary small" href={`${API}${activeJob.detections_csv_url}`} download><ArrowDownToLine size={15} /> Download detections CSV</a>
                      <a className="button button-secondary small" href={`${API}${activeJob.summary_url}`} target="_blank" rel="noreferrer"><ArrowUpRight size={15} /> View JSON summary</a>
                    </div>
                    {!!((activeJob.candidate_events || []) as EventRow[]).filter(isMainReviewEvent).length && <div className="upload-event-timeline"><div className="upload-event-timeline-heading"><strong>Event timeline</strong><span>Select a point to seek the annotated video and inspect its evidence.</span></div><ol>{(activeJob.candidate_events as EventRow[]).filter(isMainReviewEvent).slice().sort((a, b) => Number(a.timestamp_seconds ?? a.start_seconds ?? 0) - Number(b.timestamp_seconds ?? b.start_seconds ?? 0)).map((event) => <li key={event.event_id}><button className="upload-timeline-item" onClick={() => openTimelineEvent(event)} aria-label={`Seek to ${fmtTime(event.timestamp_seconds ?? event.start_seconds)} and review ${event.event_id}`}><span className="upload-timeline-marker" /><span className="upload-timeline-time">{fmtTime(event.timestamp_seconds ?? event.start_seconds)}</span><strong>{pretty(event.event_type)}</strong><span className="upload-timeline-detail">{pretty(event.class_name)} · Track #{event.track_id ?? "—"}{event.vehicle_class_name ? ` · Nearby ${pretty(event.vehicle_class_name)} #${event.vehicle_track_id ?? "—"}` : ""}</span><span className={`status-badge ${statusClass(event.status)}`}><span />{pretty(event.status)}</span></button></li>)}</ol></div>}
                  </div>}
                  {activeJob.candidate_event_status === "completed" && <div className="direction-tool">
                    <div className="direction-tool-heading"><div><strong>Wrong-way movement</strong><span>{activeJob.directional_report_url ? `Calibrated · ${activeJob.directional_candidate_count ?? 0} candidate(s) · results need review` : "Optional road calibration · results need review"}</span></div><button className="button button-secondary small" onClick={() => { setShowDirectionCalibrator((value) => !value); setDirectionAnalysisError(""); }}>{showDirectionCalibrator ? "Close" : "Calibrate"}</button></div>
                    {showDirectionCalibrator && <RoadDirectionCalibrator src={`${API}/api/v1/videos/jobs/${activeJob.job_id}/calibration-frame`} loading={directionAnalysisLoading} onSubmit={(calibration) => void analyzeCalibratedDirection(calibration)} />}
                    {directionAnalysisError && <div className="error-banner" style={{ marginTop: 10 }}><AlertCircle size={16} /><div><span>{directionAnalysisError}</span></div></div>}
                    {calibrationTimeline(
                      "Wrong-way movement",
                      directionEvents,
                      Boolean(activeJob.directional_report_url),
                      "Wrong-way analysis completed, but no tracked vehicle met the movement thresholds inside this calibrated road area. Check the road polygon, arrow direction, and footage before interpreting this as no violation."
                    )}
                  </div>}
                                    {activeJob.candidate_event_status === "completed" && <div className="direction-tool">
                    <div className="direction-tool-heading">
                      <div>
                        <strong>Illegal parking candidates</strong>
                        <span>Define a restricted parking zone · requires human review</span>
                      </div>
                      <button
                        className="button button-secondary small"
                        onClick={() => {
                          setShowParkingCalibrator((value) => !value);
                          setParkingAnalysisError("");
                        }}
                      >
                        {showParkingCalibrator ? "Close" : "Calibrate"}
                      </button>
                    </div>

                    {showParkingCalibrator && (
                      <ParkingZoneCalibrator
                        src={`${API}/api/v1/videos/jobs/${activeJob.job_id}/calibration-frame`}
                        loading={parkingAnalysisLoading}
                        onSubmit={(calibration) => void analyzeParkingZone(calibration)}
                      />
                    )}

                    {parkingAnalysisError && (
                      <div className="error-banner" style={{ marginTop: 10 }}>
                        <AlertCircle size={16} />
                        <div><span>{parkingAnalysisError}</span></div>
                      </div>
                    )}
                    {calibrationTimeline(
                      "Illegal parking",
                      parkingEvents,
                      Boolean(activeJob.parking_report_url),
                      "Parking analysis completed, but no vehicle met the stationary-duration and zone thresholds. Check the polygon and duration; a zero-candidate result does not establish that no violation occurred."
                    )}
                  </div>}
                  {activeJob.candidate_event_status === "failed" && <div className="error-banner" style={{ marginTop: 16 }}><AlertCircle size={18} /><div><strong>Review evidence generation failed</strong><span>{activeJob.candidate_event_error || "The video analysis completed, but candidate evidence could not be generated."}</span></div></div>}
                  {activeJob.candidate_event_status === "completed" && <div style={{ marginTop: 20 }}>
                    <div className="panel-heading compact"><div><div className="section-kicker">VIDEO REVIEW</div><h2>Candidate events <span className="count-chip">{((activeJob.candidate_events || []) as EventRow[]).filter(isMainReviewEvent).length}</span></h2></div><div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {activeJob.candidate_events_csv_url && <a className="button button-secondary small" href={`${API}${activeJob.candidate_events_csv_url}`} download><ArrowDownToLine size={14} /> Events CSV</a>}
                      {activeJob.candidate_events_json_url && <a className="button button-secondary small" href={`${API}${activeJob.candidate_events_json_url}`} download><ArrowDownToLine size={14} /> Events JSON</a>}
                      <a className="button button-secondary small" href={`${API}/api/v1/review-decisions.csv?job_id=${activeJob.job_id}`} download><ArrowDownToLine size={14} /> Review history</a>
                    </div></div>
                    {((activeJob.candidate_events || []) as EventRow[]).filter(isMainReviewEvent).length === 0 ? <p className="panel-subtitle">No reviewable candidate events met the current thresholds. This does not establish that the video contains no traffic events.</p> : <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: 12 }}>
                      {(activeJob.candidate_events as EventRow[]).filter(isMainReviewEvent).map((event) => <button key={event.event_id} className="job-candidate-card" onClick={() => setSelected(event)} aria-label={`Review ${event.event_id}`}>
                        {event.evidence_url ? <img src={`${API}${event.evidence_url}`} alt={`Evidence for ${event.event_id}`} loading="lazy" /> : <span className="job-candidate-placeholder"><Camera size={20} />Evidence image unavailable</span>}
                        <span className="job-candidate-meta"><strong>{pretty(event.event_type)}</strong><span>{fmtTime(event.start_seconds)} · Track #{event.track_id ?? "—"}</span><span className={`status-badge ${statusClass(event.status)}`}><span />{pretty(event.status)}</span></span>
                      </button>)}
                    </div>}
                  </div>}
                  <div className="ai-summary-panel">
                    <div className="ai-summary-heading"><div><div className="section-kicker">OPTIONAL AI EXPLANATION</div><h3>Evidence-grounded audit summary</h3></div><button className="button button-primary small" disabled={!aiConfigured || aiSummaryLoading} onClick={() => void generateAiSummary()}><Sparkles size={15} />{aiSummaryLoading ? "Generating…" : "Generate summary"}</button></div>
                    {!aiConfigured && <p className="panel-subtitle">Configure <code>OPENAI_API_KEY</code> and <code>OPENAI_MODEL</code> in the backend <code>.env</code> file to enable this option. No video frames are sent; the summary uses structured event data.</p>}
                    {aiSummaryError && <div className="error-banner" style={{ marginTop: 12 }}><AlertCircle size={18} /><div><strong>AI summary unavailable</strong><span>{aiSummaryError}</span></div></div>}
                    {aiSummary && <div className="ai-summary-content">
                      <div className="ai-summary-meta"><span>{aiSummary.model}</span><span>{aiSummary.included_event_count} of {aiSummary.source_event_count} events supplied</span><a href={`${API}${aiSummary.download_url}`} download>Download JSON</a></div>
                      <p className="ai-executive-summary">{aiSummary.summary.executive_summary}</p>
                      {!!aiSummary.summary.observed_patterns?.length && <><strong>Observed patterns</strong><ul>{aiSummary.summary.observed_patterns.map((item: string, index: number) => <li key={index}>{item}</li>)}</ul></>}
                      {!!aiSummary.summary.events_to_review?.length && <><strong>Events to review</strong><ul>{aiSummary.summary.events_to_review.map((item: { event_id: string; reason: string }) => <li key={item.event_id}><button className="ai-event-link" onClick={() => { const event = (activeJob.candidate_events || []).find((candidate: EventRow) => candidate.event_id === item.event_id); if (event) setSelected(event); }}>{item.event_id}</button> — {item.reason}</li>)}</ul></>}
                      {!!aiSummary.summary.reviewed_events?.length && <><strong>Reviewer decisions in source data</strong><ul>{aiSummary.summary.reviewed_events.map((item: { event_id: string; review_status: string; note: string }) => <li key={item.event_id}>{item.event_id}: {pretty(item.review_status)}{item.note ? ` — ${item.note}` : ""}</li>)}</ul></>}
                      {!!aiSummary.summary.limitations?.length && <><strong>Limitations</strong><ul>{aiSummary.summary.limitations.map((item: string, index: number) => <li key={index}>{item}</li>)}</ul></>}
                      <p className="muted">{aiSummary.interpretation} {aiSummary.validation_note}</p>
                    </div>}
                  </div>
                  <div className="ai-qa-panel">
                    <div className="ai-summary-heading"><div><div className="section-kicker">ASK ABOUT THIS VIDEO</div><h3>Event data Q&amp;A</h3></div><span className="ai-qa-saved">{qaSessionId ? "Conversation saved" : "Saved per video"}</span></div>
                    {!aiConfigured && <p className="panel-subtitle">Configure <code>OPENAI_API_KEY</code> and <code>OPENAI_MODEL</code> in the backend <code>.env</code> file. Questions use structured events and reviewer notes; video frames are not sent.</p>}
                    {!!qaMessages.length && <div className="ai-qa-thread" aria-live="polite">{qaMessages.map((message, index) => <div key={`${message.created_at || "m"}-${index}`} className={`ai-qa-message ${message.role}`}><strong>{message.role === "user" ? "You" : "RoadSentinel assistant"}</strong><p>{message.content}</p>{message.role === "assistant" && <span className="ai-qa-sources">Source coverage: {message.included_event_count ?? "—"} of {message.source_event_count ?? "—"} event records · no video frames</span>}{message.insufficient_data && <span className="ai-qa-insufficient">The uploaded event data could not answer this fully.</span>}{!!message.citations?.length && <div className="ai-qa-citations">{message.citations.map((citation) => <div className="ai-qa-citation" key={citation.event_id}><button className="ai-event-link" onClick={() => { const event = (activeJob.candidate_events || []).find((candidate: EventRow) => candidate.event_id === citation.event_id); if (event) setSelected(event); }} aria-label={`Open source event ${citation.event_id} at ${fmtTime(citation.timestamp_seconds)}`}>{citation.event_id} · {fmtTime(citation.timestamp_seconds)} · {pretty(citation.status)}</button><span>{citation.reason}</span></div>)}</div>}</div>)}</div>}
                    {qaError && <div className="error-banner" style={{ marginTop: 12 }}><AlertCircle size={18} /><div><strong>Q&amp;A unavailable</strong><span>{qaError}</span></div></div>}
                    <form className="ai-qa-form" onSubmit={(event) => void askVideoQuestion(event)}><textarea value={qaQuestion} onChange={(event) => setQaQuestion(event.target.value)} maxLength={1500} placeholder="Ask about observed events, timestamps, or reviewer decisions…" aria-label="Question about this video" disabled={!aiConfigured || qaLoading} /><button className="button button-primary small" type="submit" disabled={!aiConfigured || qaLoading || !qaQuestion.trim()}><Sparkles size={15} />{qaLoading ? "Thinking…" : "Ask"}</button></form>
                    <p className="muted ai-qa-disclaimer">Answers are based on up to 100 structured events from this video. Citations link to included event records; insufficient data is reported explicitly.</p>
                  </div>
                  <p className="panel-subtitle" style={{ marginTop: 12 }}>{activeJob.interpretation}</p>
                </>}
                {(activeJob.status === "completed" || activeJob.status === "failed") && <button className="button button-primary" style={{ marginTop: 12 }} onClick={startAnotherUpload}><FileVideo size={15} /> Analyze another video</button>}
              </div>}
              <div className="pipeline-steps">
                <PipelineStep n="01" title="Upload video" detail="MP4, AVI, MOV, MKV, WEBM" state="complete" />
                <PipelineStep n="02" title="Object detection" detail="Trained YOLO road-scene model" state="complete" />
                <PipelineStep n="03" title="Object tracking" detail="BoT-SORT track IDs" state="complete" />
                <PipelineStep n="04" title="Annotated video & CSV" detail="Model observations; not confirmed violations" state="complete" />
                <PipelineStep n="05" title="Candidate evidence" detail="Review-only events with frame evidence" state="complete" />
              </div>
              <div className="pipeline-footer"><span><CheckCircle2 size={15} /> Upload processing enabled</span><span className="muted">Human review still required</span></div>
            </div>
          </section>

          <section className="panel events-panel" id="events">
            <div className="panel-heading events-heading">
              <div><div className="section-kicker">INSPECT & REVIEW</div><h2>Global event records <span className="count-chip">{total}</span></h2><p className="panel-subtitle">This archive can include tracking-only observations; they are not violation findings. Open a video’s category report for safety candidates.</p></div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <button className="button button-secondary small" onClick={downloadCsv}><ArrowDownToLine size={15} /> Events CSV</button>
                <a className="button button-secondary small" href={`${API}/api/v1/reports/events.json`} download><ArrowDownToLine size={15} /> Events JSON</a>
                <a className="button button-secondary small" href={`${API}/api/v1/review-decisions.csv`} download><ArrowDownToLine size={15} /> Review history</a>
              </div>
            </div>
            <div className="filters">
              <div className="search-field"><Search size={16} /><input ref={eventSearchRef} value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search events…" aria-label="Search events" /><kbd className="search-shortcut">/</kbd></div>
              <div className="filter-select"><Filter size={15} /><select value={eventType} onChange={(e) => { setEventType(e.target.value); setOffset(0); }}><option value="all">All record types</option><option value="movement_observation">Movement observation · tracking only</option><option value="stationary_object_candidate">Stationary-object candidate</option></select></div>
              <div className="filter-select"><select value={status} onChange={(e) => { setStatus(e.target.value); setOffset(0); }}><option value="all">All statuses</option><option value="observation">Observation</option><option value="candidate_violation">Candidate violation</option><option value="needs_human_review">Needs human review</option><option value="confirmed_by_reviewer">Confirmed by reviewer</option><option value="rejected_by_reviewer">Rejected by reviewer</option></select></div>
            </div>

            <div className="table-wrap">
              <table>
                <thead><tr><th>EVENT RECORD</th><th>ROAD USER</th><th>TIMESTAMP</th><th>DETECTION SCORE</th><th>REVIEW STATUS</th><th /></tr></thead>
                <tbody>
                  {loading && events.length === 0 ? <tr><td colSpan={6} className="empty-cell"><LoaderCircle className="spin" size={20} /> Loading event data…</td></tr> :
                    filteredEvents.length === 0 ? <tr><td colSpan={6} className="empty-cell">No events match these filters on this page.</td></tr> :
                    filteredEvents.map((event) => (
                      <tr key={event.event_id} onClick={() => setSelected(event)} className="event-row">
                        <td><div className="event-id">{event.event_id}</div><div className="event-type">{pretty(event.event_type)}</div></td>
                        <td><span className="road-user"><span className={`class-icon ${event.class_name === "car" ? "car" : ""}`}><Camera size={14} /></span><span>{pretty(event.class_name)}<small>Track #{event.track_id ?? "—"}{event.vehicle_class_name ? ` · Nearby ${pretty(event.vehicle_class_name)} #${event.vehicle_track_id ?? "—"}` : ""}</small></span></span></td>
                        <td><span className="timestamp">{fmtTime(event.start_seconds)}</span><small className="subtime">Frame {event.evidence_frame ?? "—"}</small></td>
                        <td><div className="confidence-cell"><div className="confidence-track"><span style={{ width: `${Math.max(0, Math.min(100, (event.confidence || 0) * 100))}%` }} /></div><span>{event.confidence != null ? `${Math.round(event.confidence * 100)}%` : "—"}</span></div></td>
                        <td><span className={`status-badge ${statusClass(event.status)}`}><span />{pretty(event.status)}</span></td>
                        <td><button className="row-open" aria-label={`Open ${event.event_id}`} onClick={(e) => { e.stopPropagation(); setSelected(event); }}><ArrowUpRight size={16} /></button></td>
                      </tr>
                    ))
                  }
                </tbody>
              </table>
            </div>
            <div className="table-footer"><span>Showing <strong>{total ? offset + 1 : 0}–{Math.min(offset + pageSize, total)}</strong> of <strong>{total}</strong> report events{(eventType !== "all" || status !== "all" || search) ? " · filters apply to this page" : ""}</span><div className="pagination"><button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - pageSize))}><ChevronLeft size={16} /> Previous</button><span>{Math.floor(offset / pageSize) + 1} / {Math.max(1, Math.ceil(total / pageSize))}</span><button disabled={offset + pageSize >= total} onClick={() => setOffset(offset + pageSize)}>Next <ChevronRight size={16} /></button></div></div>
          </section>

          <footer className="page-footer"><span>© 2026 RoadSentinel · Traffic intelligence workspace</span><span><span className="footer-dot" /> Evidence-led · Human review required</span></footer>
        </div>
      </main>

      {reportJobId && <VideoJobReport
        api={API}
        jobId={reportJobId}
        onClose={() => setReportJobId(null)}
        onSelectEvent={(event) => {
          if (!event.event_id) return;
          setReportJobId(null);
          const trackId = Number(event.track_id);
          const eventTime = event.timestamp_seconds ?? event.timestamp ?? event.start_seconds;
          openTimelineEvent({
            ...event,
            event_id: event.event_id,
            event_type: event.event_type || "unknown",
            status: event.status || "needs_human_review",
            timestamp_seconds: eventTime,
            start_seconds: event.start_seconds ?? eventTime,
            track_id: Number.isFinite(trackId) ? trackId : undefined,
            job_id: reportJobId,
            review_history: event.review_history?.map((item) => ({
              previous_status: item.previous_status || "",
              status: item.status || "",
              reviewer: item.reviewer || "",
              note: item.note || "",
              changed_at: item.changed_at || "",
            })),
          });
        }}
      />}
      {selected && <div className="modal-backdrop" role="presentation" onMouseDown={(e) => { if (e.target === e.currentTarget) setSelected(null); }}>
        <section className="detail-modal" role="dialog" aria-modal="true" aria-labelledby="detail-title">
          <div className="modal-header"><div><div className="section-kicker">EVENT DETAIL</div><h2 id="detail-title">{selected.event_id}</h2></div><button className="icon-button" onClick={() => setSelected(null)} aria-label="Close event detail"><X size={18} /></button></div>
          <div className="modal-image-wrap">
            {selected.evidence_url ? <img src={`${API}${selected.evidence_url}`} alt={`Evidence for ${selected.event_id}`} /> : <div className="image-placeholder"><Camera size={26} /><span>No evidence image available</span></div>}
          </div>
          <div className="modal-body">
            <div className="detail-title-row"><div><span className="section-kicker">EVENT TYPE</span><strong>{pretty(selected.event_type)}</strong></div><span className={`status-badge ${statusClass(selected.status)}`}><span />{pretty(selected.status)}</span></div>
            <div className="detail-metrics"><DetailMetric label="Timestamp" value={fmtTime(selected.timestamp_seconds ?? selected.timestamp ?? selected.start_seconds)} /><DetailMetric label="Track ID" value={String(selected.track_id ?? "—")} />{selected.helmet_model_track_id != null && <DetailMetric label="Helmet model track" value={String(selected.helmet_model_track_id)} />}<DetailMetric label={selected.helmet_detector_confidence != null ? "Helmet model score" : "Confidence"} value={selected.helmet_detector_confidence != null ? `${Math.round(selected.helmet_detector_confidence * 100)}%` : selected.confidence != null ? `${Math.round(selected.confidence * 100)}%` : "—"} /><DetailMetric label="Duration" value={selected.duration_seconds != null ? `${selected.duration_seconds.toFixed(2)}s` : "—"} /></div>
            {selected.vehicle_class_name && <p className="detail-interaction-meta">Nearby vehicle: {pretty(selected.vehicle_class_name)} · Track #{selected.vehicle_track_id ?? "—"}.</p>}
            {(selected.participant_count != null || selected.related_track_ids?.length || selected.image_space_separation_pixels != null) && <p className="detail-interaction-meta">
              {selected.participant_count != null ? `Associated rider tracks: ${selected.participant_count}. ` : ""}
              {selected.related_track_ids?.length && !selected.vehicle_track_id ? `Related track IDs: ${selected.related_track_ids.join(", ")}. ` : ""}
              {selected.image_space_separation_pixels != null ? `Image-space separation: ${selected.image_space_separation_pixels} px (not physical distance).` : ""}
            </p>}
            <div className="review-note"><AlertCircle size={17} /><div><strong>Interpretation</strong><p>{selected.notes || "This event has not been independently verified."}</p><p className="muted">This record is an observation or candidate. Review the source footage before drawing conclusions.</p></div></div>
            {selected.job_id && <div className="ai-event-explanation"><div className="ai-event-explanation-heading"><div><strong>AI event explanation</strong><span>Uses this event's structured record only; no video frames are sent.</span></div><button className="button button-secondary small" disabled={!aiConfigured || aiEventExplanationLoading} onClick={() => void explainSelectedEvent()}><Sparkles size={14} />{aiEventExplanationLoading ? "Explaining…" : aiEventExplanation?.event_id === selected.event_id ? "Regenerate" : "Explain event"}</button></div>
              {!aiConfigured && <p className="muted">Configure <code>OPENAI_API_KEY</code> and <code>OPENAI_MODEL</code> in the backend <code>.env</code> file to enable this.</p>}
              {aiEventExplanationError?.event_id === selected.event_id && <p className="review-error">{aiEventExplanationError.message}</p>}
              {aiEventExplanation?.event_id === selected.event_id && <div className="ai-event-explanation-content"><p>{aiEventExplanation.explanation.explanation}</p>{!!aiEventExplanation.explanation.supporting_facts?.length && <><strong>Record supports</strong><ul>{aiEventExplanation.explanation.supporting_facts.map((fact: string, index: number) => <li key={index}>{fact}</li>)}</ul></>}{!!aiEventExplanation.explanation.uncertainties?.length && <><strong>Uncertainties</strong><ul>{aiEventExplanation.explanation.uncertainties.map((item: string, index: number) => <li key={index}>{item}</li>)}</ul></>}<strong>Reviewer check</strong><p>{aiEventExplanation.explanation.reviewer_guidance}</p><div className="ai-summary-meta"><span>{aiEventExplanation.model}</span><a href={`${API}${aiEventExplanation.download_url}`} download>Download explanation JSON</a></div><p className="muted">{aiEventExplanation.interpretation}</p></div>}
            </div>}
            <div className="review-decision-panel">
              <strong>Reviewer decision</strong>
              <label>Reviewer name<input value={reviewerName} maxLength={120} onChange={(e) => setReviewerName(e.target.value)} /></label>
              <label>Review note<textarea value={reviewNote} maxLength={2000} rows={2} onChange={(e) => setReviewNote(e.target.value)} placeholder="Optional reason or context" /></label>
              <div className="review-decision-actions">
                <button className="button button-secondary small" disabled={reviewSaving || !reviewerName.trim()} onClick={() => void saveReviewDecision("needs_human_review")}>Keep for review</button>
                <button className="button button-primary small" disabled={reviewSaving || !reviewerName.trim()} onClick={() => void saveReviewDecision("confirmed_by_reviewer")}>Confirm</button>
                <button className="button button-secondary small" disabled={reviewSaving || !reviewerName.trim()} onClick={() => void saveReviewDecision("rejected_by_reviewer")}>Reject</button>
              </div>
              {reviewError && <p className="review-error">{reviewError}</p>}
            </div>
            {!!selected.review_history?.length && <div className="review-history"><strong>Decision history</strong>{selected.review_history.map((item, index) => <div key={`${item.changed_at}-${index}`}><span>{pretty(item.previous_status)} → {pretty(item.status)}</span><small>{item.reviewer} · {new Date(item.changed_at).toLocaleString()}</small>{item.note && <p>{item.note}</p>}</div>)}</div>}
            {selected.evidence_url && <a className="button button-secondary full-button" href={`${API}${selected.evidence_url}`} target="_blank" rel="noreferrer"><ArrowUpRight size={16} /> Open full evidence image</a>}
          </div>
        </section>
      </div>}
    </div>
  );
}

function Metric({ icon, label, value, foot, accent }: { icon: React.ReactNode; label: string; value?: number; foot: string; accent: string }) {
  return <div className="metric-card"><div className="metric-top"><span className={`metric-icon ${accent}`}>{icon}</span><span className="metric-period">CURRENT REPORT</span></div><div className="metric-label">{label}</div><div className="metric-value">{value == null ? "—" : value.toLocaleString()}</div><div className="metric-foot">{foot}</div></div>;
}
function Legend({ color, label, value }: { color: string; label: string; value: number }) {
  return <div className="legend-row"><span className={`legend-dot ${color}`} /><span>{label}</span><strong>{value}</strong></div>;
}
function StatusRow({ label, count, total, color }: { label: string; count: number; total: number; color: string }) {
  const pct = total ? count / total * 100 : 0;
  return <div className="status-row"><div className="status-row-top"><span>{label}</span><strong>{count}</strong></div><div className="progress-track"><span className={color} style={{ width: `${pct}%` }} /></div></div>;
}
function PipelineStep({ n, title, detail, state }: { n: string; title: string; detail: string; state: "complete" | "pending" }) {
  return <div className="pipeline-step"><div className={`step-number ${state}`}>{state === "complete" ? <CheckCircle2 size={15} /> : n}</div><div className="step-copy"><strong>{title}</strong><span>{detail}</span></div><span className={`step-state ${state}`}>{state === "complete" ? "COMPLETE" : "PLANNED"}</span></div>;
}
function DetailMetric({ label, value }: { label: string; value: string }) {
  return <div className="detail-metric"><span>{label}</span><strong>{value}</strong></div>;
}
