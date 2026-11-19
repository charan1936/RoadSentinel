import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { AlertCircle, ArrowDownToLine, LoaderCircle, Printer, ShieldAlert, X } from "lucide-react";

type ReportEvent = {
  event_id?: string;
  category?: string;
  event_type?: string;
  status?: string;
  timestamp_seconds?: number;
  timestamp?: number;
  start_seconds?: number;
  end_seconds?: number;
  track_id?: number | string;
  class_name?: string;
  confidence?: number;
  risk_level?: string;
  uncertainty?: string;
  helmet_detector_confidence?: number;
  rider_detector_confidence?: number;
  helmet_model_track_id?: number;
  track_source?: string;
  participant_count?: number;
  related_track_ids?: number[];
  vehicle_track_id?: number;
  vehicle_class_name?: string;
  person_displacement_pixels?: number;
  person_displacement_heights?: number;
  image_space_separation_pixels?: number;
  notes?: string;
  description?: string;
  evidence_url?: string | null;
  reviewer?: string;
  review_note?: string;
  reviewed_at?: string;
  review_history?: { previous_status?: string; status?: string; reviewer?: string; note?: string; changed_at?: string }[];
};
type ReportData = {
  job_id: string;
  video?: { filename?: string; created_at?: string; updated_at?: string; status?: string; model_name?: string; model_version?: string; analysis_settings?: unknown };
  summary?: { total_events?: number; counts?: Record<string, number>; interpretation?: string; confirmed_message?: string };
  events?: ReportEvent[];
  violation_categories?: { id: string; name: string; status: string; event_count: number; event_ids: string[]; message: string; events?: ReportEvent[] }[];
  analysis_errors?: string[];
  limitations?: string[];
};

const pretty = (value?: string) => (value || "Not recorded").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
const timeLabel = (value?: number) => {
  if (value == null || !Number.isFinite(Number(value))) return "Timestamp not recorded";
  const seconds = Number(value);
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
};
const dateLabel = (value?: string) => {
  if (!value) return "Not recorded";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};
const CATEGORY_EVENT_TYPES: Record<string, string> = {
  helmetless_riding: "helmetless_riding_candidate",
  triple_riding: "triple_riding_candidate",
  illegal_parking: "parking_zone_candidate",
  overloaded_goods_vehicle: "goods_vehicle_load_risk_candidate",
  unsafe_pedestrian_crossing: "unsafe_pedestrian_interaction_candidate",
  wrong_way_movement: "wrong_way_movement_candidate",
};

export default function VideoJobReport({ api, jobId, onClose, onSelectEvent }: { api: string; jobId: string; onClose: () => void; onSelectEvent?: (event: ReportEvent) => void }) {
  const [report, setReport] = useState<ReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    fetch(`${api}/api/v1/videos/jobs/${encodeURIComponent(jobId)}/report`, { signal: controller.signal })
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not load the video report.");
        return data as ReportData;
      })
      .then(setReport)
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Could not load the video report.");
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [api, jobId, retryCount]);

  const events = report?.events || [];
  const categories = report?.violation_categories || [];
  const categoryEvents = useMemo(() => categories.map((category) => ({
    ...category,
    events: category.events ?? events.filter((event) => event.event_type === CATEGORY_EVENT_TYPES[category.id]),
  })), [categories, events]);
  const safetyEvents = useMemo(() => {
    const byId = new Map<string, ReportEvent>();
    categoryEvents.forEach((category) => category.events.forEach((event, index) => {
      byId.set(event.event_id || `${category.id}-${index}`, event);
    }));
    return [...byId.values()];
  }, [categoryEvents]);
  const confirmedCount = safetyEvents.filter((event) => event.status === "confirmed_by_reviewer").length;
  const pendingCount = safetyEvents.filter((event) => ["needs_human_review", "candidate_violation"].includes(event.status || "")).length;
  const rejectedCount = safetyEvents.filter((event) => event.status === "rejected_by_reviewer").length;

  const evidenceUrl = (url?: string | null) => url ? (/^https?:\/\//i.test(url) ? url : `${api}${url.startsWith("/") ? "" : "/"}${url}`) : "";

  const exportHtml = () => {
    if (!report) return;
    const escapeHtml = (value: unknown) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" } as Record<string, string>)[char] || char);
    const categorySections = categoryEvents.map((category) => `<section><h2>${escapeHtml(category.name)} — ${escapeHtml(pretty(category.status))}</h2><p>${escapeHtml(category.message)} · ${category.events.length} event(s)</p>${category.events.length ? category.events.map((event) => `<article><h3>${escapeHtml(timeLabel(event.timestamp_seconds ?? event.timestamp ?? event.start_seconds))} · ${escapeHtml(pretty(event.event_type))}</h3><p><b>Review:</b> ${escapeHtml(pretty(event.status))}${event.class_name ? ` · <b>Road user:</b> ${escapeHtml(pretty(event.class_name))}` : ""}${event.track_id != null ? ` · <b>Track:</b> ${escapeHtml(event.track_id)}` : ""}${event.vehicle_class_name ? ` · <b>Nearby vehicle:</b> ${escapeHtml(pretty(event.vehicle_class_name))} (track ${escapeHtml(event.vehicle_track_id ?? "not recorded")})` : ""}${event.helmet_model_track_id != null ? ` · <b>Helmet-model track:</b> ${escapeHtml(event.helmet_model_track_id)}` : ""}${event.confidence != null && Number.isFinite(Number(event.confidence)) ? ` · <b>Detection confidence:</b> ${(Number(event.confidence) * 100).toFixed(1)}%` : ""}${event.risk_level ? ` · <b>Risk:</b> ${escapeHtml(pretty(event.risk_level))}` : ""}</p><p>${escapeHtml(event.uncertainty || event.notes || event.description || "No additional explanation was saved.")}</p>${event.evidence_url ? `<p><a href="${escapeHtml(evidenceUrl(event.evidence_url))}">Open evidence image</a></p>` : "<p>Evidence image not available.</p>"}</article>`).join("") : "<p>No safety candidates were recorded for this category.</p>"}</section>`).join("");
    const categorySummary = categoryEvents.map((category) => `<li><b>${escapeHtml(category.name)}</b>: ${escapeHtml(pretty(category.status))} (${category.events.length} events). ${escapeHtml(category.message)}</li>`).join("");
    const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RoadSentinel report</title><style>body{font:16px/1.55 Arial,sans-serif;color:#172033;max-width:900px;margin:40px auto;padding:0 22px}h1{margin-bottom:4px}h2{border-bottom:1px solid #ccd3dd;padding-bottom:8px;margin-top:30px}article{border:1px solid #d9dee7;border-radius:10px;padding:14px 18px;margin:12px 0;break-inside:avoid}.notice{background:#f5f3ed;padding:14px;border-left:4px solid #5b928b}small{color:#596579}@media print{body{margin:0;max-width:none}.no-print{display:none}}</style></head><body><h1>RoadSentinel — Traffic audit report</h1><p><b>Video:</b> ${escapeHtml(report.video?.filename || "Not recorded")}<br><b>Analysis date:</b> ${escapeHtml(dateLabel(report.video?.created_at))}<br><b>Job ID:</b> ${escapeHtml(report.job_id)}</p><div class="notice"><b>Review distinction:</b> ${escapeHtml(report.summary?.confirmed_message || "No reviewer-confirmed violations recorded.")} Automated candidates are not legal findings.</div><h2>Safety event summary</h2><p>${safetyEvents.length} safety candidate(s). Reviewer-confirmed: ${confirmedCount}; needs review: ${pendingCount}; rejected: ${rejectedCount}. Generic movement observations are excluded.</p><h2>Violation category coverage</h2><ul>${categorySummary || "<li>Category analysis status is not available in this report.</li>"}</ul>${categorySections}<section><h2>What this report can and cannot tell you</h2><p>Automated detections and candidates are not legal findings. Unreviewed candidates must not be treated as confirmed violations.</p><ul>${(report.limitations || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></section><p><small>Generated by RoadSentinel. Review source footage and evidence before making conclusions.</small></p><button class="no-print" onclick="window.print()">Print / Save as PDF</button></body></html>`;
    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `RoadSentinel_Report_${jobId}.html`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  const printReport = () => {
    const cleanup = () => document.body.classList.remove("rs-report-printing");
    document.body.classList.add("rs-report-printing");
    window.addEventListener("afterprint", cleanup, { once: true });
    window.print();
    cleanup();
  };

  const section = (category: (typeof categoryEvents)[number]) => <section className="rs-report-section" key={category.id}>
    <div className="rs-report-section-heading"><h3>{category.name}</h3><span>{category.events.length}</span><span className={`rs-report-category-status ${category.status}`}>{pretty(category.status)}</span></div>
    <p className="rs-report-category-message">{category.message}</p>
    {category.events.length === 0 ? <p className="rs-report-empty">{category.status === "no_events" ? "Analysis completed; no candidates were detected. This does not establish that no violation occurred." : "No event evidence is available for this category."}</p> : <ol className="rs-report-events">{category.events.map((event, index) => <li key={event.event_id || `${category.id}-${index}`} className="rs-report-event">
      <div className="rs-report-event-top"><div><button className="rs-report-time" onClick={() => onSelectEvent?.(event)} disabled={!onSelectEvent || (event.timestamp_seconds == null && event.timestamp == null && event.start_seconds == null)} title={onSelectEvent ? "Seek video to this event" : undefined}>{timeLabel(event.timestamp_seconds ?? event.timestamp ?? event.start_seconds)}{event.end_seconds != null ? ` – ${timeLabel(event.end_seconds)}` : ""}</button><h4>{pretty(event.event_type)}</h4></div><span className={`rs-report-status ${event.status === "confirmed_by_reviewer" ? "confirmed" : event.status === "rejected_by_reviewer" ? "rejected" : "pending"}`}>{pretty(event.status)}</span></div>
      <div className="rs-report-meta">{event.class_name ? `Road user: ${pretty(event.class_name)}` : "Road user: Not recorded"}{event.track_id != null ? ` · Track #${event.track_id}` : ""}{event.vehicle_class_name ? ` · Nearby vehicle: ${pretty(event.vehicle_class_name)} · Track #${event.vehicle_track_id ?? "—"}` : ""}{event.helmet_detector_confidence != null && Number.isFinite(Number(event.helmet_detector_confidence)) ? ` · Helmet-model no-helmet class score ${(Number(event.helmet_detector_confidence) * 100).toFixed(1)}%` : event.confidence != null && Number.isFinite(Number(event.confidence)) ? ` · Detection confidence ${(Number(event.confidence) * 100).toFixed(1)}%` : ""}{event.helmet_model_track_id != null ? ` · Helmet-model track #${event.helmet_model_track_id}` : ""}{event.participant_count != null ? ` · Associated riders: ${event.participant_count}` : ""}{event.related_track_ids?.length && !event.vehicle_track_id ? ` · Related tracks: ${event.related_track_ids.join(", ")}` : ""}{event.image_space_separation_pixels != null ? ` · Image-space separation: ${event.image_space_separation_pixels} px` : ""}{event.event_id ? ` · ${event.event_id}` : ""}</div>
      <p>{event.description || event.notes || "No additional explanation was saved with this event."}</p>
      {(event.risk_level || event.uncertainty) && <p className="rs-report-event-uncertainty">{event.risk_level ? `Risk: ${pretty(event.risk_level)}. ` : ""}{event.uncertainty || ""}</p>}
      {event.reviewer && <p className="rs-report-reviewer">Reviewer: {event.reviewer}{event.reviewed_at ? ` · ${dateLabel(event.reviewed_at)}` : ""}</p>}
      {event.review_note && <p className="rs-report-reviewer">Reviewer note: {event.review_note}</p>}
      {!!event.review_history?.length && <details><summary>Review history ({event.review_history.length})</summary>{event.review_history.map((item, historyIndex) => <p key={`${item.changed_at}-${historyIndex}`}>{pretty(item.previous_status)} → {pretty(item.status)} · {item.reviewer || "Reviewer not recorded"}{item.note ? ` — ${item.note}` : ""}</p>)}</details>}
      {event.evidence_url ? <a className="rs-report-evidence-link" href={evidenceUrl(event.evidence_url)} target="_blank" rel="noreferrer"><img src={evidenceUrl(event.evidence_url)} alt={`Evidence for ${event.event_id || "event"}`} loading="eager"/><span>Open evidence image</span></a> : <div className="rs-report-no-evidence">Evidence image not available for this event.</div>}
    </li>)}</ol>}
  </section>;

  return createPortal(<div className="rs-report-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="rs-report-dialog" role="dialog" aria-modal="true" aria-labelledby="rs-report-title">
      <header className="rs-report-toolbar"><div><span className="section-kicker">VIDEO AUDIT</span><h2 id="rs-report-title">Human-readable report</h2></div><div className="rs-report-actions">{report && <><button className="button button-secondary small" onClick={exportHtml}><ArrowDownToLine size={15}/> Download HTML</button><button className="button button-secondary small" onClick={printReport}><Printer size={15}/> Print / PDF</button></>}<button className="icon-button" onClick={onClose} aria-label="Close report"><X size={18}/></button></div></header>
      <div className="rs-report-content">
        {loading && <div className="rs-report-state"><LoaderCircle className="spin"/> Loading saved job report…</div>}
        {error && <div className="error-banner"><AlertCircle size={18}/><div><strong>Report unavailable</strong><span>{error}</span><button className="button button-secondary small" onClick={() => setRetryCount((count) => count + 1)}>Retry report</button></div></div>}
        {report && <>
          <div className="rs-report-intro"><div><span className="section-kicker">ROADSENTINEL · TRAFFIC REVIEW</span><h1>{report.video?.filename || "Video analysis report"}</h1><p>Analysis date: {dateLabel(report.video?.created_at)} · Job: {report.job_id} · Analysis status: {pretty(report.video?.status)}</p></div><span className="rs-report-total">{safetyEvents.length} safety event{safetyEvents.length === 1 ? "" : "s"}</span></div>
          <div className="rs-report-confirmed-notice"><ShieldAlert size={20}/><div><strong>{report.summary?.confirmed_message || "No reviewer-confirmed violations recorded."}</strong><p>Only explicit reviewer decisions are counted as confirmed. Machine-generated candidates remain separate until reviewed.</p></div></div>
          {!!report.analysis_errors?.length && <div className="error-banner"><AlertCircle size={18}/><div><strong>Some analysis results could not be loaded</strong>{report.analysis_errors.map((message, index) => <span key={index}>{message}</span>)}</div></div>}
          <div className="rs-report-counts">
            <div><strong>{confirmedCount}</strong><span>Reviewer-confirmed</span></div><div><strong>{pendingCount}</strong><span>Needs human review</span></div><div><strong>{rejectedCount}</strong><span>Rejected</span></div><div><strong>{categoryEvents.length}</strong><span>Categories covered</span></div>
          </div>
          <section className="rs-report-category-coverage" aria-label="Violation category analysis status">
            <h3>Violation category coverage</h3>
            <div className="rs-report-category-grid">{categoryEvents.map((category) => <article key={category.id}>
              <div><strong>{category.name}</strong><span className={`rs-report-category-status ${category.status}`}>{pretty(category.status)}</span></div>
              <b>{category.events.length} safety event{category.events.length === 1 ? "" : "s"}</b>
              <p>{category.message}</p>
            </article>)}</div>
          </section>
          {report.video?.model_name && <p className="rs-report-detail">Model: {report.video.model_name}{report.video.model_version ? ` · ${report.video.model_version}` : ""}</p>}
          <div className="rs-report-chronology"><h3>Safety event evidence by category</h3><p className="rs-report-chronology-note">Only category-specific safety candidates are shown here. Generic tracking and movement observations remain in the raw analysis data and are not presented as violations.</p>{categoryEvents.map(section)}</div>
          <section className="rs-report-limitations"><h3>What this report can and cannot tell you</h3><p>This report summarises saved model observations and reviewer decisions. It does not independently establish that a traffic violation occurred.</p><ul>{(report.limitations || []).map((item, index) => <li key={index}>{item}</li>)}</ul><p>Do not treat unreviewed candidates as confirmed violations. Review the original video and supporting evidence before drawing conclusions.</p></section>
        </>}
      </div>
    </section>
  </div>, document.body);
}
