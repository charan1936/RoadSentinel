import { useState } from "react";
import { ArrowRight, RotateCcw, WandSparkles } from "lucide-react";

type Point = { x: number; y: number };
type Props = {
  src: string;
  loading: boolean;
  onSubmit: (calibration: { roi_polygon: Point[]; direction_start: Point; direction_end: Point }) => void;
};

export default function RoadDirectionCalibrator({ src, loading, onSubmit }: Props) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [polygon, setPolygon] = useState<Point[]>([]);
  const [arrow, setArrow] = useState<Point[]>([]);
  const phase = polygon.length < 3 ? "road" : arrow.length < 2 ? "direction" : "ready";

  const addPoint = (event: React.MouseEvent<HTMLDivElement>) => {
    if (phase === "ready" || !size.width) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const point = { x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)) };
    if (phase === "road") setPolygon((points) => [...points, point]);
    else setArrow((points) => [...points, point].slice(-2));
  };

  return <div className="direction-calibrator">
    <div className="direction-calibrator-heading">
      <div><strong>Calibrate road direction</strong><span>{phase === "road" ? `Tap around the road area (${polygon.length}/20)` : phase === "direction" ? `Tap the expected travel direction (${arrow.length}/2)` : "Ready to find opposite movement candidates"}</span></div>
      <button className="button button-secondary small" type="button" onClick={() => { setPolygon([]); setArrow([]); }}><RotateCcw size={14} /> Reset</button>
    </div>
    <div className="direction-calibrator-frame" style={{ aspectRatio: size.width && size.height ? `${size.width}/${size.height}` : "16 / 9" }} onClick={addPoint}>
      <img src={src} alt="Frame for road area and direction calibration" onLoad={(event) => setSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />
      {!!size.width && <svg viewBox={`0 0 ${size.width} ${size.height}`} preserveAspectRatio="none" aria-hidden="true">
        {polygon.length > 1 && <polygon points={polygon.map((point) => `${point.x * size.width},${point.y * size.height}`).join(" ")} />}
        {polygon.map((point, index) => <circle key={`p${index}`} cx={point.x * size.width} cy={point.y * size.height} r={Math.max(5, size.width / 140)} />)}
        {arrow.length > 0 && <circle className="direction-point" cx={arrow[0].x * size.width} cy={arrow[0].y * size.height} r={Math.max(6, size.width / 120)} />}
        {arrow.length > 1 && <line className="direction-arrow" x1={arrow[0].x * size.width} y1={arrow[0].y * size.height} x2={arrow[1].x * size.width} y2={arrow[1].y * size.height} markerEnd="url(#direction-head)" />}
        <defs><marker id="direction-head" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" /></marker></defs>
      </svg>}
    </div>
    <div className="direction-calibrator-footer"><span><ArrowRight size={14} /> {phase === "road" ? "Outline the road" : phase === "direction" ? "Point with traffic flow" : "Candidates stay marked for human review"}</span>
      <button className="button button-primary small" disabled={phase !== "ready" || loading} onClick={() => onSubmit({ roi_polygon: polygon, direction_start: arrow[0], direction_end: arrow[1] })}><WandSparkles size={14} />{loading ? "Checking tracks…" : "Find candidates"}</button>
    </div>
  </div>;
}
