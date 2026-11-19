import { useState } from "react";
import { RotateCcw, WandSparkles } from "lucide-react";

type Point = { x: number; y: number };

type Props = {
  src: string;
  loading: boolean;
  onSubmit: (calibration: {
    roi_polygon: Point[];
    min_stationary_seconds: number;
  }) => void;
};

export default function ParkingZoneCalibrator({
  src,
  loading,
  onSubmit,
}: Props) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [polygon, setPolygon] = useState<Point[]>([]);
  const [minDuration, setMinDuration] = useState(10);

  const ready = polygon.length >= 3 && size.width > 0;

  const addPoint = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!size.width || polygon.length >= 20) return;

    const rect = event.currentTarget.getBoundingClientRect();

    const point = {
      x: Math.max(
        0,
        Math.min(1, (event.clientX - rect.left) / rect.width)
      ),
      y: Math.max(
        0,
        Math.min(1, (event.clientY - rect.top) / rect.height)
      ),
    };

    setPolygon((points) => [...points, point]);
  };

  return (
    <div className="direction-calibrator">
      <div className="direction-calibrator-heading">
        <div>
          <strong>Define no-parking zone</strong>
          <span>
            {polygon.length < 3
              ? `Click around the restricted area (${polygon.length}/20 points)`
              : "Zone ready — review the outlined area"}
          </span>
        </div>

        <button
          className="button button-secondary small"
          type="button"
          onClick={() => setPolygon([])}
        >
          <RotateCcw size={14} /> Reset
        </button>
      </div>

      <div
        className="direction-calibrator-frame"
        style={{
          aspectRatio:
            size.width && size.height
              ? `${size.width}/${size.height}`
              : "16 / 9",
        }}
        onClick={addPoint}
      >
        <img
          src={src}
          alt="Video frame for parking-zone calibration"
          onLoad={(event) =>
            setSize({
              width: event.currentTarget.naturalWidth,
              height: event.currentTarget.naturalHeight,
            })
          }
        />

        {!!size.width && (
          <svg
            viewBox={`0 0 ${size.width} ${size.height}`}
            preserveAspectRatio="none"
            aria-hidden="true"
          >
            {polygon.length >= 3 && (
              <polygon
                points={polygon
                  .map(
                    (point) =>
                      `${point.x * size.width},${point.y * size.height}`
                  )
                  .join(" ")}
              />
            )}

            {polygon.map((point, index) => (
              <circle
                key={index}
                cx={point.x * size.width}
                cy={point.y * size.height}
                r={Math.max(5, size.width / 140)}
              />
            ))}
          </svg>
        )}
      </div>

      <div className="direction-calibrator-footer">
        <label>
          Minimum stationary duration:
          <select
            value={minDuration}
            onChange={(event) => setMinDuration(Number(event.target.value))}
          >
            <option value={5}>5 seconds</option>
            <option value={10}>10 seconds</option>
            <option value={15}>15 seconds</option>
            <option value={30}>30 seconds</option>
            <option value={60}>60 seconds</option>
          </select>
        </label>

        <button
          className="button button-primary small"
          disabled={!ready || loading}
          onClick={() =>
            onSubmit({
              roi_polygon: polygon,
              min_stationary_seconds: minDuration,
            })
          }
        >
          <WandSparkles size={14} />
          {loading ? "Analyzing..." : "Find parking candidates"}
        </button>
      </div>

      <p className="muted">
        Candidates require human review. A stationary vehicle inside this
        polygon is not, by itself, proof of illegal parking.
      </p>
    </div>
  );
}