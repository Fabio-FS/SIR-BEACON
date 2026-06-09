
import { useState } from "react";
import Slider from "../components/Slider";
import useDebouncedFetch from "../components/useDebouncedFetch";

const N = 5;
const groupColors = ["#8E0152", "#DE77AE", "#999688", "#7FBC41", "#276419"];

function blueScale(t) {
  const r = Math.round(38 + (74 - 38) * t);
  const g = Math.round(38 + (158 - 38) * t);
  const b = Math.round(36 + (255 - 36) * t);
  return `rgb(${r},${g},${b})`;
}

function DistPlot({ pop }) {
  const W = 320, H = 240;
  const pad = { l: 34, r: 12, t: 12, b: 32 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const ymax = Math.max(0.5, Math.max(...pop) * 1.15);
  const bw = pw / N;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
      {[0, 1, 2].map((i) => {
        const yv = (i / 2) * ymax;
        const y = pad.t + (1 - yv / ymax) * ph;
        return (
          <g key={i}>
            <line x1={pad.l} y1={y} x2={pad.l + pw} y2={y} stroke="rgba(240,238,230,0.10)" strokeWidth="0.5" />
            <text x={pad.l - 6} y={y + 4} textAnchor="end" fontSize="10" fill="#8a8880">{yv.toFixed(2)}</text>
          </g>
        );
      })}
      {pop.map((p, i) => {
        const h = (p / ymax) * ph;
        const x = pad.l + i * bw + bw * 0.12;
        const y = pad.t + ph - h;
        return <rect key={i} x={x} y={y} width={bw * 0.76} height={h} rx="2" fill={groupColors[i]} />;
      })}
      <line x1={pad.l} y1={pad.t + ph} x2={pad.l + pw} y2={pad.t + ph} stroke="#b3b1a8" strokeWidth="1" />
      <text x={pad.l} y={H - 8} textAnchor="start" fontSize="10" fill="#8a8880">least protective</text>
      <text x={pad.l + pw} y={H - 8} textAnchor="end" fontSize="10" fill="#8a8880">most protective</text>
    </svg>
  );
}

function ContactPlot({ C }) {
  const W = 320, H = 240;
  const pad = { l: 34, r: 18, t: 12, b: 28 };
  const size = Math.min(W - pad.l - pad.r, H - pad.t - pad.b);
  const cell = size / N;
  const x0 = pad.l, y0 = pad.t;
  let cmax = 0;
  C.forEach((row) => row.forEach((v) => { if (v > cmax) cmax = v; }));
  const cells = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const r = N - 1 - i;
      const v = C[i][j] / cmax;
      const x = x0 + j * cell;
      const y = y0 + r * cell;
      cells.push(
        <g key={i + "-" + j}>
          <rect x={x} y={y} width={cell} height={cell} fill={blueScale(v)} stroke="#1c1c1a" strokeWidth="1" />
          <text x={x + cell / 2} y={y + cell / 2 + 3} textAnchor="middle" fontSize="9"
                fill={v > 0.55 ? "#f0eee6" : "#8a8880"}>{C[i][j].toFixed(1)}</text>
        </g>
      );
    }
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
      {cells}
      <text x={x0 + size / 2} y={H - 8} textAnchor="middle" fontSize="10" fill="#8a8880">contact's group</text>
      <text x="12" y={y0 + size / 2} textAnchor="middle" fontSize="10" fill="#8a8880"
            transform={`rotate(-90 12 ${y0 + size / 2})`}>own group</text>
    </svg>
  );
}

export default function Block1Structure() {
  const [mean, setMean] = useState(0.5);
  const [pol, setPol] = useState(0.4);
  const [hom, setHom] = useState(2.0);

  const data = useDebouncedFetch("/structure", { pol, h: hom, mean });

  return (
    <>
      <h2 className="block-title">1 · The shape of a population</h2>
      <p className="block-text">
        We split the population into five behavioral groups, from least to most protective.{" "}
        <b>Polarization</b> controls how spread out the behavior is: at zero everyone sits near the average;
        at one the population splits into two opposing camps. <b>Homophily</b> controls who contacts whom:
        at zero, mixing is uniform; higher values mean people mostly contact others who behave like them.
      </p>

      <div className="container">
        <div className="panel" style={{ marginBottom: "1rem" }}>
          <div className="panel-title">Population structure</div>
          <div className="slider-grid">
            <Slider label="mean" value={mean} min={0.05} max={0.95} step={0.01} onChange={setMean} />
            <Slider label="polarization" value={pol} min={0.01} max={0.99} step={0.01} onChange={setPol} />
            <Slider label="homophily" value={hom} min={0} max={6} step={0.05} onChange={setHom} />
          </div>
        </div>

        <div className="preview-grid">
          <div className="plot-panel">
            <div className="plot-caption">Distribution of protective behavior</div>
            {data && <DistPlot pop={data.population} />}
          </div>
          <div className="plot-panel">
            <div className="plot-caption">Contact matrix between groups</div>
            {data && <ContactPlot C={data.contact_matrix} />}
          </div>
        </div>
      </div>
    </>
  );
}