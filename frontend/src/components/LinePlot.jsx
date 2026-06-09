// A simple multi-line SVG plot.
// series: [{ x: [...], y: [...], color, label }]
// Draws axes, gridlines, optional zero reference line.
export default function LinePlot({
  series,
  xLabel,
  yLabel,
  width = 660,
  height = 360,
  yMin = null,
  yMax = null,
  zeroLine = false,
  xTickFmt = (v) => v.toFixed(0),
  yTickFmt = (v) => v.toFixed(2),
}) {
  const pad = { l: 56, r: 20, t: 18, b: 44 };
  const pw = width - pad.l - pad.r;
  const ph = height - pad.t - pad.b;

  const allX = series.flatMap((s) => s.x);
  const allY = series.flatMap((s) => s.y);
  const x0 = Math.min(...allX);
  const x1 = Math.max(...allX);
  const lo = yMin !== null ? yMin : Math.min(...allY, 0);
  const hi = yMax !== null ? yMax : Math.max(...allY) * 1.08;

  const tx = (x) => pad.l + ((x - x0) / (x1 - x0)) * pw;
  const ty = (y) => pad.t + (1 - (y - lo) / (hi - lo)) * ph;

  const xTicks = 6;
  const yTicks = 4;

  const path = (s) => {
    let d = "";
    for (let i = 0; i < s.x.length; i++) {
      d += (i === 0 ? "M" : "L") + tx(s.x[i]).toFixed(1) + "," + ty(s.y[i]).toFixed(1) + " ";
    }
    return d;
  };

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto", display: "block" }}>
      {/* gridlines + y ticks */}
      {Array.from({ length: yTicks + 1 }, (_, i) => {
        const yv = lo + (i / yTicks) * (hi - lo);
        const y = ty(yv);
        return (
          <g key={"y" + i}>
            <line x1={pad.l} y1={y} x2={pad.l + pw} y2={y} stroke="rgba(240,238,230,0.10)" strokeWidth="0.5" />
            <text x={pad.l - 8} y={y + 4} textAnchor="end" fontSize="11" fill="#8a8880">{yTickFmt(yv)}</text>
          </g>
        );
      })}
      {/* x ticks */}
      {Array.from({ length: xTicks + 1 }, (_, i) => {
        const xv = x0 + (i / xTicks) * (x1 - x0);
        const x = tx(xv);
        return (
          <g key={"x" + i}>
            <line x1={x} y1={pad.t + ph} x2={x} y2={pad.t + ph + 4} stroke="#b3b1a8" strokeWidth="1" />
            <text x={x} y={pad.t + ph + 18} textAnchor="middle" fontSize="11" fill="#b3b1a8">{xTickFmt(xv)}</text>
          </g>
        );
      })}
      {/* axes */}
      <line x1={pad.l} y1={pad.t + ph} x2={pad.l + pw} y2={pad.t + ph} stroke="#b3b1a8" strokeWidth="1" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + ph} stroke="#b3b1a8" strokeWidth="1" />
      {/* zero reference */}
      {zeroLine && lo < 0 && hi > 0 && (
        <line x1={pad.l} y1={ty(0)} x2={pad.l + pw} y2={ty(0)} stroke="#b3b1a8" strokeWidth="1" strokeDasharray="4 4" />
      )}
      {/* series */}
      {series.map((s, i) => (
        <path key={i} d={path(s)} fill="none" stroke={s.color} strokeWidth="2.5" />
      ))}
      {/* axis labels */}
      <text x={pad.l + pw / 2} y={height - 8} textAnchor="middle" fontSize="12" fill="#b3b1a8">{xLabel}</text>
      <text x="16" y={pad.t + ph / 2} textAnchor="middle" fontSize="12" fill="#b3b1a8"
            transform={`rotate(-90 16 ${pad.t + ph / 2})`}>{yLabel}</text>
    </svg>
  );
}