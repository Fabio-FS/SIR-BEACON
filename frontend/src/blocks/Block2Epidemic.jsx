
import { useState } from "react";
import Slider from "../components/Slider";
import ModelSelect from "../components/ModelSelect";
import LinePlot from "../components/LinePlot";
import useDebouncedFetch from "../components/useDebouncedFetch";

const COLORS = { S: "#4a9eff", I: "#e88a5f", R: "#7FBC41", V: "#b18cff" };

export default function Block2Epidemic() {
  const [model, setModel] = useState("SIRM");
  const [r0, setR0] = useState(2.5);
  const [pol, setPol] = useState(0.4);
  const [hom, setHom] = useState(2.0);

  const data = useDebouncedFetch("/trajectory", { model, r0, pol, h: hom });

  let series = [];
  if (data) {
    const t = data.S.map((_, i) => i);
    series = [
      { x: t, y: data.S, color: COLORS.S, label: "Susceptible" },
      { x: t, y: data.I, color: COLORS.I, label: "Infected" },
      { x: t, y: data.R, color: COLORS.R, label: "Recovered" },
    ];
    if (data.V) series.push({ x: t, y: data.V, color: COLORS.V, label: "Vaccinated" });
  }

  return (
    <>
      <h2 className="block-title">2 · A single epidemic</h2>
      <p className="block-text">
        Now run an outbreak in that population. Pick an intervention, set how transmissible the disease is
        (R₀), and choose the social structure. The curves show what fraction of people are susceptible,
        infected, and recovered over time.
      </p>

      <div className="container">
        <div className="panel" style={{ marginBottom: "1rem" }}>
          <div className="panel-title">Intervention</div>
          <ModelSelect value={model} onChange={setModel} />
          <div className="slider-grid" style={{ marginTop: "0.5rem" }}>
            <Slider label="R₀" value={r0} min={0.5} max={5} step={0.05} onChange={setR0} />
            <Slider label="polarization" value={pol} min={0.01} max={0.99} step={0.01} onChange={setPol} />
            <Slider label="homophily" value={hom} min={0} max={6} step={0.05} onChange={setHom} />
          </div>
        </div>

        <div className="plot-panel">
          <div className="plot-caption">Epidemic over time</div>
          {data && (
            <>
              <LinePlot series={series} xLabel="time (days)" yLabel="fraction of population" yMin={0} yMax={1} />
              <div className="legend">
                {series.map((s) => (
                  <span className="legend-item" key={s.label}>
                    <span className="legend-swatch" style={{ background: s.color }} />
                    {s.label}
                  </span>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}