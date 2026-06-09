
import { useState } from "react";
import Slider from "../components/Slider";
import ModelSelect from "../components/ModelSelect";
import LinePlot from "../components/LinePlot";
import useDebouncedFetch from "../components/useDebouncedFetch";

export default function Block3Polarization() {
  const [model, setModel] = useState("SIRM");
  const [r0, setR0] = useState(2.5);
  const [hom, setHom] = useState(2.0);

  const data = useDebouncedFetch("/sweep_polarization", { model, r0, h: hom });

  const series = data
    ? [{ x: data.pol, y: data.infected, color: "var(--accent-red)" }]
    : [];

  return (
    <>
      <h2 className="block-title">3 · Varying polarization</h2>
      <p className="block-text">
        Hold the average behavior fixed and sweep polarization from consensus to a fully split population.
        The total infected fraction (everyone who was ever infected) often <i>rises</i> as the population
        polarizes — but whether it does, and how steeply, depends on the intervention and how transmissible
        the disease is. Try changing R₀.
      </p>

      <div className="container">
        <div className="panel" style={{ marginBottom: "1rem" }}>
          <div className="panel-title">Intervention</div>
          <ModelSelect value={model} onChange={setModel} />
          <div className="slider-grid" style={{ marginTop: "0.5rem" }}>
            <Slider label="R₀" value={r0} min={0.5} max={5} step={0.05} onChange={setR0} />
            <Slider label="homophily" value={hom} min={0} max={6} step={0.05} onChange={setHom} />
          </div>
        </div>

        <div className="plot-panel">
          <div className="plot-caption">Total infections vs polarization</div>
          {data && (
            <LinePlot
              series={series}
              xLabel="polarization"
              yLabel="total infections"
              yMin={0}
              yMax={1}
              xTickFmt={(v) => v.toFixed(1)}
            />
          )}
        </div>
      </div>
    </>
  );
}