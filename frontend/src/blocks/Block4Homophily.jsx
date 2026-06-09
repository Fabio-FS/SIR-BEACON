
import { useState } from "react";
import Slider from "../components/Slider";
import ModelSelect from "../components/ModelSelect";
import LinePlot from "../components/LinePlot";
import useDebouncedFetch from "../components/useDebouncedFetch";

export default function Block4Homophily() {
  const [model, setModel] = useState("SIRM");
  const [r0, setR0] = useState(2.5);
  const [pol, setPol] = useState(0.5);

  const data = useDebouncedFetch("/sweep_homophily", { model, r0, pol });

  const series = data
    ? [{ x: data.h, y: data.infected, color: "var(--accent-red)" }]
    : [];

  return (
    <>
      <h2 className="block-title">4 · Varying homophily</h2>
      <p className="block-text">
        Now sweep homophily — how strongly people cluster with others who behave like them — at fixed
        polarization. This is where the story flips: for mild diseases, clustering concentrates the
        unprotected into outbreak-prone pockets and makes things worse; for highly transmissible diseases,
        clustering can instead shield the protected and <i>reduce</i> total infections. Move the R₀ slider
        and watch the curve change direction.
      </p>

      <div className="container">
        <div className="panel" style={{ marginBottom: "1rem" }}>
          <div className="panel-title">Intervention</div>
          <ModelSelect value={model} onChange={setModel} />
          <div className="slider-grid" style={{ marginTop: "0.5rem" }}>
            <Slider label="R₀" value={r0} min={0.5} max={5} step={0.05} onChange={setR0} />
            <Slider label="polarization" value={pol} min={0.01} max={0.99} step={0.01} onChange={setPol} />
          </div>
        </div>

        <div className="plot-panel">
          <div className="plot-caption">Total infections vs homophily</div>
          {data && (
            <LinePlot
              series={series}
              xLabel="homophily"
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