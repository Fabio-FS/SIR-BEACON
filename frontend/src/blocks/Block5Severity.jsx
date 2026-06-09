
import { useState } from "react";
import Slider from "../components/Slider";
import ModelSelect from "../components/ModelSelect";
import LinePlot from "../components/LinePlot";
import useDebouncedFetch from "../components/useDebouncedFetch";

export default function Block5Severity() {
  const [model, setModel] = useState("SIRT");
  const [pol, setPol] = useState(0.5);
  const [hom, setHom] = useState(2.0);

  const data = useDebouncedFetch("/sweep_severity", { model, pol, h: hom });

  let series = [];
  if (data) {
    const diff = data.structured.map((s, i) => (s - data.baseline[i]) * 100);
    series = [{ x: data.r0, y: diff, color: "var(--accent-red)" }];
  }

  return (
    <>
      <h2 className="block-title">5 · Why it matters for forecasting</h2>
      <p className="block-text">
        A standard model that knows only the <i>average</i> behavior misses all this structure. Here we plot
        the gap: how many more (or fewer) people get infected in the structured population than the
        average-only model predicts, as a percentage of the population, across disease severities. Positive
        means the simple model <b>underestimates</b> the outbreak. The sign and size of this error shift with
        R₀ — which is exactly why forecasts need disease-specific calibration.
      </p>

      <div className="container">
        <div className="panel" style={{ marginBottom: "1rem" }}>
          <div className="panel-title">Intervention</div>
          <ModelSelect value={model} onChange={setModel} />
          <div className="slider-grid" style={{ marginTop: "0.5rem" }}>
            <Slider label="polarization" value={pol} min={0.01} max={0.99} step={0.01} onChange={setPol} />
            <Slider label="homophily" value={hom} min={0} max={6} step={0.05} onChange={setHom} />
          </div>
        </div>

        <div className="plot-panel">
          <div className="plot-caption">Extra infections vs disease severity</div>
          {data && (
            <LinePlot
              series={series}
              xLabel="R₀"
              yLabel="extra infections (% of population)"
              zeroLine={true}
              xTickFmt={(v) => v.toFixed(1)}
              yTickFmt={(v) => v.toFixed(0)}
            />
          )}
        </div>
      </div>
    </>
  );
}