export default function Slider({ label, value, min, max, step, decimals = 2, onChange }) {
  return (
    <>
      <label>{label}</label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <span className="slider-out">{value.toFixed(decimals)}</span>
    </>
  );
}