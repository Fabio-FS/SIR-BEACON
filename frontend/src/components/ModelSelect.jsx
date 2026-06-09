const MODELS = [
  { id: "SIRM", label: "Masks" },
  { id: "SIRT", label: "Testing" },
  { id: "SIRV", label: "Vaccination" },
];

export default function ModelSelect({ value, onChange }) {
  return (
    <div className="seg">
      {MODELS.map((m) => (
        <button
          key={m.id}
          className={value === m.id ? "active" : ""}
          onClick={() => onChange(m.id)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}