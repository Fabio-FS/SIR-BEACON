import "./styles.css";
import Block1Structure from "./blocks/Block1Structure";
import Block2Epidemic from "./blocks/Block2Epidemic";
import Block3Polarization from "./blocks/Block3Polarization";
import Block4Homophily from "./blocks/Block4Homophily";
import Block5Severity from "./blocks/Block5Severity";

export default function App() {
  return (
    <div className="page">
      <header>
        <h1>Beyond averages</h1>
        <p className="byline">
          How transmissibility and social structure shape pandemic dynamics — interactive demonstrator.
        </p>
        <p className="intro">
          Two populations can have the same <i>average</i> protective behavior yet experience very different
          epidemics. What matters is how that behavior is distributed (polarization) and whether people
          cluster with others who behave alike (homophily). Scroll through to build the intuition piece by
          piece.
        </p>
      </header>

      <Block1Structure />
      <Block2Epidemic />
      <Block3Polarization />
      <Block4Homophily />
      <Block5Severity />

      <div className="footer">
        Five behavioral groups. Distribution from a discretized Beta(a,b); contact matrix
        C<sub>ij</sub> ∝ e<sup>−h·|i−j|/N</sup>, normalized to preserve group sizes and total contact volume.
        Simulations run live via a JAX backend.<br />
        Companion to <i>Beyond Averages: How Transmissibility and Social Structure Interact to Shape Pandemic
        Dynamics</i>.
      </div>
    </div>
  );
}