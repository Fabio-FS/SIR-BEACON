from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import simulation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/structure")
def structure(pol: float, h: float, mean: float = 0.5):
    pop = simulation.population(pol, mean)
    C = simulation.matrix(pol, h, mean)
    return {"population": pop, "contact_matrix": C}


@app.get("/trajectory")
def trajectory(model: str, r0: float, pol: float, h: float, mean: float = 0.5):
    return simulation.run_trajectory(model, r0, pol, h, mean)


@app.get("/sweep_polarization")
def sweep_polarization(model: str, r0: float, h: float, mean: float = 0.5):
    return simulation.run_sweep_polarization(model, r0, h, mean)


@app.get("/sweep_homophily")
def sweep_homophily(model: str, r0: float, pol: float, mean: float = 0.5):
    return simulation.run_sweep_homophily(model, r0, pol, mean)


@app.get("/sweep_severity")
def sweep_severity(model: str, pol: float, h: float, mean: float = 0.5):
    return simulation.run_sweep_severity(model, pol, h, mean)