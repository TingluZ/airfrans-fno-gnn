---
title: AirfRANS GNN surrogate
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
short_description: What-if API for a GraphSAGE CFD surrogate on AirfRANS
---

# AirfRANS GNN surrogate API

Predicts velocity, kinematic pressure and turbulent viscosity around a 2D
airfoil with a 4-layer GraphSAGE network trained on AirfRANS. Pick one of the
test airfoils, change the angle of attack (-5 to 15 degrees) or inflow speed
(31 to 94 m/s), and get the predicted fields back.

- Interactive docs: `/docs`
- `GET /cases`, `POST /predict`, `GET /health`

This Space is deployed automatically from
[TingluZ/airfrans-fno-gnn](https://github.com/TingluZ/airfrans-fno-gnn) by
GitHub Actions; edit the code there, not here. The cases are a derivative of
the [AirfRANS dataset](https://github.com/Extrality/AirfRANS) (ODbL v1.0).
