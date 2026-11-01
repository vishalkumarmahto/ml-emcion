import os
import joblib
import pandas as pd
from pathlib import Path

from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from pulp import (
    LpProblem, LpMinimize, LpVariable,
    lpSum, LpStatus, PULP_CBC_CMD, value
)

app = FastAPI(title="EMICON ML Service")

# ----------------------------------
# LOAD TRAINED MODELS
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR/"ml-service"/"models"

try:
    energy_model = joblib.load(
        f"{MODEL_DIR}/energy_model.pkl"
    )

    anomaly_model = joblib.load(
        f"{MODEL_DIR}/anomaly_model.pkl"
    )

except FileNotFoundError:
    energy_model = None
    anomaly_model = None


# ----------------------------------
# PRODUCT-LINE MAPPING
# ----------------------------------

PRODUCT_LINES = {
    "Electric Drill": ["L1", "L2", "L3"],
    "Angle Grinder": ["L4", "L5", "L6"]
}


# ----------------------------------
# INPUT SCHEMAS
# ----------------------------------

class EnergyInput(BaseModel):
    line_id: str
    product_name: str
    production_rate: float
    availability: float
    avg_power_kw: float
    operating_minutes: float = 60


class AnomalyInput(BaseModel):
    line_id: str
    power_kw: float
    energy_delta_kwh: float
    production_rate: float
    availability: float
    scrap_rate: float


class LineInput(BaseModel):
    line_id: str
    product_name: str
    production_rate: float
    availability: float
    avg_power_kw: float


class OrderInput(BaseModel):
    order_id: str
    product_name: str
    quantity: int = Field(gt=0)
    deadline: str


class OptimizationInput(BaseModel):
    orders: List[OrderInput]
    lines: List[LineInput]
    horizon_hours: float = Field(default=24, gt=0)
    grid_factor: float = Field(default=0.716, ge=0)


# ----------------------------------
# HELPER: AVAILABILITY
# ----------------------------------

def normalize_availability(value):
    if value > 1:
        value = value / 100

    return max(0, min(value, 1))


# ----------------------------------
# HEALTH CHECK
# ----------------------------------

@app.get("/")
def home():
    return {
        "message": "EMICON ML Service Running",
        "energy_model_loaded": energy_model is not None,
        "anomaly_model_loaded": anomaly_model is not None
    }


# ----------------------------------
# ENERGY PREDICTION
# ----------------------------------

@app.post("/predict-energy")
def predict_energy(data: EnergyInput):

    if energy_model is None:
        raise HTTPException(
            status_code=503,
            detail="Energy model is not trained"
        )

    features = pd.DataFrame([{
        "line_id": data.line_id,
        "product_name": data.product_name,
        "production_rate": data.production_rate,
        "availability": normalize_availability(
            data.availability
        ),
        "avg_power_kw": data.avg_power_kw,
        "operating_minutes": data.operating_minutes
    }])

    prediction = float(energy_model.predict(features)[0])

    if prediction <= 0:
        raise HTTPException(
            status_code=422,
            detail="Invalid energy prediction"
        )

    return {
        "line_id": data.line_id,
        "predicted_energy_kwh_per_unit": round(
            prediction, 5
        )
    }


# ----------------------------------
# ANOMALY DETECTION
# ----------------------------------

@app.post("/detect-anomaly")
def detect_anomaly(data: AnomalyInput):

    if anomaly_model is None:
        raise HTTPException(
            status_code=503,
            detail="Anomaly model is not trained"
        )

    features = pd.DataFrame([{
        "power_kw": data.power_kw,
        "energy_delta_kwh": data.energy_delta_kwh,
        "production_rate": data.production_rate,
        "availability": normalize_availability(
            data.availability
        ),
        "scrap_rate": data.scrap_rate
    }])

    result = int(anomaly_model.predict(features)[0])

    is_anomaly = result == -1

    return {
        "line_id": data.line_id,
        "status": "ANOMALY" if is_anomaly else "NORMAL",
        "anomaly_detected": is_anomaly,
        "message": (
            "Unusual production or energy behaviour detected"
            if is_anomaly else
            "Production behaviour is normal"
        )
    }


# ----------------------------------
# PRODUCTION OPTIMIZATION
# ----------------------------------

@app.post("/optimize")
def optimize(data: OptimizationInput):

    if energy_model is None:
        raise HTTPException(
            status_code=503,
            detail="Energy model is not trained"
        )

    if not data.orders or not data.lines:
        raise HTTPException(
            status_code=400,
            detail="Orders and production lines are required"
        )

    now = datetime.now(timezone.utc)

    lines = []

    for line in data.lines:

        allowed = PRODUCT_LINES.get(
            line.product_name, []
        )

        if line.line_id not in allowed:
            continue

        if line.production_rate <= 0:
            continue

        availability = normalize_availability(
            line.availability
        )

        if availability <= 0:
            continue

        features = pd.DataFrame([{
            "line_id": line.line_id,
            "product_name": line.product_name,
            "production_rate": line.production_rate,
            "availability": availability,
            "avg_power_kw": line.avg_power_kw,
            "operating_minutes": 60
        }])

        predicted = float(
            energy_model.predict(features)[0]
        )

        if predicted <= 0:
            continue

        lines.append({
            "line_id": line.line_id,
            "product_name": line.product_name,
            "rate": line.production_rate,
            "availability": availability,
            "energy_per_unit": predicted
        })

    if not lines:
        raise HTTPException(
            status_code=422,
            detail="No eligible production lines available"
        )

    orders = []

    for order in data.orders:

        if order.product_name not in PRODUCT_LINES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported product: {order.product_name}"
            )

        try:
            deadline = datetime.fromisoformat(
                order.deadline.replace("Z", "+00:00")
            )

            if deadline.tzinfo is None:
                deadline = deadline.replace(
                    tzinfo=timezone.utc
                )

            due_hours = (
                deadline - now
            ).total_seconds() / 3600

        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid deadline format"
            )

        if due_hours <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"Order {order.order_id} has expired deadline"
            )

        orders.append({
            "order_id": order.order_id,
            "product_name": order.product_name,
            "quantity": order.quantity,
            "due_hours": min(
                due_hours, data.horizon_hours
            )
        })

    # ----------------------------------
    # CREATE OPTIMIZATION PROBLEM
    # ----------------------------------

    problem = LpProblem(
        "EMICON_Production_Optimization",
        LpMinimize
    )

    # Decision variables:
    # Quantity of each order assigned to each line

    x = {}

    for i, line in enumerate(lines):
        for j, order in enumerate(orders):

            if (
                line["product_name"]
                == order["product_name"]
            ):

                x[(i, j)] = LpVariable(
                    f"qty_{i}_{j}",
                    lowBound=0,
                    cat="Continuous"
                )

    # ----------------------------------
    # OBJECTIVE: MINIMUM ENERGY
    # ----------------------------------

    problem += lpSum(
        x[(i, j)] * lines[i]["energy_per_unit"]
        for (i, j) in x
    )

    # ----------------------------------
    # CONSTRAINT 1: FULFILL EACH ORDER
    # ----------------------------------

    for j, order in enumerate(orders):

        compatible = [
            x[(i, j)]
            for i in range(len(lines))
            if (i, j) in x
        ]

        if not compatible:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No available line for order "
                    f"{order['order_id']}"
                )
            )

        problem += (
            lpSum(compatible) == order["quantity"]
        )

    # ----------------------------------
    # CONSTRAINT 2: TOTAL LINE CAPACITY
    # ----------------------------------

    for i, line in enumerate(lines):

        assigned = [
            x[(i, j)]
            for j in range(len(orders))
            if (i, j) in x
        ]

        capacity = (
            line["rate"]
            * line["availability"]
            * data.horizon_hours
        )

        problem += (
            lpSum(assigned) <= capacity
        )

    # ----------------------------------
    # CONSTRAINT 3: ORDER DEADLINES
    # ----------------------------------

    deadlines = sorted(set(
        order["due_hours"] for order in orders
    ))

    for i, line in enumerate(lines):

        for deadline in deadlines:

            due_assignments = [
                x[(i, j)]
                for j, order in enumerate(orders)
                if (
                    (i, j) in x
                    and order["due_hours"] <= deadline
                )
            ]

            capacity_before_deadline = (
                line["rate"]
                * line["availability"]
                * deadline
            )

            problem += (
                lpSum(due_assignments)
                <= capacity_before_deadline
            )

    # ----------------------------------
    # SOLVE
    # ----------------------------------

    problem.solve(
        PULP_CBC_CMD(msg=False)
    )

    if LpStatus[problem.status] != "Optimal":
        raise HTTPException(
            status_code=422,
            detail=(
                "No feasible schedule found. "
                "Check production capacity and deadlines."
            )
        )

    # ----------------------------------
    # BUILD SCHEDULE
    # ----------------------------------

    schedule = []
    total_energy = 0
    line_summary = {}

    for (i, j), variable in x.items():

        quantity = float(value(variable) or 0)

        if quantity <= 0.001:
            continue

        line = lines[i]
        order = orders[j]

        energy = (
            quantity * line["energy_per_unit"]
        )

        total_energy += energy

        schedule.append({
            "order_id": order["order_id"],
            "product_name": order["product_name"],
            "line_id": line["line_id"],
            "allocated_quantity": round(quantity, 2),
            "predicted_energy_kwh": round(energy, 3)
        })

        line_id = line["line_id"]

        if line_id not in line_summary:
            line_summary[line_id] = {
                "line_id": line_id,
                "product_name": line["product_name"],
                "total_quantity": 0,
                "predicted_energy_kwh": 0
            }

        line_summary[line_id]["total_quantity"] += quantity

        line_summary[line_id]["predicted_energy_kwh"] += energy

    # ----------------------------------
    # BASELINE: EQUAL DISTRIBUTION
    # ----------------------------------

    baseline_energy = 0

    for order in orders:

        compatible_lines = [
            line for line in lines
            if line["product_name"] == order["product_name"]
        ]

        baseline_energy += (
            order["quantity"]
            * sum(
                line["energy_per_unit"]
                for line in compatible_lines
            )
            / len(compatible_lines)
        )

    energy_saved = max(
        0, baseline_energy - total_energy
    )

    carbon_saved = (
        energy_saved * data.grid_factor
    )

    carbon_total = (
        total_energy * data.grid_factor
    )

    saving_percentage = (
        energy_saved / baseline_energy * 100
        if baseline_energy > 0 else 0
    )

    return {
        "status": "OPTIMAL",
        "schedule": schedule,
        "line_summary": list(line_summary.values()),

        "total_production_quantity": sum(
            order["quantity"] for order in orders
        ),

        "predicted_energy_kwh": round(
            total_energy, 3
        ),

        "predicted_carbon_kg": round(
            carbon_total, 3
        ),

        "baseline_energy_kwh": round(
            baseline_energy, 3
        ),

        "energy_saved_kwh": round(
            energy_saved, 3
        ),

        "carbon_saved_kg": round(
            carbon_saved, 3
        ),

        "energy_saving_percentage": round(
            saving_percentage, 2
        ),

        "grid_emission_factor": data.grid_factor
    }