"""Log the latest combined_response.json to MLflow as an artifact."""
import json
import mlflow
from pathlib import Path

mlflow.set_tracking_uri("sqlite:///mldata/mlruns.db")
mlflow.set_experiment("arabic-ocr-api")

json_path = Path("combined_response.json")
data  = json.loads(json_path.read_text(encoding="utf-8"))
props = {p["name"]: p["value"] for p in data["properties"]}

with mlflow.start_run(run_name="combined_with_date_fix"):
    mlflow.set_tag("source",      "combined")
    mlflow.set_tag("front_image", "front.jpeg")
    mlflow.set_tag("back_image",  "back.jpeg")
    mlflow.set_tag("date_format", "YYYY/MM/DD")

    for name, value in props.items():
        mlflow.log_param(name, value if value is not None else "")

    filled = sum(1 for v in props.values() if v)
    total  = len(props)
    mlflow.log_metric("fields_filled", filled)
    mlflow.log_metric("fields_total",  total)
    mlflow.log_metric("fill_rate",     round(filled / total, 3))

    mlflow.log_artifact(str(json_path), artifact_path="output")

    print(f"Run logged  : {mlflow.active_run().info.run_id}")
    print(f"Fields      : {filled}/{total}  fill_rate={round(filled/total,3)}")
    print(f"Expiry_Date : {props.get('Expiry_Date')}")
    print(f"View at     : http://localhost:5001")
