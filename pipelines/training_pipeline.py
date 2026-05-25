"""Prefect training pipeline — orchestrates data validation → training → evaluation."""

import sys
from pathlib import Path

from prefect import flow, task
from prefect.logging import get_run_logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.validate import DatasetValidator
from src.training.evaluate import evaluate
from src.training.train import train


@task(name="validate-dataset", retries=1)
def validate_dataset(dataset_root: str) -> bool:
    logger = get_run_logger()
    validator = DatasetValidator(dataset_root)
    is_valid = validator.run()
    report = validator.report()

    logger.info(f"Dataset validation: {'PASSED' if is_valid else 'FAILED'}")
    for issue in report["issues"]:
        logger.warning(f"  Issue: {issue}")

    stats = report.get("stats", {}).get("split_counts", {})
    for split, counts in stats.items():
        logger.info(f"  {split}: {counts['images']} images, {counts['labels']} labels")

    if not is_valid:
        raise ValueError(f"Dataset validation failed: {report['issues']}")
    return True


@task(name="train-detector", retries=0)
def train_detector(config_path: str) -> str:
    logger = get_run_logger()
    logger.info(f"Starting training with config: {config_path}")
    run_id = train(config_path)
    logger.info(f"Training complete. MLflow run_id: {run_id}")
    return run_id


@task(name="evaluate-model", retries=0)
def evaluate_model(run_id: str, model_path: str, data_yaml: str) -> dict:
    logger = get_run_logger()
    logger.info(f"Evaluating model: {model_path}")
    results = evaluate(
        model_path=model_path,
        data_yaml=data_yaml,
        split="test",
        run_id=run_id,
    )
    logger.info(f"mAP50: {results['mAP50']:.4f}")
    logger.info(f"mAP50-95: {results['mAP50_95']:.4f}")
    return results


@task(name="check-quality-gate")
def quality_gate(metrics: dict, min_map50: float = 0.7) -> bool:
    logger = get_run_logger()
    passed = metrics["mAP50"] >= min_map50
    logger.info(f"Quality gate (mAP50 >= {min_map50}): {'PASSED' if passed else 'FAILED'}")
    if not passed:
        raise ValueError(
            f"Model did not meet quality threshold: mAP50={metrics['mAP50']:.4f} < {min_map50}"
        )
    return True


@flow(name="arabic-id-ocr-training", log_prints=True)
def training_pipeline(
    dataset_root: str = "Egyptain-Person-ID-1",
    config_path: str = "configs/training.yaml",
    min_map50: float = 0.7,
):
    validate_dataset(dataset_root)
    run_id = train_detector(config_path)

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_path = f"{cfg['output']['project']}/{cfg['output']['name']}/weights/best.pt"
    data_yaml = cfg["data"]["dataset_path"]

    metrics = evaluate_model(run_id, model_path, data_yaml)
    quality_gate(metrics, min_map50)

    print(f"Pipeline complete. Model ready at: {model_path}")
    return {"run_id": run_id, "model_path": model_path, "metrics": metrics}


if __name__ == "__main__":
    training_pipeline()
