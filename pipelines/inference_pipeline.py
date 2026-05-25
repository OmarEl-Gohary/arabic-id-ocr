"""Prefect batch inference pipeline — runs OCR on a directory of images."""

import json
import sys
from pathlib import Path
from typing import List, Optional

from prefect import flow, task
from prefect.logging import get_run_logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.detector import FieldDetector
from src.models.ocr_engine import create_ocr_engine
from src.models.pipeline import ArabicIDOCRPipeline
from src.monitoring.drift import DriftDetector


@task(name="load-pipeline")
def load_pipeline(model_path: str, device: str = "cpu") -> ArabicIDOCRPipeline:
    detector = FieldDetector(model_path=model_path, device=device)
    ocr = create_ocr_engine(engine="easyocr", languages=["ar", "en"], gpu=False)
    return ArabicIDOCRPipeline(detector=detector, ocr_engine=ocr)


@task(name="collect-images")
def collect_images(input_dir: str) -> List[str]:
    logger = get_run_logger()
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [str(p) for p in Path(input_dir).glob("**/*") if p.suffix.lower() in exts]
    logger.info(f"Found {len(paths)} images in {input_dir}")
    return paths


@task(name="run-ocr", retries=1)
def run_ocr(pipeline: ArabicIDOCRPipeline, image_paths: List[str]) -> List[dict]:
    logger = get_run_logger()
    results = []
    for i, path in enumerate(image_paths):
        try:
            result = pipeline.process_file(path)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to process {path}: {e}")
            results.append({"error": str(e), "source": path})
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(image_paths)}")
    return results


@task(name="save-results")
def save_results(results: List[dict], output_path: str):
    logger = get_run_logger()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(results)} results to {output_path}")


@task(name="check-drift")
def check_drift(results: List[dict], reference_path: Optional[str] = None) -> dict:
    logger = get_run_logger()
    valid_results = [r for r in results if "fields" in r]
    if len(valid_results) < 10:
        logger.info("Too few results for drift check")
        return {}

    detector = DriftDetector(reference_path)
    drift_report = detector.simple_stat_check(valid_results)

    if drift_report.get("drift_suspected"):
        logger.warning(f"Drift detected! Issues: {drift_report['issues']}")
    else:
        logger.info("No significant drift detected")
    return drift_report


@flow(name="arabic-id-ocr-batch-inference", log_prints=True)
def inference_pipeline(
    model_path: str,
    input_dir: str,
    output_path: str = "data/processed/batch_results.json",
    reference_path: Optional[str] = None,
    device: str = "cpu",
):
    pipeline = load_pipeline(model_path, device)
    image_paths = collect_images(input_dir)
    results = run_ocr(pipeline, image_paths)
    save_results(results, output_path)
    drift_report = check_drift(results, reference_path)
    print(f"Batch inference complete: {len(results)} images processed")
    return {"total": len(results), "output": output_path, "drift": drift_report}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="data/processed/batch_results.json")
    args = parser.parse_args()
    inference_pipeline(args.model, args.input_dir, args.output)
