#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# evaluate.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
"""
평가 스크립트 - 학습된 Pairwise Ranking Model 평가.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --test_csv checkpoints/test_metadata.csv
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import structlog
import torch
from torch.utils.data import DataLoader

from app.ml.ranking_model import PairwiseRankingModel
from training.evaluator import Evaluator
from training.datasets.pairwise_dataset import PairwiseDataset
from training.benchmarks import set_seed, PerformanceBenchmarks

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pairwise Ranking Model 평가", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="체크포인트 파일 경로")
    parser.add_argument("--test_csv", type=str, required=True, help="테스트 메타데이터 CSV 경로")
    parser.add_argument("--image_dir", type=str, default=None, help="이미지 루트 디렉토리")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 워커 수")
    parser.add_argument("--output", type=str, default=None, help="결과 JSON 저장 경로")
    parser.add_argument("--benchmark", action="store_true", help="성능 벤치마크 실행")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> PairwiseRankingModel:
    logger.info("체크포인트 로드 중", path=checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = PairwiseRankingModel(backbone_name="dinov2_vitl14", freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get("epoch", "unknown")
    best_val_loss = checkpoint.get("best_val_loss", "unknown")
    logger.info("모델 로드 완료", epoch=epoch, best_val_loss=best_val_loss)
    
    return model


def load_test_data(test_csv: str, image_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    import pandas as pd
    
    csv_path = Path(test_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"테스트 CSV 없음: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info("테스트 데이터 로드", samples=len(df))
    
    dataset = PairwiseDataset(metadata_df=df, image_dir=image_dir)
    
    def collate(batch):
        i1, i2, l = zip(*batch)
        return torch.stack(i1), torch.stack(i2), torch.tensor(l, dtype=torch.long)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    return loader


def evaluate_model(model, test_loader, device: str) -> Dict:
    evaluator = Evaluator(model=model, device=device)
    results = evaluator.evaluate_detailed(test_loader)
    return results


def run_benchmarks(model, device: str) -> Dict:
    benchmarks = PerformanceBenchmarks(model=model, device=device)
    report = benchmarks.run_full_benchmark()
    return report


def print_results(results: Dict, benchmark_results: Dict = None):
    sep = "=" * 60
    print(f"\n{sep}")
    print("평가 결과")
    print(sep)
    acc = results["accuracy"]
    print(f"Pairwise Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"전체 페어 수: {results['total_pairs']:,}")
    print(f"정확한 예측 수: {results['correct_predictions']:,}")
    
    target_acc = 0.60
    if acc >= target_acc:
        print(f"\n[PASS] 목표 정확도 {target_acc*100:.0f}% 달성!")
    else:
        print(f"\n[FAIL] 목표 정확도 {target_acc*100:.0f}% 미달 (현재: {acc*100:.2f}%)")
    
    if benchmark_results:
        print("-" * 60)
        print("성능 벤치마크")
        print("-" * 60)
        print(f"추론 시간: {benchmark_results['inference_time_ms']:.2f}ms/pair")
        print(f"메모리 사용량: {benchmark_results['memory_usage_mb']:.2f}MB")
        inf_req = "PASS" if benchmark_results["meets_inference_requirement"] else "FAIL"
        mem_req = "PASS" if benchmark_results["meets_memory_requirement"] else "FAIL"
        print(f"추론 시간 요구사항: {inf_req}")
        print(f"메모리 요구사항: {mem_req}")
    
    print(f"{sep}\n")


def main():
    args = parse_args()
    try:
        set_seed(args.seed)
        
        model = load_model(args.checkpoint, args.device)
        test_loader = load_test_data(args.test_csv, args.image_dir, args.batch_size, args.num_workers)
        
        logger.info("평가 시작")
        results = evaluate_model(model, test_loader, args.device)
        
        benchmark_results = None
        if args.benchmark:
            logger.info("벤치마크 실행")
            benchmark_results = run_benchmarks(model, args.device)
        
        print_results(results, benchmark_results)
        
        if args.output:
            output_data = {"evaluation": results}
            if benchmark_results:
                output_data["benchmark"] = benchmark_results
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info("결과 저장 완료", path=args.output)
        
        return 0 if results["accuracy"] >= 0.60 else 1
        
    except Exception as e:
        logger.error("평가 중 오류", error=str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
