"""Proxy evaluation harness for Polaris product search MCP.
Computes metrics ONLY from catalog data without network calls.
"""
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from search import Catalog


def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_query(catalog, case, product_map):
    args = case["args"]
    expected = case.get("expected", {})

    # Run 1: Timed execution
    t0 = time.perf_counter()
    res1 = catalog.search(args)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Run 2: Determinism check
    res2 = catalog.search(args)
    is_deterministic = (res1 == res2)

    results = res1.get("results", [])
    recommendations = res1.get("recommendations", [])
    result_count = len(results)
    rec_count = len(recommendations)

    # Constraint Satisfaction checks (computed strictly from catalog data)
    checks_total = 0
    checks_passed = 0

    # 1. Must be empty check (for gibberish/impossible queries)
    if expected.get("must_be_empty"):
        checks_total += 1
        if result_count == 0 and rec_count == 0:
            checks_passed += 1

    # 2. Non-empty check (for valid shopping queries)
    if expected.get("non_empty"):
        checks_total += 1
        if result_count > 0:
            checks_passed += 1

    # 3. Price constraint check
    max_price = expected.get("max_price")
    min_price = expected.get("min_price")
    if (max_price is not None or min_price is not None) and result_count > 0:
        checks_total += 1
        price_ok = True
        for r in results:
            p = product_map.get(r["id"])
            if p:
                amount = p["price"]["amount"]
                if max_price is not None and amount > max_price:
                    price_ok = False
                    break
                if min_price is not None and amount < min_price:
                    price_ok = False
                    break
        if price_ok:
            checks_passed += 1

    # 4. Color constraint check
    target_color = expected.get("color")
    if target_color and result_count > 0:
        checks_total += 1
        # Check if majority / top results reflect requested color
        color_matches = 0
        for r in results:
            p = product_map.get(r["id"])
            if p and target_color.lower() in str(p.get("color") or "").lower():
                color_matches += 1
        if color_matches >= min(1, result_count):
            checks_passed += 1

    # 5. Category / Product type constraint check
    target_type = expected.get("product_type")
    target_cat = expected.get("category")
    if (target_type or target_cat) and result_count > 0:
        checks_total += 1
        type_matches = 0
        for r in results:
            p = product_map.get(r["id"])
            if p:
                p_type = str(p.get("product_type") or "").lower()
                p_cat = str(p.get("category") or "").lower()
                p_name = str(p.get("name") or "").lower()
                if target_type and (target_type.lower() in p_type or target_type.lower() in p_name):
                    type_matches += 1
                elif target_cat and target_cat.lower() in p_cat:
                    type_matches += 1
        if type_matches >= min(1, result_count):
            checks_passed += 1

    # 6. Audience constraint check
    target_aud = expected.get("audience")
    if target_aud and result_count > 0:
        checks_total += 1
        aud_matches = 0
        for r in results:
            p = product_map.get(r["id"])
            if p:
                p_aud = str(p.get("audience") or "").lower()
                if target_aud.lower() in p_aud or ("ladies" in target_aud.lower() and ("ladies" in p_aud or "divided" in p_aud)):
                    aud_matches += 1
        if aud_matches >= min(1, result_count):
            checks_passed += 1

    # 7. Recommendations check
    if expected.get("recommendations_min"):
        checks_total += 1
        prim_ids = {r["id"] for r in results}
        valid_recs = 0
        for rec in recommendations:
            if rec["id"] not in prim_ids and rec["for_product_id"] in prim_ids:
                valid_recs += 1
        if valid_recs >= expected["recommendations_min"]:
            checks_passed += 1

    # Diversity metrics across top 10 results
    unique_categories = len(set(product_map[r["id"]].get("category", "") for r in results if r["id"] in product_map))
    unique_types = len(set(product_map[r["id"]].get("product_type", "") for r in results if r["id"] in product_map))
    unique_sections = len(set(product_map[r["id"]].get("section", "") for r in results if r["id"] in product_map))

    constraint_score = (checks_passed / checks_total) if checks_total > 0 else 1.0

    return {
        "id": case["id"],
        "category": case["category"],
        "query": args.get("query", ""),
        "latency_ms": latency_ms,
        "is_deterministic": is_deterministic,
        "result_count": result_count,
        "recommendation_count": rec_count,
        "is_empty": (result_count == 0),
        "checks_total": checks_total,
        "checks_passed": checks_passed,
        "constraint_score": constraint_score,
        "diversity": {
            "unique_categories": unique_categories,
            "unique_types": unique_types,
            "unique_sections": unique_sections,
        },
        "results": results[:5],
        "recommendations": recommendations[:5],
    }


def run_evaluation(queries_path, catalog_path, output_path=None, compare_path=None):
    print("=" * 60)
    print("POLARIS MCP SEARCH EVALUATION (PROXY METRICS)")
    print("=" * 60)
    print(f"Catalog Path: {catalog_path}")
    print(f"Queries Path: {queries_path}")

    t_load = time.perf_counter()
    catalog = Catalog.load(catalog_path)
    print(f"Catalog loaded in {time.perf_counter() - t_load:.2f}s ({len(catalog.products)} products)")

    product_map = {p["id"]: p for p in catalog.products}
    queries = load_queries(queries_path)
    print(f"Evaluating {len(queries)} queries...")

    eval_results = []
    for q in queries:
        res = evaluate_query(catalog, q, product_map)
        eval_results.append(res)

    # Compute aggregate proxy metrics
    latencies = sorted([r["latency_ms"] for r in eval_results])
    p50_ms = latencies[int(len(latencies) * 0.50)]
    p95_ms = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
    mean_latency_ms = sum(latencies) / len(latencies)

    # Determinism
    det_count = sum(1 for r in eval_results if r["is_deterministic"])
    det_rate = (det_count / len(eval_results)) * 100.0

    # Constraint satisfaction
    total_checks = sum(r["checks_total"] for r in eval_results)
    passed_checks = sum(r["checks_passed"] for r in eval_results)
    constraint_sat_rate = (passed_checks / total_checks * 100.0) if total_checks > 0 else 100.0

    # Empty result rates
    gibberish_cases = [r for r in eval_results if r["category"] == "gibberish"]
    gibberish_empty_count = sum(1 for r in gibberish_cases if r["is_empty"])
    gibberish_empty_rate = (gibberish_empty_count / len(gibberish_cases) * 100.0) if gibberish_cases else 0.0

    valid_cases = [r for r in eval_results if r["category"] != "gibberish"]
    valid_empty_count = sum(1 for r in valid_cases if r["is_empty"])
    valid_empty_rate = (valid_empty_count / len(valid_cases) * 100.0) if valid_cases else 0.0

    overall_empty_rate = (sum(1 for r in eval_results if r["is_empty"]) / len(eval_results)) * 100.0

    # Diversity in top 10 (mean across queries with results)
    non_empty_results = [r for r in eval_results if not r["is_empty"]]
    mean_cat_diversity = (sum(r["diversity"]["unique_categories"] for r in non_empty_results) / len(non_empty_results)) if non_empty_results else 0.0
    mean_type_diversity = (sum(r["diversity"]["unique_types"] for r in non_empty_results) / len(non_empty_results)) if non_empty_results else 0.0
    mean_sec_diversity = (sum(r["diversity"]["unique_sections"] for r in non_empty_results) / len(non_empty_results)) if non_empty_results else 0.0

    # Per-category summary
    cat_summary = {}
    for cat in set(r["category"] for r in eval_results):
        cat_items = [r for r in eval_results if r["category"] == cat]
        cat_checks = sum(r["checks_total"] for r in cat_items)
        cat_passed = sum(r["checks_passed"] for r in cat_items)
        cat_summary[cat] = {
            "count": len(cat_items),
            "constraint_sat": (cat_passed / cat_checks * 100.0) if cat_checks > 0 else 100.0,
            "empty_count": sum(1 for r in cat_items if r["is_empty"]),
            "mean_latency_ms": sum(r["latency_ms"] for r in cat_items) / len(cat_items),
        }

    summary_metrics = {
        "total_queries": len(eval_results),
        "constraint_satisfaction_rate": round(constraint_sat_rate, 2),
        "determinism_rate": round(det_rate, 2),
        "gibberish_empty_rate": round(gibberish_empty_rate, 2),
        "valid_queries_empty_rate": round(valid_empty_rate, 2),
        "overall_empty_rate": round(overall_empty_rate, 2),
        "diversity_in_top10": {
            "mean_unique_categories": round(mean_cat_diversity, 2),
            "mean_unique_product_types": round(mean_type_diversity, 2),
            "mean_unique_sections": round(mean_sec_diversity, 2),
        },
        "latency_ms": {
            "mean": round(mean_latency_ms, 2),
            "p50": round(p50_ms, 2),
            "p95": round(p95_ms, 2),
        },
        "category_breakdown": cat_summary,
    }

    print("\n" + "=" * 60)
    print("PROXY EVALUATION SUMMARY (Strictly Catalog Data)")
    print("=" * 60)
    print(f"Total Evaluated Queries:       {summary_metrics['total_queries']}")
    print(f"Determinism Rate:              {summary_metrics['determinism_rate']}% (100% required)")
    print(f"Constraint Satisfaction:       {summary_metrics['constraint_satisfaction_rate']}%")
    print(f"Gibberish Empty Rate:          {summary_metrics['gibberish_empty_rate']}% (higher is better for abstention)")
    print(f"Valid Queries Empty Rate:      {summary_metrics['valid_queries_empty_rate']}% (lower is better for recall)")
    print(f"Diversity - Unique Types/10:   {summary_metrics['diversity_in_top10']['mean_unique_product_types']}")
    print(f"Diversity - Unique Cats/10:    {summary_metrics['diversity_in_top10']['mean_unique_categories']}")
    print(f"Latency p50:                   {summary_metrics['latency_ms']['p50']} ms")
    print(f"Latency p95:                   {summary_metrics['latency_ms']['p95']} ms")
    print("=" * 60)

    # Optional Comparison with Baseline
    if compare_path and Path(compare_path).exists():
        print("\n" + "-" * 60)
        print(f"COMPARISON DELTAS (vs {compare_path})")
        print("-" * 60)
        with open(compare_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
            base_metrics = base_data.get("summary", {})
            for key in ["constraint_satisfaction_rate", "determinism_rate", "gibberish_empty_rate", "valid_queries_empty_rate"]:
                if key in base_metrics:
                    delta = summary_metrics[key] - base_metrics[key]
                    sign = "+" if delta >= 0 else ""
                    print(f"  {key:30}: {base_metrics[key]:6.2f}% -> {summary_metrics[key]:6.2f}% ({sign}{delta:.2f}%)")
            if "latency_ms" in base_metrics:
                for lk in ["p50", "p95"]:
                    delta = summary_metrics["latency_ms"][lk] - base_metrics["latency_ms"][lk]
                    sign = "+" if delta >= 0 else ""
                    print(f"  latency_{lk:22}: {base_metrics['latency_ms'][lk]:6.2f}ms -> {summary_metrics['latency_ms'][lk]:6.2f}ms ({sign}{delta:.2f}ms)")
        print("-" * 60)

    # Save to output file if requested
    if output_path:
        full_output = {
            "summary": summary_metrics,
            "queries": eval_results,
        }
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(full_output, f, indent=2)
        print(f"\nSaved evaluation output to: {out_p.resolve()}")

    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run proxy evaluation for Polaris MCP search.")
    parser.add_argument("--queries", default=str(ROOT / "eval/queries.json"), help="Path to queries.json")
    parser.add_argument("--catalog", default=str(ROOT / "data/products.json"), help="Path to products.json")
    parser.add_argument("--output", default=str(ROOT / "eval/baseline.json"), help="Output path for evaluation JSON")
    parser.add_argument("--compare", default=None, help="Optional previous baseline JSON to compare against")
    args = parser.parse_args()

    run_evaluation(args.queries, args.catalog, args.output, args.compare)
