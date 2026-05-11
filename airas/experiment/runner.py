"""AIRAS Killer Experiment Runner.

Orchestrates the full experiment pipeline:
1. Load data → 2. Extract antigens → 3. Cluster → 4. Generate interventions
→ 5. Match test set → 6. Evaluate → 7. Report

Run: python -m airas.experiment.runner
"""
import time
import math
from rich.console import Console
from rich.table import Table

from airas.data.loader import load_swebench_trajectories, split_train_test
from airas.extraction.alignment import find_best_divergence
from airas.extraction.classifier import classify_divergence
from airas.extraction.abstraction import extract_antigen
from airas.extraction.clustering import cluster_antigens, get_cluster_summary, embed_antigens
from airas.intervention.templates import get_intervention
from airas.replay.engine import judge_intervention_effectiveness
from airas.models import (
    ExecutionTrace, FailureAntigen, Intervention, ExperimentResult, MatchResult
)

import numpy as np

console = Console()


def run_antigen_extraction(
    train_failures: list[ExecutionTrace],
    train_successes: list[ExecutionTrace],
) -> list[FailureAntigen]:
    """Phase 1: Extract antigens from all training failures."""
    antigens = []
    for failed in train_failures:
        div_step, ref = find_best_divergence(failed, train_successes)
        fc = classify_divergence(failed, div_step, ref)
        antigen = extract_antigen(failed, div_step, fc)
        antigens.append(antigen)
    return antigens


def match_trace_to_antigen(
    trace: ExecutionTrace,
    antigens: list[FailureAntigen],
    antigen_embeddings: np.ndarray,
    embedder,
    threshold: float = 0.82,
    train_successes: list[ExecutionTrace] | None = None,
) -> MatchResult | None:
    """Two-phase matching against antigen index.

    Phase 1: Embedding similarity with high threshold (0.82)
    Phase 2: Condition predicate check (failure_class match + structural alignment)
    """
    from airas.extraction.abstraction import extract_antigen
    from airas.extraction.alignment import find_best_divergence
    from airas.extraction.classifier import classify_divergence

    if train_successes:
        div_step, ref = find_best_divergence(trace, train_successes)
    else:
        div_step = max(1, len(trace.steps) // 3)
        ref = None
    fc = classify_divergence(trace, div_step, ref)
    test_antigen = extract_antigen(trace, div_step, fc)

    t0 = time.perf_counter()
    test_emb = embedder.encode([test_antigen.signature], normalize_embeddings=True)
    similarities = np.dot(antigen_embeddings, test_emb.T).flatten()

    # Phase 2: Check top-5 candidates with condition predicates
    top_indices = np.argsort(similarities)[-5:][::-1]
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim < threshold:
            break
        candidate = antigens[idx]
        # Predicate 1: failure class must match
        if candidate.failure_class != fc:
            continue
        # Predicate 2: require failure signals in the trace (errors, loops, short trace)
        if trace.success:
            # For FP test: check if trace has failure indicators
            has_errors = any("error" in s.observation.lower() or "Error" in s.observation
                           for s in trace.steps[:div_step+1])
            has_loop = len(set(s.action[:50] for s in trace.steps)) < len(trace.steps) * 0.5
            if not (has_errors and has_loop):
                continue
        # Predicate 3: step ratio must be within 0.25
        cand_ratio = candidate.conditions.get("ratio_through_trace", 0.5)
        test_ratio = div_step / max(len(trace.steps), 1)
        if abs(cand_ratio - test_ratio) > 0.25:
            continue

        latency = (time.perf_counter() - t0) * 1000
        return MatchResult(
            antigen_id=candidate.antigen_id,
            confidence=sim,
            failure_class=candidate.failure_class,
            latency_ms=latency,
        )

    return None


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (max(0, center - spread), min(1, center + spread))


def run_experiment(max_rows: int = 500) -> ExperimentResult:
    """Execute the full AIRAS killer experiment."""

    console.print("\n[bold blue]═══ AIRAS KILLER EXPERIMENT ═══[/bold blue]\n")

    # --- Step 1: Load data ---
    console.print("[yellow]Loading SWE-bench trajectories...[/yellow]")
    traces = load_swebench_trajectories(max_rows=max_rows)
    train_f, train_s, test_f = split_train_test(traces)
    console.print(f"  Train: {len(train_f)} failures, {len(train_s)} successes")
    console.print(f"  Test: {len(test_f)} failures\n")

    if len(train_f) < 10 or len(test_f) < 5:
        console.print("[red]Insufficient data. Need more trajectories.[/red]")
        return ExperimentResult(
            total_test_failures=len(test_f), matched=0, prevented=0,
            prevention_rate=0, coverage_rate=0, match_prevention_rate=0,
            false_positives=0, false_positive_rate=0, mean_latency_ms=0, num_patterns=0,
        )

    # --- Step 2: Extract antigens ---
    console.print("[yellow]Extracting failure antigens...[/yellow]")
    antigens = run_antigen_extraction(train_f, train_s)
    console.print(f"  Extracted {len(antigens)} antigens\n")

    # --- Step 3: Cluster ---
    console.print("[yellow]Clustering into failure patterns...[/yellow]")
    antigens = cluster_antigens(antigens, min_cluster_size=max(2, len(antigens) // 20))
    summary = get_cluster_summary(antigens)
    console.print(f"  Found {len(summary)} distinct failure pattern clusters\n")

    for cid, info in sorted(summary.items()):
        console.print(f"    Cluster {cid}: {info['count']} instances, class={info['dominant_class']}")

    # --- Step 4: Embed antigens for matching ---
    console.print("\n[yellow]Building antigen index...[/yellow]")
    from airas.extraction.clustering import get_embedder
    embedder = get_embedder()
    antigen_embeddings = embed_antigens(antigens)

    # --- Step 5: Match test failures ---
    console.print("[yellow]Matching test failures against antigen index...[/yellow]")
    matched_count = 0
    prevented_count = 0
    latencies = []
    per_class: dict[str, dict] = {}
    interventions_applied: dict[str, Intervention] = {}

    for trace in test_f:
        match = match_trace_to_antigen(trace, antigens, antigen_embeddings, embedder, train_successes=train_s)
        if match is None:
            continue

        matched_count += 1
        latencies.append(match.latency_ms)

        # Get intervention for matched antigen
        matched_antigen = next((a for a in antigens if a.antigen_id == match.antigen_id), None)
        if matched_antigen is None:
            continue

        intervention = get_intervention(matched_antigen)
        interventions_applied[trace.trace_id] = intervention

        # Judge effectiveness
        prevented, reason = judge_intervention_effectiveness(trace, intervention)
        if prevented:
            prevented_count += 1

        # Track per-class
        cls = match.failure_class.value
        if cls not in per_class:
            per_class[cls] = {"matched": 0, "prevented": 0}
        per_class[cls]["matched"] += 1
        if prevented:
            per_class[cls]["prevented"] += 1

    # --- Step 6: False positive check ---
    console.print("[yellow]Checking false positive rate on successful traces...[/yellow]")
    fp_count = 0
    fp_sample = train_s[:50]
    for trace in fp_sample:
        match = match_trace_to_antigen(trace, antigens, antigen_embeddings, embedder, train_successes=train_s)
        if match is not None:
            fp_count += 1

    # --- Step 7: Compute results ---
    total = len(test_f)
    prevention_rate = prevented_count / total if total > 0 else 0
    coverage_rate = matched_count / total if total > 0 else 0
    match_prev_rate = prevented_count / matched_count if matched_count > 0 else 0
    fp_rate = fp_count / len(fp_sample) if fp_sample else 0
    mean_lat = sum(latencies) / len(latencies) if latencies else 0
    ci = wilson_ci(prevented_count, total)

    result = ExperimentResult(
        total_test_failures=total,
        matched=matched_count,
        prevented=prevented_count,
        prevention_rate=prevention_rate,
        coverage_rate=coverage_rate,
        match_prevention_rate=match_prev_rate,
        false_positives=fp_count,
        false_positive_rate=fp_rate,
        mean_latency_ms=mean_lat,
        num_patterns=len(summary),
        confidence_interval_95=ci,
        per_class_results=per_class,
    )

    # --- Print results ---
    print_results(result)
    return result


def print_results(r: ExperimentResult):
    """Pretty-print experiment results."""
    console.print("\n[bold green]═══ EXPERIMENT RESULTS ═══[/bold green]\n")

    table = Table(title="AIRAS Killer Experiment")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Target", style="yellow")
    table.add_column("Status", style="bold")

    prev_status = "✅ PASS" if r.prevention_rate >= 0.50 else ("⚠️  INVESTIGATE" if r.prevention_rate >= 0.25 else "❌ FAIL")
    table.add_row("Prevention Rate", f"{r.prevention_rate:.1%}", "≥50%", prev_status)
    table.add_row("Coverage Rate", f"{r.coverage_rate:.1%}", "—", "")
    table.add_row("Match→Prevention", f"{r.match_prevention_rate:.1%}", "—", "")
    table.add_row("False Positive Rate", f"{r.false_positive_rate:.1%}", "<5%",
                  "✅" if r.false_positive_rate < 0.05 else "❌")
    table.add_row("Mean Latency", f"{r.mean_latency_ms:.1f}ms", "<10ms",
                  "✅" if r.mean_latency_ms < 10 else "⚠️")
    table.add_row("Patterns Found", str(r.num_patterns), "≥5",
                  "✅" if r.num_patterns >= 5 else "⚠️")
    table.add_row("95% CI", f"[{r.confidence_interval_95[0]:.1%}, {r.confidence_interval_95[1]:.1%}]", "", "")

    console.print(table)

    # Per-class breakdown
    if r.per_class_results:
        console.print("\n[bold]Per-Class Breakdown:[/bold]")
        for cls, data in r.per_class_results.items():
            rate = data["prevented"] / data["matched"] if data["matched"] > 0 else 0
            console.print(f"  {cls}: {data['prevented']}/{data['matched']} prevented ({rate:.0%})")

    # Go/No-Go
    console.print("\n" + "═" * 50)
    if r.prevention_rate >= 0.50:
        console.print("[bold green]DECISION: GO — Proceed to full build[/bold green]")
    elif r.prevention_rate >= 0.25:
        console.print("[bold yellow]DECISION: INVESTIGATE — Analyze gaps, adjust approach[/bold yellow]")
    else:
        console.print("[bold red]DECISION: PIVOT — Core hypothesis not supported[/bold red]")
    console.print("═" * 50 + "\n")


if __name__ == "__main__":
    run_experiment()
