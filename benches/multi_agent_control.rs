//! Benchmarks: control-plane gate hot path (design doc §12 stage 4, optional).
//!
//! Every spawn crosses these primitives — the limiter CAS, the budget
//! reserve/commit ticket dance, and `usage_total` on the usage-reporting leg.
//! They are synchronous, allocation-light, and stable enough to be a
//! regression signal; the async spawn/close round-trip is covered by the
//! stress tests instead (too scheduling-noisy to bench honestly).
//!
//! Run: cargo bench --bench multi_agent_control --features multi_agent

use std::sync::Arc;

use agent_base::UsageInfo;
use agent_works::multi_agent::AgentExecutionLimiter;
use agent_works::multi_agent::{RolloutBudget, usage_total};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_control_plane(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_agent_control");

    group.bench_function("limiter_acquire_release", |b| {
        let limiter = Arc::new(AgentExecutionLimiter::unlimited());
        b.iter(|| {
            let slot = limiter.try_acquire().expect("unlimited");
            black_box(slot.limiter().current());
            // slot drops here → fetch_sub
        });
    });

    group.bench_function("budget_reserve_commit", |b| {
        let budget = Arc::new(RolloutBudget::new(None, None));
        b.iter(|| {
            let ticket = budget.try_reserve_spawn().expect("unlimited");
            ticket.commit();
            black_box(budget.spawn_count());
        });
    });

    group.bench_function("budget_reserve_rollback", |b| {
        let budget = Arc::new(RolloutBudget::new(None, None));
        b.iter(|| {
            let ticket = budget.try_reserve_spawn().expect("unlimited");
            drop(ticket); // uncommitted → fetch_sub rollback
            black_box(budget.spawn_count());
        });
    });

    group.bench_function("usage_total", |b| {
        let u = UsageInfo {
            prompt_tokens: Some(1200),
            completion_tokens: Some(340),
            total_tokens: None,
            reasoning_tokens: None,
        };
        b.iter(|| black_box(usage_total(&u)));
    });

    group.finish();
}

criterion_group!(benches, bench_control_plane);
criterion_main!(benches);
