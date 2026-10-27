## BEFORE step 5: legacy single-column user_id indexes (3.1s to swap)

Rows: mm_memory_preferences=10,000, mm_beliefs=1,000,000, mm_belief_events=2,000,000, mm_evidence=1,000,000, mm_belief_evidence_links=1,000,000, mm_evolution_jobs=10,000, mm_extraction_stats=30,000, mm_snapshots=10,000
; load 62.6s

| query | mode | scan | index | est rows | rows | removed by filter | ms | shared hit/read |
|---|---|---|---|---:|---:|---:|---:|---|
| recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150) | C | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 5 | 6 | 94 | 1.12 | 0/316 |
| recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150) | A | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 1 | 6 | 94 | 0.32 | 426/0 |
| recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150) | B | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 1 | 6 | 94 | 0.12 | 316/0 |
| recall: recent slice (ORDER BY last_evidence_at LIMIT 25) | C | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 25 | 25 | 0 | 0.1 | 316/0 |
| recall: recent slice (ORDER BY last_evidence_at LIMIT 25) | A | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 25 | 25 | 0 | 0.12 | 420/0 |
| recall: recent slice (ORDER BY last_evidence_at LIMIT 25) | B | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 25 | 25 | 0 | 0.12 | 316/0 |
| belief by id (session.get, PK) | C | Index Scan | mm_beliefs_pkey | 1 | 1 | 0 | 0.07 | 1/3 |
| belief by id (session.get, PK) | A | Index Scan | mm_beliefs_pkey | 1 | 1 | 0 | 0.03 | 8/0 |
| belief by id (session.get, PK) | B | Index Scan | mm_beliefs_pkey | 1 | 1 | 0 | 0.01 | 4/0 |
| events by belief (ORDER BY created_at DESC LIMIT 50) | C | Index Scan | ix_mm_belief_events_belief_id | 1 | 2 | 0 | 0.07 | 0/15 |
| events by belief (ORDER BY created_at DESC LIMIT 50) | A | Index Scan | ix_mm_belief_events_belief_id | 1 | 2 | 0 | 0.02 | 20/0 |
| events by belief (ORDER BY created_at DESC LIMIT 50) | B | Index Scan | ix_mm_belief_events_belief_id | 1 | 2 | 0 | 0.01 | 15/0 |
| evidence by belief (links JOIN evidence) | C | Index Scan | mm_evidence_pkey, uq_mm_belief_evidence_link | 1 | 1 | 0 | 0.1 | 6/10 |
| evidence by belief (links JOIN evidence) | A | Index Scan | mm_evidence_pkey, uq_mm_belief_evidence_link | 1 | 1 | 0 | 0.04 | 24/0 |
| evidence by belief (links JOIN evidence) | B | Index Scan | mm_evidence_pkey, uq_mm_belief_evidence_link | 1 | 1 | 0 | 0.01 | 16/0 |
| job claim (conditional UPDATE pending->running) | C | Index Scan | mm_evolution_jobs_pkey | 0 | 0 | 0 | 0.64 | 20/0 |
| job claim (conditional UPDATE pending->running) | A | Index Scan | mm_evolution_jobs_pkey | 0 | 0 | 0 | 0.1 | 31/0 |
| job claim (conditional UPDATE pending->running) | B | Index Scan | mm_evolution_jobs_pkey | 0 | 0 | 0 | 0.03 | 25/0 |
| pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100) | C | Index Scan | ix_mm_evolution_jobs_pending | 100 | 100 | 0 | 0.04 | 34/0 |
| pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100) | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evolution_jobs_pending, uq_mm_evolution_jobs_idempotency | 10 | 10 | 0 | 0.05 | 24/0 |
| pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100) | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evolution_jobs_pending, uq_mm_evolution_jobs_idempotency | 10 | 10 | 0 | 0.03 | 24/0 |
| delete path: beliefs by user | C | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 0 | 0 | 0 | 0.53 | 312/0 |
| delete path: beliefs by user | A | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 0 | 0 | 0 | 0.15 | 416/0 |
| delete path: beliefs by user | B | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 0 | 0 | 0 | 0.1 | 312/0 |
| delete path: events by user | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_belief_events_user_id, mm_belief_events_pkey | 0 | 0 | 0 | 8.4 | 204/1036 |
| delete path: events by user | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_belief_events_user_id, mm_belief_events_pkey | 0 | 0 | 0 | 4.29 | 1600/0 |
| delete path: events by user | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_belief_events_user_id, mm_belief_events_pkey | 0 | 0 | 0 | 2.46 | 1240/0 |
| delete path: evidence by user | C | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_evidence_user_ref | 0 | 0 | 0 | 1.74 | 102/207 |
| delete path: evidence by user | A | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_evidence_user_ref | 0 | 0 | 0 | 0.24 | 412/0 |
| delete path: evidence by user | B | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_evidence_user_ref | 0 | 0 | 0 | 0.11 | 309/0 |
| evidence re-find by (user, ref) (record_evidence) | C | Index Scan | uq_mm_evidence_user_ref | 1 | 0 | 0 | 0.01 | 3/0 |
| evidence re-find by (user, ref) (record_evidence) | A | Index Scan | uq_mm_evidence_user_ref | 1 | 0 | 0 | 0.01 | 6/0 |
| evidence re-find by (user, ref) (record_evidence) | B | Index Scan | uq_mm_evidence_user_ref | 1 | 0 | 0 | 0.01 | 3/0 |
| unlinked evidence ids by user | C | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_evidence_user_ref | 100 | 100 | 0 | 0.09 | 106/0 |
| unlinked evidence ids by user | A | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_evidence_user_ref | 100 | 100 | 0 | 0.05 | 209/0 |
| unlinked evidence ids by user | B | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_evidence_user_ref | 100 | 100 | 0 | 0.04 | 106/0 |
| extraction stats by user (/stats: ORDER BY created_at DESC LIMIT 20) | C | Index Scan | ix_mm_extraction_stats_user_id | 3 | 3 | 297 | 0.05 | 18/0 |
| extraction stats by user (/stats: ORDER BY created_at DESC LIMIT 20) | A | Index Scan | ix_mm_extraction_stats_user_id | 3 | 3 | 297 | 0.07 | 24/0 |
| extraction stats by user (/stats: ORDER BY created_at DESC LIMIT 20) | B | Index Scan | ix_mm_extraction_stats_user_id | 3 | 3 | 297 | 0.05 | 18/0 |
| snapshots by user (K3 summaries: status IN (...) ORDER BY created_at DESC LIMIT 5) | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_snapshots_user_id, mm_snapshots_pkey | 1 | 1 | 0 | 0.04 | 23/0 |
| snapshots by user (K3 summaries: status IN (...) ORDER BY created_at DESC LIMIT 5) | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_snapshots_user_id, mm_snapshots_pkey | 1 | 1 | 0 | 0.04 | 28/0 |
| snapshots by user (K3 summaries: status IN (...) ORDER BY created_at DESC LIMIT 5) | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_snapshots_user_id, mm_snapshots_pkey | 1 | 1 | 0 | 0.02 | 23/0 |

## AFTER step 5: (tenant_id, app_id, user_id) indexes as shipped (5.6s to swap back)

Rows: mm_memory_preferences=10,000, mm_beliefs=1,000,000, mm_belief_events=2,000,000, mm_evidence=1,000,000, mm_belief_evidence_links=1,000,000, mm_evolution_jobs=10,000, mm_extraction_stats=30,000, mm_snapshots=10,000
; load 62.6s

| query | mode | scan | index | est rows | rows | removed by filter | ms | shared hit/read |
|---|---|---|---|---:|---:|---:|---:|---|
| recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150) | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user | 5 | 6 | 94 | 0.2 | 312/0 |
| recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150) | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user | 1 | 6 | 94 | 0.23 | 415/0 |
| recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150) | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user | 1 | 6 | 94 | 0.14 | 312/0 |
| recall: recent slice (ORDER BY last_evidence_at LIMIT 25) | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user | 25 | 25 | 0 | 0.08 | 312/0 |
| recall: recent slice (ORDER BY last_evidence_at LIMIT 25) | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user | 25 | 25 | 0 | 0.12 | 415/0 |
| recall: recent slice (ORDER BY last_evidence_at LIMIT 25) | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user | 25 | 25 | 0 | 0.07 | 312/0 |
| belief by id (session.get, PK) | C | Index Scan | mm_beliefs_pkey | 1 | 1 | 0 | 0.02 | 4/0 |
| belief by id (session.get, PK) | A | Index Scan | mm_beliefs_pkey | 1 | 1 | 0 | 0.02 | 8/0 |
| belief by id (session.get, PK) | B | Index Scan | mm_beliefs_pkey | 1 | 1 | 0 | 0.01 | 4/0 |
| events by belief (ORDER BY created_at DESC LIMIT 50) | C | Index Scan | ix_mm_belief_events_belief_id | 1 | 2 | 0 | 0.02 | 15/0 |
| events by belief (ORDER BY created_at DESC LIMIT 50) | A | Index Scan | ix_mm_belief_events_belief_id | 1 | 2 | 0 | 0.01 | 20/0 |
| events by belief (ORDER BY created_at DESC LIMIT 50) | B | Index Scan | ix_mm_belief_events_belief_id | 1 | 2 | 0 | 0.01 | 15/0 |
| evidence by belief (links JOIN evidence) | C | Index Scan | mm_evidence_pkey, uq_mm_belief_evidence_link | 1 | 1 | 0 | 0.03 | 16/0 |
| evidence by belief (links JOIN evidence) | A | Index Scan | mm_evidence_pkey, uq_mm_belief_evidence_link | 1 | 1 | 0 | 0.01 | 24/0 |
| evidence by belief (links JOIN evidence) | B | Index Scan | mm_evidence_pkey, uq_mm_belief_evidence_link | 1 | 1 | 0 | 0.01 | 16/0 |
| job claim (conditional UPDATE pending->running) | C | Index Scan | mm_evolution_jobs_pkey | 0 | 0 | 0 | 0.09 | 26/0 |
| job claim (conditional UPDATE pending->running) | A | Index Scan | mm_evolution_jobs_pkey | 0 | 0 | 0 | 0.09 | 31/0 |
| job claim (conditional UPDATE pending->running) | B | Index Scan | mm_evolution_jobs_pkey | 0 | 0 | 0 | 0.03 | 25/0 |
| pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100) | C | Index Scan | ix_mm_evolution_jobs_pending | 100 | 100 | 0 | 0.04 | 34/0 |
| pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100) | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evolution_jobs_pending, ix_mm_evolution_jobs_tenant_user | 10 | 10 | 0 | 0.04 | 24/0 |
| pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100) | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evolution_jobs_pending, ix_mm_evolution_jobs_tenant_user | 10 | 10 | 0 | 0.03 | 24/0 |
| delete path: beliefs by user | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user | 0 | 0 | 0 | 0.11 | 309/0 |
| delete path: beliefs by user | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user | 0 | 0 | 0 | 0.14 | 412/0 |
| delete path: beliefs by user | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user | 0 | 0 | 0 | 0.09 | 309/0 |
| delete path: events by user | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_belief_events_tenant_user | 0 | 0 | 0 | 0.44 | 612/0 |
| delete path: events by user | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_belief_events_tenant_user | 0 | 0 | 0 | 0.31 | 816/0 |
| delete path: events by user | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_belief_events_tenant_user | 0 | 0 | 0 | 0.22 | 612/0 |
| delete path: evidence by user | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evidence_tenant_user | 0 | 0 | 0 | 0.35 | 309/0 |
| delete path: evidence by user | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evidence_tenant_user | 0 | 0 | 0 | 0.22 | 412/0 |
| delete path: evidence by user | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evidence_tenant_user | 0 | 0 | 0 | 0.18 | 309/0 |
| evidence re-find by (user, ref) (record_evidence) | C | Index Scan | uq_mm_evidence_user_ref | 1 | 0 | 0 | 0.01 | 3/0 |
| evidence re-find by (user, ref) (record_evidence) | A | Index Scan | uq_mm_evidence_user_ref | 1 | 0 | 0 | 0.02 | 6/0 |
| evidence re-find by (user, ref) (record_evidence) | B | Index Scan | uq_mm_evidence_user_ref | 1 | 0 | 0 | 0.01 | 3/0 |
| unlinked evidence ids by user | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evidence_tenant_user | 108 | 100 | 0 | 0.1 | 106/0 |
| unlinked evidence ids by user | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evidence_tenant_user | 108 | 100 | 0 | 0.05 | 209/0 |
| unlinked evidence ids by user | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evidence_tenant_user | 108 | 100 | 0 | 0.04 | 106/0 |
| extraction stats by user (/stats: ORDER BY created_at DESC LIMIT 20) | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_extraction_stats_tenant_user | 3 | 3 | 0 | 0.03 | 17/0 |
| extraction stats by user (/stats: ORDER BY created_at DESC LIMIT 20) | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_extraction_stats_tenant_user | 3 | 3 | 0 | 0.04 | 22/0 |
| extraction stats by user (/stats: ORDER BY created_at DESC LIMIT 20) | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_extraction_stats_tenant_user | 3 | 3 | 0 | 0.02 | 17/0 |
| snapshots by user (K3 summaries: status IN (...) ORDER BY created_at DESC LIMIT 5) | C | Index Scan | ix_mm_snapshots_tenant_user | 1 | 1 | 0 | 0.02 | 9/0 |
| snapshots by user (K3 summaries: status IN (...) ORDER BY created_at DESC LIMIT 5) | A | Index Scan | ix_mm_snapshots_tenant_user | 1 | 1 | 0 | 0.04 | 12/0 |
| snapshots by user (K3 summaries: status IN (...) ORDER BY created_at DESC LIMIT 5) | B | Index Scan | ix_mm_snapshots_tenant_user | 1 | 1 | 0 | 0.01 | 9/0 |
