## Baseline indexes (as shipped)

Rows: mm_memory_preferences=10,000, mm_beliefs=1,000,000, mm_belief_events=2,000,000, mm_evidence=1,000,000, mm_belief_evidence_links=1,000,000, mm_evolution_jobs=10,000
; load 58.6s

| query | mode | scan | index | est rows | rows | removed by filter | ms | shared hit/read |
|---|---|---|---|---:|---:|---:|---:|---|
| recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150) | C | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 6 | 6 | 94 | 0.64 | 0/316 |
| recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150) | A | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 1 | 6 | 94 | 0.52 | 426/0 |
| recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150) | B | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 1 | 6 | 94 | 0.22 | 316/0 |
| recall: recent slice (ORDER BY last_evidence_at LIMIT 25) | C | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 25 | 25 | 0 | 0.14 | 316/0 |
| recall: recent slice (ORDER BY last_evidence_at LIMIT 25) | A | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 25 | 25 | 0 | 0.2 | 420/0 |
| recall: recent slice (ORDER BY last_evidence_at LIMIT 25) | B | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 25 | 25 | 0 | 0.14 | 316/0 |
| belief by id (session.get, PK) | C | Index Scan | mm_beliefs_pkey | 1 | 1 | 0 | 0.03 | 1/3 |
| belief by id (session.get, PK) | A | Index Scan | mm_beliefs_pkey | 1 | 1 | 0 | 0.03 | 8/0 |
| belief by id (session.get, PK) | B | Index Scan | mm_beliefs_pkey | 1 | 1 | 0 | 0.02 | 4/0 |
| events by belief (ORDER BY created_at DESC LIMIT 50) | C | Index Scan | ix_mm_belief_events_belief_id | 1 | 2 | 0 | 0.04 | 0/15 |
| events by belief (ORDER BY created_at DESC LIMIT 50) | A | Index Scan | ix_mm_belief_events_belief_id | 1 | 2 | 0 | 0.03 | 20/0 |
| events by belief (ORDER BY created_at DESC LIMIT 50) | B | Index Scan | ix_mm_belief_events_belief_id | 1 | 2 | 0 | 0.03 | 15/0 |
| evidence by belief (links JOIN evidence) | C | Index Scan | mm_evidence_pkey, uq_mm_belief_evidence_link | 1 | 1 | 0 | 0.05 | 6/10 |
| evidence by belief (links JOIN evidence) | A | Index Scan | mm_evidence_pkey, uq_mm_belief_evidence_link | 1 | 1 | 0 | 0.05 | 24/0 |
| evidence by belief (links JOIN evidence) | B | Index Scan | mm_evidence_pkey, uq_mm_belief_evidence_link | 1 | 1 | 0 | 0.03 | 16/0 |
| job claim (conditional UPDATE pending->running) | C | Index Scan | mm_evolution_jobs_pkey | 0 | 0 | 0 | 0.12 | 20/0 |
| job claim (conditional UPDATE pending->running) | A | Index Scan | mm_evolution_jobs_pkey | 0 | 0 | 0 | 0.12 | 31/0 |
| job claim (conditional UPDATE pending->running) | B | Index Scan | mm_evolution_jobs_pkey | 0 | 0 | 0 | 0.07 | 25/0 |
| pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100) | C | Index Scan | ix_mm_evolution_jobs_pending | 100 | 100 | 0 | 0.04 | 34/0 |
| pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100) | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evolution_jobs_pending, uq_mm_evolution_jobs_idempotency | 10 | 10 | 0 | 0.07 | 24/0 |
| pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100) | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evolution_jobs_pending, uq_mm_evolution_jobs_idempotency | 10 | 10 | 0 | 0.06 | 24/0 |
| delete path: beliefs by user | C | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 0 | 0 | 0 | 0.31 | 312/0 |
| delete path: beliefs by user | A | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 0 | 0 | 0 | 0.21 | 416/0 |
| delete path: beliefs by user | B | Bitmap Heap Scan / Bitmap Index Scan | uq_mm_beliefs_user_key | 0 | 0 | 0 | 0.19 | 312/0 |
| delete path: events by user | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_belief_events_user_id, mm_belief_events_pkey | 0 | 0 | 0 | 5.13 | 204/1032 |
| delete path: events by user | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_belief_events_user_id, mm_belief_events_pkey | 0 | 0 | 0 | 4.87 | 1595/0 |
| delete path: events by user | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_belief_events_user_id, mm_belief_events_pkey | 0 | 0 | 0 | 4.33 | 1236/0 |

## With candidate composite indexes (7.3s to build CONCURRENTLY)

Rows: mm_memory_preferences=10,000, mm_beliefs=1,000,000, mm_belief_events=2,000,000, mm_evidence=1,000,000, mm_belief_evidence_links=1,000,000, mm_evolution_jobs=10,000
; load 58.6s

| query | mode | scan | index | est rows | rows | removed by filter | ms | shared hit/read |
|---|---|---|---|---:|---:|---:|---:|---|
| recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150) | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user_recent | 7 | 6 | 94 | 0.27 | 300/12 |
| recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150) | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user_recent | 1 | 6 | 94 | 0.28 | 415/0 |
| recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150) | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user_recent | 1 | 6 | 94 | 0.2 | 312/0 |
| recall: recent slice (ORDER BY last_evidence_at LIMIT 25) | C | Index Scan | ix_mm_beliefs_tenant_user_recent | 25 | 25 | 0 | 0.05 | 56/0 |
| recall: recent slice (ORDER BY last_evidence_at LIMIT 25) | A | Index Scan | ix_mm_beliefs_tenant_user_recent | 25 | 25 | 0 | 0.06 | 84/0 |
| recall: recent slice (ORDER BY last_evidence_at LIMIT 25) | B | Index Scan | ix_mm_beliefs_tenant_user_recent | 25 | 25 | 0 | 0.05 | 56/0 |
| belief by id (session.get, PK) | C | Index Scan | mm_beliefs_pkey | 1 | 1 | 0 | 0.02 | 4/0 |
| belief by id (session.get, PK) | A | Index Scan | mm_beliefs_pkey | 1 | 1 | 0 | 0.03 | 8/0 |
| belief by id (session.get, PK) | B | Index Scan | mm_beliefs_pkey | 1 | 1 | 0 | 0.02 | 4/0 |
| events by belief (ORDER BY created_at DESC LIMIT 50) | C | Index Scan | ix_mm_belief_events_belief_id | 1 | 2 | 0 | 0.02 | 15/0 |
| events by belief (ORDER BY created_at DESC LIMIT 50) | A | Index Scan | ix_mm_belief_events_belief_id | 1 | 2 | 0 | 0.02 | 20/0 |
| events by belief (ORDER BY created_at DESC LIMIT 50) | B | Index Scan | ix_mm_belief_events_belief_id | 1 | 2 | 0 | 0.03 | 15/0 |
| evidence by belief (links JOIN evidence) | C | Index Scan | mm_evidence_pkey, uq_mm_belief_evidence_link | 1 | 1 | 0 | 0.03 | 16/0 |
| evidence by belief (links JOIN evidence) | A | Index Scan | mm_evidence_pkey, uq_mm_belief_evidence_link | 1 | 1 | 0 | 0.03 | 24/0 |
| evidence by belief (links JOIN evidence) | B | Index Scan | mm_evidence_pkey, uq_mm_belief_evidence_link | 1 | 1 | 0 | 0.03 | 16/0 |
| job claim (conditional UPDATE pending->running) | C | Index Scan | mm_evolution_jobs_pkey | 0 | 0 | 0 | 0.06 | 25/0 |
| job claim (conditional UPDATE pending->running) | A | Index Scan | mm_evolution_jobs_pkey | 0 | 0 | 0 | 0.06 | 30/0 |
| job claim (conditional UPDATE pending->running) | B | Index Scan | mm_evolution_jobs_pkey | 0 | 0 | 0 | 0.05 | 25/0 |
| pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100) | C | Index Scan | ix_mm_evolution_jobs_pending | 100 | 100 | 0 | 0.05 | 34/0 |
| pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100) | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evolution_jobs_pending, uq_mm_evolution_jobs_idempotency | 10 | 10 | 0 | 0.06 | 24/0 |
| pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100) | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_evolution_jobs_pending, uq_mm_evolution_jobs_idempotency | 10 | 10 | 0 | 0.06 | 24/0 |
| delete path: beliefs by user | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user_recent | 0 | 0 | 0 | 0.13 | 309/0 |
| delete path: beliefs by user | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user_recent | 0 | 0 | 0 | 0.21 | 412/0 |
| delete path: beliefs by user | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_beliefs_tenant_user_recent | 0 | 0 | 0 | 0.2 | 309/0 |
| delete path: events by user | C | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_belief_events_tenant_user | 0 | 0 | 0 | 0.27 | 612/0 |
| delete path: events by user | A | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_belief_events_tenant_user | 0 | 0 | 0 | 0.51 | 816/0 |
| delete path: events by user | B | Bitmap Heap Scan / Bitmap Index Scan | ix_mm_belief_events_tenant_user | 0 | 0 | 0 | 0.42 | 612/0 |
