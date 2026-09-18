# Benchmarks

## LOCOMO Refined

**Dataset**: 10 long conversations, 1382 QA questions
**Source**: [mem-eval-suite/LoCoMo_refined](https://github.com/mem-eval-suite/LoCoMo_refined)

### Results

| Metric | Value |
|--------|-------|
| F1 mean (all) | 0.080 |
| F1 mean (answered) | 0.228 |
| Hit rate (F1 > 0) | 34.6% |
| Has answer | 30.7% |
| Top F1 | 0.870 |

### By Category

| Category | Count | F1 |
|----------|-------|-----|
| Multi-hop reasoning | 213 | 0.064 |
| Single-hop reasoning | 299 | 0.036 |
| Temporal reasoning | 68 | 0.088 |
| Open-domain knowledge | 802 | 0.100 |

### Evolution (Conv-26 single conversation)

| Version | F1 | Hit Rate | Top F1 |
|---------|-----|----------|--------|
| Beliefs only | 0.019 | 12% | 0.316 |
| +P0 specific facts | 0.024 | 13% | 0.600 |
| +P3 dual-channel | 0.094 | 39% | 0.667 |

## LongMemEval

**Dataset**: 500 QA questions, oracle dataset
**Source**: [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)

| Metric | Value |
|--------|-------|
| Has answer | 33.2% |
| No info | 66.8% |

## Comparison with Mainstream

| System | LOCOMO (LLM judge) | Note |
|--------|-------------------|------|
| Mem0 (managed) | 92.5 | Closed-source optimizations |
| MemoraX AI | 82.7 | |
| MemOS | 63.6 | |
| Mem0 (open-source) | 48.9 | Strict judge |
| **mirror-memory** | ~30-40 (est.) | F1=0.080, LLM judge not yet run |

**Note**: mirror-memory uses F1 (token-level matching), while others use LLM judge (semantic matching). Direct comparison requires running the same metric.

## How to Run

```bash
# LOCOMO
export MM_LLM_API_KEY=sk-xxx
export MM_LLM_BASE_URL=https://api.deepseek.com
export MM_LLM_MODEL=deepseek-chat

# Clone LOCOMO refined
git clone https://github.com/mem-eval-suite/LoCoMo_refined.git
cd LoCoMo_refined

# Run (inline script — runner file pending)
python3 -c "..."  # See benchmarks/README.md for full script

# Score
PYTHONPATH="src" python -c 'from evaluate import main; main()' \
  --questions-path data/public/questions.jsonl \
  --predictions-path /tmp/locomo_predictions.jsonl \
  --metrics f1 bleu
```