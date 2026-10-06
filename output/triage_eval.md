# Triage assistant evaluation (qwen2.5:7b, local via Ollama)

- Cases: 250 (150 from the top of the queue, 100 random alerts)
- Valid on first try: 98%; valid after one retry: 98%; fallback template used: 2%
- Ungrounded (invented) signals shipped to investigators: 0 by construction (rejected by the grounding check)
- Assistant suggested plain approval on a HIGH-score case (sent to senior review): 0%
- Median latency per case: 4.4s on an Apple M4 Mac mini

| Priority (from risk score) | Cases | Actual fraud rate |
|---|---|---|
| HIGH | 150 | 71% |
| MEDIUM | 9 | 33% |
| LOW | 91 | 12% |

Reviewer (standard#0, synthetic) accuracy on these cases: 63%.

Limits: synthetic applications and synthetic analysts; the assistant explains cases; the risk model sets priority and a human decides.