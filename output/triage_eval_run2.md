# Triage assistant evaluation (qwen2.5:7b, local via Ollama)

- Cases: 250 (150 from the top of the queue, 100 random alerts)
- Valid on first try: 57%; valid after one retry: 60%; fallback template used: 40%
- Ungrounded (invented) signals shipped to investigators: 0 by construction (rejected by the grounding check)
- Priority disagreed with score band (sent to senior review): 23%
- Median latency per case: 6.4s on an Apple M4 Mac mini

| Assistant priority | Cases | Actual fraud rate |
|---|---|---|
| HIGH | 96 | 73% |
| MEDIUM | 61 | 66% |
| LOW | 93 | 12% |

Reviewer (standard#0, synthetic) accuracy on these cases: 63%.

Limits: synthetic applications and synthetic analysts; the assistant summarizes and prioritizes, it does not decide.