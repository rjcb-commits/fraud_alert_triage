# Triage assistant evaluation (qwen2.5:7b, local via Ollama)

- Cases: 250 (150 from the top of the queue, 100 random alerts)
- Valid on first try: 89%; valid after one retry: 98%; fallback template used: 2%
- Ungrounded (invented) signals shipped to investigators: 0 by construction (rejected by the grounding check)
- Priority disagreed with score band (sent to senior review): 5%
- Median latency per case: 4.9s on an Apple M4 Mac mini

| Assistant priority | Cases | Actual fraud rate |
|---|---|---|
| HIGH | 149 | 71% |
| MEDIUM | 21 | 19% |
| LOW | 80 | 14% |

Reviewer (standard#0, synthetic) accuracy on these cases: 63%.

Limits: synthetic applications and synthetic analysts; the assistant summarizes and prioritizes, it does not decide.