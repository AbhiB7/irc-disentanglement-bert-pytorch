---
description: Review a proposed or completed patch for correctness risks.
---

Review the proposed or current patch critically:

1. Assume the patch may be wrong.
2. Look for logical bugs, incorrect assumptions, edge cases, and mismatches with surrounding code.
3. Check whether the change is consistent with the existing architecture.
4. Identify missing tests or validation steps.
5. Point out any unnecessary code churn or unrelated refactoring.
6. Give a short verdict: likely safe, risky, or unclear.
7. If risky or unclear, say exactly what should be checked next.