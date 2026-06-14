# Experimental NLA Token Strategy Analysis

This report compares token-selection strategies only. It does not change the GEPA runner or main NLA pipeline.

## Summary

- `current_fixed_6`: avg_tokens=5.83, candidate_rows=108, source_rows=36, reference_rows=66, duplicate_pct=72.38, weak_pct=17.14
- `candidate_only_6`: avg_tokens=5.86, candidate_rows=211, source_rows=0, reference_rows=0, duplicate_pct=29.38, weak_pct=20.85
- `candidate_source_6`: avg_tokens=5.97, candidate_rows=143, source_rows=72, reference_rows=0, duplicate_pct=58.60, weak_pct=19.53
- `candidate_reference_6`: avg_tokens=5.81, candidate_rows=143, source_rows=0, reference_rows=66, duplicate_pct=57.42, weak_pct=20.10
- `balanced_6`: avg_tokens=5.83, candidate_rows=72, source_rows=72, reference_rows=66, duplicate_pct=79.05, weak_pct=10.95
- `candidate_only_10`: avg_tokens=8.33, candidate_rows=300, source_rows=0, reference_rows=0, duplicate_pct=26.00, weak_pct=20.00
- `candidate_source_10`: avg_tokens=9.69, candidate_rows=241, source_rows=108, reference_rows=0, duplicate_pct=51.86, weak_pct=21.78
- `candidate_no_first_6`: avg_tokens=5.69, candidate_rows=205, source_rows=0, reference_rows=0, duplicate_pct=19.51, weak_pct=16.10
- `candidate_content_6`: avg_tokens=5.19, candidate_rows=187, source_rows=0, reference_rows=0, duplicate_pct=17.65, weak_pct=0.00
- `candidate_content_10`: avg_tokens=7.03, candidate_rows=253, source_rows=0, reference_rows=0, duplicate_pct=17.00, weak_pct=0.00
- `candidate_source_content_8`: avg_tokens=7.03, candidate_rows=187, source_rows=66, reference_rows=0, duplicate_pct=39.13, weak_pct=0.00
- `hybrid_context_dedup_6`: avg_tokens=3.92, candidate_rows=129, source_rows=6, reference_rows=6, duplicate_pct=19.15, weak_pct=0.00
- `hybrid_context_dedup_8`: avg_tokens=5.53, candidate_rows=187, source_rows=6, reference_rows=6, duplicate_pct=16.58, weak_pct=0.00

## Decision Use

- Prefer strategies with high candidate coverage and low weak/duplicate token rates.
- Do not merge any strategy into the main pipeline from this analysis alone.
- Use this analysis only to choose isolated GEPA strategy jobs.
