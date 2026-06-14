# Experimental NLA Token Strategy Analysis

This report compares token-selection strategies only. It does not change the GEPA runner or main NLA pipeline.

## Summary

- `current_fixed_6`: avg_tokens=5.84, candidate_rows=888, source_rows=300, reference_rows=564, duplicate_pct=80.88, weak_pct=19.18
- `candidate_only_6`: avg_tokens=5.61, candidate_rows=1684, source_rows=0, reference_rows=0, duplicate_pct=53.44, weak_pct=22.09
- `candidate_source_6`: avg_tokens=5.90, candidate_rows=1170, source_rows=600, reference_rows=0, duplicate_pct=70.79, weak_pct=21.81
- `candidate_reference_6`: avg_tokens=5.78, candidate_rows=1170, source_rows=0, reference_rows=564, duplicate_pct=70.18, weak_pct=18.11
- `balanced_6`: avg_tokens=5.87, candidate_rows=598, source_rows=600, reference_rows=564, duplicate_pct=83.77, weak_pct=13.00
- `candidate_only_10`: avg_tokens=8.06, candidate_rows=2417, source_rows=0, reference_rows=0, duplicate_pct=51.22, weak_pct=21.02
- `candidate_source_10`: avg_tokens=9.34, candidate_rows=1908, source_rows=894, reference_rows=0, duplicate_pct=68.17, weak_pct=22.20
- `candidate_no_first_6`: avg_tokens=5.39, candidate_rows=1618, source_rows=0, reference_rows=0, duplicate_pct=44.75, weak_pct=17.06
- `candidate_content_6`: avg_tokens=4.89, candidate_rows=1467, source_rows=0, reference_rows=0, duplicate_pct=35.58, weak_pct=0.00
- `candidate_content_10`: avg_tokens=6.65, candidate_rows=1995, source_rows=0, reference_rows=0, duplicate_pct=34.89, weak_pct=0.00
- `candidate_source_content_8`: avg_tokens=6.53, candidate_rows=1467, source_rows=492, reference_rows=0, duplicate_pct=51.76, weak_pct=0.00

## Decision Use

- Prefer strategies with high candidate coverage and low weak/duplicate token rates.
- Do not merge any strategy into the main pipeline from this analysis alone.
- Use this analysis only to choose isolated GEPA strategy jobs.
