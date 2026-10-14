Verification run, 14 Oct 2026

On 14 Oct 2026 the __20260921S snapshot in data/raw was overwritten by re-running 01d_fetch_scope.py
with END_DATE = 2026-10-01 (102 of 104 files no longer match manifest__20260921S.json; the two hourly
Bitcoin files still match). No other copy of the original files exists.

20_scope_expansion.py was then run unchanged on the overwritten files (it restricts the sample to
12 Jan 2024 - 15 Jun 2026). Its output (this folder) reproduces every number reported in Section 4.10,
Tables 13-16 and Appendix B (Tables B1-B3) of the revised JRFM manuscript exactly. The difference is
in the file bytes only (prices after 15 Jun 2026, and Yahoo's re-scaled adjusted prices).
