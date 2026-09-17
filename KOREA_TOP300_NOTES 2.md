# Korea Top 300 creator extension

Created: 2026-09-14

## Files
- `korea_top300_creators.csv`: 300 rows, exact same 9-column schema as the original creators.csv.
- `creators_korea_5300.csv`: original 5,000 rows + Korea 300 rows = 5,300 rows.
- `korea_top300_provenance.csv`: ranking/source notes for the added 300 rows.

## Sources
- Instagram Top 100: https://hafi.pro/top/most-followed-instagram/south-korea
- TikTok Top 100: https://hafi.pro/top/most-followed-tiktok/south-korea
- YouTube Top 100 by subscribers: https://socialblade.com/youtube/lists/top/100/subscribers/all/KR

## Important data notes
1. Instagram/TikTok follower counts and engagement rates are copied from the public ranking snapshot accessed on 2026-09-14.
2. YouTube ranking provides subscriber counts, but not a directly comparable engagement rate. `avg_engagement_rate` is therefore blank for YouTube rather than fabricating a metric.
3. Public ranking pages do not provide reliable `verified` or `fake_followers_pct` for all 300 rows. Those fields are left blank.
4. `niche` is standardized into the existing project's broad categories using transparent keyword/identity heuristics. Treat it as a project feature, not an official platform category.
5. Social Blade's ranking page exposes channel names rather than actual YouTube @handles. The YouTube `handle` values beginning with `@yt_` are internal unique IDs generated for the dashboard, not claimed platform handles.
6. The original uploaded `creators.csv` contains a small number of invalid UTF-8 bytes. The merged UTF-8-SIG file preserves all rows but replaces undecodable bytes with the Unicode replacement character.
