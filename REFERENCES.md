# References

Data sources, licenses, and any facts/quotes drawn from external articles used
in this project, logged as they're used.

## Data sources

- **nflverse / nflreadpy** — play-by-play, schedules/results, and player
  stats for the **1999–2025** seasons, pulled via the `nflreadpy` Python package
  from nflverse's GitHub-hosted data releases. Licensed CC-BY-4.0.
  https://github.com/nflverse/nflreadpy · https://github.com/nflverse/nflverse-data
  No FTN Data charting fields are used in this project (confirmed no
  `load_ftn_charting`/FTN-derived columns are pulled), so no separate
  CC-BY-SA 4.0 attribution is needed.

## Articles (claims tested in Phases 13–16)

These pieces supplied the specific public claims the all-time comparison checks
against the data. Where a computed result disagreed, the computed value is what
this project reports — see REPORT.md, "Were they actually all-time great?".

- John Boyle, "12 Numbers of Note From The Seahawks Super Bowl Winning 2025
  Season," Seahawks.com, 17 Feb 2026. Source of the +191/+246 point-differential,
  17.2 points-allowed, 3.7 yards-per-carry, and 26-game 100-yard-rusher figures.
  https://www.seahawks.com/news/12-numbers-of-note-from-the-seahawks-super-bowl-winning-2025-season
- John Boyle, "The Case For The 2025 Seahawks To Be Remembered As An 'All-Time
  Great' Team," Seahawks.com, 4 Mar 2026. Source of the postseason-path and
  best-champion-since-1999-Rams claims.
  https://www.seahawks.com/news/the-case-for-the-2025-seahawks-to-be-remembered-as-an-all-time-great-team
- "One Telling Statistic Proves 2025 Seattle Seahawks Are Historically Dominant,"
  Sports Illustrated, citing @AcccountStat on X. Source of the average
  end-of-game point differential claim tested in Phase 14.
  https://www.si.com/nfl/seahawks/onsi/seahawks-news/one-telling-statistic-proves-2025-seattle-seahawks-are-historically-dominant-01kh5a51hvf3
- FTN Fantasy, "Final 2025 DVOA Ratings" and "Week 9 DVOA Ratings: Seahawks
  Making History." Source of the DVOA claims, which are **cited as external
  context only** — DVOA is proprietary and is not reproduced anywhere in this
  project. https://ftnfantasy.com/nfl/final-2025-dvoa-ratings
- Sports Info Solutions, "The Seahawks Are Down, But Their Defense Has Shown
  They're Not Out," 20 Nov 2025. Source of the −0.34 EPA/play eight-week streak
  and the 40.1% pressure / 19.2% blitz rate figures. The pressure and blitz
  numbers come from charting data this project does not have and are **not
  reproduced**; see REPORT.md's caveats.
  https://www.sportsinfosolutions.com/2025/11/20/the-seahawks-are-down-but-their-defense-has-shown-theyre-not-out/
- Match Quarters, "How the Seahawks Manufactured the NFL's Most Efficient
  Pressures" and "Mike Macdonald's Super Bowl LX Defense: All-22 Masterclass" —
  scheme context for Phase 15's framing.
  https://www.matchquarters.com/p/seahawks-checkdown-loop-efficient-blitz-system

## Phase 17-18 — player and coach deep dives

External context cited but **not reproduced** from the play-by-play. Every figure
below is a published fact about awards, voting, biography, or a league record; the
statistical claims in these two phases are computed from `data/processed/` and are
listed separately in each phase's output JSON.

- Seahawks.com, "Seahawks WR Jaxon Smith-Njigba Named PFWA Offensive Player of the
  Year," 21 Jan 2026. Source of the franchise-record context (1,793 yards breaking
  DK Metcalf's 1,303; second Seahawk to lead the NFL in receiving after Steve
  Largent), the 44.1% share of team receiving yards, and the "566 team pass
  attempts" framing that Phase 17 tests directly against the reference set.
  https://www.seahawks.com/news/seahawks-wr-jaxon-smith-njigba-named-pfwa-offensive-player-of-the-year
- Fox Sports, "AP NFL Offensive Player of the Year voting," 5 Feb 2026. Source of
  the OPOY margin — Smith-Njigba 272 points and 14 first-place votes to Christian
  McCaffrey's 223 and 12. Award voting is not in any dataset here.
  https://www.foxsports.com/articles/nfl/ap-nfl-offensive-player-of-the-year-voting
- The Ringer, Danny Kelly, "Jaxon Smith-Njigba Is Having a Historic Season,"
  13 Nov 2025. In-season context on Seattle's league-low pass rate and the
  yards-per-team-pass-attempt framing this project adopts and extends to a full
  historical reference set.
  https://www.theringer.com/2025/11/13/nfl/jaxon-smith-njigba-historic-wide-receiver-season-seattle-seahawks
- NFL.com team stats, 2025 regular season. Source of the official 481 team pass
  attempts used to validate Phase 17's denominator against nflverse's `pass_attempt`
  flag, which also counts sacks and two-point conversions.
  https://www.nfl.com/stats/team-stats/offense/passing/2025/reg/all
- ESPN, "Sam Darnold perseveres through oblique injury, delivers Seahawks Super
  Bowl LX win," 9 Feb 2026. Source of the oblique strain and of the independent
  confirmation that Darnold led all players with 20 regular-season turnovers.
  https://www.espn.com/nfl/story/_/id/47874504/
- SI, "Sam Darnold 'seeing ghosts' on ESPN broadcast," 25 Oct 2019, and CBS News
  Boston, 7 Apr 2022. Sources for the Week 7 2019 game (11/32, 86 yards, four
  interceptions, 33-0 to New England), how the remark was captured and cleared for
  broadcast, and Darnold's own later explanation of it.
  https://www.si.com/nfl/2019/10/25/sam-darnold-seeing-ghosts-jets-patriots-monday-night-football-espn-broadcast
- CBS Sports, "Mike Macdonald becomes first head coach to win a Super Bowl while
  calling defensive plays," 9 Feb 2026, and AP via MyNorthwest, 3 Feb 2026. Sources
  for the play-calling record, for only two head coaches calling their own defense
  in 2025, and for Macdonald being the third-youngest Super Bowl-winning head coach
  at 38.
  https://www.cbssports.com/nfl/news/mike-macdonald-seahawks-super-bowl-lx-coach-third-youngest-defensive-playcaller/
- Seahawks.com, "Mike Macdonald Named Head Coach of the Seattle Seahawks,"
  31 Jan 2024. Source for his hire at 36 as the NFL's youngest head coach, and for
  Baltimore's 2022 defensive rankings.
  https://www.seahawks.com/news/mike-macdonald-named-head-coach-of-the-seattle-seahawks
- BaltimoreRavens.com, "Ravens Defense Completes Historic Triple Crown." Source for
  the 2023 Ravens leading the NFL in scoring defense, sacks and takeaways in the
  same season — the first team in league history to do so.
  https://www.baltimoreravens.com/news/ravens-defense-triple-crown-nfl-record-best-of-all-time
- NFL.com, "Mike Vrabel named AP Coach of the Year," 2026. Source for the voting
  order in which Macdonald finished third, behind Vrabel and Liam Coen.
  https://www.nfl.com/news/mike-vrabel-patriots-coach-of-the-year-2025
- Seahawks.com, "12 Numbers of Note From the Seahawks' Super Bowl-Winning 2025
  Season," 17 Feb 2026. Source for the in-season 26-game streak without allowing an
  individual 100-yard rusher. Phase 18 recomputes the streak through the Super Bowl
  and reports 29; the difference is the endpoint, not a disagreement.
  https://www.seahawks.com/news/12-numbers-of-note-from-the-seahawks-super-bowl-winning-2025-season

**Not reproducible from this cache, and therefore never asserted:** routes run and
any route-based target share; snap counts and participation; blitz rate,
pass-rusher counts and simulated-pressure rate; coverage shells, man/zone splits
and pre- or post-snap safety rotation; individual offensive-line grades. See
REPORT.md's Limitations.
