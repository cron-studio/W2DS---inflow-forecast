# Working notes

Decisions we have made and assumptions we still need to challenge.

The chain we are building, for a past period:

1. height → caudal
2. caudal − esgoto → afluências indevidas
3. afluências indevidas → indireta + direta

## Decisions

### Step 1: height → caudal

**Current best model:** the two-number curve `caudal = a × height^b` does the main work, and a machine-learning model learns how far off the curve is (caudal / curve) from height information only: the hour's mean, max, min and spread, how much height changed within the hour and since the previous hour, the previous 3 hours, and the mean, min and max over the last 6 and 24 hours. No rain, calendar or temperature.

**Why:** among eight models trained on the same hours, it had the lowest average daily volume error on October data none of them had seen: 0.16% at Antanhol and 8.8% at Ameal. Because the curve carries the prediction, it is not capped at the highest height seen in training, which plain tree models are.

**What did not help:** the curve as an extra input (same information as height), calendar (height already shows the day's pattern), water temperature (made October at Ameal much worse; the summer temperature pattern does not carry over).

**Limits:** Antanhol is solved (every model under 1% per day). Ameal stays 7–9% low in October whatever we add, which points to the data (about 300 usable hours, and a height-to-caudal relationship that moved), not to the inputs. The Ameal comparison rests on 4.5 October days, so differences of a point or two are noise.

**Code:** `predict inflow/experiments/experiment_features.py` and `experiment_features_2.py`.

**Next, when needed:** find out how and when the relationship moved at Ameal (slow drift, jumps after rain, or a drifting speed reading before the 6 Oct failure) before adding more inputs.

### Steps 2 and 3: esgoto table and split (first version)

**Method** (`predict inflow/afluencias.py`, plots and checks in `predict inflow/08_esgoto_e_separacao.ipynb`):

- A dry day has less than 1 mm of rain that day and on each of the 2 days before.
- We read the indireta on every night with less than 0.5 mm of rain in the 12 hours before 07:00, as the lowest hourly caudal between 02:00 and 07:00, and draw a straight line between those nights.
- Esgoto is a table, the mean of caudal − indireta on dry days, for each 3-hour block, on working days and on rest days (weekends and holidays).
- Afluências indevidas = caudal − esgoto, and direta = afluências indevidas − indireta. We do not clip anything to zero hour by hour.

**Results of the checks** (meter caudal, Jul–Oct 2026):

- **Direta on dry days, with each day left out of the table:** on average 1% of caudal at Antanhol and −3% at Ameal (the current method gives 17% and 22%). A single day is typically off by about 19%, because esgoto varies a lot from day to day. Only totals over a week or more are reliable.
- **Made-up storms (Antanhol, 30 storms):** the method found 100% of the added direta (96–103%) and 102% of the added indireta (97–107%). The current method found 86% of the direta, 2% of the indireta and 58% of the total. We could not run this test at Ameal because there are too few dry days.
- **Daily totals:** at Antanhol, 15 of 53 days have negative afluências indevidas, from 15 to 23 August and from 2 to 8 October. On those days the caudal was below the typical esgoto. Possible reasons are holidays (people away in mid-August) or something in the network, such as a pumping station. At Ameal no day is negative.
- **Real storms:** at Ameal the 23–27 August storm (55 mm) gives about 2,700 m² of connected area, which is plausible. At Antanhol the results are inconsistent: 7,200 m² and 18,100 m² for the two August storms, 128,000 m² for a 2.3 mm shower in July, and negative for 5 October. We do not yet understand the Antanhol network well enough to trust its direta.

### How we check steps 2 and 3

Nothing in our data measures esgoto, indireta or direta on their own, so there is no ground truth for steps 2 and 3. We check them at three levels.

1. **Checks that rule out wrong answers.** Direta should be close to zero on dry days. Daily afluências indevidas should not be negative. Each storm should give a similar amount of direta per mm of rain, and that amount, read as the area of roofs and drains connected to the sewer, should be plausible for the area the pipe serves.
2. **A made-up storm on real data.** We take the meter's real caudal on dry days, add a storm whose volume we choose (a fast direta spike and a slow rise in indireta), add the matching rain, and run steps 2 and 3. The method passes if it finds the volumes we added. This tests the method, not whether real storms behave like the made-up one.
3. **Independent confirmation, with the water company.** Water consumption data for the area served by each pipe gives an independent estimate of esgoto. A known-volume test, in which a measured amount of water is released into a drain upstream, gives real ground truth.

## Assumptions to challenge

### 1. Esgoto is constant (the average)

**Where it came from:** a domain expert told us to assume esgoto is a constant equal to its average.

**What the code does:** takes the dry days (no rain that day or the days before), subtracts the indireta, and averages all readings. One number per station.

**What the data shows so far** (Antanhol, October, hours with no rain in the previous 48 h):

- The average is 15.3 m³/h, which matches the value used (15.1).
- Within a day it varies a lot: most hours fall between 4 and 29 m³/h, lowest around 5 am, highest around 9 pm.
- Between days it varies a lot: the only two full dry days averaged 11.8 and 22.8 m³/h.
- Between seasons: 22.6 m³/h in summer, 15.1 in October.

**Why it matters:** the code subtracts the average from every hour and sets negative results to zero. Evening hours above the average count as afluências indevidas; night hours below it are dropped instead of cancelling out. So afluências indevidas come out too high even on dry days. The same constant also makes the rain-only prediction model unable to predict below esgoto + indireta (23.8 m³/h at Antanhol), so it is badly wrong every night.

**Decision (to try):** replace the single number with a small lookup table, esgoto by block of hours and weekday/weekend. Start coarse because there are few dry days. Recompute it every so often from recent dry days to follow the seasons. To avoid feedback loops:

- choose dry days by rain only;
- build the table from measured (or height-derived) caudal only, never from a prediction that already includes esgoto;
- when testing, build it from training days only;
- if there are no dry days in the window, keep the last good table.

**Still open:** why two dry days a week apart differed by a factor of two.

### 2. Indireta is constant

**What the code does:** one number per station, the single lowest 2-minute reading between 02:00 and 04:00 across all dry nights.

**Why it matters:** one odd reading sets it for everything, and groundwater really does rise after wet weeks. At Ameal it is 2.70 m³/h from the training days and 1.66 from all days.

**Idea:** let it vary slowly, from a typical night low smoothed over several days, and carry the last dry-night value through rainy nights. If we only report the total afluências indevidas (= caudal − esgoto), indireta does not affect that number; it only matters for the split, which we do want.

**What we do in the new method (`predict inflow/afluencias.py`):** we read the indireta on every night with less than 0.5 mm of rain in the previous 12 hours, as the lowest hourly caudal of the night, and draw a straight line between those nights. It can therefore rise after rain and fall again.

### 3. The night low is between 02:00 and 04:00

**Where it came from:** our professor gave us 02:00–04:00 as the window in which to read the night minimum, which we use as the indireta.

**What the data shows** (Jul–Oct 2026, mean caudal on dry days, m³/h):

| Hour | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
|---|---|---|---|---|---|---|---|---|---|
| Antanhol | 22.9 | 18.0 | 13.9 | 10.1 | 8.2 | 7.9 | **7.5** | 9.3 | 15.6 |
| Ameal | 3.65 | 3.21 | 2.77 | 2.71 | **2.70** | 2.76 | 2.99 | 4.34 | 4.67 |

- At Antanhol the low is later than the window. On 25 of 32 dry days the lowest hour is between 04:00 and 07:00, and at 02:00 the caudal is still almost double the low.
- At Ameal the caudal is flat from 02:00 to 06:00, so the window makes little difference there.

**Why it matters:** if we read the indireta between 02:00 and 04:00 at Antanhol, the value still contains sewage. Subtracting it gave a negative esgoto between 03:00 and 09:00, which is impossible.

**What we do for now:** we read the night low between **02:00 and 07:00**, which covers the low at both stations. This departs from the window our professor gave us, so we should discuss it with them. The window is the setting `HORAS_MADRUGADA` in `afluencias.py`.

**Questions for the professor:** was 02:00–04:00 a general rule of thumb, or specific to these sites? Is there a reason the low at Antanhol could be later, for example a pumping station upstream that empties in the early morning?
