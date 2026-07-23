# Data Profiling — Full Findings

Detailed findings from [`stm_data_profiling.ipynb`](../stm_data_profiling.ipynb).
Summary and context: [README](../README.md#3-data-profiling).

The notebook works through nine steps:

| Step | Focus |
|---|---|
| 1 | Load the translated files (as text, to preserve missingness tokens) |
| 2 | Missingness — three distinct kinds |
| 3 | Categorical levels |
| 4 | Service-time parsing and duration-band consistency |
| 5 | Multi-line incidents |
| 6 | Cleaning the mileage file |
| 7 | Merging (date handling and the `route` column) |
| 8 | Rates, and picking the right denominator |
| 9 | Visualisation |

## Data quality

1. `incident_duration_band` is train-only in practice. Every station incident is
   filed as `02 min and under` despite a median real duration of roughly
   41 minutes.
2. Times run past 24:00 — `calendar_day` is a **service day**, not a calendar
   day. `pd.to_datetime` fails on these; hours are parsed manually and left
   unwrapped so that subtraction still works across midnight.
3. `primary_cause = 'other'` is largely an artefact of how station incidents are
   recorded — the published methodology confirms station incidents receive no
   cause analysis, so `primary_cause` and `secondary_cause` are effectively
   train-only fields.
4. `calendar_year_month` is populated in both files but formatted differently,
   so it cannot be used as a join key as-is. Month is derived from the date
   column in both files instead.
5. The mileage file contains zero-value padding rows that duplicate 2019 and
   early 2020.

## Reporting practices and classification consistency

These findings are about how incidents are *recorded* rather than what happened.
They are separated out because they bear directly on the data-collection
recommendations.

- **Missingness is not one thing.** `N/A` (field doesn't apply),
  `unassigned` (field applies, nobody filled it in), and an empty cell (a real
  null) mean different things and are counted separately. Files are read with
  `keep_default_na=False` so the distinction survives loading, and missingness
  is read *per incident type* — `hardware_type` looks 91% missing overall, but
  station incidents have no train, so the field cannot apply.
- **Completeness degrades year on year** for `route`, `vehicle`, and
  `hardware_type`.
- **Incidents cluster in time, and the data cannot say why.** Grouping by line
  and service day with a 10-minute window surfaces sequences of separate
  incident numbers opened close together. Nothing in the schema distinguishes a
  single event fragmented across several records from a genuine cascade where
  one fault triggers the next. One incident number is therefore not reliably one
  interruption. This is left in the profiling deliberately: it is a gap in
  collection rather than a finding about the network, and closing it —
  by linking related incident numbers at the point of entry — would improve
  prediction accuracy directly, since duplicates and cascades carry opposite
  implications for how a live incident should be escalated. Carried forward to
  the data-collection recommendations.
- **Band-versus-clock agreement drifts by year**, and some rows filed as
  `02 min and under` carry clock durations of several hours — tickets left open
  rather than short incidents.

## Joining

- **`route` is not a join key.** The mileage file uses a single value, `001`;
  the incidents file uses 168 distinct values spanning 001–779, which are train
  run numbers. A sanity check against published line lengths shows `001` implies
  thousands of one-way trips per day and an annual total matching STM's
  network-wide published figure — so it denotes all revenue service, not one
  run. Joining on it retains roughly 2% of rows and matches those to the wrong
  mileage.
- **Join on line + day.** Mileage stops September 2023; incidents run to
  September 2025.
- **Line-days with mileage but no incidents are real zeros** and must be filled,
  not dropped.
- **Multi-line incidents are exploded, not dropped.** They are 0.8% of rows but
  10% of major incidents — a 13× enrichment. One row per affected line, full
  credit to each, with `n_lines_affected` retained so credit can be made
  fractional later if cost has to be attributed without double-counting. The
  103 `unassigned`-line rows are held out of line-level analysis.

## Signal

- **Ranking lines by raw count is misleading.** Orange carries the highest
  incident counts but among the lowest rates per planned km. Ranking by count
  points resources at the busiest line rather than the least reliable one.
- **Major incidents are under 2% of rows**, so any model must handle class
  imbalance.
- **`emergency_metro`, `evacuation`, and `material_damage` are strong escalation
  signals
- **Volume and severity peak at different times.** Incident volume peaks at rush
  hour, but the share running 20+ minutes peaks late in the service day, when
  recovery options are thin. An escalation rule keyed to volume would look in
  the wrong place.
- **Major incidents concentrate by location**, giving a short list of sites that
  account for a disproportionate share.
- **Planned mileage is a fair yardstick across lines and years, but a poor one
  day to day.** The pooled correlation with incident counts is strong; within
  each line it nearly vanishes, explaining only a few percent of daily variance.
  Two limitations drive this:
  - It records *intent*. Through the 2020 shutdown, planned km barely moved
    while incidents fell 44% — planned mileage cannot see a pandemic.
  - Passenger-driven incidents do not scale with train kilometres at all. Train
    faults roughly halve at weekends, tracking reduced service; passenger
    incidents barely move, so weekends look less reliable than they are.

  This motivates **splitting the denominator**: rolling stock, train operations,
  and fixed equipment normalised per 100k planned km; customer and station
  operations reported as counts per day, since no ridership data is available.