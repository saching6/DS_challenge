import pandas as pd
import numpy as np


def to_minutes(col):
    """Service day runs past midnight — hours reach 25+. pd.to_datetime fails on
    these, so parse by hand and leave values unwrapped so ordering stays correct."""
    parts = col.str.split(':', expand=True)
    return parts[0].astype(float) * 60 + parts[1].astype(float)



## Methods used by the Financial use case

def severity_table(train, min_n=10):
    """
    Median and P90 of major and minor (non-major) incidents grouped by
    symptom in historical/retrospective data. Symptoms with fewer than min_n samples
    get dropped and are reverted to a fallback of median values
    """
    def summarise(rows):
        t = rows.groupby('symptom', observed=True).incident_minutes.agg(
            n='count', median_inc_minutes='median', p90_inc_minutes=lambda s: s.quantile(0.9))
        # ignore small sample noise
        return t[t.n >= min_n]
 
    major_inc = summarise(train[train.label == 1])
    minor_inc = summarise(train[train.label == 0])
    
    fallback = {'major_median_inc_min': train[train.label == 1].incident_minutes.median(),
                'minor_median_inc_min': train[train.label == 0].incident_minutes.median(),
                'major_p90_inc_min': train[train.label == 1].incident_minutes.quantile(0.9)
                }
    return major_inc, minor_inc, fallback

def typical_minutes(df, table, column, fallback):
    """Map each row to its symptom's figure, filling low freq. symptoms with the
    global minutes value."""
    values = df.symptom.astype(str).map(table[column])
    return values.fillna(fallback).values

def expected_burden(df, p, tables):
    """Expected delay minutes for each incident.
    """
    major, minor, fb = tables
    major_median = typical_minutes(df, major, 'median_inc_minutes', fb['major_median_inc_min'])
    major_p90 = typical_minutes(df, major, 'p90_inc_minutes', fb['major_p90_inc_min'])
    minor_median = typical_minutes(df, minor, 'median_inc_minutes', fb['minor_median_inc_min'])
 
    expected_inc_min_median = p * major_median + (1 - p) * minor_median

    ## This isn't meant to be "the 90th percentile of expected burden." 
    #  It's meant to be "expected burden if the major branch goes badly"
    expected_inc_min_p90 = p * major_p90 + (1 - p) * minor_median

    return pd.DataFrame({'p_major': p, 'expected_min': expected_inc_min_median,
                         'expected_min_p90': expected_inc_min_p90}, index=df.index)


def exposure_by_line(mileage, start, end):
    """Total planned km per line over [start, end] must match
    the date range of the incidents"""
    m = mileage[(mileage.service_day >= start) & (mileage.service_day <= end)]
    return m.groupby('line').planned_km.sum()
 
 
def burden_per_km_validation(df, p, tables, mileage):
    """Validation view: does the model's PREDICTED burden-per-km rank lines the
    same way the ACTUAL burden-per-km
    """
    burden = expected_burden(df, p, tables)
    line = df['line_primary'].astype(str)
    km = exposure_by_line(mileage, df.service_day.min(), df.service_day.max())
    out = pd.DataFrame({'actual_min': df.incident_minutes.groupby(line.values).sum(),
                        'predicted_min': burden.expected_min.groupby(line.values).sum(),
                        'predicted_min_with_p90': burden.expected_min_p90.groupby(line.values).sum(),
                        'planned_km': km}).dropna()
    out['actual_per_100k'] = (1e5 * out.actual_min / out.planned_km).round(1)
    out['pred_per_100k'] = (1e5 * out.predicted_min / out.planned_km).round(1)
    out['pred_p90_per_100k'] = (1e5 * out.predicted_min_with_p90 / out.planned_km).round(1)
    out['actual_rank'] = out.actual_per_100k.rank(ascending=False).astype(int)
    out['pred_rank'] = out.pred_per_100k.rank(ascending=False).astype(int)
    out['pred_p90_rank'] = out.pred_p90_per_100k.rank(ascending=False).astype(int)
    out = out.sort_values('actual_per_100k', ascending=False)
    rank_corr = out.actual_per_100k.corr(out.pred_per_100k, method='spearman')
    rank_corr_p90 = out.actual_per_100k.corr(out.pred_p90_per_100k, method='spearman')
    return out[['actual_per_100k', 'pred_per_100k', 'pred_p90_per_100k','actual_rank',
                'pred_rank', 'pred_p90_rank']], rank_corr, rank_corr_p90


