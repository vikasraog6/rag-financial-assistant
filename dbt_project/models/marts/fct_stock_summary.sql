{{
  config(
    materialized  = 'table',
    partition_by  = {
      "field"       : "trade_date",
      "data_type"   : "date",
      "granularity" : "month"
    },
    cluster_by    = ["ticker"],
    labels        = {"domain": "finance", "tier": "mart", "rag_source": "true"}
  )
}}

with prices as (

    select * from {{ ref('stg_stock_prices') }}

),

windowed as (

    select
        *,

        -- ── Moving averages ────────────────────────────────────────────────
        avg(close_price) over (
            partition by ticker
            order by trade_date
            rows between 6 preceding and current row
        )                                               as ma_7d,

        avg(close_price) over (
            partition by ticker
            order by trade_date
            rows between 29 preceding and current row
        )                                               as ma_30d,

        avg(close_price) over (
            partition by ticker
            order by trade_date
            rows between 89 preceding and current row
        )                                               as ma_90d,

        -- ── Risk metrics ───────────────────────────────────────────────────
        stddev(daily_return_pct) over (
            partition by ticker
            order by trade_date
            rows between 29 preceding and current row
        )                                               as volatility_30d,

        -- ── Volume baseline ────────────────────────────────────────────────
        avg(volume) over (
            partition by ticker
            order by trade_date
            rows between 29 preceding and current row
        )                                               as avg_volume_30d,

        -- ── Lag prices for return calculations ────────────────────────────
        lag(close_price, 1)  over (partition by ticker order by trade_date) as prev_close,
        lag(close_price, 7)  over (partition by ticker order by trade_date) as close_7d_ago,
        lag(close_price, 30) over (partition by ticker order by trade_date) as close_30d_ago,

        row_number() over (
            partition by ticker order by trade_date desc
        )                                               as recency_rank

    from prices

),

enriched as (

    select
        stock_date_key,
        ticker,
        trade_date,
        trade_timestamp,

        -- ── OHLCV ─────────────────────────────────────────────────────────
        open_price,
        high_price,
        low_price,
        close_price,
        volume,

        -- ── Returns ───────────────────────────────────────────────────────
        daily_return_pct,
        safe_divide(close_price - close_7d_ago,  close_7d_ago)  as return_7d_pct,
        safe_divide(close_price - close_30d_ago, close_30d_ago) as return_30d_pct,

        -- ── Intraday range ────────────────────────────────────────────────
        intraday_range,
        safe_divide(intraday_range, close_price)                 as intraday_range_pct,

        -- ── Moving averages ───────────────────────────────────────────────
        round(ma_7d,  2) as ma_7d,
        round(ma_30d, 2) as ma_30d,
        round(ma_90d, 2) as ma_90d,

        -- ── Trend signal (Golden/Death Cross proxy) ───────────────────────
        case
            when close_price > ma_7d  and ma_7d  > ma_30d then 'bullish'
            when close_price < ma_7d  and ma_7d  < ma_30d then 'bearish'
            else 'neutral'
        end                                                      as trend_signal,

        -- ── Volatility ────────────────────────────────────────────────────
        round(volatility_30d, 6)                                 as volatility_30d,
        case
            when volatility_30d > 0.030 then 'high'
            when volatility_30d > 0.015 then 'medium'
            else 'low'
        end                                                      as volatility_bucket,

        -- ── Volume ────────────────────────────────────────────────────────
        round(avg_volume_30d)                                    as avg_volume_30d,
        round(safe_divide(volume, avg_volume_30d), 3)            as volume_ratio,

        -- ── Freshness flag ────────────────────────────────────────────────
        recency_rank = 1                                         as is_latest,

        ingested_at,
        current_timestamp()                                      as dbt_updated_at

    from windowed

)

select * from enriched
