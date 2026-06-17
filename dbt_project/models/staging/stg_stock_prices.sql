{{
  config(
    materialized        = 'incremental',
    unique_key          = 'stock_date_key',
    partition_by        = {
      "field"           : "trade_date",
      "data_type"       : "date",
      "granularity"     : "month"
    },
    cluster_by          = ["ticker"],
    incremental_strategy= 'merge',
    on_schema_change    = 'sync_all_columns'
  )
}}

with source as (

    select
        *,
        -- Deduplicate within each file: keep the most recently ingested row
        -- per ticker+day in case the DAG is re-run intraday.
        row_number() over (
            partition by ticker, date(timestamp(datetime))
            order by timestamp(ingested_at) desc
        ) as _row_num
    from {{ source('raw', 'stock_prices') }}

    {% if is_incremental() %}
        -- On incremental runs only process partitions newer than what we have.
        where date(timestamp(datetime)) > (select max(trade_date) from {{ this }})
    {% endif %}

),

deduped as (

    select * from source where _row_num = 1

),

renamed as (

    select
        -- Surrogate key — stable across re-runs
        {{ dbt_utils.generate_surrogate_key(['ticker', 'datetime']) }}
                                                            as stock_date_key,

        ticker,
        date(timestamp(datetime))                           as trade_date,
        timestamp(datetime)                                 as trade_timestamp,

        -- OHLCV with defensive casts from JSON-sourced strings
        safe_cast(open   as float64)                        as open_price,
        safe_cast(high   as float64)                        as high_price,
        safe_cast(low    as float64)                        as low_price,
        safe_cast(close  as float64)                        as close_price,
        safe_cast(volume as int64)                          as volume,

        -- Derived columns computed once here so marts stay clean
        safe_divide(
            safe_cast(close as float64) - safe_cast(open as float64),
            safe_cast(open  as float64)
        )                                                   as daily_return_pct,

        safe_cast(high as float64)
            - safe_cast(low as float64)                     as intraday_range,

        -- Audit
        timestamp(ingested_at)                              as ingested_at,
        current_timestamp()                                 as dbt_updated_at

    from deduped

)

select * from renamed
