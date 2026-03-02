# TLS Profiling 

The code structure:
```
    tls_profiling/
      __init__.py

      config/                # defaults + schemas
        __init__.py
        defaults.yaml
        logging.yaml

      io/                    # loading/saving datasets, parquet schemas
        __init__.py
        readers.py
        writers.py
        schema.py

      preprocessing/         # your folder -> library module
        __init__.py
        cleaning.py
        parsing.py
        filtering.py
        feature_engineering.py
        windows.py           # session/windowization for time-based / flow-based

      exploration/           # your folder -> EDA utilities (not notebooks)
        __init__.py
        cardinality.py       # get_df_tls_field_card, etc.
        summaries.py
        drift.py

      baselines/             # your folder -> classical models + rules
        __init__.py
        ja4.py               # if you add fingerprint baseline
        stats_models.py
        sklearn_models.py
        evaluation.py

      autoencoder/           # your folder -> AE models, training, inference
        __init__.py
        datasets.py          # tf.data builders, batching, padding
        models.py            # architecture definitions
        losses.py
        train.py
        infer.py             # embeddings/anomaly scores
        calibration.py       # thresholds, quantiles
        evaluation.py

      visualization/         # your folder -> plotting/reporting
        __init__.py
        styles.py
        tls_plots.py
        timeseries_plots.py
        report_plots.py

      metrics/               # shared evaluation metrics across approaches
        __init__.py
        classification.py
        anomaly.py
        ranking.py

      utils/
        __init__.py
        typing.py
        hashing.py           # stable hashing for array fields, tuples, etc.
        time.py
        pandas_helpers.py
        polars_helpers.py
```