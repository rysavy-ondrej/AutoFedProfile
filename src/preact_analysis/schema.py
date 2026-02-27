import pyarrow as pa

dataset_schema = pa.schema([
    pa.field("bs", pa.int64()),
    pa.field("ps", pa.int64()),
    pa.field("br", pa.int64()),
    pa.field("pr", pa.int64()),
    pa.field("dp", pa.int64()),
    pa.field("sp", pa.int64()),
    pa.field("da", pa.string(), nullable=True),
    pa.field("sa", pa.string(), nullable=True),
    pa.field("ts", pa.float64()),
    pa.field("td", pa.float64()),
    pa.field("tls.cver", pa.string(), nullable=True),
    pa.field("tls.sver", pa.string(), nullable=True),
    pa.field("tls.sext", pa.list_(pa.string())),
    pa.field("tls.csg", pa.list_(pa.string())),
    pa.field("tls.ccs", pa.list_(pa.string())),
    pa.field("tls.cext", pa.list_(pa.string())),
    pa.field("tls.ssv", pa.list_(pa.string())),
    pa.field("tls.csv", pa.list_(pa.string())),
    pa.field("tls.scs", pa.string(), nullable=True),
    pa.field("tls.alpn", pa.list_(pa.string())),
    pa.field("tls.sni", pa.string(), nullable=True),
    pa.field("tls.ja3", pa.string(), nullable=True),
    pa.field("tls.ja4", pa.string(), nullable=True),
    pa.field("tls.ja3s", pa.string(), nullable=True),
    pa.field("tls.ja4s", pa.string(), nullable=True),
    pa.field("meta.sample.id", pa.string(), nullable=True),
    pa.field("meta.malware.family", pa.string(), nullable=True),
    pa.field("meta.system.os", pa.string(), nullable=True),
    pa.field("meta.system.service", pa.string(), nullable=True),
    pa.field("meta.application.name", pa.string(), nullable=True),
    pa.field("meta.application.process", pa.string(), nullable=True),
    pa.field("tls.rec", pa.list_(pa.int64())),
])

def get_numeric_field_names(schema: pa.Schema) -> list[str]:
    """
    Return names of all numeric fields (int, uint, float).
    """
    numeric_types = (
        pa.types.is_integer,
        pa.types.is_floating,
        pa.types.is_decimal,
    )

    return [
        field.name
        for field in schema
        if any(check(field.type) for check in numeric_types)
    ]


def get_categorical_field_names(schema: pa.Schema) -> list[str]:
    """
    Return names of all string / categorical-like fields.
    """
    return [
        field.name
        for field in schema
        if pa.types.is_string(field.type)
        or pa.types.is_large_string(field.type)
        or pa.types.is_dictionary(field.type)
    ]


def dataset_summary(dataset):
    print("=== DATASET SUMMARY ===")
    print(f"Format: {dataset.format}")
    print(f"Columns: {len(dataset.schema)}")
    print(f"Column names: {dataset.schema.names}")
    fragments = list(dataset.get_fragments())
    print(f"Files: {len(fragments)}")        
    total_rows = dataset.count_rows()
    print(f"Total rows: {total_rows:,}")
