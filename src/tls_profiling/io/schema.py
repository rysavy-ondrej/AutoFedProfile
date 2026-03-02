import pyarrow as pa

def get_tls_schema(version: str="v1") -> pa.Schema:
    """
    Build and return the project TLS dataset schema.

    Parameters
    ----------
    version:
        Schema version identifier. Only ``"v1"`` is currently supported.

    Returns
    -------
    pa.Schema
        PyArrow schema describing flow, TLS, and metadata columns.

    Raises
    ------
    Exception
        If an unsupported schema version is requested.
    """
    if version != "v1": 
        raise Exception(f"Unsupported version {version}")
    # the schema of annotated TLS connection parquet data files
    tls_schema = pa.schema([
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
            pa.field("tls.rec", pa.list_(pa.int64()))
        ])
    return tls_schema



def _get_numeric_field_names(schema: pa.Schema) -> list[str]:
    """
    Return names of numeric fields from a schema.

    Numeric fields include integer, floating point, and decimal types.

    Parameters
    ----------
    schema:
        Input schema to inspect.

    Returns
    -------
    list[str]
        Names of all numeric columns.
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


def _get_categorical_field_names(schema: pa.Schema) -> list[str]:
    """
    Return names of categorical-like fields from a schema.

    Categorical-like fields are treated as string, large string,
    or dictionary-encoded columns.

    Parameters
    ----------
    schema:
        Input schema to inspect.

    Returns
    -------
    list[str]
        Names of categorical-like columns.
    """
    return [
        field.name
        for field in schema
        if pa.types.is_string(field.type)
        or pa.types.is_large_string(field.type)
        or pa.types.is_dictionary(field.type)
    ]

def _get_field_names_by_prefix(schema: pa.Schema, prefix: str) -> list[str]:
    """
    Return names of all fields whose name starts with a prefix.

    Parameters
    ----------
    schema:
        Input schema to inspect.
    prefix:
        Prefix to match (for example ``"tls."`` or ``"meta."``).

    Returns
    -------
    list[str]
        Names of fields matching the prefix.
    """
    return [
        field.name
        for field in schema
        if field.name.startswith(prefix)
    ]

def list_columns_by_role(schema: pa.Schema, role: str) -> list[str] | None:
    """
    Return a column list for a predefined feature role.

    Supported roles:
    - ``"numeric"``: numeric columns inferred from schema dtypes
    - ``"categorical"``: string/dictionary-like columns
    - ``"tls"``: columns prefixed with ``"tls."``
    - ``"meta"``: columns prefixed with ``"meta."``
    - ``"flow"``: predefined base flow fields

    Parameters
    ----------
    schema:
        Input schema to inspect.
    role:
        Column role selector.

    Returns
    -------
    list[str] | None
        Matching column names, or ``None`` for unsupported role values.
    """
    if role =="numeric": return _get_numeric_field_names(schema)
    if role =="categorical": return _get_categorical_field_names(schema)
    if role =="tls": return _get_field_names_by_prefix(schema, "tls.")
    if role =="meta": return _get_field_names_by_prefix(schema, "meta.")
    if role =="flow": return  ["ps", "br", "pr", "dp", "sp", "da", "sa", "ts", "td"]
    return None

