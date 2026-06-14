from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

traffic_source = FileSource(
    path="data/traffic_features.parquet",
    timestamp_field="event_timestamp",
)

traffic_entity = Entity(
    name="traffic_record",
    join_keys=["traffic_record_id"],
)

traffic_features = FeatureView(
    name="traffic_features",
    entities=[traffic_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="lane_count", dtype=Int64),
        Field(name="speed_limit_kmph", dtype=Int64),
        Field(name="blackspot_score", dtype=Float32),
        Field(name="vehicle_count_per_hr", dtype=Float32),
        Field(name="avg_speed_kmph", dtype=Float32),
        Field(name="green_duration_s", dtype=Float32),
        Field(name="red_duration_s", dtype=Float32),
        Field(name="yellow_duration_s", dtype=Float32),
        Field(name="cycle_time_s", dtype=Float32),
        Field(name="violations_count", dtype=Float32),
        Field(name="speed_ratio", dtype=Float32),
        Field(name="traffic_density", dtype=Float32),
        Field(name="signal_efficiency", dtype=Float32),
        Field(name="is_weekend", dtype=Int64),
        Field(name="is_night", dtype=Int64),
    ],
    source=traffic_source,
)