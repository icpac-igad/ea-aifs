import coiled

# lightweight environment for ETL tasks
coiled.create_software_environment(
    name="aifs-etl-v2",
    workspace='geosfm',
    pip="requirements-etl.txt",
)
