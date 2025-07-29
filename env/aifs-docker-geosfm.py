import coiled 

coiled.create_software_environment(
    name="flashattn-dockerv1",
    workspace='geosfm', 
    container='us-east1-docker.pkg.dev/e4drr-crafd/coiled/flash-attn-notebook:v1.0'
)
