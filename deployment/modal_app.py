"""Persistent public deployment: modal deploy -m deployment.modal_app.

The existing FastAPI app, model files, PDF fonts, and frontend are shipped
together. No tunnel, local computer, GPU, secret, or training dataset is used.
"""
import sys
import modal

image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('libgomp1')
)
if modal.is_local():
    from .runtime_bundle import SITE, runtime_files
    image = image.pip_install_from_requirements(str(SITE / 'requirements.txt'))
    for source, relative in runtime_files():
        image = image.add_local_file(source, f'/app/site/{relative}')

app = modal.App('tynys', image=image)


@app.function(
    cpu=1,
    memory=2048,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=60,
    timeout=300,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label='tynys')
def web():
    sys.path.insert(0, '/app/site')
    from app import app as website
    return website
