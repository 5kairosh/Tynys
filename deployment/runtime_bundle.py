"""Select only the public runtime files needed by Tynys cloud deployments."""
import hashlib
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SITE = REPOSITORY / 'site'
PYTHON_FILES = (
    'app.py', 'assessment.py', 'calibration.py', 'dicom_input.py',
    'ensemble.py', 'image_quality.py', 'ood.py', 'pad_geometry.py',
    'pdf_report.py', 'selective_evaluation.py', 'service_runtime.py',
    'upload_memory.py',
)
MODEL_FILES = (
    'cnn_best.keras', 'external_validation.json', 'image_quality_config.json',
    'mcnemar.json', 'model_config.json', 'ood_params.npz',
    'pneumonia_mobilenetv2.keras', 'project_summary.txt',
    'selective_evaluation.json', 'test_metrics.csv',
)
PUBLIC_SUFFIXES = {'.html', '.js', '.css', '.png', '.jpg', '.jpeg', '.webp', '.gif', '.svg', '.json'}


def runtime_files(site=SITE):
    site = Path(site).resolve(strict=True)
    paths = [site / name for name in PYTHON_FILES]
    paths += [site / 'requirements.txt']
    paths += [site / 'model' / name for name in MODEL_FILES]
    paths += [site / 'fonts' / name for name in ('NotoSans-Regular.ttf', 'NotoSans-Bold.ttf', 'LICENSE.txt')]
    paths += [p for p in (site / 'static').rglob('*') if p.is_file() and p.suffix.lower() in PUBLIC_SUFFIXES and not any(part.startswith('.') for part in p.relative_to(site).parts)]
    result = []
    for path in sorted(set(paths)):
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(site):
            raise ValueError(f'Runtime file is outside site/: {path.name}')
        result.append((resolved, path.relative_to(site).as_posix()))
    return result


def manifest(site=SITE):
    files = runtime_files(site)
    return {'file_count': len(files), 'total_bytes': sum(p.stat().st_size for p, _ in files),
            'files': {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in files}}
