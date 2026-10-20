"""Prepare a fixed patient-separated RSNA proxy-label cohort using HTTP ZIP ranges.

Original pixels/metadata remain in ignored cache. Public manifests contain only
source IDs, labels, hashes, split and attribution, never uploaded user images.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import zipfile
import zlib

import requests
from download_validation import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'training/generalization_stage6'
CACHE = ROOT / 'training/cache/external_stage6'
SOURCE = 'https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/'
ARCHIVE = SOURCE + 'pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip'
ANNOTATIONS = 'pneumonia-challenge-annotations-adjudicated-kaggle_2018.json'
MAPPING = 'pneumonia-challenge-dataset-mappings_2018.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metadata(name):
    path = CACHE / name
    if not path.exists():
        response = requests.get(SOURCE + name, timeout=(15, 90))
        response.raise_for_status()
        temporary = path.with_suffix('.part')
        temporary.write_bytes(response.content); temporary.replace(path)
    return json.loads(path.read_text(encoding='utf-8'))


def select_cohort(annotations, mapping, protocol):
    calculated = next(g for g in annotations['labelGroups'] if g['name'] == 'Calculated')
    names = {label['id']: label['name'].strip() for label in calculated['labels']}
    labels = {}
    for item in annotations['datasets'][0]['annotations']:
        name = names.get(item['labelId'])
        if name:
            labels.setdefault(item['StudyInstanceUID'], set()).add(name)
    groups = {}
    exclusions = {}
    for row in sorted(mapping, key=lambda r: r['subset_img_id']):
        terms = labels.get(row['StudyInstanceUID'], set())
        if terms == {'Normal'}:
            label = 0
        elif terms == {'Lung Opacity'}:
            label = 1
        else:
            reason = 'non_normal_other_finding' if terms == {'No Lung Opacity / Not Normal'} else 'missing_or_ambiguous_calculated_label'
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        original = Path(row['img_id']).stem
        patient = original.split('_')[0]
        if len(patient) != 8 or not patient.isdigit():
            raise ValueError('Invalid original NIH patient ID.')
        groups.setdefault(patient, []).append({
            'source_id': row['subset_img_id'], 'patient_group': 'nih:' + patient,
            'nih_image': row['img_id'], 'label': label,
            'target': 'NORMAL' if label == 0 else 'RSNA_LUNG_OPACITY',
            'source': 'RSNA-STR 2018 / NIH Clinical Center',
        })
    rng = random.Random(protocol['seed'])
    # Sample one examination per patient before class quotas, including patients
    # whose finding changed between studies. No label-based patient cherry-pick.
    unique = [rng.choice(groups[key]) for key in sorted(groups)]
    pools = [[r for r in unique if r['label'] == label] for label in (0, 1)]
    for pool in pools:
        rng.shuffle(pool)
    needed = sum(protocol['rsna_counts_per_class'].values())
    if min(map(len, pools)) < needed:
        raise ValueError('Insufficient original NIH patients for frozen quotas.')
    splits = {}; position = 0
    for split, count in protocol['rsna_counts_per_class'].items():
        splits[split] = [{**r, 'split': split} for pool in pools for r in pool[position:position+count]]
        rng.shuffle(splits[split]); position += count
    all_rows = sum(splits.values(), [])
    assert len({r['patient_group'] for r in all_rows}) == len(all_rows)
    return splits, {'eligible_patients': len(unique), 'eligible_class_counts': list(map(len, pools)), 'exclusions': exclusions}


class OfficialZip(RemoteZip):
    def __init__(self):
        response = requests.head(ARCHIVE, timeout=(15, 45))
        response.raise_for_status()
        self.url = ARCHIVE
        self.size = int(response.headers['Content-Length'])
        self.position = 0


def download_images(splits, workers):
    remote = OfficialZip()
    index_path = CACHE / 'zip_index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        assert index['archive_size'] == remote.size
    else:
        with zipfile.ZipFile(remote) as archive:
            entries = [{key: getattr(info, key) for key in ('filename','header_offset','compress_size','compress_type','file_size','CRC')}
                       for info in archive.infolist() if not info.is_dir()]
        index = {'archive_size': remote.size, 'entries': entries}
        index_path.write_text(json.dumps(index), encoding='utf-8')
    # The official RSNA ZIP uses DICOM SOP UIDs, whereas Kaggle uses UUIDs.
    # Resolve through the publisher's mapping, never by directory order.
    by_instance = {Path(row['filename']).stem: row for row in index['entries']}
    entries = {row['subset_img_id']: by_instance[row['SOPInstanceUID']]
               for row in metadata(MAPPING)}
    sys.path.insert(0, str(ROOT / 'site'))
    from dicom_input import dicom_to_png
    image_dir = CACHE / 'images'; image_dir.mkdir(exist_ok=True)
    def fetch(row):
        path = image_dir / (row['source_id'] + '.png')
        digest_path = path.with_suffix('.sha256')
        if path.exists() and digest_path.exists() and sha(path) == digest_path.read_text():
            return {**row, 'sha256': sha(path)}
        entry = entries[row['source_id']]
        data = remote.range(entry['header_offset'], entry['compress_size'] + 65536)
        header = struct.unpack('<4s5H3I2H', data[:30])
        if header[0] != b'PK\x03\x04':
            raise ValueError('Invalid ZIP local header.')
        offset = 30 + header[-2] + header[-1]
        block = data[offset:offset+entry['compress_size']]
        raw = zlib.decompress(block, -15) if entry['compress_type'] == 8 else block
        if len(raw) != entry['file_size'] or zlib.crc32(raw) != entry['CRC']:
            raise ValueError('RSNA DICOM CRC/length mismatch.')
        converted = dicom_to_png(raw)
        temporary = path.with_suffix('.part'); temporary.write_bytes(converted); temporary.replace(path)
        digest_path.write_text(sha(path), encoding='ascii')
        return {**row, 'sha256': sha(path)}
    rows = sum(splits.values(), []); results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, row): row for row in rows}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result(); results[result['source_id']] = result
            if i % 100 == 0 or i == len(rows):
                print(f'RSNA verified images {i}/{len(rows)}', flush=True)
    resolved = {split: [results[r['source_id']] for r in part] for split, part in splits.items()}
    hashes = [r['sha256'] for r in sum(resolved.values(), [])]
    assert len(hashes) == len(set(hashes)), 'Duplicate pixels across frozen RSNA cohort; stop before fitting.'
    prior = set()
    for path in [*(ROOT/'training/manifests').glob('*_manifest.csv'), ROOT/'training/external_predictions.csv']:
        with path.open(encoding='utf-8-sig') as stream:
            prior.update(r['sha256'] for r in csv.DictReader(stream))
    assert not prior.intersection(hashes), 'Existing cohort/RSNA exact-byte overlap.'
    return resolved, remote.size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=range(1,9), default=6)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); OUT.mkdir(exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((OUT/'protocol.json').read_text())
    splits, audit = select_cohort(metadata(ANNOTATIONS), metadata(MAPPING), protocol)
    selection_path = OUT / 'selection.json'
    selection = {k: [{key: r[key] for key in ('source_id','patient_group','label','split')} for r in rows] for k, rows in splits.items()}
    if selection_path.exists():
        assert json.loads(selection_path.read_text()) == selection, 'Frozen selection changed.'
    else:
        selection_path.write_text(json.dumps(selection, indent=2)+'\n', encoding='utf-8')
    print('Frozen selection: ' + json.dumps({k: len(v) for k,v in splits.items()}), flush=True)
    if args.prepare_only:
        return
    resolved, archive_size = download_images(splits, args.workers)
    for split, rows in resolved.items():
        write_csv(OUT/(split+'_manifest.csv'), rows)
    audit.update({'source': ARCHIVE, 'archive_size': archive_size, 'downloaded_images': sum(map(len, resolved.values())),
                  'counts': {k: len(v) for k,v in resolved.items()}, 'one_image_per_verified_nih_patient': True,
                  'sha256_overlap_prior_cohorts': 0, 'label_definition': protocol['target_definition'],
                  'annotation_sha256': sha(CACHE/ANNOTATIONS), 'mapping_sha256': sha(CACHE/MAPPING),
                  'image_conversion': 'Existing site.dicom_input.dicom_to_png, then same model preprocessing',
                  'attribution': 'NIH Clinical Center; Wang et al., CVPR 2017. RSNA-STR; Shih et al., Radiology AI 2019, doi:10.1148/ryai.2019180041.',
                  'terms_url': 'https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018'})
    (OUT/'data_report.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print('RSNA_COHORT_READY', flush=True)


if __name__ == '__main__':
    main()
