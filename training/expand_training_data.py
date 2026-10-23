"""Prepare new patient-separated development data and an independent-source pilot.

No prediction is computed here. Medical pixels and PadChest metadata remain in
ignored cache; only anonymous RSNA manifests and aggregate statistics are public.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import struct
import zipfile
import zlib

import requests

from prepare_rsna_stage6 import (ROOT, CACHE, metadata, ANNOTATIONS, MAPPING,
    select_cohort, download_images, write_csv, sha)
from download_validation import RemoteZip

OUT = ROOT / 'training/data_expansion_20261023'
PAD = ROOT / 'training/cache/padchest_pilot'


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def fresh_rsna(protocol, previous):
    pool_protocol = {'seed': protocol['rsna_selection']['original_selection_seed'],
                     'rsna_counts_per_class': {'all': protocol['rsna_selection']['pool_per_class']}}
    all_rows, _ = select_cohort(metadata(ANNOTATIONS), metadata(MAPPING), pool_protocol)
    used = {row['patient_group'] for row in previous}
    pools = [[r for r in all_rows['all'] if r['label'] == label and r['patient_group'] not in used] for label in (0, 1)]
    rng = random.Random(protocol['seed'])
    for pool in pools:
        rng.shuffle(pool)
    quotas = protocol['rsna_selection']['new_counts_per_class']
    if min(map(len, pools)) < sum(quotas.values()):
        raise ValueError('Insufficient fresh patients; do not borrow old evaluation patients.')
    splits = {}; position = 0
    for split, count in quotas.items():
        splits[split] = [{**r, 'split': split} for pool in pools for r in pool[position:position+count]]
        rng.shuffle(splits[split]); position += count
    return splits


class PadChestZip(RemoteZip):
    def __init__(self):
        with requests.get('https://www.kaggle.com/api/v1/datasets/download/raddar/padchest-chest-xrays-sample',
                          params={'datasetVersionNumber': 1}, stream=True, timeout=(20, 45)) as response:
            response.raise_for_status()
            self.url = response.url
            self.size = int(response.headers['Content-Length'])
            if not 0 < self.size <= 250 * 1024 * 1024:
                raise ValueError('Unexpected sample archive size.')
        self.position = 0


def member_bytes(remote, entry):
    data = remote.range(entry.header_offset, entry.compress_size + 65536)
    header = struct.unpack('<4s5H3I2H', data[:30])
    if header[0] != b'PK\x03\x04':
        raise ValueError('Invalid ZIP member header.')
    start = 30 + header[-2] + header[-1]
    compressed = data[start:start+entry.compress_size]
    raw = zlib.decompress(compressed, -15) if entry.compress_type == 8 else compressed
    if len(raw) != entry.file_size or zlib.crc32(raw) != entry.CRC:
        raise ValueError('PadChest CRC/length mismatch.')
    return raw


def padchest_candidates(rows):
    groups = {}; excluded = Counter()
    for row in rows:
        terms = {str(t).strip().lower() for t in ast.literal_eval(row['Labels'])}
        if row['Projection'] not in {'PA', 'AP', 'AP_horizontal'}:
            excluded['non_frontal'] += 1; continue
        if 'pneumonia' in terms and 'normal' not in terms:
            label = 1
        elif terms == {'normal'}:
            label = 0
        else:
            excluded['other_or_ambiguous'] += 1; continue
        if not row.get('PatientID'):
            raise ValueError('No patient grouping ID: do not invent independent patients.')
        groups.setdefault(row['PatientID'], []).append((row, label))
    selected = [min(values, key=lambda item: hashlib.sha256(item[0]['ImageID'].encode()).hexdigest())
                for _, values in sorted(groups.items())]
    return selected, dict(excluded)


def prepare_padchest():
    PAD.mkdir(parents=True, exist_ok=True)
    report_path = OUT/'padchest_data_report.json'
    manifest_path = PAD/'manifest.csv'
    if report_path.exists() and manifest_path.exists():
        report = json.loads(report_path.read_text())
        cached = read_rows(manifest_path)
        if sha(manifest_path) == report['manifest_sha256'] and all(sha(ROOT/r['path']) == r['sha256'] for r in cached):
            return report
    remote = PadChestZip()
    with zipfile.ZipFile(remote) as archive:
        entries = {Path(info.filename).name: info for info in archive.infolist() if not info.is_dir()}
    metadata_bytes = member_bytes(remote, entries['chest_x_ray_images_labels_sample.csv'])
    rows = list(csv.DictReader(io.StringIO(metadata_bytes.decode('utf-8-sig'))))
    selected, excluded = padchest_candidates(rows)
    images = PAD / 'images'; images.mkdir(exist_ok=True)
    manifest = []
    from PIL import Image
    for row, label in selected:
        name = row['ImageID']
        path = images / name
        # Read only exact members selected from the publisher-linked sample.
        raw = member_bytes(remote, entries[name])
        digest = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as image:
            image.load(); width, height = image.size
        path.write_bytes(raw)
        manifest.append({'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'label': label,
            'sha256': digest, 'patient_group': hashlib.sha256(('padchest:'+row['PatientID']).encode()).hexdigest(),
            'method_label': row['MethodLabel'], 'projection': row['Projection'], 'width': width, 'height': height})
    if len({r['sha256'] for r in manifest}) != len(manifest):
        raise ValueError('Duplicate pilot images.')
    write_csv(PAD / 'manifest.csv', manifest)  # Never redistribute rows/reports/pixels.
    return {'source': 'PadChest official-linked public sample', 'archive_bytes': remote.size,
        'sample_rows': len(rows), 'eligible_images': len(manifest),
        'class_counts': dict(Counter(r['label'] for r in manifest)), 'exclusions': excluded,
        'method_label_counts': dict(Counter(r['method_label'] for r in manifest)),
        'role': 'external descriptive pilot only; never training or selection',
        'license': 'PadChest research use agreement; no raw data redistribution',
        'terms_url': 'https://bimcv.cipf.es/bimcv-projects/padchest/padchest-dataset-research-use-agreement/',
        'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
        'manifest_sha256': sha(PAD / 'manifest.csv'), 'clinically_validated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('rsna', 'padchest', 'all'), default='all')
    args = parser.parse_args()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    previous = [r for split in ('train', 'validation', 'holdout', 'holdout_confirmation')
                for r in read_rows(ROOT/'training/generalization_stage6'/(split+'_manifest.csv'))]
    if args.source in {'rsna', 'all'}:
        splits = fresh_rsna(protocol, previous)
        selection = {k: [{f: r[f] for f in ('source_id', 'patient_group', 'label', 'split')} for r in v] for k, v in splits.items()}
        path = OUT / 'selection.json'
        if path.exists() and json.loads(path.read_text()) != selection:
            raise ValueError('Frozen patient selection changed.')
        save_json(path, selection)
        resolved, size = download_images(splits, 6)
        prior_hashes = {r['sha256'] for r in previous}
        all_rows = [r for part in resolved.values() for r in part]
        if {r['sha256'] for r in all_rows} & prior_hashes:
            raise ValueError('Prior RSNA image overlap.')
        for split, rows in resolved.items():
            write_csv(OUT / (split+'_manifest.csv'), rows)
        report = {'new_images': len(all_rows), 'new_unique_patients': len({r['patient_group'] for r in all_rows}),
            'counts': {k: len(v) for k, v in resolved.items()}, 'prior_patient_overlap': 0, 'prior_hash_overlap': 0,
            'archive_size': size, 'existing_source_expansion_not_new_institution': True,
            'new_training_total': 6601, 'new_validation_total': 1552,
            'fresh_confirmation_count': 400, 'model_updated': False}
        save_json(OUT/'data_report.json', report)
        print('RSNA_EXPANSION_READY '+json.dumps(report), flush=True)
    if args.source in {'padchest', 'all'}:
        report = prepare_padchest(); save_json(OUT/'padchest_data_report.json', report)
        print('PADCHEST_PILOT_READY '+json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
