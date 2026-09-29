"""Verify or download the exact prepared residential input without training."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def verify_file(path, descriptor):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    digest = h.hexdigest()
    if digest != descriptor['sha256']:
        raise ValueError(f'Residential SHA-256 differs: {digest}')
    import pyarrow.parquet as pq
    table = pq.ParquetFile(path)
    rows, columns = table.metadata.num_rows, len(table.schema_arrow.names)
    if (rows, columns) != (descriptor['rows'], descriptor['physical_columns']):
        raise ValueError(f'Residential shape differs: {rows} rows, {columns} columns')
    return {'status': 'PASS', 'file': str(path.resolve()), 'sha256': digest,
            'bytes': path.stat().st_size, 'rows': rows, 'physical_columns': columns}


def download(url, output, descriptor):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('Use an actual HTTPS data-asset URL without embedded credentials.')
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Output already exists. Verify it with --file; it will not be overwritten.')
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + '.part')
    # Exclusive creation prevents clobbering another unfinished download.
    with partial.open('xb') as target:
        try:
            with urllib.request.urlopen(url, timeout=120) as source:
                if urllib.parse.urlsplit(source.geturl()).scheme != 'https':
                    raise ValueError('Download redirected outside HTTPS.')
                while True:
                    block = source.read(8 * 1024 * 1024)
                    if not block:
                        break
                    target.write(block)
        except BaseException:
            target.close()
            partial.unlink(missing_ok=True)
            raise
    try:
        result = verify_file(partial, descriptor)
        # Atomic no-overwrite publication on the same filesystem.
        os.link(partial, output)
        result['file'] = str(output)
        return result
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=Path, help='Verify an existing prepared Parquet')
    group.add_argument('--url', help='Download from an actual HTTPS asset URL')
    parser.add_argument('--output', type=Path, help='Destination for a new download')
    args = parser.parse_args()
    descriptor = json.loads((ROOT / 'datasets/residential/dataset.json').read_text())
    if args.url:
        if args.output is None:
            parser.error('--url requires --output')
        result = download(args.url, args.output, descriptor)
    else:
        if args.output is not None:
            parser.error('--output is only used with --url')
        result = verify_file(args.file.expanduser(), descriptor)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
