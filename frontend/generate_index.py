from pathlib import Path
import json

root = Path(__file__).resolve().parent
kb_root = (root.parent / 'knowledge-base').resolve()
kb_algorithms = (kb_root / 'algorithms').resolve()
kb_projects = (kb_root / 'projects').resolve()
output_dir = root / 'data'
output_dir.mkdir(parents=True, exist_ok=True)

algorithms = []
for metadata_path in sorted(kb_algorithms.rglob('metadata.json')):
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as exc:
        print(f'Warning: failed to load {metadata_path}: {exc}')
        continue

    algorithm_dir = metadata_path.parent
    relative_path = algorithm_dir.relative_to(kb_algorithms).as_posix()
    algorithms.append({
        'id': relative_path,
        'path': relative_path,
        'name': metadata.get('name', relative_path),
        'description': metadata.get('description', ''),
        'metadata': metadata,
    })

projects = []
if kb_projects.exists():
    for readme_path in sorted(kb_projects.rglob('README.md')):
        project_dir = readme_path.parent
        relative_path = project_dir.relative_to(kb_projects).as_posix()
        metadata = {}
        metadata_path = project_dir / 'metadata.json'
        title = ''
        description = ''

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                title = metadata.get('name', '')
                description = metadata.get('description', '')
            except Exception as exc:
                print(f'Warning: failed to load {metadata_path}: {exc}')

        if not title or not description:
            text = readme_path.read_text(encoding='utf-8', errors='ignore')
            for line in text.splitlines():
                stripped = line.strip()
                if not title and stripped.startswith('# '):
                    title = stripped[2:].strip()
                    continue
                if title and not description and stripped and not stripped.startswith('#') and not stripped.startswith('- '):
                    description = stripped
                    break

        projects.append({
            'id': relative_path,
            'path': relative_path,
            'name': title or relative_path,
            'description': description,
            'metadata': metadata,
        })

with open(output_dir / 'algorithms.json', 'w', encoding='utf-8') as f:
    json.dump(algorithms, f, ensure_ascii=False, indent=2)
with open(output_dir / 'projects.json', 'w', encoding='utf-8') as f:
    json.dump(projects, f, ensure_ascii=False, indent=2)

public_data_dir = root / 'public' / 'data'
public_data_dir.mkdir(parents=True, exist_ok=True)
with open(public_data_dir / 'algorithms.json', 'w', encoding='utf-8') as f:
    json.dump(algorithms, f, ensure_ascii=False, indent=2)
with open(public_data_dir / 'projects.json', 'w', encoding='utf-8') as f:
    json.dump(projects, f, ensure_ascii=False, indent=2)

print(f'Generated {len(algorithms)} algorithm entries into {output_dir / "algorithms.json"}')
print(f'Generated {len(projects)} project entries into {output_dir / "projects.json"}')
print(f'Copied JSON files into {public_data_dir} for Vite public assets.')
