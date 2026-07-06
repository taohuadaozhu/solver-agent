from pathlib import Path


def chunk_readme(readme_path: Path) -> dict:
    text = readme_path.read_text(encoding='utf-8', errors='ignore')
    return {
        'id': readme_path.stem,
        'path': str(readme_path),
        'content': text,
    }


def collect_readmes(source_dir: Path):
    for readme_path in sorted(source_dir.rglob('README.md')):
        yield chunk_readme(readme_path)
