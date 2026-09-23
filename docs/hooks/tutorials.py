"""Include original tutorial notebooks and assets without source duplication."""

from copy import copy
from pathlib import Path
import re

from mkdocs.plugins import event_priority
from mkdocs.structure.files import get_files


def on_page_content(html, **kwargs):
    """Give CamelCase class references the same token as class declarations."""
    return re.sub(
        r'<span class="(?:n|na|nn)">([A-Z][A-Za-z0-9_]*[a-z][A-Za-z0-9_]*)</span>',
        r'<span class="nc">\1</span>',
        html,
    )


@event_priority(100)
def on_files(files, config):
    """Add tutorial files before mkdocs-jupyter converts notebooks to pages."""
    tutorial_config = copy(config)
    tutorial_config['docs_dir'] = str(
        Path(config['config_file_path']).resolve().parent.parent / 'Tutorials'
    )
    for file in get_files(tutorial_config):
        if files.get_file_from_path(file.src_uri) is not None:
            raise ValueError(f'Documentation and Tutorials share a path: {file.src_uri}')
        files.append(file)
    return files
