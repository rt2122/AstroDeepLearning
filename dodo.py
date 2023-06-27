"""Doit file."""

DOIT_CONFIG = {'default_tasks': ['all']}


def task_html():
    """Make HTML documentation."""
    return {
            'actions': ['sphinx-build -b html docs/source docs/build'],
           }


def task_sdist():
    """Create source distribution."""
    return {
            'actions': ['python -m build -s'],
           }


def task_wheel():
    """Create binary wheel distribution."""
    return {
            'actions': ['python -m build -w'],
           }


def task_style():
    """Check style with flake8."""
    return {
            'actions': ['black ADL']
           }


def task_docstyle():
    """Check docstrings with pydocstyle."""
    return {
            'actions': ['pydocstyle ADL']
           }


def task_check():
    """Perform all checks."""
    return {
            'actions': None,
            'task_dep': ['style', 'docstyle']
           }


def task_all():
    """Perform all build task."""
    return {
            'actions': None,
            'task_dep': ['check', 'html', 'wheel']
           }
