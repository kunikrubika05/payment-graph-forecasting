from pathlib import Path


def test_setup_v100_installs_project_package_dependencies():
    script = Path("scripts/setup_v100.sh").read_text()

    assert 'pip install -e ".[dev]" -q' in script
    assert "import joblib" in script
    assert "import yaml" in script
