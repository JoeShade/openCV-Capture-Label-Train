from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE_FILES = sorted(REPO_ROOT.glob("*.py"))
TEST_FILES = sorted((REPO_ROOT / "tests").glob("*.py"))


def test_required_governance_files_exist():
    required_paths = [
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "systemDesign.md",
        REPO_ROOT / "docs" / "architecture-notes.md",
        REPO_ROOT / "docs" / "deviations.md",
        REPO_ROOT / "source-code-footer.txt",
        REPO_ROOT / "pytest.ini",
    ]
    missing = [path for path in required_paths if not path.exists()]
    assert not missing, f"Missing required governance files: {missing}"


def test_deviations_file_is_explicitly_empty_or_uses_template():
    text = (REPO_ROOT / "docs" / "deviations.md").read_text(encoding="utf-8")
    if "_None currently tracked._" in text:
        assert "### DEV-" not in text
        return

    sections = [section for section in text.split("### DEV-") if section.strip()]
    assert sections, "Expected either the explicit empty marker or at least one DEV section."
    for section in sections:
        assert "Design expectation:" in section
        assert "Current implementation:" in section
        assert "Reason for deviation:" in section
        assert "Intended resolution:" in section
        assert "Owner:" in section
        assert "Review trigger:" in section


def test_footer_policy_is_documented_and_applied_to_python_files():
    agents_text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "Canonical footer text: `source-code-footer.txt`" in agents_text
    assert "Applies to: checked-in `.py` files in the repository root" in agents_text
    assert "Does not apply to: `tests/`" in agents_text
    assert (
        "Implementation form: append the canonical footer text inside a trailing standalone triple-quoted string literal"
        in agents_text
    )

    footer_text = (REPO_ROOT / "source-code-footer.txt").read_text(encoding="utf-8").rstrip("\n")
    expected_suffix = "\n\n'''\n" + footer_text + "\n'''\n"
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in PYTHON_SOURCE_FILES
        if not path.read_text(encoding="utf-8").endswith(expected_suffix)
    ]
    assert not offenders, f"Python source files missing the canonical footer suffix: {offenders}"

    excluded = [
        path.relative_to(REPO_ROOT)
        for path in TEST_FILES
        if path.read_text(encoding="utf-8").endswith(expected_suffix)
    ]
    assert not excluded, f"Test files should not carry the canonical footer suffix: {excluded}"


def test_gitignore_covers_generated_test_artifacts():
    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in [".venv/", ".venv312/", ".pytest_cache/", "build/", "dist/", "runs/", "captures/.yolo_training_cache/"]:
        assert entry in gitignore_text


def test_source_footer_artifact_has_expected_markers():
    text = (REPO_ROOT / "source-code-footer.txt").read_text(encoding="utf-8")
    assert text.lstrip().startswith("/*")
    assert "Copyright (c) JoeShade" in text
    assert "Licensed under the GNU Affero General Public License v3.0" in text
    assert text.rstrip().endswith("*/")
