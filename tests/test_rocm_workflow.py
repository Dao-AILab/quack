"""Concurrency and event-routing invariants for the ROCm workflow."""

from pathlib import Path

_WORKFLOW = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "rocm.yml").read_text()


def test_rocm_workflow_separates_event_concurrency():
    assert "github.event_name" in _WORKFLOW
    assert "github.event.pull_request.number || github.ref" in _WORKFLOW


def test_rocm_workflow_routes_internal_and_fork_prs_once():
    assert "pull_request_target:" in _WORKFLOW
    assert "github.event.pull_request.head.repo.full_name == github.repository" in _WORKFLOW
    assert "github.event.pull_request.head.repo.full_name != github.repository" in _WORKFLOW
