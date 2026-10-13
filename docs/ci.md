# Documentation-only CI

The server compatibility workflow skips compilation and tests when a pull request
or push changes only documentation. Manual workflow dispatch always runs full CI.

Documentation includes root Markdown/reStructuredText files and license notices,
documents and static images under `docs/`, and README files under `scripts/`.
Source examples (`.py`, `.js`, `.cpp`), test fixtures, submodule revisions,
build scripts, `VERSION` and CI configuration still run full CI, even when mixed
with documentation edits.

Pull requests compare the merge base with their head; pushes compare the previous
and new commits. Renames include the deleted path, so moving source into a Markdown
filename cannot hide a source change. Missing history or an unknown event runs full CI.

Only quick change detection and the three existing required status reports remain
for documentation. The UI status jobs do not install Node, build the UI or run tests.
This preserves normal merging on protected `master`; skipping the entire workflow
with `paths-ignore` would leave required checks pending. Change detection failures
fail the required reports instead of allowing an unchecked merge.

Validate the classifier locally with `python tests/test_ci_changes.py`.
