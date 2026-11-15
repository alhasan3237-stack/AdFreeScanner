#!/usr/bin/env bash
# Saves all changes to GitHub: commits everything that changed and pushes it.
# Run by Claude Code after each reply (see .claude/settings.local.json). Safe to run by hand.
# Does nothing if the folder isn't a Git repository yet or nothing changed.
cd "$(dirname "$0")/.." || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0
[ -z "$(git status --porcelain)" ] && exit 0

git add -A
files=$(git diff --cached --name-only | head -5 | xargs -n1 basename | paste -sd, -)
count=$(git diff --cached --name-only | wc -l | tr -d ' ')
git commit -q -m "Auto-save: ${count} file(s) changed (${files})" || exit 0

if git remote get-url origin >/dev/null 2>&1; then
  if git push -q origin HEAD 2>/dev/null; then
    echo '{"systemMessage": "Saved to GitHub"}'
  else
    echo '{"systemMessage": "Committed locally, but pushing to GitHub failed (open GitHub Desktop and press Push origin)"}'
  fi
else
  echo '{"systemMessage": "Committed locally (no GitHub repository linked yet: Publish it in GitHub Desktop)"}'
fi
