@echo off
chcp 65001 >nul
cd /d %~dp0
python deep_research_literature_agent.py --year-from 2024 --year-to 2026 --max-results 80 --batch-size 4 --citation-seed-limit 8 --force-github-enrichment
pause
