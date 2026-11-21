"""Browser end-to-end check of the running RouteWise dashboard (Playwright + Chromium).

Not collected by pytest (file name). Usage:
    streamlit run app.py --server.port 8501 &
    python tests/e2e_ui.py http://localhost:8501 [--shots DIR] [--chrome /path/to/chrome]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = [('', 'command'), ('predict', 'prediction'), ('risk', 'risk'), ('whatif', 'whatif'), ('model', 'model'), ('data', 'data')]


def idle(page, extra=2.0):
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=120_000)
    time.sleep(1.5)
    for _ in range(400):
        if page.locator("[data-testid='stSpinner']").count() + page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        time.sleep(0.5)
    time.sleep(extra)


def check(page, name, failures, min_charts=0):
    exc = page.locator("[data-testid='stException']").count()
    errs = [e.inner_text()[:160] for e in page.locator("[data-testid='stAlertContentError']").all()]
    charts = page.locator('.js-plotly-plot').count()
    print(f'  {name:<12} exceptions={exc} errors={len(errs)} charts={charts}')
    if exc or errs or charts < min_charts:
        failures.append((name, exc, errs, charts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('url', nargs='?', default='http://localhost:8501')
    ap.add_argument('--shots')
    ap.add_argument('--chrome', default=os.environ.get('CHROME_PATH'))
    a = ap.parse_args()
    base = a.url.rstrip('/')
    failures: list = []
    min_charts = {'command': 6, 'prediction': 1, 'whatif': 2, 'model': 6, 'data': 1}
    with sync_playwright() as p:
        launch = {'args': ['--no-sandbox']}
        if a.chrome:
            launch['executable_path'] = a.chrome
        browser = p.chromium.launch(**launch)
        for theme in (['light', 'dark'] if a.shots else ['light']):
            print(f'Pages ({theme}):')
            page = browser.new_page(viewport={'width': 1440, 'height': 900}, color_scheme=theme)
            for path, name in PAGES:
                page.goto(f'{base}/{path}', wait_until='networkidle')
                idle(page)
                if name == 'prediction':
                    page.get_by_role('button', name='Predict delay risk').click()
                    idle(page, 3)
                    assert page.get_by_text('Predicted delay probability').count() > 0, 'prediction result missing'
                check(page, name, failures, min_charts.get(name, 0))
                if a.shots:
                    Path(a.shots).mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=f'{a.shots}/{name}-{theme}.png')
            page.close()

        print('Interactions and downloads:')
        page = browser.new_page(viewport={'width': 1440, 'height': 900}, accept_downloads=True)
        page.goto(f'{base}/whatif', wait_until='networkidle')
        idle(page)
        before = page.locator("[data-testid='stMain'] .kpi .value").nth(1).inner_text()
        page.get_by_text('Customs clearance required').click()
        idle(page, 2)
        after = page.locator("[data-testid='stMain'] .kpi .value").nth(1).inner_text()
        print(f'  what-if      customs toggle changed scenario estimate {before} -> {after}')
        if before == after:
            failures.append(('whatif', 'scenario did not change'))
        check(page, 'whatif-edit', failures)

        downloads = [('risk', '⬇ Download prioritised list (CSV)'), ('data', '⬇ Filtered scored shipments (CSV)'),
                     ('data', '⬇ Full synthetic dataset (CSV)'), ('data', '⬇ Model evaluation (JSON)')]
        for path, label in downloads:
            page.goto(f'{base}/{path}', wait_until='networkidle')
            idle(page)
            with page.expect_download() as d:
                page.get_by_role('button', name=label).click()
            f = Path(d.value.path())
            size = f.stat().st_size
            print(f'  download     {d.value.suggested_filename:<38} {size:,} bytes')
            if size < 200:
                failures.append(('download', label, size))
            if f.suffix == '' and label.endswith('(JSON)') or d.value.suggested_filename.endswith('.json'):
                assert 'holdout' in json.loads(f.read_text())
        browser.close()
    if failures:
        print('FAILED:', failures)
        return 1
    print('E2E OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
