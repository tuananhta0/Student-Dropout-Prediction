"""Playwright browser acceptance for the localhost demo, not a unit-test fixture.

Requires: pip install playwright, and Microsoft Edge (default on Windows) or
`python -m playwright install chromium`. This test does not call Gemini.
Never writes credentials, bearer tokens, chat contents or student records.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from scripts.report_provenance import provenance

ROLES = (
    ('student', 'BIT250093', 'panelStudentData'),
    ('advisor', 'GV001', 'panelAdvisorData'),
    ('admin', 'ADMIN', 'panelAdminData'),
)
VIEWPORTS = (('desktop', 1280, 800), ('mobile', 375, 812))


def localhost_url(value):
    parsed = urlparse(value)
    if (parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1')
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or parsed.path not in ('', '/')):
        raise ValueError('Only http://localhost:<port> or http://127.0.0.1:<port> is accepted')
    return value.rstrip('/')


async def inspect(browser, base_url, role, username, panel, viewport, output_dir):
    name, width, height = viewport
    context = await browser.new_context(viewport={'width': width, 'height': height})
    page = await context.new_page()
    defects = []
    page.on('pageerror', lambda exc: defects.append(type(exc).__name__))
    checks = []
    def check(title, condition, detail=None):
        checks.append({'name': title, 'status': 'PASS' if condition else 'FAIL', 'detail': detail})
    stage = "navigation"
    try:
        response = await page.goto(base_url, wait_until='domcontentloaded', timeout=30000)
        check('Initial HTTP', response is not None and response.status == 200)
        stage = 'login'
        await page.locator('#inpUsername').fill(username)
        await page.locator('#inpPassword').fill(username)  # documented demo policy
        await page.locator('#loginForm button[type="submit"]').click()
        await page.locator('#mainApp').wait_for(state='visible', timeout=30000)
        check('Login and workspace', await page.locator('#loginScreen').is_hidden())
        stage = 'data_panel'
        await page.locator('#btnTabData').click()
        try:
            await page.locator('#' + panel).wait_for(state='visible', timeout=25000)
            check('Role-specific data panel', True)
        except Exception:
            check('Role-specific data panel', False)
        check('Workspace error absent', await page.locator('#workspaceError').is_hidden(),
              (await page.locator('#workspaceError').inner_text())[:120] if await page.locator('#workspaceError').is_visible() else None)
        if role == 'student':
            check('Student course table exists', await page.locator('#stuCoursesTable').count() == 1)
            check('Student GPA area exists', await page.locator('#stuGpaBars').count() == 1)
        elif role == 'advisor':
            check('Advisor class selector exists', await page.locator('#selAdvisorClass').count() == 1)
        else:
            check('Admin faculty list exists', await page.locator('#admFacultiesTable').count() == 1)
        stage = 'chat'
        await page.locator('#btnTabChat').click()
        await page.locator('#viewChat').wait_for(state='visible', timeout=10000)
        await page.locator('#chatInput').fill('Bao luu vi bi om dau theo dieu nao?')
        try:
            async with page.expect_response(lambda r: r.url.endswith('/api/chat'), timeout=35000) as req:
                await page.locator('#chatSend').click()
            chat_response = await req.value
            check('Chat API via browser', chat_response.status == 200)
        except Exception as exc:
            check('Chat API via browser', False, type(exc).__name__)
        overflow = await page.evaluate('''() => ({scroll: document.documentElement.scrollWidth,
            client: document.documentElement.clientWidth})''')
        check('No page-level horizontal overflow', overflow['scroll'] <= overflow['client'] + 3,
              f"{overflow['scroll']}px / {overflow['client']}px")
        check('No JavaScript page errors', not defects, ','.join(defects[:5]) if defects else None)
        stage = 'screenshot'
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Synthetic-data interface, local only. Avoid storing page HTML or cookies.
            await page.screenshot(path=str(output_dir / (role + '_' + name + '.png')), full_page=False)
    except Exception as exc:
        check('Browser journey completed', False, stage + ': ' + type(exc).__name__ + ': ' + str(exc).splitlines()[0][:180])
    finally:
        await context.close()
    return {'role': role, 'viewport': name, 'checks': checks,
            'passed': sum(x['status'] == 'PASS' for x in checks),
            'failed': sum(x['status'] == 'FAIL' for x in checks)}


async def run(args):
    from playwright.async_api import async_playwright
    base_url = localhost_url(args.url)
    async with async_playwright() as pw:
        launch = {'headless': not args.headed}
        if args.direct_local:
            launch['args']=['--no-proxy-server']
        if args.executable_path:
            launch['executable_path'] = args.executable_path
        else:
            launch['channel'] = args.channel
        browser = await pw.chromium.launch(**launch)
        try:
            results = []
            for role, username, panel in ROLES:
                for viewport in VIEWPORTS:
                    results.append(await inspect(browser, base_url, role, username, panel,
                                                 viewport, args.screenshots))
        finally:
            await browser.close()
    return {'checks': results, 'passed': sum(x['passed'] for x in results),
            'failed': sum(x['failed'] for x in results),
            'notes': ['Real browser against localhost, not a stub or API fixture',
                      'Human visual/responsive and accessibility judgment remains required',
                      'Use only synthetic student accounts and local demo; no Gemini requests'],
            'provenance': provenance(check='local_browser_playwright',url=base_url,direct_local=args.direct_local)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--channel', default='msedge' if os.name == 'nt' else 'chromium',
                   help='Playwright browser channel; use msedge with installed Edge on Windows')
    p.add_argument('--executable-path', help='Explicit Chromium executable instead of browser channel')
    p.add_argument('--direct-local', action='store_true', help='Use direct loopback connection in this isolated smoke browser only')
    p.add_argument('--headed', action='store_true', help='Show the real browser for manual inspection')
    p.add_argument('--screenshots', type=Path, help='Optional screenshots, demo synthetic data only')
    p.add_argument('--output', type=Path, default=Path('data/kiem_tra_ban_giao/browser_roles.json'))
    args = p.parse_args()
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'failed': report['failed'],
                      'roles': [{k: r[k] for k in ('role','viewport','passed','failed')} for r in report['checks']],
                      'output': str(args.output)}, ensure_ascii=False, indent=2))
    for session in report['checks']:
        for check in session['checks']:
            if check['status']=='FAIL':
                print(f"{session['role']}/{session['viewport']}: {check['name']}: {check.get('detail')}")
    return 0 if report['failed'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
