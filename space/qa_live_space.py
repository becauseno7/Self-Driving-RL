"""Desktop/mobile browser QA for the locally running live Space."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Browser, Page, sync_playwright

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "runs"
URL = "http://127.0.0.1:7860"


def wait_for_live(page: Page) -> None:
    page.goto(URL, wait_until="networkidle")
    page.locator("#load-state").wait_for(state="hidden", timeout=20_000)
    page.locator("#play").wait_for(state="visible")
    page.wait_for_function(
        "document.querySelector('#speed-value').textContent.trim() !== '—'"
    )


def desktop_qa(browser: Browser) -> list[str]:
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors: list[str] = []
    page.on(
        "console",
        lambda message: errors.append(message.text) if message.type == "error" else None,
    )
    page.on("pageerror", lambda error: errors.append(str(error)))
    wait_for_live(page)

    assert page.get_by_text("Genuine live policy session").is_visible()
    assert page.locator("#run-state").inner_text().strip().casefold() == "policy deciding live"
    assert page.locator("#play").is_enabled()
    assert page.locator("#new-seed").is_enabled()

    page.locator("#sensors").click()
    assert page.locator("#sensors").get_attribute("aria-pressed") == "true"
    page.locator("#rate").select_option("2")
    page.wait_for_timeout(350)

    page.locator("#play").click()
    page.wait_for_timeout(250)
    paused_elapsed = page.locator("#elapsed").inner_text()
    page.wait_for_timeout(450)
    assert page.locator("#elapsed").inner_text() == paused_elapsed
    assert "simulation paused" in page.locator("#run-state").inner_text().casefold()
    page.locator("#play").click()
    page.wait_for_timeout(400)
    assert page.locator("#elapsed").inner_text() != paused_elapsed

    initial_seed = page.locator("#seed-value").inner_text()
    page.locator("#new-seed").click()
    page.wait_for_function(
        "seed => document.querySelector('#seed-value').textContent.trim() !== seed",
        arg=initial_seed,
    )
    page.wait_for_timeout(800)

    page.screenshot(path=OUTPUT / "space-qa-desktop.png", full_page=True)
    assert not errors, errors
    page.close()
    return errors


def mobile_qa(browser: Browser) -> list[str]:
    page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
    errors: list[str] = []
    page.on(
        "console",
        lambda message: errors.append(message.text) if message.type == "error" else None,
    )
    page.on("pageerror", lambda error: errors.append(str(error)))
    wait_for_live(page)

    dimensions = page.evaluate(
        """() => ({
          viewport: document.documentElement.clientWidth,
          scroll: document.documentElement.scrollWidth,
          pauseHeight: document.querySelector('#play').getBoundingClientRect().height,
          sensorsHeight: document.querySelector('#sensors').getBoundingClientRect().height,
        })"""
    )
    assert dimensions["scroll"] <= dimensions["viewport"] + 1, dimensions
    assert dimensions["pauseHeight"] >= 44, dimensions
    assert dimensions["sensorsHeight"] >= 44, dimensions
    assert page.locator("#intent").is_visible()
    assert page.locator("#difficulty").is_visible()
    page.screenshot(path=OUTPUT / "space-qa-mobile.png", full_page=True)
    assert not errors, errors
    page.close()
    return errors


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        desktop_qa(browser)
        mobile_qa(browser)
        browser.close()
    print("live Space desktop/mobile QA passed")


if __name__ == "__main__":
    main()
