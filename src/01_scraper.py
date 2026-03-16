"""
01_scraper.py – ViFood-OntoGraph Data Acquisition
===================================================
Reads dish names from config/dishes.txt, scrapes culinary data from
Vietnamese Wikipedia (primary) with web fallback, and saves structured
Markdown files to data/raw_data/{dish_slug}.md.

Usage:
    python src/01_scraper.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
import wikipediaapi
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from slugify import slugify

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DISHES_FILE = ROOT_DIR / "config" / "dishes.txt"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw_data"

# ── Constants ────────────────────────────────────────────────────────────────
MIN_WORD_COUNT = 200  # Minimum words to consider Wikipedia content sufficient
REQUEST_DELAY = 1.5   # Seconds between web requests (be polite)
USER_AGENT = (
    "ViFood-OntoGraph/0.1 "
    "(https://github.com/gugOfBoat/vifoodvqa; educational project) "
    "Python/requests"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "vi,en;q=0.9",
}

console = Console()


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_dishes(filepath: Path) -> list[str]:
    """Read dish names from a text file (one per line)."""
    if not filepath.exists():
        console.print(f"[red]ERROR:[/red] File not found: {filepath}")
        sys.exit(1)
    dishes = [
        line.strip()
        for line in filepath.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    console.print(f"[green]✓[/green] Loaded [bold]{len(dishes)}[/bold] dishes from {filepath.name}")
    return dishes


def word_count(text: str) -> int:
    """Count words in a string."""
    return len(text.split())


# ── Wikipedia Scraper ────────────────────────────────────────────────────────


def fetch_wikipedia(dish_name: str) -> str | None:
    """
    Fetch article content from Vietnamese Wikipedia.
    Returns full text or None if page not found / too short.
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent=USER_AGENT,
        language="vi",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )

    page = wiki.page(dish_name)
    if not page.exists():
        # Try alternate: title case, replace spaces with underscores
        alt_title = dish_name.replace(" ", "_")
        page = wiki.page(alt_title)
        if not page.exists():
            return None

    text = page.text.strip()
    if word_count(text) < MIN_WORD_COUNT:
        return None

    return text


# ── Web Fallback Scraper ────────────────────────────────────────────────────


def _extract_text_from_url(url: str) -> str:
    """Download a URL and extract readable text content."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove script, style, nav, footer, header tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Get text from article or main content area
        article = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r"content|post|entry|article", re.I))
        if article:
            text = article.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception:
        return ""


def _search_web(query: str, num_results: int = 3) -> list[str]:
    """
    Search for Vietnamese culinary content using DuckDuckGo HTML.
    Returns a list of URLs.
    """
    search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        urls: list[str] = []
        for link in soup.select("a.result__a"):
            href = link.get("href", "")
            if href and href.startswith("http"):
                urls.append(href)
            if len(urls) >= num_results:
                break
        return urls
    except Exception:
        return []


def fetch_web_fallback(dish_name: str) -> str | None:
    """
    Fallback: search the web for dish info and scrape top results.
    Targets Vietnamese culinary sites.
    """
    queries = [
        f"{dish_name} món ăn Việt Nam nguyên liệu cách làm",
        f"{dish_name} công thức nấu ăn nguồn gốc",
    ]

    all_text_parts: list[str] = []

    for query in queries:
        urls = _search_web(query)
        for url in urls:
            time.sleep(REQUEST_DELAY)
            text = _extract_text_from_url(url)
            if text and word_count(text) > 50:
                all_text_parts.append(text)
                break  # One good source per query is enough

    if not all_text_parts:
        return None

    combined = "\n\n".join(all_text_parts)
    # Truncate if excessively long (> 5000 words)
    words = combined.split()
    if len(words) > 5000:
        combined = " ".join(words[:5000])

    return combined


# ── Markdown Formatter ───────────────────────────────────────────────────────


def _split_into_sections(raw_text: str) -> dict[str, str]:
    """
    Attempt to intelligently split raw text into ontology-friendly sections.
    Uses keyword matching to assign paragraphs to sections.
    """
    sections: dict[str, list[str]] = {
        "Overview": [],
        "Ingredients & Side Dishes": [],
        "Origin & History": [],
        "Cooking Technique & Taste": [],
    }

    KEYWORD_MAP: dict[str, list[str]] = {
        "Ingredients & Side Dishes": [
            "nguyên liệu", "thành phần", "gia vị", "nước chấm", "ăn kèm",
            "rau", "thịt", "cá", "tôm", "bún", "mì", "gạo", "bột",
            "hành", "tỏi", "ớt", "nước mắm", "muối", "đường", "dầu",
            "kèm", "chấm", "rau sống", "rau thơm", "dùng kèm",
        ],
        "Origin & History": [
            "nguồn gốc", "lịch sử", "xuất xứ", "vùng miền", "miền",
            "truyền thống", "đặc sản", "địa phương", "tỉnh", "thành phố",
            "miền bắc", "miền trung", "miền nam", "huế", "hà nội",
            "sài gòn", "hội an", "từ xưa", "thế kỷ", "thời",
        ],
        "Cooking Technique & Taste": [
            "cách làm", "cách nấu", "chế biến", "nấu", "chiên", "xào",
            "luộc", "hấp", "nướng", "kho", "rim", "quay", "rang",
            "hương vị", "vị", "ngọt", "mặn", "cay", "chua", "béo",
            "thơm", "giòn", "mềm", "dai", "kỹ thuật", "bước",
        ],
    }

    paragraphs = re.split(r"\n{2,}", raw_text)

    for para in paragraphs:
        para_lower = para.lower()
        assigned = False

        # Check for Wikipedia-style headings
        heading_match = re.match(r"^={2,}\s*(.+?)\s*={2,}$", para.strip(), re.MULTILINE)
        if heading_match:
            heading = heading_match.group(1).lower()
            for section, keywords in KEYWORD_MAP.items():
                if any(kw in heading for kw in keywords):
                    sections[section].append(para)
                    assigned = True
                    break
            if not assigned:
                sections["Overview"].append(para)
            continue

        # Score paragraphs by keyword density
        best_section = "Overview"
        best_score = 0
        for section, keywords in KEYWORD_MAP.items():
            score = sum(1 for kw in keywords if kw in para_lower)
            if score > best_score:
                best_score = score
                best_section = section

        sections[best_section].append(para)

    return {k: "\n\n".join(v) for k, v in sections.items()}


def format_markdown(dish_name: str, raw_text: str, source: str) -> str:
    """
    Format scraped text into a structured Markdown document
    with sections aligned to the ontology config.
    """
    sections = _split_into_sections(raw_text)

    md_parts: list[str] = [f"# {dish_name}\n"]

    for section_name in ["Overview", "Ingredients & Side Dishes", "Origin & History", "Cooking Technique & Taste"]:
        content = sections.get(section_name, "").strip()
        md_parts.append(f"## {section_name}\n")
        if content:
            md_parts.append(content + "\n")
        else:
            md_parts.append("_Chưa có thông tin._\n")

    md_parts.append("## References\n")
    md_parts.append(f"- Source: {source}\n")
    md_parts.append(f"- Scraped at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    return "\n".join(md_parts)


# ── Main Pipeline ────────────────────────────────────────────────────────────


def scrape_dish(dish_name: str) -> tuple[str, str]:
    """
    Scrape a single dish. Returns (markdown_content, source_label).
    """
    # Try Wikipedia first
    wiki_text = fetch_wikipedia(dish_name)
    if wiki_text:
        md = format_markdown(dish_name, wiki_text, source="Vietnamese Wikipedia")
        return md, "wikipedia"

    # Fallback to web scraping
    time.sleep(REQUEST_DELAY)
    web_text = fetch_web_fallback(dish_name)
    if web_text:
        md = format_markdown(dish_name, web_text, source="Web Scraping (DuckDuckGo)")
        return md, "web"

    # Last resort: create a stub
    stub = format_markdown(
        dish_name,
        f"{dish_name} là một món ăn Việt Nam. Chưa tìm được thông tin chi tiết trên internet.",
        source="Stub (no data found)",
    )
    return stub, "stub"


def main() -> None:
    """Main entry point: scrape all dishes and save Markdown files."""
    console.print("\n[bold blue]━━━ ViFood-OntoGraph: Data Acquisition ━━━[/bold blue]\n")

    # Ensure output directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load dish list
    dishes = load_dishes(DISHES_FILE)
    if not dishes:
        console.print("[red]No dishes to scrape. Exiting.[/red]")
        sys.exit(1)

    stats: dict[str, int] = {"wikipedia": 0, "web": 0, "stub": 0}
    errors: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scraping dishes...", total=len(dishes))

        for i, dish in enumerate(dishes):
            slug = slugify(dish, lowercase=True)
            output_path = RAW_DATA_DIR / f"{slug}.md"

            progress.update(task, description=f"[cyan]{dish}[/cyan] ({i + 1}/{len(dishes)})")

            try:
                md_content, source = scrape_dish(dish)
                output_path.write_text(md_content, encoding="utf-8")
                stats[source] += 1
            except Exception as e:
                errors.append(f"{dish}: {e}")
                stats["stub"] += 1
                # Write an error stub so the file still exists
                error_md = format_markdown(dish, f"Lỗi khi thu thập dữ liệu: {e}", source="Error")
                output_path.write_text(error_md, encoding="utf-8")

            progress.update(task, advance=1)

            # Rate limiting between dishes
            if i < len(dishes) - 1:
                time.sleep(REQUEST_DELAY)

    # ── Summary ──────────────────────────────────────────────────────────
    console.print("\n[bold green]━━━ Scraping Complete ━━━[/bold green]\n")
    console.print(f"  📂 Output directory: [bold]{RAW_DATA_DIR}[/bold]")
    console.print(f"  📊 Total dishes:     [bold]{len(dishes)}[/bold]")
    console.print(f"  🌐 Wikipedia:        [green]{stats['wikipedia']}[/green]")
    console.print(f"  🔍 Web fallback:     [yellow]{stats['web']}[/yellow]")
    console.print(f"  ⚠️  Stubs/errors:     [red]{stats['stub']}[/red]")

    if errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for err in errors:
            console.print(f"  • {err}")

    # List generated files
    md_files = sorted(RAW_DATA_DIR.glob("*.md"))
    console.print(f"\n  ✅ Generated [bold]{len(md_files)}[/bold] Markdown files\n")


if __name__ == "__main__":
    main()
