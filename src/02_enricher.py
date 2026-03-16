"""
02_enricher.py – ViFood-OntoGraph Data Enrichment
===================================================
Iterates through raw Markdown files, identifies missing sections 
("_Chưa có thông tin._"), and uses Gemini API to fill gaps with
high-quality culinary data.

Usage:
    python src/02_enricher.py
"""

import os
import re
import time
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# --- Configuration ---
load_dotenv()
# Check for common environment variable names for the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw_data"

console = Console()

if not GEMINI_API_KEY:
    console.print("[bold red]ERROR:[/bold red] Không tìm thấy API Key.")
    console.print("Vui lòng tạo file [bold].env[/bold] trong thư mục ViFoodVQA với nội dung:")
    console.print("[cyan]GEMINI_API_KEY=your_actual_key_here[/cyan]")
    console.print("\nHoặc thiết lập biến môi trường [bold]GOOGLE_API_KEY[/bold].")
    import sys
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)
# We can use gemini-1.5-flash or gemini-2.0-flash-exp (or standard stable names)
# The user modified this to gemini-3.1-flash-lite-preview earlier
model_name = 'gemini-3.1-flash-lite-preview' 
try:
    model = genai.GenerativeModel(model_name)
except Exception:
    # Fallback to a stable name if preview fails
    model = genai.GenerativeModel('gemini-1.5-flash')

# --- Enrichment Logic ---

def get_missing_sections(content: str) -> list[str]:
    """Identify sections that have the 'Chưa có thông tin' placeholder."""
    # Using a simpler pattern that doesn't rely on literal dots with escape sequences if possible
    # or using double backslashes in non-raw strings.
    missing_pattern = r"## (.*?)\n\n_Chưa có thông tin\._"
    return re.findall(missing_pattern, content)

def generate_missing_content(dish_name: str, section_name: str, existing_context: str) -> str:
    """Use Gemini to generate content for a specific culinary section."""
    prompt = f"""
Bạn là một chuyên gia ẩm thực Việt Nam lão luyện. Nhiệm vụ của bạn là cung cấp thông tin chi tiết và chính xác cho món ăn: "{dish_name}".
Cụ thể, hãy viết nội dung cho mục: "{section_name}".

Yêu cầu:
1. Nội dung phải viết bằng tiếng Việt, giọng văn chuyên nghiệp, giàu hình ảnh.
2. Cung cấp thông tin sâu về văn hóa, kỹ thuật, và nguyên liệu (không viết hời hợt).
3. Đảm bảo thông tin giúp ích cho việc xây dựng cơ sở dữ liệu tri thức (Knowledge Graph) sau này.
4. KHÔNG lặp lại những gì đã có (nếu có context).
5. Chỉ trả về nội dung của mục "{section_name}", không thêm lời chào hay giải thích.

Dưới đây là một số thông tin hiện có để bạn tham khảo (nếu có):
{existing_context}
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        console.print(f"[red]Error generating content for {dish_name} - {section_name}: {e}[/red]")
        return "_Lỗi khi tự động bổ sung thông tin._"

def enrich_file(file_path: Path):
    content = file_path.read_text(encoding="utf-8")
    dish_name_match = re.search(r"^# (.*)", content)
    if not dish_name_match:
        return
    dish_name = dish_name_match.group(1).strip()
    
    missing_sections = get_missing_sections(content)
    if not missing_sections:
        return # Everything is already there
        
    updated_content = content
    for section in missing_sections:
        console.print(f"  [yellow]Enriching Section:[/yellow] {section} for [cyan]{dish_name}[/cyan]")
        # Provide some existing context to avoid redundancy (basic overview if we have it)
        context = ""
        context_match = re.search(f"## Overview\n\n(.*?)\n\n##", content, re.DOTALL)
        if context_match and "_Chưa có thông tin" not in context_match.group(1):
            context = context_match.group(1).strip()

        new_info = generate_missing_content(dish_name, section, context)
        
        # Replace the placeholder with the new content
        placeholder = f"## {section}\n\n_Chưa có thông tin._"
        replacement = f"## {section}\n\n{new_info}"
        updated_content = updated_content.replace(placeholder, replacement)
    
    # Update reference section to mention Gemini enrichment
    ref_update = "- Enriched by: Google Gemini API (AI Synthesis)\n"
    if "## References" in updated_content:
        updated_content = updated_content.replace("## References\n\n", f"## References\n\n{ref_update}")
    else:
        updated_content += f"\n## References\n{ref_update}"
        
    file_path.write_text(updated_content, encoding="utf-8")

def main():
    console.print("\n[bold blue]━━━ ViFood-OntoGraph: Data Enrichment Pipeline ━━━[/bold blue]\n")
    
    md_files = list(RAW_DATA_DIR.glob("*.md"))
    if not md_files:
        console.print("[red]No markdown files found to enrich.[/red]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Enriching files...", total=len(md_files))
        
        for file_path in md_files:
            progress.update(task, description=f"Processing [cyan]{file_path.name}[/cyan]")
            try:
                enrich_file(file_path)
            except Exception as e:
                console.print(f"[red]Failed to enrich {file_path.name}: {e}[/red]")
            progress.update(task, advance=1)
            # small delay to prevent rate limit
            time.sleep(1)

    console.print("\n[bold green]✅ Enrichment Complete![/bold green]\n")

if __name__ == "__main__":
    main()
