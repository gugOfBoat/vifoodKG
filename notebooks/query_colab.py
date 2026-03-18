# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ViFoodKG — Multi-Entity Knowledge Retriever (Google Colab Version)      ║
# ║  Runtime: T4 GPU | Run từng cell từ trên xuống dưới                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: Clone repo & install dependencies
# ─────────────────────────────────────────────────────────────────────────────
# @title Cell 1: Setup

import subprocess

# Clone repo (nếu chưa có)
subprocess.run(["git", "clone", "https://github.com/gugOfBoat/vifoodKG.git",
                "/content/vifoodKG"], check=False)

# Install
subprocess.run(["pip", "install", "-q",
                "neo4j", "sentence-transformers", "python-dotenv"], check=True)

import sys
sys.path.insert(0, "/content/vifoodKG/src")

print("✓ Setup complete")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: Credentials — điền vào đây
# ─────────────────────────────────────────────────────────────────────────────
# @title Cell 2: Neo4j Credentials

NEO4J_URI      = "neo4j+s://aa4eacb4.databases.neo4j.io"  # ← URI Aura của bạn
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "YOUR_PASSWORD_HERE"                      # ← Password

# Ghi vào env để query.py tự đọc
import os
os.environ["NEO4J_URI"]      = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

print(f"✓ URI: {NEO4J_URI}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: Kiểm tra GPU
# ─────────────────────────────────────────────────────────────────────────────
# @title Cell 3: GPU Check

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: Import retriever từ query.py
# ─────────────────────────────────────────────────────────────────────────────
# @title Cell 4: Import KGRetriever

from query import KGRetriever, print_results

print("✓ Retriever imported")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: ★ CHẠY TRUY VẤN — Khởi tạo mô hình và chạy (Singleton) ★
# ─────────────────────────────────────────────────────────────────────────────
# @title Cell 5: Initialize & Run Query ★

# ---- Nhập tham số ----
ITEMS    = ["Phở Bò", "Thịt Bò", "Bánh Phở"]   # danh sách thực thể từ ảnh
QUESTION = "món này có nguyên liệu chính là gì"  # câu hỏi tự do
TOP_K    = 5                                      # số kết quả

# Khởi tạo mô hình (chỉ chạy 1 lần, các ô dưới có thể dùng lại biến `kg`)
print("Đang tải mô hình embedding và kết nối Neo4j...")
kg = KGRetriever(device=device)

print(f"\nTruy vấn Neo4j...")
results = kg.retrieve(items=ITEMS, question=QUESTION, top_k=TOP_K)

# In kết quả
print_results(results, ITEMS, QUESTION)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6: Xuất JSON (tuỳ chọn)
# ─────────────────────────────────────────────────────────────────────────────
# @title Cell 6: Export JSON (optional)

import json

output = {
    "items": ITEMS,
    "question": QUESTION,
    "top_k": TOP_K,
    "results": results,
}

out_path = "/content/query_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

print(f"✓ Saved to {out_path}")

# Download về máy (Colab)
from google.colab import files
files.download(out_path)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7: Batch query — nhiều câu hỏi cùng lúc mà KIẾN TRÚC MÔ HÌNH KHÔNG LOAD LẠI
# ─────────────────────────────────────────────────────────────────────────────
# @title Cell 7: Batch Queries (Fast)

BATCH_QUERIES = [
    {
        "items": ["Bánh Xèo", "Tôm", "Thịt Lợn"],
        "question": "chất gây dị ứng trong món này",
        "top_k": 3,
    },
    {
        "items": ["Bún Chả", "Chả Lụa"],
        "question": "ăn kèm với gì",
        "top_k": 3,
    },
    {
        "items": ["Cơm Tấm"],
        "question": "nguồn gốc vùng miền",
        "top_k": 3,
    },
]

all_results = []
for q in BATCH_QUERIES:
    # Dùng lại instance `kg` đã khởi tạo ở Cell 5
    rows = kg.retrieve(items=q["items"], question=q["question"], top_k=q["top_k"])
    all_results.append({**q, "results": rows})
    print_results(rows, q["items"], q["question"])

print(f"\n✓ Batch complete: {len(BATCH_QUERIES)} queries processed instantly.")
