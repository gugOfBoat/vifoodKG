"""
Export KG Triples with missing (NULL) evidence and split evenly for 4 members.
Output: ViFoodKG/data/null_evidence_tasks.csv

Usage:
    cd ViFoodKG
    python src/utils/export_null_evidence.py
"""

import os
import math
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

def get_supabase_client():
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY missing in .env")
    return create_client(url, key)

def main():
    # Construct paths to load .env relative to ViFoodKG
    # Assuming script is placed in ViFoodKG/src/utils/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    dotenv_path = PROJECT_ROOT / ".env"
    
    if not load_dotenv(dotenv_path):
        print(f"Warning: Could not load .env from {dotenv_path}")

    try:
        supabase = get_supabase_client()
    except Exception as e:
        print(f"Failed to create Supabase client: {e}")
        return
    
    print("Fetching triples where evidence IS NULL from kg_triple_catalog...")
    
    all_records = []
    limit = 1000
    offset = 0
    
    # Loop to bypass the 1000 record query limit in PostgREST
    while True:
        response = supabase.table("kg_triple_catalog") \
            .select("triple_id, subject, relation, target, evidence, source_url") \
            .is_("evidence", "null") \
            .order("triple_id") \
            .range(offset, offset + limit - 1) \
            .execute()
        
        data = response.data
        if not data:
            break
            
        all_records.extend(data)
        
        if len(data) < limit:
            break
            
        offset += limit

    total_records = len(all_records)
    print(f"Found {total_records} triples with missing evidence.")

    if total_records == 0:
        print("No triples to verify!")
        return

    # Convert to pandas DataFrame for easy splitting and manipulation
    df = pd.DataFrame(all_records)
    
    # Create 4 equal ranges to assign to members
    num_members = 4
    chunk_size = math.ceil(total_records / num_members)
    
    # Add an assignee column (member_1 through member_4)
    # Using min(...) handles the last chunk nicely even if it's slightly smaller
    df['assignee'] = [f"Member_{min((i // chunk_size) + 1, num_members)}" for i in range(total_records)]
    
    # Output file path configuration
    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "null_evidence_tasks.csv"
    
    # Save as CSV with utf-8-sig to preserve Vietnamese characters in Excel
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Successfully exported assignments to: {output_path}")
    
    # Print a summary to the console
    print("\n📊 Task Distribution Summary:")
    summary = df.groupby('assignee').agg(
        count=('triple_id', 'count'),
        start_id=('triple_id', 'min'),
        end_id=('triple_id', 'max')
    )
    print(summary.to_string())

if __name__ == "__main__":
    main()
