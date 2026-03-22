import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

def run():
    # 1. Count JSON
    try:
        with open('data/triples/extracted_triples.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            # If data is a dict containing "all_triples", extract it. Otherwise, assume it's just the list itself.
            if isinstance(data, dict):
                dishes = data.get("all_triples", [])
            else:
                dishes = data
                
            total_triples = sum(len(d.get("triples", [])) for d in dishes)
            print(f"📊 Trong file JSON hiện tại: {len(dishes)} dishes - {total_triples} triples (edges)")
    except Exception as e:
        print(f"Lỗi đọc JSON: {e}")

    # 2. Neo4j Blind Spots
    load_dotenv()
    driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
    
    def get_blind_spots(tx):
        # Tổng null
        query_total = """
        MATCH ()-[r]-()
        WHERE r.verbalized_text IS NOT NULL AND r.embedding IS NULL
        RETURN count(r) AS c
        """
        total_null = tx.run(query_total).single()["c"]
        print(f"\n🔍 VÙNG MÙ TRÊN NEO4J (Có Text nhưng mất Embeddings):")
        print(f"   => {total_null} relations")
        
        # Chi tiết
        query_dist = """
        MATCH ()-[r]-()
        WHERE r.verbalized_text IS NOT NULL AND r.embedding IS NULL
        RETURN type(r) AS rel_type, count(r) AS c
        ORDER BY c DESC
        """
        dist = tx.run(query_dist).data()
        
        if len(dist) > 0:
            print("\n   [Chi tiết theo loại quan hệ]:")
            for row in dist:
                print(f"      + {row['rel_type']}: {row['c']}")
            
            # Món ăn bị ảnh hưởng
            query_dish = """
            MATCH (d:Dish)-[r]-()
            WHERE r.verbalized_text IS NOT NULL AND r.embedding IS NULL
            RETURN d.name as dish, count(r) as c
            ORDER BY c DESC LIMIT 5
            """
            dishes = tx.run(query_dish).data()
            print("\n   [Top 5 Món ăn bị thiếu embedding nhiều nhất]:")
            for row in dishes:
                print(f"      + {row['dish']} (Thiếu {row['c']} edges)")
        else:
             print("\n   Tuyệt vời, không có quan hệ nào bị thiếu vector nhúng!")
             
    with driver.session() as session:
        session.execute_read(get_blind_spots)

    driver.close()

if __name__ == '__main__':
    run()
