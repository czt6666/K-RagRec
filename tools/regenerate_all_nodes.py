"""
重新生成 all_nodes.csv 和 all_edges.csv
策略：从 filtered_full_fb.txt 中提取前 14669 个实体（按出现顺序）
和它们之间的前 63203 条边，尽量映射 MID → 名称。
"""
import pandas as pd

# 1. 加载 fb_entity_names
print("Loading fb_entity_names.tsv ...")
name_map = {}
with open('dataset/fb/fb_entity_names.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            name_map[parts[0]] = parts[1]

# 2. 从 filtered_full_fb.txt 提取前 14669 个实体
print("Extracting first 14669 entities from filtered_full_fb.txt ...")
entities = []
seen = set()
with open('dataset/fb/filtered_full_fb.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            s, p, o = parts
            for e in [s, o]:
                if e not in seen:
                    seen.add(e)
                    entities.append(e)
                    if len(entities) >= 14669:
                        break
            if len(entities) >= 14669:
                break

# 3. 构建 all_nodes.csv
print("Building all_nodes.csv ...")
nodes = []
for i, e in enumerate(entities):
    mid = e.replace('/m/', 'm.')
    name = name_map.get(mid, e)  # 如果找不到名称，保留原始 MID
    nodes.append({'node_id': i, 'node_attr': name})

nodes_df = pd.DataFrame(nodes)
nodes_df.to_csv('dataset/fb/nodes/all_nodes.csv', index=False)
print(f"Saved all_nodes.csv with {len(nodes_df)} nodes")

# 4. 构建 all_edges.csv：取两个端点都在前 14669 个实体中的前 63203 条边
print("Building all_edges.csv ...")
edges = []
entity_set = set(entities)
with open('dataset/fb/filtered_full_fb.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            s, p, o = parts
            if s in entity_set and o in entity_set:
                edges.append({'src': entities.index(s), 'edge_attr': p, 'dst': entities.index(o)})
                if len(edges) >= 63203:
                    break

edges_df = pd.DataFrame(edges)
edges_df.to_csv('dataset/fb/edges/all_edges.csv', index=False)
print(f"Saved all_edges.csv with {len(edges_df)} edges")

# 5. 统计覆盖率
found = sum(1 for e in entities if name_map.get(e.replace('/m/', 'm.')) is not None)
print(f"Name coverage: {found}/{len(entities)} ({found/len(entities)*100:.1f}%)")
