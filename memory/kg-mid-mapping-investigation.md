---
name: kg-mid-mapping-investigation
description: 调查发现本地 fb_entity_names.tsv 与 all_nodes.csv 的 MID 集合完全不重叠，无法用于修复 KG 节点名称
type: project
---

## 调查结论

- `fb_entity_names.tsv` 包含 524,287 条 Freebase MID → 名称映射
- `all_nodes.csv` 中 10,923 个节点（74.5%）是 Freebase MID 格式（如 `/m/01006ysx`）
- **但两者的 MID 集合完全不重叠！命中率 = 0%**
- `movies_with_mids.txt` 中的电影 MID 与 `fb_entity_names.tsv` 也完全不重叠

## 推论

`fb_entity_names.tsv` 不是为这个数据集的 KG 准备的映射文件。
本地无法修复 KG 节点的 MID → 名称映射问题，因为缺少正确的映射源（如 `mapped_filtered_fb.txt`）。

服务器上可能有经过映射处理的数据文件。如果服务器上的 `all_nodes.csv` 已经是名称而非 MID，说明问题只存在于本地环境。

## 下一步

在服务器上检查：
1. `dataset/fb/nodes/all_nodes.csv` 中 node_attr 是否仍是大量 MID
2. 是否存在 `mapped_filtered_fb.txt`
3. 服务器上的 `0.pt` 和本地的 `0.pt` 是否相同（用 md5sum 对比）
