# Freebase 与其开发者页面

## `https://developers.google.com/freebase?hl=zh-cn` 是什么？

这是 **Google 的 Freebase 开发者页面**（中文版）。

### Freebase 是什么？

- **Freebase** 是一个开源的、协作式的知识图谱数据库
- 由 **Metaweb** 公司于 2007 年发布
- **2010 年被 Google 收购**
- **2016 年关闭**，数据迁移到了 **Wikidata**

### 为什么关闭了？

Google 把 Freebase 的数据整合进了 **Google Knowledge Graph**（谷歌知识图谱，你在谷歌搜索时右侧看到的那个信息卡片就是它的产物）。

### 现在还能用吗？

- Freebase API 已经关闭，不能查询了
- 但 Google 发布了完整的 **Freebase Dump**（数据快照）
- Wikidata 也有 **Freebase MID → Wikidata QID → 多语言标签** 的映射
- 所以如果需要做 MID → 名称 的映射，可以通过 Wikidata 实现

### 和本项目的关系

K-RagRec 论文使用的知识图谱就是 Freebase 的一个子集，经过筛选后保留了与 MovieLens 电影相关的三元组。
