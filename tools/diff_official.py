"""
对比本地 baseline/ 与官方 GitHub 版本的差异。
用法：先把官方仓库 clone 到 /tmp/official_kragrec，然后跑：
    $env:PYTHONPATH = "methods/baseline"
    python tools/diff_official.py /tmp/official_kragrec
"""
import sys
import os
import difflib
import filecmp

FILES_TO_CHECK = [
    "train.py",
    "evaluate.py",
    "retrieve.py",
    "src/config.py",
    "src/model/graph_llm.py",
    "src/model/gnn.py",
    "src/model/__init__.py",
    "src/utils/ckpt.py",
    "src/utils/seed.py",
]

def diff_file(path1, path2):
    with open(path1, 'r', encoding='utf-8', errors='ignore') as f:
        lines1 = f.readlines()
    with open(path2, 'r', encoding='utf-8', errors='ignore') as f:
        lines2 = f.readlines()
    diff = list(difflib.unified_diff(lines1, lines2, fromfile=path1, tofile=path2))
    return diff

def main(official_dir):
    local_dir = "methods/baseline"
    print(f"对比: {local_dir}  vs  {official_dir}")
    print("=" * 60)

    for fname in FILES_TO_CHECK:
        p1 = os.path.join(local_dir, fname)
        p2 = os.path.join(official_dir, fname)

        if not os.path.exists(p1):
            print(f"\n[MISS] 本地缺失: {fname}")
            continue
        if not os.path.exists(p2):
            print(f"\n[MISS] 官方缺失: {fname}")
            continue

        if filecmp.cmp(p1, p2, shallow=False):
            print(f"[SAME] {fname}")
        else:
            print(f"\n[DIFF] {fname}")
            diff = diff_file(p1, p2)
            for line in diff[:30]:
                print(line.rstrip())
            if len(diff) > 30:
                print(f"... ({len(diff)-30} more lines)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/diff_official.py <path_to_official_kragrec>")
        print("Note: 如果你没 clone 官方仓库，可以先在服务器上跑:")
        print("  git clone https://github.com/Sjay-Wang/K-ragrec.git /tmp/official_kragrec")
        sys.exit(1)
    main(sys.argv[1])
