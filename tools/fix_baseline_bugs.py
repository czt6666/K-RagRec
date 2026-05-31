"""
一键修复 baseline 中已知的 train/eval 不一致和工程 bug。
修改内容：
  1. evaluate.py:64  whether_retrieval 调用对齐 train.py
  2. evaluate.py:68  prompt 模板对齐 train.py (### Instruction)
  3. train.py:90   改 best ckpt 保存逻辑（保留 loss 最低的，而非最后一轮）
  4. train.py      加上 adjust_learning_rate 调用

用法：
    python tools/fix_baseline_bugs.py
会在 methods/baseline/ 原地修改，并备份 .bak。
"""
import shutil
import os

def patch_evaluate():
    path = "methods/baseline/evaluate.py"
    backup = path + ".bak"
    shutil.copy(path, backup)

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix 1: whether_retrieval 对齐 train.py
    # 原: retrieval_model.whether_retrieval(args.adaptive_ratio*sequence_id, 5)
    # 改: retrieval_model.whether_retrieval(sequence_id, args.adaptive_ratio*len(sequence_id))
    old1 = "retrieve_movies_list = retrieval_model.whether_retrieval(args.adaptive_ratio*sequence_id, 5)"
    new1 = "retrieve_movies_list = retrieval_model.whether_retrieval(sequence_id, args.adaptive_ratio*len(sequence_id))"
    if old1 in content:
        content = content.replace(old1, new1)
        print("[FIXED] evaluate.py: whether_retrieval 已对齐 train.py")
    else:
        print("[WARN] evaluate.py: 没找到旧的 whether_retrieval 调用，可能已改或位置变了")

    # Fix 2: prompt 模板统一用 ### Instruction
    old2 = "## Instruction: Given the user's watching history"
    new2 = "### Instruction: Given the user's watching history"
    if old2 in content:
        content = content.replace(old2, new2)
        print("[FIXED] evaluate.py: prompt 已对齐 train.py (### Instruction)")
    else:
        print("[WARN] evaluate.py: 没找到旧的 prompt 模板")

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def patch_train():
    path = "methods/baseline/train.py"
    backup = path + ".bak"
    shutil.copy(path, backup)

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix 3: 加上 learning rate schedule 调用
    # 在 optimizer.step() 之后加一行
    old3 = """                optimizer.step()
                loss_val = loss.item()"""
    new3 = """                optimizer.step()
                adjust_learning_rate(optimizer, i, args.lr, args.warmup_epochs, len(inputs_train)//args.batch_size)
                loss_val = loss.item()"""
    if old3 in content:
        content = content.replace(old3, new3)
        print("[FIXED] train.py: 已加上 adjust_learning_rate 调用")
    else:
        print("[WARN] train.py: 没找到 optimizer.step() 插入点")

    # Fix 4: best ckpt 保存改为保留 loss 最低的
    # 原: 每 epoch 末直接 _save_checkpoint(..., is_best=True)
    # 改: 记录最低 loss 的 epoch 才存 best
    old4 = """            print("Epoch %s is finished"%(epoch))
            epoch_log_path = f"{args.output_dir}/{args.dataset}/baseline_train_log.json"""
    new4 = """            print("Epoch %s is finished"%(epoch))
            # 只在 loss 最低时存 best ckpt
            epoch_losses = [r["loss"] for r in train_log if r["epoch"] == epoch]
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
            best_loss_path = f"{args.output_dir}/{args.dataset}/best_loss.txt"
            best_loss = float('inf')
            if os.path.exists(best_loss_path):
                with open(best_loss_path, 'r') as f:
                    best_loss = float(f.read().strip())
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                with open(best_loss_path, 'w') as f:
                    f.write(str(best_loss))
                is_best = True
                print(f"  New best loss: {best_loss:.4f}")
            else:
                is_best = False

            epoch_log_path = f"{args.output_dir}/{args.dataset}/baseline_train_log.json"""
    if old4 in content:
        content = content.replace(old4, new4)
        # 还要把下面的 _save_checkpoint 的 is_best=True 改成 is_best=is_best
        content = content.replace("_save_checkpoint(model, optimizer, epoch, args, is_best=True)",
                                  "_save_checkpoint(model, optimizer, epoch, args, is_best=is_best)")
        print("[FIXED] train.py: best ckpt 改为按最低 loss 保存")
    else:
        print("[WARN] train.py: 没找到 epoch 结束块")

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


if __name__ == "__main__":
    print("开始修复 methods/baseline/ 的已知 bug...")
    patch_evaluate()
    patch_train()
    print("\n完成。备份文件: .bak")
    print("建议接下来重新训练 + 评估，看 Recall@1 是否从 4.9% 提升到接近论文水平。")
