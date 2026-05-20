import os
import re

METHODS = ["baseline", "h1_qformer", "h2_gate", "h3_pcst", "h4_temporal", "h5_rel_gnn"]
BASE = "/root/workspace/python/K-RagRec/methods"


def patch_train(path, method_name):
    with open(path, "r") as f:
        text = f.read()

    # avoid double-patching
    if "train_log.json" in text:
        return

    # add json/os/time imports after the first import block
    first_import = text.splitlines()[0]
    extra_imports = "import json\nimport os\nimport time\n"
    if "import json" not in text:
        text = extra_imports + text

    # insert log init right after "def main(args):"
    log_init = '''    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    train_log = []
    start_time = time.time()

'''
    text = text.replace("def main(args):\n", "def main(args):\n" + log_init)

    # replace the loss print line to also record in train_log
    old_loss_print = "print(f'{i}-th LOSS:', loss.item())"
    new_loss_print = """loss_val = loss.item()
                train_log.append({"epoch": epoch, "step": i, "loss": loss_val})
                print(f'{i}-th LOSS:', loss_val)"""
    text = text.replace(old_loss_print, new_loss_print)

    # after epoch finished print, save epoch log
    old_epoch_end = 'print("Epoch %s is finished"%(epoch))'
    new_epoch_end = '''print("Epoch %s is finished"%(epoch))
            epoch_log_path = f"{args.output_dir}/{args.dataset}/{method_name}_train_log.json"
            with open(epoch_log_path, "w") as f:
                json.dump(train_log, f, indent=2)'''
    text = text.replace(old_epoch_end, new_epoch_end)

    # after the final _save_checkpoint, append training summary
    old_ckpt = "_save_checkpoint(model, optimizer, epoch, args, is_best=True)"
    new_ckpt = '''_save_checkpoint(model, optimizer, epoch, args, is_best=True)
            elapsed = time.time() - start_time
            summary = {
                "method": method_name,
                "dataset": args.dataset,
                "llm_model_name": args.llm_model_name,
                "gnn_model_name": args.gnn_model_name,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "total_steps": len(train_log),
                "final_epoch": epoch,
                "elapsed_seconds": elapsed,
            }
            if train_log:
                losses = [r["loss"] for r in train_log]
                summary["avg_loss"] = sum(losses) / len(losses)
                summary["min_loss"] = min(losses)
                summary["max_loss"] = max(losses)
            summary_path = f"{args.output_dir}/{args.dataset}/{method_name}_train_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Training summary saved to {summary_path}")'''
    text = text.replace(old_ckpt, new_ckpt)

    with open(path, "w") as f:
        f.write(text)
    print(f"Patched train.py: {path}")


def patch_evaluate(path, method_name):
    with open(path, "r") as f:
        text = f.read()

    if "results.json" in text:
        return

    # ensure json/os imports
    if "import json" not in text:
        text = "import json\nimport os\n" + text

    # replace the commented-out accuracy line with live accuracy + save block
    old_block = '''    # print(f'Final ACC: ', accuracy_score(gold, pred))


if __name__ == "__main__":'''

    new_block = f'''    # compute accuracy from top-1 prediction (first element of each ranked list)
    top1_pred = [p[0] if p else -1 for p in pred]
    acc = accuracy_score(gold, top1_pred) if pred else 0.0
    final_results = {{
        "method": "{method_name}",
        "dataset": args.dataset,
        "llm_model_name": args.llm_model_name,
        "gnn_model_name": args.gnn_model_name,
        "num_samples": len(gold),
        "accuracy": acc,
        "recall@1": sum(recalls_1) / len(recalls_1) if recalls_1 else 0.0,
        "recall@3": sum(recalls_3) / len(recalls_3) if recalls_3 else 0.0,
        "recall@5": sum(recalls_5) / len(recalls_5) if recalls_5 else 0.0,
        "recall@10": sum(recalls_10) / len(recalls_10) if recalls_10 else 0.0,
    }}
    os.makedirs(f"{{args.output_dir}}/{{args.dataset}}", exist_ok=True)
    result_path = f"{{args.output_dir}}/{{args.dataset}}/{method_name}_results.json"
    with open(result_path, "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Evaluation results saved to {{result_path}}")
    print(json.dumps(final_results, indent=2, ensure_ascii=False))

if __name__ == "__main__":'''

    # some files may have slightly different spacing; try common variants
    text = text.replace(old_block, new_block)
    # fallback for files where spacing differs slightly
    if "results.json" not in text:
        text = text.replace(
            '''    # print(f'Final ACC: ', accuracy_score(gold, pred))\n    \n            \nif __name__ == "__main__":''',
            new_block
        )
        text = text.replace(
            '''    # print(f'Final ACC: ', accuracy_score(gold, pred))\n    \nif __name__ == "__main__":''',
            new_block
        )
        text = text.replace(
            '''    # print(f'Final ACC: ', accuracy_score(gold, pred))\n\nif __name__ == "__main__":''',
            new_block
        )

    with open(path, "w") as f:
        f.write(text)
    print(f"Patched evaluate.py: {path}")


if __name__ == "__main__":
    for m in METHODS:
        train_path = os.path.join(BASE, m, "train.py")
        eval_path = os.path.join(BASE, m, "evaluate.py")
        if os.path.exists(train_path):
            patch_train(train_path, m)
        if os.path.exists(eval_path):
            patch_evaluate(eval_path, m)
