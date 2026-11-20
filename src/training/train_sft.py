"""Full fine-tune of a small Qwen3.5 on the teacher's SFT examples, with the loss on the reply only.

python -m src.training.train_sft data/sft/<prompt_version> [--model Qwen/Qwen3.5-0.8B] [--max-steps N]
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

CHECKPOINT_DIR = Path(__file__).parents[2] / "checkpoints"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="folder with train.jsonl and val.jsonl from build_sft_data")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="Hugging Face model to fine-tune")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after this many updates, for a smoke test")
    args = parser.parse_args()

    data = load_dataset("json", data_files={split: str(args.data_dir / f"{split}.jsonl") for split in ("train", "val")})
    output_dir = CHECKPOINT_DIR / f"{args.model.split('/')[-1].lower()}-sft-{args.data_dir.name}"

    config = SFTConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        assistant_only_loss=True,
        max_length=None,               # the default 1024 would cut off the reply of long examples
        per_device_train_batch_size=2, # the logits over the 248k vocabulary are large for long examples
        eval_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,          # one checkpoint per epoch, to compare them on dev
    )
    trainer = SFTTrainer(
        model=AutoModelForCausalLM.from_pretrained(args.model),
        args=config,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=AutoTokenizer.from_pretrained(args.model),
    )
    trainer.train()
    trainer.save_model()
    print(f"saved to {output_dir}")
