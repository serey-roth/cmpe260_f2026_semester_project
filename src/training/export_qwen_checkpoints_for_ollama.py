"""Convert a Qwen3.5 checkpoint to a GGUF file that Ollama can import, with a Modelfile.

python -m src.training.export_qwen_checkpoints_for_ollama <checkpoint folder or Hugging Face model> <output folder> --llama-cpp <llama.cpp folder>
then: ollama create <name> -f <output folder>/Modelfile

The GGUF must come from the llama.cpp release that Ollama bundles (Ollama's LLAMA_CPP_VERSION file): newer
converters write the linear-attention weights in a layout older Ollama engines can't load. On Linux, Ollama can't
import the safetensors checkpoint itself, because it sends safetensors to its MLX engine, which is Mac-only.
The base model goes through the same export, so base and student differ only in training.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

IM_END_TOKEN_ID = 248046  # <|im_end|>, which ends every chat reply

# Ollama's settings for its own Qwen3.5 models, so imported models sample the same way
MODELFILE_PARAMETERS = {"temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0, "presence_penalty": 1.5, "repeat_penalty": 1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="checkpoint folder or Hugging Face model")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout at the release Ollama bundles")
    args = parser.parse_args()
    output_dir, hf_dir = args.output_dir.resolve(), args.output_dir.resolve() / "hf"

    # Text-only, the same class as the student, so base and student convert the same way
    AutoModelForCausalLM.from_pretrained(args.source).save_pretrained(hf_dir)
    AutoTokenizer.from_pretrained(args.source).save_pretrained(hf_dir)

    # The config's end token is <|endoftext|>; replies end with <|im_end|>, so make both end tokens
    for name in ("config.json", "generation_config.json"):
        path = hf_dir / name
        config = json.loads(path.read_text())
        eos = config["eos_token_id"] if isinstance(config["eos_token_id"], list) else [config["eos_token_id"]]
        config["eos_token_id"] = sorted(set(eos) | {IM_END_TOKEN_ID})
        path.write_text(json.dumps(config, indent=2))

    llama_cpp = args.llama_cpp.resolve()
    env = {**os.environ, "PYTHONPATH": f"{llama_cpp / 'gguf-py'}:{llama_cpp}"}
    subprocess.run([sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(hf_dir), "--outtype", "bf16",
                    "--outfile", str(output_dir / "model.gguf")], env=env, check=True)

    parameters = "".join(f"PARAMETER {key} {value}\n" for key, value in MODELFILE_PARAMETERS.items())
    # Ollama's own Qwen3.5 renderer and parser, as on its official models. Without them, Ollama renders the prompt
    # with the GGUF's chat template, which renders the same text but silently drops the JSON schema (response_format)
    (output_dir / "Modelfile").write_text(f"FROM {output_dir / 'model.gguf'}\nRENDERER qwen3.5\nPARSER qwen3.5\n{parameters}")
    print(f"exported to {output_dir}; import with: ollama create <name> -f {output_dir / 'Modelfile'}")
