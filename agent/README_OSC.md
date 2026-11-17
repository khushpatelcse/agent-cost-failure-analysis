# F1 agent: Llama fine-tuning on OSC

Put these files in the same folder as `agent.py`, `tools.py` and `model.py`.

| File | What it does | Where it runs |
|---|---|---|
| `prompts.py` | The plan and answer prompts from `model.py`, shared so training matches run time | anywhere |
| `llama_model.py` | `LlamaModel`, a drop-in for `GeminiModel` (same `plan_tools` / `generate_answer`) | OSC GPU |
| `make_dataset.py` | Builds training/val/test data from question templates with known correct plans | laptop or OSC login node |
| `train_lora.py` | LoRA fine-tune with TRL | OSC GPU |
| `evaluate.py` | Plan validity, exact match, latency and tokens per model | laptop (Gemini), OSC (Llama) |
| `smoke_test.py` | First GPU job: loads Llama and prints 3 plans | OSC GPU |
| `osc/*.sh, *.sbatch` | Environment setup and Slurm job scripts | OSC |

Replace `PAS1234` with your class project code in every file under `osc/`.

## Workflow

1. **Hugging Face access (everyone, now):** accept the license on the
   `meta-llama/Llama-3.2-3B-Instruct` page and create an access token.
2. **Copy the project to OSC:** use `git clone` on a login node, or upload through OnDemand's file browser.
3. **Set up the environment once:** run `bash osc/setup_env.sh` on an Ascend login node.
4. **Build the data:** run `python make_dataset.py` (planning data, no API calls).
   For answer data, run this on a laptop with internet and `GEMINI_API_KEY`:
   `python make_dataset.py --answers 150`. Then copy `data/` to OSC.
5. **Smoke test:** `mkdir -p logs && sbatch osc/smoke.sbatch`, then
   `squeue -u $USER` and `cat logs/f1-smoke-*.out`. This is also the zero-shot baseline.
6. **Train:** `sbatch osc/train.sbatch`. The adapter is saved to `outputs/f1-lora/`.
7. **Evaluate:** `sbatch osc/eval.sbatch` (Llama before and after training) and
   `python evaluate.py --model gemini` on a laptop. Compare `results/*_summary.json`.
8. **Swap the model in `main.py`:**
   ```python
   from llama_model import LlamaModel
   model = LlamaModel(adapter_path="outputs/f1-lora")
   ```

## What the evaluation measures

- `valid_json_rate`: did the model return parseable JSON? Gemini is forced by
  `response_schema`; Llama has to learn it.
- `valid_plan_rate`: did `F1Agent._read_plan` and `_check_tool_call` accept the plan?
- `exact_match_rate`: same tools and arguments as the correct plan, in any order,
  with race names normalized.
- `exact_match_held_out_wording`: accuracy on phrasings that never appeared in
  training, which shows whether the model generalizes rather than memorizes.
- `avg_input_tokens` / `avg_output_tokens` / `avg_plan_seconds`: the inputs for the cost comparison.

To get Gemini token counts too, add this to `GeminiModel.plan_tools` after the API call:
```python
self.last_usage = {
    "input_tokens": response.usage_metadata.prompt_token_count,
    "output_tokens": response.usage_metadata.candidates_token_count,
}
```

## Notes

- Do the downloads on login nodes. The `.sbatch` files set `HF_HUB_OFFLINE=1`,
  so jobs use the cached weights instead of trying to download.
- `evaluate.py --full` runs the real OpenF1 tools, so it needs internet. Before relying
  on it inside a GPU job, test that with `curl -I https://api.openf1.org/v1/meetings`.
- Llama 3.2 3B with LoRA fits on one A100. To try 8B later:
  `--model meta-llama/Llama-3.1-8B-Instruct` (accept that license too).
