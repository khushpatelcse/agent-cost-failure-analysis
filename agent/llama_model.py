# llama_model - drop-in replacement for GeminiModel
#
# F1Agent only needs two methods from a model:
#   plan_tools(question)              -> JSON text (or dict) with "tool_calls"
#   generate_answer(question, results) -> answer text
# LlamaModel has the same two methods, so in main.py you only swap:
#   model = GeminiModel()   ->   model = LlamaModel(adapter_path="outputs/f1-lora")

import json
import time

from prompts import build_plan_prompt, build_answer_prompt

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def extract_json_object(text):
    """Return the first complete {...} block in text, or None.

    Gemini is forced into JSON by response_schema. Llama has no such switch,
    so it may wrap the JSON in extra words or a ```json fence. This pulls the
    object out. If there is none, the raw text goes to the agent, which raises
    "The model did not return valid JSON." That counts as a planning failure.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class LlamaModel:
    """Uses a local Llama model (optionally with a LoRA adapter)."""

    def __init__(
        self,
        base_model=DEFAULT_BASE_MODEL,
        adapter_path=None,
        max_new_tokens_plan=256,
        max_new_tokens_answer=400,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,   # older transformers (<4.56): torch_dtype=
            device_map="auto",
        )

        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        self.model = model
        self.model_name = adapter_path or base_model
        self.max_new_tokens_plan = max_new_tokens_plan
        self.max_new_tokens_answer = max_new_tokens_answer

        # Filled in after every call so the cost analysis can read it.
        self.last_usage = None

    def _chat(self, prompt, max_new_tokens):
        """Send one user message through the chat template, return the reply."""
        messages = [{"role": "user", "content": prompt}]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        input_tokens = inputs["input_ids"].shape[1]
        started = time.perf_counter()

        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy: same question -> same plan
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output[0][input_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        self.last_usage = {
            "input_tokens": int(input_tokens),
            "output_tokens": int(len(new_tokens)),
            "generation_seconds": round(time.perf_counter() - started, 4),
        }
        return text.strip()

    def plan_tools(self, question):
        """Ask Llama which F1 tools should be called."""
        text = self._chat(build_plan_prompt(question), self.max_new_tokens_plan)

        if not text:
            raise ValueError("Llama returned an empty tool plan.")

        # The agent still validates this JSON before running anything.
        return extract_json_object(text) or text

    def generate_answer(self, question, tool_results):
        """Ask Llama to answer using the tool results."""
        text = self._chat(
            build_answer_prompt(question, tool_results),
            self.max_new_tokens_answer,
        )
        return text or "I couldn't generate an answer from the tool results."
