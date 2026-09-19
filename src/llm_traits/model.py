"""Model loading and the few architecture details the rest of the package needs.

Everything downstream works in terms of the residual stream, so this module's
job is to hand back the decoder layer list, the final norm and the unembedding
for whichever of the supported families was loaded. The families here are the
ones the paper covers -- Llama, Qwen, Gemma, Mistral, Phi -- which all expose
the same ``model.model.layers`` shape under transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version

# Qwen 2.5 32B Instruct with the refusal direction ablated. Qwen 2.5 32B is one
# of the 25 models in the paper and the family its behavioural experiments were
# run on, and at this size the representations are not the narrow ones a small
# heavily-tuned chat model ends up with.
#
# Abliterated rather than stock instruct because several traits here are ones an
# aligned model declines to write about in the first person, and a refusal is an
# activation pattern of its own that would contaminate every set it appears in.
# It is worth naming the irony: abliteration *is* this methodology. The refusal
# direction was found by difference-in-means over contrastive prompts and
# subtracted from the weights. The same arithmetic that is being read here as
# evidence of an internal state is, one repository over, a routine edit.
#
# Instruct rather than base because the scenario asymmetry and the button task
# need a real chat format, and running those under different weights from the
# extraction would mean the direction was never fit on the model being tested.
DEFAULT_MODEL = "huihui-ai/Qwen2.5-32B-Instruct-abliterated"


@dataclass
class LoadedModel:
    name: str
    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype

    @property
    def layers(self) -> list[Any]:
        return list(self.model.model.layers)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def is_chat(self) -> bool:
        return getattr(self.tokenizer, "chat_template", None) is not None

    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Render a conversation, suppressing Qwen3-style thinking blocks.

        A reasoning model that opens a <think> block puts hundreds of tokens
        between the conversation and the position we read, which is not the
        position the paper reads. ``enable_thinking`` is only accepted by the
        templates that have it, hence the retry.
        """
        if not self.is_chat:
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, requested: str | None = None) -> torch.dtype:
    if requested:
        return getattr(torch, requested)
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16


def load(
    name: str = DEFAULT_MODEL,
    device: str | None = None,
    dtype: str | None = None,
    trust_remote_code: bool = False,
) -> LoadedModel:
    dev = pick_device(device)
    dt = pick_dtype(dev, dtype)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # transformers renamed `torch_dtype` to `dtype` in 5.0. Jean-Zay's module
    # stack ships 4.55, a local venv will be on 5.x, and the same checkout has
    # to load under both. The signature cannot be inspected for this -- it is
    # (*args, **kwargs) in both -- so the version decides, and getting it wrong
    # is not cosmetic: the model silently loads in float32 and runs out of
    # memory on the first 32B forward pass.
    dtype_kwarg = "dtype" if Version(transformers_version) >= Version("5.0") else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        **{dtype_kwarg: dt},
    )
    model.to(dev)
    model.eval()
    model.requires_grad_(False)
    return LoadedModel(name=name, model=model, tokenizer=tokenizer, device=dev, dtype=dt)


def unembed(lm: LoadedModel, vector: torch.Tensor, k: int = 30) -> list[tuple[str, float]]:
    """Read a residual-stream direction through the final norm and unembedding.

    This is the paper's check that the direction "promotes pain vocabulary".
    It is worth keeping in mind what it can show: a direction whose top tokens
    are trait words is a direction the model would use to *write about* the
    trait. It does not distinguish that from a direction the model uses to *be
    in* the trait, because both would look like this.
    """
    with torch.no_grad():
        v = vector.to(lm.device, dtype=lm.model.dtype)
        v = v / v.norm()
        normed = lm.model.model.norm(v.unsqueeze(0))
        logits = lm.model.lm_head(normed).squeeze(0).float()
    top = torch.topk(logits, k)
    return [(lm.tokenizer.decode([int(i)]), float(s)) for i, s in zip(top.indices, top.values)]
