import torch
import torch.distributed as dist
import json
import logging
import os
from pathlib import Path
from dataclasses import asdict, dataclass, field
from datetime import datetime
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval import simple_evaluate
from omegaconf import OmegaConf

from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
# Import your GPT2 model definition here
from apps.gpt2.transformer import GPT2Transformer, GPT2TransformerArgs
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.args import dump_config
from lingua.data import init_choice_state, setup_sources
from lingua.distributed import (
    DistributedArgs,
    dist_mean_dict,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
)
from lingua.stool import StoolArgs, launch_job

logger = logging.getLogger()

EVAL_FOLDER_NAME = "{:010d}"


@dataclass
class LMHarnessArgs:
    tasks: Optional[List[Any]] = None
    num_fewshot: Optional[int] = None
    device: Optional[str] = None
    use_cache: Optional[str] = None
    cache_requests: bool = False
    rewrite_requests_cache: bool = False
    delete_requests_cache: bool = False
    limit: Optional[Union[int, float]] = None
    bootstrap_iters: int = 100000
    check_integrity: bool = False
    write_out: bool = False
    log_samples: bool = True
    system_instruction: Optional[str] = None
    apply_chat_template: Union[bool, str] = False
    fewshot_as_multiturn: bool = False
    gen_kwargs: Optional[str] = None
    verbosity: str = "INFO"
    predict_only: bool = False
    random_seed: int = 0
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 1234
    compute_loss: bool = False


@dataclass
class ValidationArgs:
    max_steps: Optional[int] = None 
    use_val_from_train_src: bool = True 
    root_dir: str = ""
    sources: List[str] = field(default_factory=list)

@dataclass
class EvalArgs:
    name: str = "evals"
    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None
    ckpt_dir: str = ""
    generator: PackedCausalTransformerGeneratorArgs = field(
        default_factory=PackedCausalTransformerGeneratorArgs
    )
    harness: Optional[LMHarnessArgs] = field(default_factory=LMHarnessArgs)
    validation: Optional[ValidationArgs] = field(default_factory=ValidationArgs)

    wandb: Optional[Any] = None
    global_step: Optional[int] = None


def all_dicts_same(dict_list):
    if not dict_list:
        return True
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)


class MockAccelerator:
    def gather(self, tensor):
        l = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        torch.distributed.all_gather(l, tensor)
        return torch.stack(l)

    def wait_for_everyone(self):
        torch.distributed.barrier()


class EvalHarnessLM(LM):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.accelerator = MockAccelerator()
        self._rank = get_global_rank()
        self._world_size = get_world_size()
        self.device = generator.device
        self.losses = defaultdict(list)
        self.compute_loss = False

    def generate_until(self, requests: List[Instance]) -> List[str]:
        prompts, gen_args = zip(*[req.args for req in requests])
        assert all_dicts_same(gen_args), "Doesn't support different gen args for now"
        gen_args = gen_args[0]
        temperature = gen_args.get("temperature", 0.0)
        top_p = gen_args.get("top_p", None)
        top_k = gen_args.get("top_k", None)
        until = gen_args.get("until", [])

        self.generator.temperature = temperature
        self.generator.top_p = top_p
        self.generator.top_k = top_k
        self.generator.until = until
        generations, _, _ = self.generator.generate(prompts)
        filtered_gen = []
        for g in generations:
            for e in until:
                g = g.replace(e, "")
            filtered_gen.append(g)
        return filtered_gen

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        prompts, continuations = zip(*[req.args for req in requests])
        inputs = [req.args[0] + req.args[1] for req in requests]
        max_gen_len = self.generator.max_gen_len
        self.generator.max_gen_len = 1
        _, lls, greedy = self.generator.generate(inputs)
        results = []
        
        for p, ll, gr, req in zip(prompts, lls, greedy, requests):
            p_len = len(self.generator.tokenizer.encode(p, add_bos=False, add_eos=False))
            cont_ll = ll[p_len:].sum().item()
            cont_tokens = len(ll[p_len:])
            if self.compute_loss and hasattr(req, 'task_name'):
                self.losses[req.task_name].append(-cont_ll / cont_tokens)
            results.append((cont_ll, gr[p_len:].all().item()))

        self.generator.max_gen_len = max_gen_len
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        prompts = [req.args[0] for req in requests]
        max_gen_len = self.generator.max_gen_len
        self.generator.max_gen_len = 1
        _, lls, _ = self.generator.generate(prompts)
        results = []
        for ll in lls:
            results.append((ll.sum().item(),))
        self.generator.max_gen_len = max_gen_len

        return results


def eval_on_val(generator, val_args: ValidationArgs, train_cfg):
    path_to_iter = {}
    srcs = {}
    for src in val_args.sources:
        path = os.path.join(val_args.root_dir, src)
        srcs[path] = 1.0

    if val_args.use_val_from_train_src:
        for src in train_cfg.data.sources:
            path = os.path.join(train_cfg.data.root_dir, src)
            srcs[path] = 1.0

    multi_state = init_choice_state("", srcs, 0, get_global_rank(), get_world_size(), "*.val.jsonl")
    path_to_iter.update(setup_sources(multi_state))

    original_max_len = generator.max_gen_len
    generator.max_gen_len = 1

    all_val_metrics = {}
    logger.info("Running GPT2 validation...")

    for src in path_to_iter:
        jsonl_iterator = path_to_iter[src]
        texts = []
        logger.info(f"Running validation on {src}...")

        for step, (content, state) in enumerate(jsonl_iterator):
            if val_args.max_steps is not None and step >= val_args.max_steps:
                break
            if content is None:
                continue
            if not isinstance(content, dict):
                logger.warning(f"Skipping non-dict content at step {step} in {src}: {content}")
                continue

            text_val = None
            for key_candidate in ["text", "content", "doc"]:
                if key_candidate in content:
                    text_val = content[key_candidate]
                    break
            if not text_val or not isinstance(text_val, str) or not text_val.strip():
                continue

            texts.append(text_val)

        if not texts:
            logger.warning(f"No valid texts found in {src}, skipping...")
            continue

        _, loglikelihood, _ = generator.generate(texts)
        from collections import defaultdict
        metrics = defaultdict(list)
        for txt, ll in zip(texts, loglikelihood):
            neg_ll = -ll.sum().item()
            metrics["nll"].append(neg_ll)
            metrics["nll_per_token"].append(neg_ll / len(ll))
            metrics["nll_per_char"].append(neg_ll / len(txt))
            metrics["avg_seqlen"].append(len(ll))

        for m in metrics:
            metrics[m] = sum(metrics[m]) / len(metrics[m])

        metrics.update(dist_mean_dict(metrics))
        logger.info(f"Validation on {src} done. Metrics: {metrics}")

        name = os.path.basename(src)
        if name in all_val_metrics:
            logger.warning(f"Duplicate source name {name}, path {src}, renaming to {name}_1")
            name = f"{name}_1"
        all_val_metrics[name] = metrics

    generator.max_gen_len = original_max_len
    return all_val_metrics


def launch_eval(cfg: EvalArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())
    consolidate_path = Path(cfg.ckpt_dir)
    if (
        consolidate_path.exists()
        and (consolidate_path / "params.json").exists()
        and next(consolidate_path.glob("*.pth"), None) is not None
    ):
        pass
    else:
        consolidate_path = consolidate_path / CONSOLIDATE_FOLDER
        if not consolidate_path.exists() and get_global_rank() == 0:
            consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)

    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)

    consolidate_path = str(consolidate_path)
    torch.distributed.barrier()

    logger.info("Loading GPT2 model")
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
        model_cls=GPT2Transformer,
        model_args_cls=GPT2TransformerArgs,
    )
    logger.info("Model loaded")
    model.eval()
    generator = PackedCausalTransformerGenerator(cfg.generator, model, tokenizer)

    wrap = EvalHarnessLM(generator)
    wrap.compute_loss = cfg.harness.compute_loss
    harness_args = asdict(cfg.harness)
    harness_args.pop('compute_loss', None)

    results = simple_evaluate(wrap, **harness_args)

    if dist.get_rank() == 0 and results is not None:
        if cfg.harness.compute_loss:
            for task_name, task_losses in wrap.losses.items():
                loss_value = sum(task_losses) / len(task_losses)
                if task_name in results["results"]:
                    results["results"][task_name]["loss"] = loss_value
                else:
                    results["results"][task_name] = {
                        "loss": loss_value
                    }

    val_results = None
    if cfg.validation:
        val_results = eval_on_val(generator, cfg.validation, train_cfg)

    if get_global_rank() == 0:
        with open(Path(cfg.dump_dir) / "results.json", "w") as f:
            f.write(json.dumps(results))
        logger.info(f"All evaluation results: {results['results']}")
        if val_results is not None:
            with open(Path(cfg.dump_dir) / "validation.json", "w") as f:
                f.write(json.dumps(val_results))
            logger.info(f"All validation results: {val_results}")

    if cfg.metric_log_dir and get_global_rank() == 0:
        metric_log_path = Path(cfg.metric_log_dir) / "metrics.eval.jsonl"

        logger.info(f"Writing metric logs to {metric_log_path}")
        timestamp = {
            "created_at": datetime.utcnow().isoformat(),
        }
        if cfg.global_step is not None:
            timestamp["global_step"] = cfg.global_step
        print(
            json.dumps(timestamp | results["results"]),
            file=open(metric_log_path, mode="a"),
            flush=True,
        )

        val_log_path = Path(cfg.metric_log_dir) / "metrics.validation.jsonl"
        if val_results is not None:
            print(
                json.dumps(timestamp | val_results),
                file=open(val_log_path, mode="a"),
                flush=True,
            )

    del generator