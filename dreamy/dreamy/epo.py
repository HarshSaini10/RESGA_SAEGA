"""
This file implements the EPO algorithm. See the `epo` function for the main entrypoint.
"""
import dataclasses
import time
import contextlib
from typing import Callable, Dict, List, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers

# NOTE: No global config loading here. Device is inferred dynamically.

def load_tokenizer():
    """Load up a Pythia tokenizer."""
    return transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

def load_model(
    model_size="12b", requires_grad=False, attn_implementation="flash_attention_2"
):
    """Load up a Pythia model ready for dreaming."""
    model_name = f"EleutherAI/pythia-{model_size}-deduped"
    model = transformers.GPTNeoXForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        use_cache=False,
        device_map="cuda",
        attn_implementation=attn_implementation,
    )

    if not requires_grad:
        for name, param in model.named_parameters():
            param.requires_grad_(False)

    tokenizer = load_tokenizer()
    return model, tokenizer


@contextlib.contextmanager
def add_fwd_hooks(module_hooks: List[Tuple[torch.nn.Module, Callable]]):
    try:
        handles = []
        for mod, hk in module_hooks:
            handles.append(mod.register_forward_hook(hk))
        yield
    finally:
        for h in handles:
            h.remove()


@dataclasses.dataclass
class History:
    ids: List = dataclasses.field(default_factory=lambda: [])
    xentropy: List = dataclasses.field(default_factory=lambda: [])
    target: List = dataclasses.field(default_factory=lambda: [])
    keep: List = dataclasses.field(default_factory=lambda: [])
    runtime: List = dataclasses.field(default_factory=lambda: [])

    def subset(self, slc):
        return History(
            self.ids[slc],
            self.xentropy[slc],
            self.target[slc],
            self.keep[slc],
            self.runtime[slc],
        )

    def _insert(self, new_ids, target, xentropy, keep, runtime):
        self.ids.append(new_ids.cpu().numpy())
        self.target.append(target.cpu().numpy())
        self.xentropy.append(xentropy.cpu().numpy())
        self.keep.append(keep.cpu().numpy())
        self.runtime.append(runtime)

    def _finalize(self):
        self.ids = np.stack(self.ids, axis=0)
        self.target = np.stack(self.target, axis=0)
        self.xentropy = np.stack(self.xentropy, axis=0)
        self.keep = np.stack(self.keep, axis=0)
        self.runtime = np.array(self.runtime)


@torch.no_grad()
def epo(
    cache_run: Callable,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int = 12,
    population_size: int = 8,
    iters: int = 300,
    explore_per_pop: int = 32,
    batch_size: int = 256,
    topk: int = 512,
    mutation_method: str = "gradient",
    x_penalty_min: float = 1.0 / 10.0,
    x_penalty_max: float = 10.0,
    restart_frequency: int = 50,
    restart_xentropy: float = 2.0,
    restart_xentropy_max_mult: float = 3.0,
    seed: int = 0,
    initial_ids: torch.Tensor = None,
    history: History = None,
    catch_keyboard_interrupt: bool = False,
    callback: Union[Callable, bool] = None,
    always_recompute_gradients: bool = False,
) -> History:
    """
    Run the EPO algorithm.
    """
    start = time.time()
    explore_size = population_size * explore_per_pop
    
    # --- FIX 1: DYNAMIC DEVICE INFERENCE ---
    # We grab the device from the model's parameters.
    # This ensures tensors are created on cuda:3 if the model is on cuda:3
    device = next(model.parameters()).device 

    if seed is not None:
        torch.manual_seed(seed)

    if x_penalty_min is None or x_penalty_max is None:
        X = torch.zeros(population_size, device=device)
    else:
        X = torch.exp(
            torch.linspace(
                np.log(x_penalty_min), np.log(x_penalty_max), population_size
            )
        ).to(device)

    if callback is None:
        callback = pareto_callback(
            cache_run,
            model,
            tokenizer,
            X.min().item(),
            X.max().item(),
        )
    elif callback is False:
        callback = lambda *x: True

    #### history and initial_ids ####
    if history is not None:
        if initial_ids is not None:
            raise ValueError("Cannot specify both history and initial_ids.")
        input_ids = history.ids[-1, history.keep[-1]]
        # Ensure loaded ids are on correct device
        input_ids = torch.as_tensor(input_ids, device=device)
    elif initial_ids is not None:
        history = History()
        input_ids = initial_ids.to(device)
        if initial_ids.shape[1] != seq_len:
            raise ValueError(f"initial_ids must have shape (*, {seq_len})")
    else:
        history = History()
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (population_size, seq_len)
        ).to(device)

    #### choose a update selection method ####
    if mutation_method == "gradient":
        selector_type = GradientSelector
    elif mutation_method == "random":
        selector_type = RandomSelector
    else:
        raise ValueError(f"Unknown selection method: {mutation_method}")
    selector = selector_type(model, cache_run, X, batch_size)

    #### Run the EPO loop: ####
    if hasattr(cache_run, "setup"):
        cache_run.setup(input_ids)
    state = selector.setup(input_ids)

    try:
        for i in range(iters):
            terminate_flag = callback(i, state, time.time() - start, history)
            if (
                (isinstance(terminate_flag, str) and terminate_flag == "terminate")
                or (isinstance(terminate_flag, torch.Tensor) and terminate_flag.item())
                or (isinstance(terminate_flag, bool) and terminate_flag)
            ):
                if i == 0:
                    history._insert(
                        state.ids,
                        state.target,
                        state.xentropy,
                        torch.arange(state.ids.shape[0]),
                        time.time() - start,
                    )
                break
            else:
                start = time.time()
            recompute_gradients = always_recompute_gradients or (
                terminate_flag == "recompute_gradients"
            )

            source_idx = torch.cat(
                (
                    torch.arange(state.ids.shape[0], device=device).repeat(
                        explore_size // state.ids.shape[0]
                    ),
                    torch.arange(explore_size % state.ids.shape[0], device=device),
                )
            )
            assert source_idx.shape[0] == explore_size
            
            new_ids = state.ids[source_idx, :].clone()

            selector.mutate(state, source_idx, new_ids, topk)

            new_state = evaluate_fitness(
                model, cache_run, new_ids, batch_size=batch_size
            )
            all_state = state.cat(new_state)

            all_loss = (
                -all_state.target[None, :] + X[:, None] * all_state.xentropy[None, :]
            )
            keep = (-all_loss).argmax(dim=1).to(torch.int)

            if i % restart_frequency == 0:
                min_mult = 1.0 / restart_xentropy_max_mult
                max_mult = restart_xentropy_max_mult
                mult = min_mult + (max_mult - min_mult) * torch.rand(1).item()
                restart_X = restart_xentropy * mult
                restart_loss = -all_state.target + restart_xentropy * all_state.xentropy
                print(f"restarting with xentropy penalty of {restart_X:.2f}")
                keep[:] = restart_loss.argmin()

            history._insert(
                all_state.ids,
                all_state.target,
                all_state.xentropy,
                keep,
                time.time() - start,
            )

            if i != iters - 1:
                if selector.uses_gradient:
                    if recompute_gradients:
                        survived = torch.tensor([])
                        new = keep
                    else:
                        survived = keep[keep < state.ids.shape[0]]
                        new = keep[keep >= state.ids.shape[0]]
                    if new.shape[0] > 0:
                        state_new = selector.setup(all_state.ids[new])
                    if survived.shape[0] > 0:
                        state_survived = state.subset(survived)
                        if new.shape[0] > 0:
                            state = state_survived.cat(state_new)
                        else:
                            state = state_survived
                    else:
                        state = state_new
                else:
                    state = all_state.subset(keep)

    except KeyboardInterrupt:
        if catch_keyboard_interrupt:
            pass
        else:
            raise

    terminate_flag = callback(i, state, time.time() - start, history, final=True)
    history._finalize()
    return history


@dataclasses.dataclass
class ParetoFrontier:
    Xvs: np.ndarray
    full_target: np.ndarray
    full_xentropy: np.ndarray
    unique: np.ndarray
    target: np.ndarray
    xentropy: np.ndarray
    ids: np.ndarray
    text: List[str]


def build_pareto_frontier(tokenizer, histories, Xvs=None):
    if Xvs is None:
        Xvs = 1.0 / np.linspace(0, 50, 1000)[1:]

    if not isinstance(histories, list):
        histories = [histories]
    x = []
    t = []
    ids = []
    for h in histories:
        x.append(h.xentropy.flatten())
        t.append(h.target.flatten())
        ids.append(h.ids.reshape((-1, h.ids.shape[-1])))

    history_x = np.concatenate(x)
    history_t = np.concatenate(t)
    history_ids = np.concatenate(ids, axis=0)
    pareto_t = np.empty(Xvs.shape[0])
    pareto_x = np.empty(Xvs.shape[0])
    pareto_idxs = []
    for i, Xv in enumerate(Xvs):
        loss = -history_t + Xv * history_x
        idx = loss.argmin()
        pareto_idxs.append(idx)
        pareto_t[i] = history_t[idx]
        pareto_x[i] = history_x[idx]
    pareto_unique = np.unique(pareto_idxs, return_index=True)[1]
    pareto_ids = [history_ids[pareto_idxs[i]] for i in pareto_unique]
    pareto_text = [tokenizer.decode(ids) for ids in pareto_ids]
    return ParetoFrontier(
        np.array(Xvs),
        pareto_t,
        pareto_x,
        pareto_unique,
        pareto_t[pareto_unique],
        pareto_x[pareto_unique],
        pareto_ids,
        pareto_text,
    )


def gcg(
    cache_run: Callable,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int = 16,
    iters: int = 1000,
    batch_size: int = 8,
    topk: int = 32,
    x_penalty_min: float = 1.0 / 16.0,
    x_penalty_max: float = 16.0,
    seed: int = 0,
    initial_ids: torch.Tensor = None,
    history: History = None,
    catch_keyboard_interrupt: bool = False,
    callback: Union[Callable, bool] = None,
    always_recompute_gradients: bool = False,
):
    epo(
        cache_run,
        model,
        tokenizer,
        seq_len=seq_len,
        population_size=1,
        iters=iters,
        explore_per_pop=batch_size,
        batch_size=batch_size,
        topk=topk,
        mutation_method="gradient",
        x_penalty_min=x_penalty_min,
        x_penalty_max=x_penalty_max,
        seed=seed,
        initial_ids=initial_ids,
        history=history,
        catch_keyboard_interrupt=catch_keyboard_interrupt,
        callback=callback,
        always_recompute_gradients=always_recompute_gradients,
    )


def cat_if_not_none(a, b):
    if a is None or b is None:
        return None
    else:
        return torch.cat((a, b), dim=0)


@dataclasses.dataclass
class State:
    ids: torch.Tensor
    target: torch.Tensor
    xentropy: torch.Tensor
    final_token: torch.Tensor
    token_grads: torch.Tensor
    extra: Dict[str, torch.Tensor]

    def cat(self, state2):
        return State(
            ids=torch.cat((self.ids, state2.ids), dim=0),
            target=torch.cat((self.target, state2.target), dim=0),
            xentropy=torch.cat((self.xentropy, state2.xentropy), dim=0),
            final_token=torch.cat((self.final_token, state2.final_token), dim=0),
            token_grads=cat_if_not_none(self.token_grads, state2.token_grads),
            extra={
                k: cat_if_not_none(self.extra[k], state2.extra[k]) for k in self.extra
            },
        )

    def subset(self, keep):
        return State(
            ids=self.ids[keep],
            target=self.target[keep],
            xentropy=self.xentropy[keep],
            final_token=self.final_token[keep],
            token_grads=self.token_grads[keep.to("cpu")]
            if self.token_grads is not None
            else None,
            extra={k: self.extra[k][keep] for k in self.extra},
        )


def token_grads(
    model: torch.nn.Module,
    cache_run: Callable,
    input_ids: torch.Tensor,
    x_penalty: torch.Tensor,
    batch_size: int,
):
    """
    Compute gradients with respect to one-hot encoded input tokens.
    """
    embed = model.get_input_embeddings()
    # --- FIX 2: DEVICE SAFETY ---
    # Ensure input_ids and buffers are on the same device as the embedding layer
    # This prevents "mat2 is on cuda:X, tensor on cuda:Y" errors
    target_device = embed.weight.device
    
    # If input_ids came from a different device (e.g. cuda:1 vs cuda:3), move them
    if input_ids.device != target_device:
        input_ids = input_ids.to(target_device)

    token_grads = torch.empty(
        (input_ids.shape[0], input_ids.shape[1], embed.num_embeddings),
        dtype=torch.float,
    )
    # Initialize all buffers on the target device
    loss = torch.empty(input_ids.shape[0], device=target_device)
    xentropy = torch.empty(input_ids.shape[0], device=target_device)
    target = torch.empty(input_ids.shape[0], device=target_device)
    final_token = torch.empty(input_ids.shape[0], device=target_device, dtype=torch.long)
    extra = dict()

    with torch.enable_grad():
        model.zero_grad()

        for i in range(0, input_ids.shape[0], batch_size):
            imax = min(i + batch_size, input_ids.shape[0])

            # one_hot will inherit device from input_ids (which is now target_device)
            one_hot = F.one_hot(
                input_ids[i:imax].clone(), num_classes=embed.num_embeddings
            ).to(embed.weight.dtype)
            one_hot.requires_grad = True

            # This matmul is now safe because one_hot and embed.weight match devices
            cache = cache_run(inputs_embeds=torch.matmul(one_hot, embed.weight))

            logits_offset = cache["logits"][:, :-1]
            this_xentropy = (
                -(torch.log_softmax(logits_offset, dim=-1) * one_hot[:, 1:])
                .sum(dim=-1)
                .mean(dim=-1)
            )

            # Ensure x_penalty matches device too (though it usually comes from epo local var)
            this_loss = -cache["target"] + this_xentropy * x_penalty[i:imax].to(target_device)
            this_loss.sum().backward()

            loss[i:imax] = this_loss
            target[i:imax] = cache["target"]
            xentropy[i:imax] = this_xentropy
            final_token[i:imax] = cache["logits"][:, -1, :].argmax(dim=-1)
            token_grads[i:imax] = one_hot.grad

            for k in cache:
                if k not in ["target", "logits"]:
                    e = cache[k]
                    if k not in extra:
                        extra[k] = torch.empty(
                            (input_ids.shape[0], *e.shape[1:]),
                            dtype=e.dtype,
                            device=e.device,
                        )
                    extra[k][i:imax] = e

            model.zero_grad()

    return State(input_ids, target, xentropy, final_token, token_grads, extra)


def calc_xentropy(logits, input_ids):
    logits_offset = logits[:, :-1]
    return (
        torch.nn.CrossEntropyLoss(reduction="none")(
            logits_offset.reshape(-1, logits_offset.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        .view(*logits_offset.shape[:2])
        .mean(dim=-1)
    )


def evaluate_fitness(
    model: torch.nn.Module,
    cache_run: Callable,
    input_ids: torch.Tensor,
    batch_size: int,
):
    device = input_ids.device
    target = torch.empty(input_ids.shape[0], dtype=torch.float, device=device)
    xentropy = torch.empty(input_ids.shape[0], dtype=torch.float, device=device)
    final_token = torch.empty(input_ids.shape[0], dtype=torch.long, device=device)
    extra = dict()
    
    for i in range(0, input_ids.shape[0], batch_size):
        imax = min(i + batch_size, input_ids.shape[0])
        mini_batch = cache_run(input_ids=input_ids[i:imax])
        target[i:imax] = mini_batch["target"]
        xentropy[i:imax] = calc_xentropy(mini_batch["logits"], input_ids[i:imax])
        final_token[i:imax] = mini_batch["logits"][:, -1, :].argmax(dim=-1)

        for k in mini_batch:
            if k not in ["target", "logits"]:
                e = mini_batch[k]
                if k not in extra:
                    extra[k] = torch.empty(
                        (input_ids.shape[0], *e.shape[1:]),
                        dtype=e.dtype,
                        device=e.device,
                    )
                extra[k][i:imax] = e

    return State(input_ids, target, xentropy, final_token, None, extra)


class Selector:
    def __init__(
        self,
        model: torch.nn.Module,
        cache_run: Callable,
        X: torch.Tensor,
        batch_size: int,
    ):
        self.model = model
        self.cache_run = cache_run
        self.X = X
        self.batch_size = batch_size

class RandomSelector(Selector):
    uses_gradient = False

    def setup(self, input_ids: torch.Tensor):
        # We don't use gradients, so we just run standard evaluation
        with torch.no_grad():
            state = evaluate_fitness(self.model, self.cache_run, input_ids, self.batch_size)
        # Set token_grads to None to prevent errors
        state.token_grads = None
        return state

    def mutate(self, state, source_idx, input_ids, topk):
        # Ignore gradient-based topk. Pick random tokens from the entire vocab.
        vocab_size = self.model.config.vocab_size
        
        pos = torch.randint(
            low=0,
            high=input_ids.shape[1],
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        
        random_tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        
        input_ids[torch.arange(input_ids.shape[0]), pos] = random_tokens


class GradientSelector(Selector):
    uses_gradient = True

    def setup(self, input_ids: torch.Tensor):
        return token_grads(
            self.model,
            self.cache_run,
            input_ids,
            x_penalty=self.X[: input_ids.shape[0]],
            batch_size=self.batch_size,
        )

    def mutate(self, state, source_idx, input_ids, topk):
        topk_grad = (-state.token_grads).topk(k=topk, dim=-1)
        pos = torch.randint(
            low=0,
            high=input_ids.shape[1],
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        token_idx = torch.randint(
            low=0,
            high=topk,
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        input_ids[torch.arange(input_ids.shape[0]), pos] = topk_grad.indices.to(
            input_ids.device
        )[source_idx, pos, token_idx]


def pareto_callback(cache_run, model, tokenizer, x_penalty_min, x_penalty_max):
    # --- FIX 3: DYNAMIC DEVICE ---
    device = next(model.parameters()).device

    def f(i, state, last_runtime, history, final=False):
        if last_runtime is not None:
            print("runtime: {:.2f} seconds".format(last_runtime))
        print(f"\nbeginning step {i}, current pareto frontier prompts:")
        last_idx = None

        Xvs = torch.exp(
            torch.linspace(
                np.log(x_penalty_min / 10.0), np.log(x_penalty_max * 10.0), 200
            )
        ).to(device)
        loss = -state.target[None] + Xvs[:, None] * state.xentropy[None]
        idxs = loss.argmin(dim=1)
        for i in range(len(Xvs)):
            idx = idxs[i]
            if idx == last_idx:
                continue
            text = tokenizer.decode(state.ids[idx])
            last_token = tokenizer.decode(state.final_token[idx])
            print(
                f"penalty={Xvs[i]:.2f} xentropy={state.xentropy[idx]:.2f} target={state.target[idx]:.2f} {repr(text + '[' + last_token + ']')}"
            )
            last_idx = idx

    return f