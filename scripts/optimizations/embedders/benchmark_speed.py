from argparse import ArgumentParser
from time import perf_counter

import intel_extension_for_pytorch as ipex
import numpy as np
import torch
from aim import Run
from embedders import EmbedderModel
from optimum.intel import INCModel
from tqdm import trange
from transformers import AutoModel, AutoTokenizer


def generate_random_sequences(vocab_size, batch_size, sequence_length):
    input_ids = torch.randint(0, vocab_size - 1, (batch_size, sequence_length))
    token_type_ids = torch.zeros((batch_size, sequence_length), dtype=torch.int64)
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.int64)
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }


class PerformanceBenchmark:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = {}

    def full_sequence_benchmark(
        self, batch_size=1, num_samples=1000, warmup=3000, sequence_length=512
    ):
        # Warmup
        for _ in trange(warmup):
            inputs = generate_random_sequences(
                batch_size=batch_size,
                sequence_length=sequence_length,
                vocab_size=self.model.vocab_size,
            )
            _ = self.model.embed(inputs)

        latencies = []
        for _ in trange(num_samples // batch_size):
            inputs = generate_random_sequences(
                batch_size=batch_size,
                sequence_length=sequence_length,
                vocab_size=self.model.vocab_size,
            )

            start = perf_counter()
            _ = self.model.embed(inputs)
            latency = perf_counter() - start
            latencies.append(latency / batch_size)

        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(f"Average latency (ms) - {time_avg_ms:.2f} +/- {time_std_ms:.2f}")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

    def run_benchmark(self, batch_size=1, num_samples=1000, warmup=50, sequence_length=512):
        print("Full sequence latencies:")
        print(f"batch_size {batch_size}   num_samples: {num_samples}")
        full_seq = self.full_sequence_benchmark(batch_size, num_samples, warmup, sequence_length)
        self.metrics.update(full_seq)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", required=True, help="HF model name")
    parser.add_argument(
        "--mode",
        choices=["inc", "hf", "ipex", "ipex-ts"],
        required=True,
        help="Type of model to load",
    )
    parser.add_argument("--ipex", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--torch_dynamo", action="store_true")
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to run the benchmark on",
    )
    parser.add_argument("--bs", type=int, default=1, help="batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="sequence length")
    parser.add_argument("--warmup", type=int, default=5, help="num of warmup cycels (*batch size)")
    args = parser.parse_args()

    run = Run(experiment="embedders_performance", capture_terminal_logs=True)
    params = vars(args)
    params["cores"] = torch.get_num_threads()
    run["hparams"] = params

    # load the right type of model
    if "inc" == args.mode:
        # benchmark optimized model
        model = INCModel.from_pretrained(args.model_name)
    else:
        model = AutoModel.from_pretrained(args.model_name)

    # load the tokenizer and Embedder model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    opt_model = EmbedderModel(model, tokenizer)

    benchmark = PerformanceBenchmark(opt_model, opt_model.tokenizer)

    if "inc" == args.mode:
        with torch.no_grad(), torch.cpu.amp.autocast():
            print("INC model benchmark + ipex + cpu.amp")
            benchmark.run_benchmark(
                batch_size=args.bs,
                num_samples=args.samples,
                warmup=args.warmup,
                sequence_length=args.seq_len,
            )
    elif "hf" == args.mode:
        with torch.no_grad():
            benchmark.run_benchmark(
                batch_size=args.bs,
                num_samples=args.samples,
                warmup=args.warmup,
                sequence_length=args.seq_len,
            )
    elif "ipex" == args.mode:
        model.model = ipex.optimize(
            opt_model.model, dtype=torch.bfloat16 if args.bf16 else torch.float32
        )
        if args.torch_dynamo:
            opt_model.model = torch.compile(opt_model.model, backend="ipex")
        if args.bf16:
            with torch.no_grad(), torch.cpu.amp.autocast():
                benchmark.run_benchmark(
                    batch_size=args.bs,
                    num_samples=args.samples,
                    warmup=args.warmup,
                    sequence_length=args.seq_len,
                )
        else:
            with torch.no_grad():
                benchmark.run_benchmark(
                    batch_size=args.bs,
                    num_samples=args.samples,
                    warmup=args.warmup,
                    sequence_length=args.seq_len,
                )
    elif "ipex-ts" == args.mode:
        opt_model.model = ipex.optimize(
            opt_model.model, dtype=torch.bfloat16 if args.bf16 else torch.float32
        )
        vocab_size = opt_model.model.config.vocab_size
        batch_size = 1
        seq_length = args.seq_len
        if args.bf16:
            with torch.no_grad(), torch.cpu.amp.autocast():
                d = torch.randint(vocab_size, size=[batch_size, seq_length])
                opt_model.model = torch.jit.trace(
                    opt_model.model, (d,), check_trace=False, strict=False
                )
                opt_model.model = torch.jit.freeze(opt_model.model)

                benchmark.run_benchmark(
                    batch_size=args.bs,
                    num_samples=args.samples,
                    warmup=args.warmup,
                    sequence_length=args.seq_len,
                )
        else:
            with torch.no_grad():
                d = torch.randint(vocab_size, size=[batch_size, seq_length])
                opt_model.model = torch.jit.trace(
                    opt_model.model, (d,), check_trace=False, strict=False
                )
                opt_model.model = torch.jit.freeze(opt_model.model)

                benchmark.run_benchmark(
                    batch_size=args.bs,
                    num_samples=args.samples,
                    warmup=args.warmup,
                    sequence_length=args.seq_len,
                )
    run.track(benchmark.metrics["time_avg_ms"], name="time_avg_ms")
    run.track(benchmark.metrics["time_std_ms"], name="time_std_ms")
    run.report_successful_finish()
