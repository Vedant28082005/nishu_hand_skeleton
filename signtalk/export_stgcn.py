"""
One-time export: ASL Citizen ST-GCN checkpoint (.pt) -> ONNX. Needs PyTorch + onnx
HERE ONLY; the runtime (aslcitizen.py) needs neither.

    pip install torch onnx          # export machine only
    python export_stgcn.py          # -> models/aslcitizen_stgcn.onnx

Builds the model from vendor/ASL-citizen-code/ST-GCN/architecture with the exact
config stgcn_testing.py uses, loads the checkpoint, exports with dynamic batch
and time axes, then checks ONNX against the checkpoint's own torch forward pass
(max abs logit diff must be < 1e-4).

One deliberate deviation, mathematically identical: the encoder ends with
F.avg_pool2d(x, x.size()[2:]) -- a global average. Traced, that bakes the
example T into the graph as a fixed kernel, which breaks the dynamic time axis.
We swap it for x.mean((2, 3)). The parity check below compares against the
UNMODIFIED original forward at several T, so a mistake here cannot hide.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
VENDOR = HERE / "vendor" / "ASL-citizen-code" / "ST-GCN"

GRAPH_ARGS = {
    "num_nodes": 27, "center": 0,
    "inward_edges": [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17], [7, 8],
                     [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], [7, 15], [15, 16],
                     [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], [17, 23], [23, 24],
                     [17, 25], [25, 26]],
}


def build(ckpt: Path):
    import torch
    sys.path.insert(0, str(VENDOR))
    from architecture.fc import FC
    from architecture.network import Network
    from architecture.st_gcn import STGCN

    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    n_classes = sd["decoder.classifier.weight"].shape[0]
    enc = STGCN(in_channels=2, graph_args=GRAPH_ARGS, edge_importance_weighting=True)
    net = Network(encoder=enc, decoder=FC(n_features=256, num_class=n_classes, dropout_ratio=0.05))
    net.load_state_dict(sd)               # strict: every one of the 172 tensors must match
    net.eval()
    return net, sd, n_classes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "models" / "ASL_citizen_stgcn_weights.pt"))
    ap.add_argument("--out", default=str(HERE / "models" / "aslcitizen_stgcn.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import torch
    import onnxruntime as ort

    net64, sd, n_classes = build(Path(args.ckpt))
    n_params = sum(p.numel() for p in net64.parameters())
    print(f"checkpoint: {len(sd)} tensors, classes={n_classes}, params={n_params / 1e6:.2f} M, "
          f"dtype={next(iter(sd.values())).dtype}, A={tuple(sd['encoder.A'].shape)}, "
          f"data_bn={tuple(sd['encoder.data_bn.weight'].shape)}")

    # Reference = original forward, in the checkpoint's own dtype.
    # Export model = float32 copy with the pooling swap.
    import copy
    net32 = copy.deepcopy(net64).float().eval()
    enc = net32.encoder

    import types
    import torch.nn.functional as F

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        for gcn, imp in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * imp)
        return x.mean(dim=(2, 3))         # == avg_pool2d over the full (T, V) extent

    enc.forward = types.MethodType(forward, enc)

    dummy = torch.randn(1, 2, 128, 27)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net32, dummy, args.out, opset_version=args.opset,
        input_names=["pose"], output_names=["logits"],
        dynamic_axes={"pose": {0: "batch", 2: "time"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    print("\nparity: ONNX(float32) vs ORIGINAL torch forward (checkpoint dtype)")
    for batch, T in [(1, 128), (1, 64), (1, 100), (1, 37), (3, 128)]:
        x = rng.standard_normal((batch, 2, T, 27)).astype(np.float32)
        with torch.no_grad():
            ref = net64(torch.from_numpy(x).to(next(net64.parameters()).dtype)).numpy()
        got = sess.run(None, {"pose": x})[0]
        d = float(np.abs(ref - got).max())
        worst = max(worst, d)
        print(f"  batch={batch} T={T:<4} max|diff|={d:.3e}   (logit range {ref.min():.1f}..{ref.max():.1f})"
              f"  argmax-equal={bool((ref.argmax(1) == got.argmax(1)).all())}")
    ok = worst < 1e-4
    print(f"\nWORST max abs diff = {worst:.3e}  -> {'PASS (< 1e-4)' if ok else 'FAIL (>= 1e-4)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
