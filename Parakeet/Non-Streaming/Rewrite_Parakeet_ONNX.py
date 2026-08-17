"""Apply narrow ONNX Runtime fusions to the source-optimized Parakeet encoder.

The raw export remains provider-neutral and immutable. This script writes a separate graph set and replaces
only two exporter decompositions that ONNX has no equivalent standard fused operator for:

* ``x * sigmoid(x)`` -> ``com.microsoft::QuickGelu(alpha=1.0)``
* residual ``Add`` + ``LayerNormalization`` -> ``com.microsoft::SkipLayerNormalization``

Both contrib operators use domain version 1 and are supported by ONNX Runtime 1.27 CPUExecutionProvider.
The rewrite intentionally performs no generic cleanup, simplification, quantization, or dtype conversion.
"""

import collections
import json
import shutil
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper


_ENCODER_NAME = "Parakeet_ASR_Encoder.onnx"

_HIDDEN_SIZE = 1024
_FF_SIZE = 4096
_STANDARD_DOMAINS = {"", "ai.onnx"}


def _tensor_specs(model: onnx.ModelProto) -> dict[str, tuple[int, tuple[int | str | None, ...]]]:
    specs = {}
    values = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    for value in values:
        tensor_type = value.type.tensor_type
        dims = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        specs[value.name] = (tensor_type.elem_type, tuple(dims))
    return specs


def _graph_links(nodes):
    producers = {}
    consumers = collections.defaultdict(list)
    for node in nodes:
        for output in node.output:
            if not output:
                continue
            producers[output] = node
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)
    return producers, consumers


def _match_silu(nodes, producers, consumers, specs):
    matches = []
    for mul in nodes:
        if mul.domain not in _STANDARD_DOMAINS or mul.op_type != "Mul" or len(mul.input) != 2:
            continue
        if mul.attribute or len(mul.output) != 1:
            continue
        for sigmoid_output, x_name in ((mul.input[0], mul.input[1]), (mul.input[1], mul.input[0])):
            sigmoid = producers.get(sigmoid_output)
            if sigmoid is None or sigmoid.domain not in _STANDARD_DOMAINS or sigmoid.op_type != "Sigmoid":
                continue
            if sigmoid.attribute or len(sigmoid.input) != 1 or len(sigmoid.output) != 1:
                continue
            if sigmoid.input[0] != x_name or consumers[sigmoid_output] != [mul]:
                continue
            width = specs.get(x_name, (None, ()))[1][-1:]
            if width not in {(_HIDDEN_SIZE,), (_FF_SIZE,)}:
                continue
            matches.append((sigmoid, mul, x_name, width[0]))
            break
    return matches


def _match_skip_layer_norm(nodes, producers, consumers, specs, initializers):
    matches = []
    for norm in nodes:
        if norm.domain not in _STANDARD_DOMAINS or norm.op_type != "LayerNormalization":
            continue
        if len(norm.input) not in {2, 3} or len(norm.output) != 1:
            continue
        add = producers.get(norm.input[0])
        if add is None or add.domain not in _STANDARD_DOMAINS or add.op_type != "Add":
            continue
        if add.attribute or len(add.input) != 2 or len(add.output) != 1:
            continue
        if any(input_name in initializers for input_name in add.input):
            continue

        attrs = {attr.name: helper.get_attribute_value(attr) for attr in norm.attribute}
        if set(attrs) != {"axis", "epsilon", "stash_type"}:
            continue
        if attrs["axis"] != -1 or abs(float(attrs["epsilon"]) - 1e-5) > 1e-12 or attrs["stash_type"] != 1:
            continue

        scale = initializers.get(norm.input[1])
        if scale is None or scale.data_type != TensorProto.FLOAT or tuple(scale.dims) != (_HIDDEN_SIZE,):
            continue
        if len(norm.input) == 3:
            bias = initializers.get(norm.input[2])
            if bias is None or bias.data_type != TensorProto.FLOAT or tuple(bias.dims) != (_HIDDEN_SIZE,):
                continue

        sum_consumers = consumers[add.output[0]]
        other_consumers = [node for node in sum_consumers if node is not norm]
        if other_consumers and (len(other_consumers) != 1 or other_consumers[0].op_type != "Add"):
            continue
        matches.append((add, norm, bool(other_consumers)))
    return matches


def _add_contrib_opset(model: onnx.ModelProto) -> None:
    contrib = [opset for opset in model.opset_import if opset.domain == "com.microsoft"]
    if not contrib:
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))


def _rewrite_encoder(model: onnx.ModelProto) -> dict:
    nodes = list(model.graph.node)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    producers, consumers = _graph_links(nodes)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)
    specs = _tensor_specs(inferred)
    silu_matches = _match_silu(nodes, producers, consumers, specs)
    skip_matches = _match_skip_layer_norm(nodes, producers, consumers, specs, initializers)

    quick_by_sigmoid = {}
    removed_quick_mul = set()
    for index, (sigmoid, mul, x_name, _) in enumerate(silu_matches):
        name = f"Parakeet_FusedSiLU_{index:03d}"
        quick_by_sigmoid[id(sigmoid)] = helper.make_node(
            "QuickGelu", [x_name], [mul.output[0]], name=name, domain="com.microsoft", alpha=1.0
        )
        removed_quick_mul.add(id(mul))

    skip_by_add = {}
    removed_norm = set()
    for index, (add, norm, expose_sum) in enumerate(skip_matches):
        name = f"Parakeet_SkipLayerNormalization_{index:03d}"
        inputs = [add.input[0], add.input[1], norm.input[1], *norm.input[2:]]
        outputs = [norm.output[0], "", "", add.output[0]] if expose_sum else [norm.output[0]]
        epsilon = next(helper.get_attribute_value(attr) for attr in norm.attribute if attr.name == "epsilon")
        skip_by_add[id(add)] = helper.make_node(
            "SkipLayerNormalization", inputs, outputs, name=name, domain="com.microsoft", epsilon=epsilon
        )
        removed_norm.add(id(norm))

    rewritten_nodes = []
    for node in nodes:
        node_id = id(node)
        if node_id in quick_by_sigmoid:
            rewritten_nodes.append(quick_by_sigmoid[node_id])
        elif node_id in removed_quick_mul:
            continue
        elif node_id in skip_by_add:
            rewritten_nodes.append(skip_by_add[node_id])
        elif node_id in removed_norm:
            continue
        else:
            rewritten_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    _add_contrib_opset(model)

    fusions = {"QuickGelu": len(silu_matches), "SkipLayerNormalization": len(skip_matches)}

    return {
        "raw_nodes": len(nodes),
        "final_nodes": len(rewritten_nodes),
        "inserted": fusions,
        "deleted": {"Sigmoid": len(silu_matches), "Mul": len(silu_matches),
                "Add": len(skip_matches), "LayerNormalization": len(skip_matches)},
        "rewired_sum_outputs": sum(expose for _, _, expose in skip_matches),
        "transformed_initializers": 0,
        "deleted_initializers": 0,
    }


def rewrite(raw_folder: Path, output_folder: Path) -> dict:
    raw_folder = raw_folder.expanduser().resolve()
    output_folder = output_folder.expanduser().resolve()
    raw_encoder = raw_folder / _ENCODER_NAME

    model = onnx.load(str(raw_encoder), load_external_data=False)
    report = _rewrite_encoder(model)

    output_folder.parent.mkdir(parents=True, exist_ok=True)
    temp_folder = Path(tempfile.mkdtemp(prefix=f".{output_folder.name}.", dir=output_folder.parent))
    try:
        for source in raw_folder.iterdir():
            if source.is_file() and source.name != _ENCODER_NAME:
                shutil.copy2(source, temp_folder / source.name)
        final_encoder = temp_folder / _ENCODER_NAME
        onnx.save_model(model, str(final_encoder))
        temp_folder.rename(output_folder)
    except BaseException:
        shutil.rmtree(temp_folder, ignore_errors=True)
        raise

    report.update({
        "raw_folder": str(raw_folder),
        "output_folder": str(output_folder),
        "opset_imports": {opset.domain or "ai.onnx": opset.version for opset in model.opset_import},
        "graph_inputs": [value.name for value in model.graph.input],
        "graph_outputs": [value.name for value in model.graph.output],
    })
    print(json.dumps(report, indent=2, sort_keys=True))
    return report
