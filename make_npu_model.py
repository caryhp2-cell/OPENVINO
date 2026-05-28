import numpy as np
import openvino as ov
from openvino import opset11 as ops

input_node = ops.parameter([1, 3, 224, 224], dtype=np.float16, name="input")

x = input_node
in_channels = 3

for i in range(12):
    out_channels = 64
    weights = np.random.randn(out_channels, in_channels, 3, 3).astype(np.float16) * 0.02
    const = ops.constant(weights)

    x = ops.convolution(
        x,
        const,
        strides=[1, 1],
        pads_begin=[1, 1],
        pads_end=[1, 1],
        dilations=[1, 1],
    )
    x = ops.relu(x)
    in_channels = out_channels

model = ov.Model([x], [input_node], "npu_stress_conv")
ov.save_model(model, "npu_stress.xml")

print("Saved npu_stress.xml")