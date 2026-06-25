import sys

import numpy as np
import onnxruntime as ort


NP = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16,
    "tensor(double)": np.float64, "tensor(int64)": np.int64,
    "tensor(int32)": np.int32, "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8, "tensor(bool)": np.bool_,
}


def main():
    s = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
    for i in s.get_inputs():
        print("IN ", i.name, i.shape, i.type)
    for o in s.get_outputs():
        print("OUT", o.name, o.shape, o.type)
    try:
        feed = {
            i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape],
                             NP.get(i.type, np.float32))
            for i in s.get_inputs()
        }
        for o, a in zip(s.get_outputs(), s.run(None, feed)):
            if a.size:
                print("RANGE", o.name, a.dtype, float(a.min()), float(a.max()))
            else:
                print("RANGE", o.name, a.dtype, "empty")
    except Exception as e:
        print("RANGE skipped:", e)


if __name__ == "__main__":
    main()
