import numpy as np
from tritonclient.http import (InferenceServerClient, InferInput,
                               InferRequestedOutput)
from tritonclient.utils import np_to_triton_dtype


def call_triton_infer(input):
    triton_client = InferenceServerClient(url="localhost:8000")

    triton_input = InferInput(
        name="input__0",
        shape=input.shape,
        datatype=np_to_triton_dtype(input.dtype),
    )
    triton_input.set_data_from_numpy(input)

    triton_output = InferRequestedOutput("output__0")
    query_response = triton_client.infer(
        "base_model", [triton_input], outputs=[triton_output]
    )

    output = query_response.as_numpy("output__0")

    return output


def main():
    inputs = np.zeros((1, 785), dtype="double")
    outs = np.array(
        [
            -2.38495271,
            -2.34510387,
            -2.38707664,
            -2.50841805,
            -2.49321042,
            -2.29477965,
            -2.48320836,
            -2.58118737,
            -1.69008942,
            -2.19207585,
        ],
        dtype="double",
    )
    triton_out = call_triton_infer(inputs)
    print("Mean diff: " + str(np.mean(np.abs(outs - triton_out))))


if __name__ == "__main__":
    main()
