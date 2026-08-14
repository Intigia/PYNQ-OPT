
# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import os
from qonnx.core.datatype import DataType
from driver_base_latency_200MHz import FINNExampleOverlay
from pynq.pl_server.device import Device

# dictionary describing the I/O of the FINN-generated accelerator
io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : [DataType['INT8']],
    "odt" : [DataType['UINT4']],
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : [(1, 320, 320, 3)],
    "oshape_normal" : [(1, 10, 10, 208)],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded" : [(1, 320, 320, 1, 3)],
    "oshape_folded" : [(1, 10, 10, 208, 1)],
    "ishape_packed" : [(1, 320, 320, 1, 3)],
    "oshape_packed" : [(1, 10, 10, 208, 1)],
    "input_dma_name" : ['idma0'],
    "output_dma_name" : ['odma0'],
    "number_of_external_weights": 0,
    "num_inputs" : 1,
    "num_outputs" : 1,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute FINN-generated accelerator on numpy inputs, or run throughput test')
    parser.add_argument('--exec_mode', help='Please select functional verification ("execute") or throughput test ("throughput_test")', default="execute")
    parser.add_argument('--platform', help='Target platform: zynq-iodma alveo', default="zynq-iodma")
    parser.add_argument('--batchsize', help='number of samples for inference', type=int, default=1)
    parser.add_argument('--device', help='FPGA device to be used', type=int, default=0)
    parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit")
    parser.add_argument('--inputfile', help='name(s) of input npy file(s) (i.e. "input.npy")', nargs="*", type=str, default=["input.npy"])
    parser.add_argument('--outputfile', help='name(s) of output npy file(s) (i.e. "output.npy")', nargs="*", type=str, default=["output.npy"])
    parser.add_argument('--runtime_weight_dir', help='path to folder containing runtime-writable .dat weights', default="runtime_weights/")
    parser.add_argument('--repetitions', type=int, default=100, help='repeticiones de latencia')
    parser.add_argument('--warmup', type=int, default=5, help='ejecuciones de calentamiento')
    # parse arguments
    args = parser.parse_args()
    exec_mode = args.exec_mode
    platform = args.platform
    batch_size = args.batchsize
    bitfile = args.bitfile
    inputfile = args.inputfile
    outputfile = args.outputfile
    runtime_weight_dir = args.runtime_weight_dir
    repetitions = args.repetitions
    warmup = args.warmup
    devID = args.device
    device = Device.devices[devID]

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    accel = FINNExampleOverlay(
        bitfile_name = bitfile, platform = platform,
        io_shape_dict = io_shape_dict, batch_size = batch_size,fclk_mhz=200.0,
        runtime_weight_dir = runtime_weight_dir, device=device
    )

    # for the remote execution the data from the input npy file has to be loaded,
    # packed and copied to the PYNQ buffer
    if exec_mode == "execute":
        # load desired input .npy file(s)
        ibuf_normal = []
        for ifn in inputfile:
            ibuf_normal.append(np.load(ifn))
        obuf_normal = accel.execute(ibuf_normal)
        if not isinstance(obuf_normal, list):
            obuf_normal = [obuf_normal]
        for o, obuf in enumerate(obuf_normal):
            np.save(outputfile[o], obuf)
    elif exec_mode == "throughput_test":
        res = accel.throughput_test()
        filename = f"nw_metrics_batch_{batch_size}.txt"
        with open(filename, "w") as file:
            for key, value in res.items():
                file.write(f"{key:<45}: {value:.6f}\n")
        print(f"Results written to {filename}")
        with open(filename, "r") as file:
            print(file.read())

    elif exec_mode == "latency_test":
        if batch_size != 1:
            raise ValueError("Para latencia real usa --batchsize 1")

        res = accel.latency_test(
            repetitions=repetitions,
            warmup=warmup,
        )

        filename = "latency_metrics_200MHz.txt"
        with open(filename, "w") as file:
            for key, value in res.items():

                display_key = key

                if key == "hardware_mean[ms]":
                    display_key = "Latency HW[ms]"

                elif key == "end_to_end_mean[ms]":
                    display_key = "Latency End-to-End[ms]"

                if isinstance(value, (int, np.integer)):
                    file.write(f"{display_key:<45}: {int(value)}\n")
                else:
                    file.write(f"{display_key:<45}: {float(value):.6f}\n")

        print(f"Results written to {filename}")
        with open(filename, "r") as file:
            print(file.read())

    else:
        raise Exception(
            "Exec mode has to be execute, throughput_test or latency_test"
        )
