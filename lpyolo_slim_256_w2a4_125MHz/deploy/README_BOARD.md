# slim it10 SIMD=4 @ 125 MHz (WNS +0.011) — ZedBoard
NOTE: idma0 = WEIGHTS (external), the image enters via idma1.
runtime_weights/ resolves relative to the cwd: ALWAYS run from driver/.

    cd /home/xilinx/zed_4/driver
    sudo python3 driver.py --exec_mode throughput_test --batchsize 100 \
        --bitfile ../bitfile/finn-accel.bit --platform zynq-iodma
    cat nw_metrics.txt

    sudo python3 driver.py --exec_mode execute --bitfile ../bitfile/finn-accel.bit \
        --inputfile input.npy --outputfile out0.npy out1.npy
    python3 -c "import numpy as np; [print(f, (np.load(f)!=np.load('expected_'+f)).sum()) for f in ['out0.npy','out1.npy']]"

After a DMA hang: do NOT reload the overlay — power-cycle.
