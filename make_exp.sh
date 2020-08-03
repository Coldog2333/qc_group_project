mkdir -p errs
for seed in 111 222 333 444 555
do
    for method in qrac conv_h conv_v conv_h_overlap conv_all 16px
    do
        python qasm_nn_mnist.py --method $method --seed $seed --epochs 5 --layer 2
    done
done
