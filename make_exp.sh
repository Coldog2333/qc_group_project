mkdir -p errs
for seed in 111 222 333 444 555
do
    for method in 8px  #qrac conv_h conv_v conv_h_overlap conv_all 16px
    do
        python qasm_nn_mnist.py --method $method --seed $seed --epochs 10 --layer 1
    done
done
