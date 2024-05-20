cd ../..


echo "=> generate data"

cd data/cifar10 || exit 1
rm -rf all_data
python generate_data.py \
    --n_tasks 2 \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345

cd ../..

echo "Train centralized model"
python centralized.py \
    cifar10 \
    --n_rounds 10 \
    --participation_probs 1.0 1.0 \
    --bz 64 \
    --lr 0.001 \
    --log_freq 1 \
    --device cuda \
    --optimizer sgd \
    --server_optimizer sgd \
    --logs_dir logs/cifar10/fedavg/lr_0.001/seed_1234 \
    --seed 1234 \
    --verbose 1
