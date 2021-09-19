python3 model.py \
    --model_name_or_path="bert-base-cased" \
    --train_file="finaltrain_regressor.csv" \
    --validation_file="finalvalidate_regressor.csv" \
    --output_dir="output_dir" \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \