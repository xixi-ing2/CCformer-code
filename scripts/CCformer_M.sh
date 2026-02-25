

model_name=TimeXer

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TY/ \
  --data_path TY2015.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 256 \
  --batch_size 32 \
  --des 'exp' \
  --itr 1


python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TY/ \
  --data_path TY2015.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 72 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des 'Exp' \
  --d_model 128 \
  --batch_size 32 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TY/ \
  --data_path TY2015.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 112 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 32 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TY/ \
  --data_path TY2015.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 144 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 1024 \
  --batch_size 16 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TY/ \
  --data_path TY2016.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 256 \
  --batch_size 32 \
  --des 'exp' \
  --itr 1


python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TY/ \
  --data_path TY2016.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 72 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des 'Exp' \
  --d_model 128 \
  --batch_size 32 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TY/ \
  --data_path TY2016.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 112 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 32 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TY/ \
  --data_path TY2016.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 144 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 1024 \
  --batch_size 16 \
  --itr 1

  python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TY/ \
  --data_path TY2017.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 256 \
  --batch_size 32 \
  --des 'exp' \
  --itr 1


python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TY/ \
  --data_path TY2017.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 72 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des 'Exp' \
  --d_model 128 \
  --batch_size 32 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TY/ \
  --data_path TY2017.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 112 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 32 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TY/ \
  --data_path TY2017.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 144 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 1024 \
  --batch_size 16 \
  --itr 1