
# LING
python train.py -train1 ../processed_data/new_sl/kem.csv -train2 ../processed_data/new_sl/vet.csv -val ../processed_data/new_sl/bim.csv -test ../processed_data/new_sl/jez.csv -gold_val ../termlists_2/rsdo5bim.terms2 -gold_test ../termlists_2/rsdo5jez.terms2 -store ./res12/ -preds ../results/bim_ling1.txt  -log ../results/bim_ling_log1.txt
python train.py -train1 ../processed_data/new_sl/kem.csv -train2 ../processed_data/new_sl/bim.csv -val ../processed_data/new_sl/vet.csv -test ../processed_data/new_sl/jez.csv -gold_val ../termlists_2/rsdo5vet.terms2 -gold_test ../termlists_2/rsdo5jez.terms2 -store ./res12/ -preds ../results/vet_ling1.txt  -log ../results/vet_ling_log1.txt
python train.py -train1 ../processed_data/new_sl/bim.csv -train2 ../processed_data/new_sl/vet.csv -val ../processed_data/new_sl/kem.csv -test ../processed_data/new_sl/jez.csv -gold_val ../termlists_2/rsdo5kem.terms2 -gold_test ../termlists_2/rsdo5jez.terms2 -store ./res12/ -preds ../results/kem_ling1.txt  -log ../results/kem_ling_log1.txt

# KEM
python train.py -train1 ../processed_data/new_sl/bim.csv -train2 ../processed_data/new_sl/vet.csv -val ../processed_data/new_sl/jez.csv -test ../processed_data/new_sl/kem.csv -gold_val ../termlists_2/rsdo5jez.terms2 -gold_test ../termlists_2/rsdo5kem.terms2 -store ./res12/ -preds ../results/ling_kem1.txt  -log ../results/ling_kem_log1.txt
python train.py -train1 ../processed_data/new_sl/jez.csv -train2 ../processed_data/new_sl/vet.csv -val ../processed_data/new_sl/bim.csv -test ../processed_data/new_sl/kem.csv -gold_val ../termlists_2/rsdo5bim.terms2 -gold_test ../termlists_2/rsdo5kem.terms2 -store ./res12/ -preds ../results/bim_kem1.txt  -log ../results/bim_kem_log1.txt
python train.py -train1 ../processed_data/new_sl/bim.csv -train2 ../processed_data/new_sl/jez.csv -val ../processed_data/new_sl/vet.csv -test ../processed_data/new_sl/kem.csv -gold_val ../termlists_2/rsdo5vet.terms2 -gold_test ../termlists_2/rsdo5kem.terms2 -store ./res12/ -preds ../results/vet_kem1.txt  -log ../results/vet_kem_log1.txt

# VET
python train.py -train1 ../processed_data/new_sl/bim.csv -train2 ../processed_data/new_sl/kem.csv -val ../processed_data/new_sl/jez.csv -test ../processed_data/new_sl/vet.csv -gold_val ../termlists_2/rsdo5jez.terms2 -gold_test ../termlists_2/rsdo5vet.terms2 -store ./res12/ -preds ../results/ling_vet1.txt  -log ../results/ling_vet_log1.txt
python train.py -train1 ../processed_data/new_sl/bim.csv -train2 ../processed_data/new_sl/jez.csv -val ../processed_data/new_sl/kem.csv -test ../processed_data/new_sl/vet.csv -gold_val ../termlists_2/rsdo5kem.terms2 -gold_test ../termlists_2/rsdo5vet.terms2 -store ./res12/ -preds ../results/kem_vet1.txt  -log ../results/kem_vet_log1.txt
python train.py -train1 ../processed_data/new_sl/jez.csv -train2 ../processed_data/new_sl/kem.csv -val ../processed_data/new_sl/bim.csv -test ../processed_data/new_sl/vet.csv -gold_val ../termlists_2/rsdo5bim.terms2 -gold_test ../termlists_2/rsdo5vet.terms2 -store ./res12/ -preds ../results/bim_vet1.txt  -log ../results/bim_vet_log1.txt

# BIM
python train.py -train1 ../processed_data/new_sl/kem.csv -train2 ../processed_data/new_sl/vet.csv -val ../processed_data/new_sl/jez.csv -test ../processed_data/new_sl/bim.csv -gold_val ../termlists_2/rsdo5jez.terms2 -gold_test ../termlists_2/rsdo5bim.terms2 -store ./res12/ -preds ../results/ling_bim1.txt  -log ../results/ling_bim_log1.txt
python train.py -train1 ../processed_data/new_sl/jez.csv -train2 ../processed_data/new_sl/vet.csv -val ../processed_data/new_sl/kem.csv -test ../processed_data/new_sl/bim.csv -gold_val ../termlists_2/rsdo5kem.terms2 -gold_test ../termlists_2/rsdo5bim.terms2 -store ./res12/ -preds ../results/kem_bim1.txt  -log ../results/kem_bim_log1.txt
python train.py -train1 ../processed_data/new_sl/kem.csv -train2 ../processed_data/new_sl/jez.csv -val ../processed_data/new_sl/vet.csv -test ../processed_data/new_sl/bim.csv -gold_val ../termlists_2/rsdo5vet.terms2 -gold_test ../termlists_2/rsdo5bim.terms2 -store ./res12/ -preds ../results/ling_bim1.txt  -log ../results/vet_bim_log1.txt