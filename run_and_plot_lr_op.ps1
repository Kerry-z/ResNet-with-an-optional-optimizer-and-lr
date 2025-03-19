# conda activate your_env

$python = "python"  

$depth = 3
$n_iter = 64000
$lrs_sgd = @(0.01, 0.1, 0.5)
$lrs_momentum = @(0.01, 0.1, 0.5)
$lrs_adam = @(0.001, 0.0005, 0.0001)

New-Item -ItemType Directory -Force -Path logs | Out-Null

foreach ($lr in $lrs_sgd) {
    Write-Host "Running SGD with lr=$lr"
    & $python train.py --n $depth --n_iter $n_iter --optimizer sgd --lr $lr --checkpoint_dir "checkpoint_sgd_$lr" | Out-File "logs/sgd_lr$lr.log"
}

foreach ($lr in $lrs_momentum) {
    Write-Host "Running SGD + Momentum with lr=$lr"
    & $python train.py --n $depth --n_iter $n_iter --optimizer momentum --lr $lr --checkpoint_dir "checkpoint_momentum_$lr" | Out-File "logs/momentum_lr$lr.log"
}

foreach ($lr in $lrs_adam) {
    Write-Host "Running Adam with lr=$lr"
    & $python train.py --n $depth --n_iter $n_iter --optimizer adam --lr $lr --checkpoint_dir "checkpoint_adam_$lr" | Out-File "logs/adam_lr$lr.log"
}

Write-Host "✅ All experiments finished!"
