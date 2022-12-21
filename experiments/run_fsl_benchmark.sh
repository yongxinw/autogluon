
datasets=("caltech256" "cifar10" "cifar100" "oxfordiiipet" "sun397")
for f in ${datasets[@]}; do
  echo running benchmark $f
  python fsl_benchmarks.py $f 2>&1 | tee logs/fsl_benchmarks_$f.log
done
#bash fsl_benchmarks.py
