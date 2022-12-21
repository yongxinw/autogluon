
datasets=("belgalogos" "cub200" "dtd47" "eflooddepth" "fgvc_aircrafts" "magnetictiledefects" "nike" "food101" "ifood" "minc2500" "nwpu_resisc45" "stanforddogs" "stanfordcars" "semartschool" "malariacellimages" "mit67" "oxfordflowers")
#datasets=("belgalogos")
timeStamp=`date '+%m-%d-%YT%H-%m-%S'`
for f in ${datasets[@]}; do
  echo running benchmark $f
#  echo saving to logs/fewshot/fsl_cl_benchmarks_${f}_${timeStamp}.log
#  python fsl_cl_benchmarks.py --dataset $f --fewshot 2>&1 | tee logs/fsl_cl_benchmarks_$f.log
  python fsl_cl_benchmarks.py --dataset $f --fewshot 2>&1 | tee logs/fewshot/fsl_cl_benchmarks_${f}_${timeStamp}.log
#  python fsl_cl_benchmarks.py --dataset $f 2>&1 | tee logs/fullshot/fsl_cl_benchmarks_$f_$timeStamp.log
done
#bash fsl_benchmarks.py 
