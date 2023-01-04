for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"
   export "$KEY"="$VALUE"
done

# use here your expected variables
#echo "STEPS = $STEPS"
#echo "REPOSITORY_NAME = $REPOSITORY_NAME"
#echo "EXTRA_VALUES = $EXTRA_VALUES"

if [ -z "${DATASETS}" ]; then
  echo Using all datasets
  datasets=("belgalogos" "cub200" "dtd47" "eflooddepth" "fgvc_aircrafts" "magnetictiledefects" "nike" "food101" "ifood" "minc2500" "nwpu_resisc45" "stanforddogs" "stanfordcars" "semartschool" "malariacellimages" "mit67" "oxfordflowers")
else
  echo Using datasets: ${DATASETS}
  datasets=${DATASETS}
fi

if [ ! -z "${MODE}" ]; then
  if [ "${MODE}" = "fewshot" ]; then
    echo Using fewshot mode
    mode=fewshot
  elif [ "${MODE}" = "fullshot" ]; then
    echo Using fullshot mode
    mode=fullshot
  else
    echo MODE ${MODE} not recognized. Please specify either fewshot or fullshot
    exit
  fi
else
  echo MODE not set. Please specify either fewshot or fullshot
  exit
fi

if [ ! -z "${LOGDIR}" ]; then
  logDir=${LOGDIR}
  if [ ! -d ${logDir} ]; then
    echo Directory doesn\'t exit: ${logDir}
    exit
  fi
  echo logDir: ${logDir}
else
  timeStamp=`date '+%m-%d-%YT%H-%m-%S'`
  if [ ! -z "${SUFFIX}" ]; then
    suffix="${SUFFIX}"
    logDir=./logs/${MODE}/${timeStamp}_${suffix}
  else
    suffix=""
    logDir=./logs/${MODE}/${timeStamp}
  fi

  echo logDir: ${logDir}
  if [ ! -d "${logDir}" ]; then
    mkdir -p ${logDir}
  fi
fi

for f in ${datasets[@]}; do
  if [ "${mode}" = "fewshot" ]; then
    python fsl_cl_benchmarks.py --dataset $f --fewshot --logdir ${logDir} 2>&1 | tee ${logDir}/$f.log
  else
    python fsl_cl_benchmarks.py --dataset $f --logdir ${logDir} 2>&1 | tee ${logDir}/$f.log
  fi
done
#for f in ${datasets[@]}; do
#  echo running benchmark $f
##  echo saving to logs/fewshot/fsl_cl_benchmarks_${f}_${timeStamp}.log
##  python fsl_cl_benchmarks.py --dataset $f --fewshot 2>&1 | tee logs/fsl_cl_benchmarks_$f.log
#  python fsl_cl_benchmarks.py --dataset $f --fewshot 2>&1 | tee logs/fewshot/fsl_cl_benchmarks_${f}_${timeStamp}.log
##  python fsl_cl_benchmarks.py --dataset $f 2>&1 | tee logs/fullshot/fsl_cl_benchmarks_$f_$timeStamp.log
#done
#bash fsl_benchmarks.py 
