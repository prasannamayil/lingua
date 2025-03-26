NPROC=$1
API_KEY=$2
PATH_LIST=$3
DUMP_DIR=$4
METRIC_LOG_DIR=$5

for path in $(cat $PATH_LIST); do
    step=$(basename $path)
    model_dir=$(dirname $(dirname $path))
    model=$(basename $model_dir)

    bash run_eval.sh $NPROC $API_KEY ckpt_dir=$path dump_dir=$DUMP_DIR/$model metric_log_dir=$METRIC_LOG_DIR/$model global_step=$step
done

