#!/bin/bash
dir=$PWD
source /home/luca/environments/spark/bin/activate
cd /home/luca/cococcia-shop

for i in {2..2}
do
    for j in {1..1}
    do
        docker-compose down
        docker-compose -f docker-compose.tracing.yml stop zipkin
        python generate_injections.py $i
        cd config
        mvn clean package
        cd ..
        docker-compose up -d
        sleep 2m
        locust --host=http://localhost  --no-web -c 40 -r 1 --run-time 10s
        docker-compose -f docker-compose.tracing.yml up -d zipkin
        sleep 30s
        t1=$( date +%s )
        locust --host=http://localhost  --no-web -c 40 -r 1 --run-time 10m
        t2=$( date +%s )
        echo $i';'$t1';'$t2  >> $dir/experiments.csv
    done
done