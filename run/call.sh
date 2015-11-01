LD_LIBRARY_PATH=../build:$LD_LIBRARY_PATH

algo=ostmr #gibbs-lda pa-i osgd-topic-flow osgd-topic-fixed octr
K=20 #5 10 20 100
U=69878
V=10681
T=7689
I=1
J=4
Jb=1
e=0
c=0.02
a0=0.5
b0=0.45
lu=2
lv=2
su=0.5
sv=8
train=train_in.dat
test=test_in.dat
learn_cnt=9000000
test_cnt=1000000
basedir='../data/'
test_interval=50000
cdk_file=null_file
ofm_method='in-matrix'

#./build/cppWrapper ctr $K $U $V $T $I $J $Jburnin $e $c $a0 $b0 $lambda_u $lambda_v $__ $__ $train_file $test_file $learn_cnt $test_cnt $basedir $test_interval $cdk_file $ofm_method

#./build/cppWrapper ostmr $K $U $V $T $I $J $Jburnin $e $c $a0 $b0 $sigma_r $sigma_r $sigma_u $sigma_v $train_file $test_file $learn_cnt $test_cnt $basedir $test_interval $cdk_file $ofm_method

for r in 1 2 4
do
	for u in 4
	do
		for v in 0.5 1 2 4 8
		#for v in "0.125" "0.25" "0.5" 1 2 4 8 16
		do
			#if [ "$r" -gt "$u" ] && [ "$v" -gt "$u" ] && [ "$v" -gt "$r" ]
			#then
				echo "$r $train $test"
				../build/cppWrapper $algo $K $U $V $T $I $J $Jb $e $c $a0 $b0 $r $r $u $v $train $test $learn_cnt $test_cnt $basedir $test_interval $cdk_file $ofm_method &
                sleep 2
			#fi
		done
	done
done

wait
mkdir tmp
mv *mean* *cdk* *phi* *train* *time* tmp
