	B`??""@B`??""@!B`??""@	???)?@???)?@!???)?@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:B`??""@?Zd;??AZd;? @YJ+???rEagerKernelExecute 0*	     ??@2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle?;?O??n??!ogZ{?P@)?Zd;???1????L@:Preprocessing2?
`Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::TensorSlice?
ףp=
??!????C@)
ףp=
??1????C@:Preprocessing2F
Iterator::ModelˡE?????!???6q?@)ˡE?????1???6q?@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????Mb??!??'?????)????Mb??1??'?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s5.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???)?@IF?`b}?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Zd;???Zd;??!?Zd;??      ?!       "      ?!       *      ?!       2	Zd;? @Zd;? @!Zd;? @:      ?!       B      ?!       J	J+???J+???!J+???R      ?!       Z	J+???J+???!J+???b      ?!       JCPU_ONLYY???)?@b qF?`b}?W@