	y?&1,F@y?&1,F@!y?&1,F@	??S?,?????S?,???!??S?,???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:y?&1,F@??n??@A??S??C@Y?p=
ף??rEagerKernelExecute 0*	     ?@@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?I+???!??????@@)?I+???1??????@@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache?~j?t???!/?袋.B@);?O??n??1E]t?E;@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??~j?t??!?&?l??L@)????Mb??1>???>8@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl?~j?t?h?!/?袋."@)?~j?t?h?1/?袋."@:Preprocessing2F
Iterator::Model/?$???!?E]t?O@)????Mb`?1>???>@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??S?,???I֓i??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??n??@??n??@!??n??@      ?!       "      ?!       *      ?!       2	??S??C@??S??C@!??S??C@:      ?!       B      ?!       J	?p=
ף???p=
ף??!?p=
ף??R      ?!       Z	?p=
ף???p=
ף??!?p=
ף??b      ?!       JCPU_ONLYY??S?,???b q֓i??X@