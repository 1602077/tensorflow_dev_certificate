	????ƛD@????ƛD@!????ƛD@	'?ع???'?ع???!'?ع???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:????ƛD@??~j??@A㥛? ?A@Y?&1???rEagerKernelExecute 0*	     @Q@2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCachey?&1???!??(?3JD@)????????1??v`?B@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch/?$???!?7??Mo>@)/?$???1?7??Mo>@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism;?O??n??!!Y?BJ@)???Q???1????7?5@:Preprocessing2F
Iterator::ModelˡE?????!;0?̵M@){?G?zt?1?(?3J?@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl?~j?t?h?!???,d@)?~j?t?h?1???,d@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9&?ع???I??p)?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??~j??@??~j??@!??~j??@      ?!       "      ?!       *      ?!       2	㥛? ?A@㥛? ?A@!㥛? ?A@:      ?!       B      ?!       J	?&1????&1???!?&1???R      ?!       Z	?&1????&1???!?&1???b      ?!       JCPU_ONLYY&?ع???b q??p)?X@