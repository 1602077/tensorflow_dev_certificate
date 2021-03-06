?	u?V?E@u?V?E@!u?V?E@	?gVF'????gVF'???!?gVF'???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:u?V?E@ffffff@AR????B@Y?Zd;??rEagerKernelExecute 0*	     @R@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???Mb??!?#F?P@)???Mb??1?#F?P@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Zd;??!?~????T@)y?&1???1ٲe˖-3@:Preprocessing2F
Iterator::ModelL7?A`???!?I?&M?V@){?G?zt?1[?lٲe@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCachey?&1?|?!ٲe˖-#@)????Mbp?1?^?z??@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl?~j?t?h?!8p@)?~j?t?h?18p@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?gVF'???I????%?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ffffff@ffffff@!ffffff@      ?!       "      ?!       *      ?!       2	R????B@R????B@!R????B@:      ?!       B      ?!       J	?Zd;???Zd;??!?Zd;??R      ?!       Z	?Zd;???Zd;??!?Zd;??b      ?!       JCPU_ONLYY?gVF'???b q????%?X@Y      Y@qg??????"?
both?Your program is POTENTIALLY input-bound because 15.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 