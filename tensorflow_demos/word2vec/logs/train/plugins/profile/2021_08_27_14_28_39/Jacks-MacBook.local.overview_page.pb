?	??ʡE?!@??ʡE?!@!??ʡE?!@	?ƃm?D0@?ƃm?D0@!?ƃm?D0@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??ʡE?!@㥛? ???AX9??v>@Y?K7?A`??rEagerKernelExecute 0*	     h?@2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle???Q???!?C????Q@)?/?$??1$?6??wJ@:Preprocessing2?
`Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::TensorSlice???"??~??!??[?՘D@)??"??~??1??[?՘D@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????????!?F??@)????????1?F??@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?v??/??!???¯@)y?&1???1۴??I??:Preprocessing2F
Iterator::Modelh??|?5??!:Blӊ{@)????Mbp?1?C???x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 16.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t10.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?ƃm?D0@IH?d??T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	㥛? ???㥛? ???!㥛? ???      ?!       "      ?!       *      ?!       2	X9??v>@X9??v>@!X9??v>@:      ?!       B      ?!       J	?K7?A`???K7?A`??!?K7?A`??R      ?!       Z	?K7?A`???K7?A`??!?K7?A`??b      ?!       JCPU_ONLYY?ƃm?D0@b qH?d??T@Y      Y@q?? Q?@"?	
both?Your program is MODERATELY input-bound because 16.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nomoderate"t10.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.2no:
Refer to the TF2 Profiler FAQ2"CPU: B 