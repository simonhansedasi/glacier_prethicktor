
À
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ø¦
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:	*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:	*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
z
dense_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	%* 
shared_namedense_45/kernel
s
#dense_45/kernel/Read/ReadVariableOpReadVariableOpdense_45/kernel*
_output_shapes

:	%*
dtype0
r
dense_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:%*
shared_namedense_45/bias
k
!dense_45/bias/Read/ReadVariableOpReadVariableOpdense_45/bias*
_output_shapes
:%*
dtype0
z
dense_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:%* 
shared_namedense_46/kernel
s
#dense_46/kernel/Read/ReadVariableOpReadVariableOpdense_46/kernel*
_output_shapes

:%*
dtype0
r
dense_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_46/bias
k
!dense_46/bias/Read/ReadVariableOpReadVariableOpdense_46/bias*
_output_shapes
:*
dtype0
z
dense_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_47/kernel
s
#dense_47/kernel/Read/ReadVariableOpReadVariableOpdense_47/kernel*
_output_shapes

:*
dtype0
r
dense_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_47/bias
k
!dense_47/bias/Read/ReadVariableOpReadVariableOpdense_47/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/dense_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	%*'
shared_nameAdam/dense_45/kernel/m

*Adam/dense_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_45/kernel/m*
_output_shapes

:	%*
dtype0

Adam/dense_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:%*%
shared_nameAdam/dense_45/bias/m
y
(Adam/dense_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_45/bias/m*
_output_shapes
:%*
dtype0

Adam/dense_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:%*'
shared_nameAdam/dense_46/kernel/m

*Adam/dense_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_46/kernel/m*
_output_shapes

:%*
dtype0

Adam/dense_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_46/bias/m
y
(Adam/dense_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_46/bias/m*
_output_shapes
:*
dtype0

Adam/dense_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_47/kernel/m

*Adam/dense_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_47/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_47/bias/m
y
(Adam/dense_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_47/bias/m*
_output_shapes
:*
dtype0

Adam/dense_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	%*'
shared_nameAdam/dense_45/kernel/v

*Adam/dense_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_45/kernel/v*
_output_shapes

:	%*
dtype0

Adam/dense_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:%*%
shared_nameAdam/dense_45/bias/v
y
(Adam/dense_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_45/bias/v*
_output_shapes
:%*
dtype0

Adam/dense_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:%*'
shared_nameAdam/dense_46/kernel/v

*Adam/dense_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_46/kernel/v*
_output_shapes

:%*
dtype0

Adam/dense_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_46/bias/v
y
(Adam/dense_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_46/bias/v*
_output_shapes
:*
dtype0

Adam/dense_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_47/kernel/v

*Adam/dense_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_47/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_47/bias/v
y
(Adam/dense_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_47/bias/v*
_output_shapes
:*
dtype0
z
ConstConst*
_output_shapes

:	*
dtype0*=
value4B2	"$m:BQ/AAnøE²õ2E¸HE6%@;C%êE
|
Const_1Const*
_output_shapes

:	*
dtype0*=
value4B2	"$WwÀCÓE:iBE¡IsIrIölA;eF÷AJ

NoOpNoOp
/
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*Ö.
valueÌ.BÉ. BÂ.
è
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¾

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
	keras_api
_adapt_function*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
¦

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
°
/iter

0beta_1

1beta_2
	2decay
3learning_ratemNmOmP mQ'mR(mSvTvUvV vW'vX(vY*
C
0
1
2
3
4
5
 6
'7
(8*
.
0
1
2
 3
'4
(5*
* 
°
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

9serving_default* 
* 
* 
* 
* 
RL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEcount5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
_Y
VARIABLE_VALUEdense_45/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_45/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_46/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_46/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 

?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_47/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_47/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 

Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*
 
0
1
2
3*

I0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	Jtotal
	Kcount
L	variables
M	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

J0
K1*

L	variables*
|
VARIABLE_VALUEAdam/dense_45/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_45/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_46/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_46/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_47/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_47/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_45/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_45/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_46/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_46/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_47/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_47/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

'serving_default_normalization_159_inputPlaceholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ã
StatefulPartitionedCallStatefulPartitionedCall'serving_default_normalization_159_inputConstConst_1dense_45/kerneldense_45/biasdense_46/kerneldense_46/biasdense_47/kerneldense_47/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_122360
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ì

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp#dense_45/kernel/Read/ReadVariableOp!dense_45/bias/Read/ReadVariableOp#dense_46/kernel/Read/ReadVariableOp!dense_46/bias/Read/ReadVariableOp#dense_47/kernel/Read/ReadVariableOp!dense_47/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_45/kernel/m/Read/ReadVariableOp(Adam/dense_45/bias/m/Read/ReadVariableOp*Adam/dense_46/kernel/m/Read/ReadVariableOp(Adam/dense_46/bias/m/Read/ReadVariableOp*Adam/dense_47/kernel/m/Read/ReadVariableOp(Adam/dense_47/bias/m/Read/ReadVariableOp*Adam/dense_45/kernel/v/Read/ReadVariableOp(Adam/dense_45/bias/v/Read/ReadVariableOp*Adam/dense_46/kernel/v/Read/ReadVariableOp(Adam/dense_46/bias/v/Read/ReadVariableOp*Adam/dense_47/kernel/v/Read/ReadVariableOp(Adam/dense_47/bias/v/Read/ReadVariableOpConst_2*)
Tin"
 2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_122574

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_45/kerneldense_45/biasdense_46/kerneldense_46/biasdense_47/kerneldense_47/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount_1Adam/dense_45/kernel/mAdam/dense_45/bias/mAdam/dense_46/kernel/mAdam/dense_46/bias/mAdam/dense_47/kernel/mAdam/dense_47/bias/mAdam/dense_45/kernel/vAdam/dense_45/bias/vAdam/dense_46/kernel/vAdam/dense_46/bias/vAdam/dense_47/kernel/vAdam/dense_47/bias/v*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_122668¬
Ò

¶
.__inference_sequential_15_layer_call_fn_122056
normalization_159_input
unknown
	unknown_0
	unknown_1:	%
	unknown_2:%
	unknown_3:%
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallnormalization_159_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_15_layer_call_and_return_conditional_losses_122037o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1
_user_specified_namenormalization_159_input:$ 

_output_shapes

:	:$ 

_output_shapes

:	
=
²
__inference__traced_save_122574
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	.
*savev2_dense_45_kernel_read_readvariableop,
(savev2_dense_45_bias_read_readvariableop.
*savev2_dense_46_kernel_read_readvariableop,
(savev2_dense_46_bias_read_readvariableop.
*savev2_dense_47_kernel_read_readvariableop,
(savev2_dense_47_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_45_kernel_m_read_readvariableop3
/savev2_adam_dense_45_bias_m_read_readvariableop5
1savev2_adam_dense_46_kernel_m_read_readvariableop3
/savev2_adam_dense_46_bias_m_read_readvariableop5
1savev2_adam_dense_47_kernel_m_read_readvariableop3
/savev2_adam_dense_47_bias_m_read_readvariableop5
1savev2_adam_dense_45_kernel_v_read_readvariableop3
/savev2_adam_dense_45_bias_v_read_readvariableop5
1savev2_adam_dense_46_kernel_v_read_readvariableop3
/savev2_adam_dense_46_bias_v_read_readvariableop5
1savev2_adam_dense_47_kernel_v_read_readvariableop3
/savev2_adam_dense_47_bias_v_read_readvariableop
savev2_const_2

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: °
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ù
valueÏBÌB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH§
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¢
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop*savev2_dense_45_kernel_read_readvariableop(savev2_dense_45_bias_read_readvariableop*savev2_dense_46_kernel_read_readvariableop(savev2_dense_46_bias_read_readvariableop*savev2_dense_47_kernel_read_readvariableop(savev2_dense_47_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_45_kernel_m_read_readvariableop/savev2_adam_dense_45_bias_m_read_readvariableop1savev2_adam_dense_46_kernel_m_read_readvariableop/savev2_adam_dense_46_bias_m_read_readvariableop1savev2_adam_dense_47_kernel_m_read_readvariableop/savev2_adam_dense_47_bias_m_read_readvariableop1savev2_adam_dense_45_kernel_v_read_readvariableop/savev2_adam_dense_45_bias_v_read_readvariableop1savev2_adam_dense_46_kernel_v_read_readvariableop/savev2_adam_dense_46_bias_v_read_readvariableop1savev2_adam_dense_47_kernel_v_read_readvariableop/savev2_adam_dense_47_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *+
dtypes!
2		
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Å
_input_shapes³
°: :	:	: :	%:%:%:::: : : : : : : :	%:%:%::::	%:%:%:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:	: 

_output_shapes
:	:

_output_shapes
: :$ 

_output_shapes

:	%: 

_output_shapes
:%:$ 

_output_shapes

:%: 

_output_shapes
::$ 

_output_shapes

:: 	

_output_shapes
::


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:	%: 

_output_shapes
:%:$ 

_output_shapes

:%: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:	%: 

_output_shapes
:%:$ 

_output_shapes

:%: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
­
Ò
I__inference_sequential_15_layer_call_and_return_conditional_losses_122037

inputs
normalization_159_sub_y
normalization_159_sqrt_x!
dense_45_121998:	%
dense_45_122000:%!
dense_46_122015:%
dense_46_122017:!
dense_47_122031:
dense_47_122033:
identity¢ dense_45/StatefulPartitionedCall¢ dense_46/StatefulPartitionedCall¢ dense_47/StatefulPartitionedCallo
normalization_159/subSubinputsnormalization_159_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a
normalization_159/SqrtSqrtnormalization_159_sqrt_x*
T0*
_output_shapes

:	`
normalization_159/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_159/MaximumMaximumnormalization_159/Sqrt:y:0$normalization_159/Maximum/y:output:0*
T0*
_output_shapes

:	
normalization_159/truedivRealDivnormalization_159/sub:z:0normalization_159/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 dense_45/StatefulPartitionedCallStatefulPartitionedCallnormalization_159/truediv:z:0dense_45_121998dense_45_122000*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_45_layer_call_and_return_conditional_losses_121997
 dense_46/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0dense_46_122015dense_46_122017*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_122014
 dense_47/StatefulPartitionedCallStatefulPartitionedCall)dense_46/StatefulPartitionedCall:output:0dense_47_122031dense_47_122033*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_122030x
IdentityIdentity)dense_47/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
NoOpNoOp!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:	:$ 

_output_shapes

:	
á
ã
I__inference_sequential_15_layer_call_and_return_conditional_losses_122201
normalization_159_input
normalization_159_sub_y
normalization_159_sqrt_x!
dense_45_122185:	%
dense_45_122187:%!
dense_46_122190:%
dense_46_122192:!
dense_47_122195:
dense_47_122197:
identity¢ dense_45/StatefulPartitionedCall¢ dense_46/StatefulPartitionedCall¢ dense_47/StatefulPartitionedCall
normalization_159/subSubnormalization_159_inputnormalization_159_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a
normalization_159/SqrtSqrtnormalization_159_sqrt_x*
T0*
_output_shapes

:	`
normalization_159/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_159/MaximumMaximumnormalization_159/Sqrt:y:0$normalization_159/Maximum/y:output:0*
T0*
_output_shapes

:	
normalization_159/truedivRealDivnormalization_159/sub:z:0normalization_159/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 dense_45/StatefulPartitionedCallStatefulPartitionedCallnormalization_159/truediv:z:0dense_45_122185dense_45_122187*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_45_layer_call_and_return_conditional_losses_121997
 dense_46/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0dense_46_122190dense_46_122192*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_122014
 dense_47/StatefulPartitionedCallStatefulPartitionedCall)dense_46/StatefulPartitionedCall:output:0dense_47_122195dense_47_122197*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_122030x
IdentityIdentity)dense_47/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
NoOpNoOp!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall:i e
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1
_user_specified_namenormalization_159_input:$ 

_output_shapes

:	:$ 

_output_shapes

:	


õ
D__inference_dense_45_layer_call_and_return_conditional_losses_122426

inputs0
matmul_readvariableop_resource:	%-
biasadd_readvariableop_resource:%
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	%*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:%*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
¹'
Ò
__inference_adapt_step_122406
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:	'
readvariableop_2_resource:	¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢IteratorGetNext¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢add/ReadVariableOp±
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*&
output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
output_types
2h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:	*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:	
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:	*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:	*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:	*
squeeze_dims
 a
ShapeShapeIteratorGetNext:components:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: K
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: I
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0P
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:	X
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:	G
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:	d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0V
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:	J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:	f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:	*
dtype0V
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:	E
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:	V
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:	L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @N
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:	Z
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:	I
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:	I
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:	
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
Å

)__inference_dense_46_layer_call_fn_122435

inputs
unknown:%
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_122014o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ%: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
 
_user_specified_nameinputs


õ
D__inference_dense_46_layer_call_and_return_conditional_losses_122014

inputs0
matmul_readvariableop_resource:%-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:%*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
 
_user_specified_nameinputs
­
Ò
I__inference_sequential_15_layer_call_and_return_conditional_losses_122135

inputs
normalization_159_sub_y
normalization_159_sqrt_x!
dense_45_122119:	%
dense_45_122121:%!
dense_46_122124:%
dense_46_122126:!
dense_47_122129:
dense_47_122131:
identity¢ dense_45/StatefulPartitionedCall¢ dense_46/StatefulPartitionedCall¢ dense_47/StatefulPartitionedCallo
normalization_159/subSubinputsnormalization_159_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a
normalization_159/SqrtSqrtnormalization_159_sqrt_x*
T0*
_output_shapes

:	`
normalization_159/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_159/MaximumMaximumnormalization_159/Sqrt:y:0$normalization_159/Maximum/y:output:0*
T0*
_output_shapes

:	
normalization_159/truedivRealDivnormalization_159/sub:z:0normalization_159/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 dense_45/StatefulPartitionedCallStatefulPartitionedCallnormalization_159/truediv:z:0dense_45_122119dense_45_122121*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_45_layer_call_and_return_conditional_losses_121997
 dense_46/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0dense_46_122124dense_46_122126*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_122014
 dense_47/StatefulPartitionedCallStatefulPartitionedCall)dense_46/StatefulPartitionedCall:output:0dense_47_122129dense_47_122131*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_122030x
IdentityIdentity)dense_47/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
NoOpNoOp!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:	:$ 

_output_shapes

:	
Å

)__inference_dense_47_layer_call_fn_122455

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_122030o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç	
õ
D__inference_dense_47_layer_call_and_return_conditional_losses_122465

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
p
µ
"__inference__traced_restore_122668
file_prefix#
assignvariableop_mean:	)
assignvariableop_1_variance:	"
assignvariableop_2_count:	 4
"assignvariableop_3_dense_45_kernel:	%.
 assignvariableop_4_dense_45_bias:%4
"assignvariableop_5_dense_46_kernel:%.
 assignvariableop_6_dense_46_bias:4
"assignvariableop_7_dense_47_kernel:.
 assignvariableop_8_dense_47_bias:&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: #
assignvariableop_14_total: %
assignvariableop_15_count_1: <
*assignvariableop_16_adam_dense_45_kernel_m:	%6
(assignvariableop_17_adam_dense_45_bias_m:%<
*assignvariableop_18_adam_dense_46_kernel_m:%6
(assignvariableop_19_adam_dense_46_bias_m:<
*assignvariableop_20_adam_dense_47_kernel_m:6
(assignvariableop_21_adam_dense_47_bias_m:<
*assignvariableop_22_adam_dense_45_kernel_v:	%6
(assignvariableop_23_adam_dense_45_bias_v:%<
*assignvariableop_24_adam_dense_46_kernel_v:%6
(assignvariableop_25_adam_dense_46_bias_v:<
*assignvariableop_26_adam_dense_47_kernel_v:6
(assignvariableop_27_adam_dense_47_bias_v:
identity_29¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9³
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ù
valueÏBÌB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHª
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B °
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_45_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_45_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_46_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_46_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_47_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_47_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp&assignvariableop_13_adam_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_45_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_45_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_46_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_46_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_47_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_47_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_45_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_45_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_46_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_46_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_47_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_47_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ·
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: ¤
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
á
ã
I__inference_sequential_15_layer_call_and_return_conditional_losses_122227
normalization_159_input
normalization_159_sub_y
normalization_159_sqrt_x!
dense_45_122211:	%
dense_45_122213:%!
dense_46_122216:%
dense_46_122218:!
dense_47_122221:
dense_47_122223:
identity¢ dense_45/StatefulPartitionedCall¢ dense_46/StatefulPartitionedCall¢ dense_47/StatefulPartitionedCall
normalization_159/subSubnormalization_159_inputnormalization_159_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a
normalization_159/SqrtSqrtnormalization_159_sqrt_x*
T0*
_output_shapes

:	`
normalization_159/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_159/MaximumMaximumnormalization_159/Sqrt:y:0$normalization_159/Maximum/y:output:0*
T0*
_output_shapes

:	
normalization_159/truedivRealDivnormalization_159/sub:z:0normalization_159/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 dense_45/StatefulPartitionedCallStatefulPartitionedCallnormalization_159/truediv:z:0dense_45_122211dense_45_122213*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_45_layer_call_and_return_conditional_losses_121997
 dense_46/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0dense_46_122216dense_46_122218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_46_layer_call_and_return_conditional_losses_122014
 dense_47/StatefulPartitionedCallStatefulPartitionedCall)dense_46/StatefulPartitionedCall:output:0dense_47_122221dense_47_122223*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_122030x
IdentityIdentity)dense_47/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
NoOpNoOp!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall:i e
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1
_user_specified_namenormalization_159_input:$ 

_output_shapes

:	:$ 

_output_shapes

:	
°*
ò
!__inference__wrapped_model_121972
normalization_159_input)
%sequential_15_normalization_159_sub_y*
&sequential_15_normalization_159_sqrt_xG
5sequential_15_dense_45_matmul_readvariableop_resource:	%D
6sequential_15_dense_45_biasadd_readvariableop_resource:%G
5sequential_15_dense_46_matmul_readvariableop_resource:%D
6sequential_15_dense_46_biasadd_readvariableop_resource:G
5sequential_15_dense_47_matmul_readvariableop_resource:D
6sequential_15_dense_47_biasadd_readvariableop_resource:
identity¢-sequential_15/dense_45/BiasAdd/ReadVariableOp¢,sequential_15/dense_45/MatMul/ReadVariableOp¢-sequential_15/dense_46/BiasAdd/ReadVariableOp¢,sequential_15/dense_46/MatMul/ReadVariableOp¢-sequential_15/dense_47/BiasAdd/ReadVariableOp¢,sequential_15/dense_47/MatMul/ReadVariableOp
#sequential_15/normalization_159/subSubnormalization_159_input%sequential_15_normalization_159_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	}
$sequential_15/normalization_159/SqrtSqrt&sequential_15_normalization_159_sqrt_x*
T0*
_output_shapes

:	n
)sequential_15/normalization_159/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¹
'sequential_15/normalization_159/MaximumMaximum(sequential_15/normalization_159/Sqrt:y:02sequential_15/normalization_159/Maximum/y:output:0*
T0*
_output_shapes

:	º
'sequential_15/normalization_159/truedivRealDiv'sequential_15/normalization_159/sub:z:0+sequential_15/normalization_159/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	¢
,sequential_15/dense_45/MatMul/ReadVariableOpReadVariableOp5sequential_15_dense_45_matmul_readvariableop_resource*
_output_shapes

:	%*
dtype0¼
sequential_15/dense_45/MatMulMatMul+sequential_15/normalization_159/truediv:z:04sequential_15/dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ% 
-sequential_15/dense_45/BiasAdd/ReadVariableOpReadVariableOp6sequential_15_dense_45_biasadd_readvariableop_resource*
_output_shapes
:%*
dtype0»
sequential_15/dense_45/BiasAddBiasAdd'sequential_15/dense_45/MatMul:product:05sequential_15/dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%~
sequential_15/dense_45/ReluRelu'sequential_15/dense_45/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%¢
,sequential_15/dense_46/MatMul/ReadVariableOpReadVariableOp5sequential_15_dense_46_matmul_readvariableop_resource*
_output_shapes

:%*
dtype0º
sequential_15/dense_46/MatMulMatMul)sequential_15/dense_45/Relu:activations:04sequential_15/dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_15/dense_46/BiasAdd/ReadVariableOpReadVariableOp6sequential_15_dense_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_15/dense_46/BiasAddBiasAdd'sequential_15/dense_46/MatMul:product:05sequential_15/dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
sequential_15/dense_46/ReluRelu'sequential_15/dense_46/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
,sequential_15/dense_47/MatMul/ReadVariableOpReadVariableOp5sequential_15_dense_47_matmul_readvariableop_resource*
_output_shapes

:*
dtype0º
sequential_15/dense_47/MatMulMatMul)sequential_15/dense_46/Relu:activations:04sequential_15/dense_47/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_15/dense_47/BiasAdd/ReadVariableOpReadVariableOp6sequential_15_dense_47_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_15/dense_47/BiasAddBiasAdd'sequential_15/dense_47/MatMul:product:05sequential_15/dense_47/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'sequential_15/dense_47/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
NoOpNoOp.^sequential_15/dense_45/BiasAdd/ReadVariableOp-^sequential_15/dense_45/MatMul/ReadVariableOp.^sequential_15/dense_46/BiasAdd/ReadVariableOp-^sequential_15/dense_46/MatMul/ReadVariableOp.^sequential_15/dense_47/BiasAdd/ReadVariableOp-^sequential_15/dense_47/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2^
-sequential_15/dense_45/BiasAdd/ReadVariableOp-sequential_15/dense_45/BiasAdd/ReadVariableOp2\
,sequential_15/dense_45/MatMul/ReadVariableOp,sequential_15/dense_45/MatMul/ReadVariableOp2^
-sequential_15/dense_46/BiasAdd/ReadVariableOp-sequential_15/dense_46/BiasAdd/ReadVariableOp2\
,sequential_15/dense_46/MatMul/ReadVariableOp,sequential_15/dense_46/MatMul/ReadVariableOp2^
-sequential_15/dense_47/BiasAdd/ReadVariableOp-sequential_15/dense_47/BiasAdd/ReadVariableOp2\
,sequential_15/dense_47/MatMul/ReadVariableOp,sequential_15/dense_47/MatMul/ReadVariableOp:i e
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1
_user_specified_namenormalization_159_input:$ 

_output_shapes

:	:$ 

_output_shapes

:	


¥
.__inference_sequential_15_layer_call_fn_122254

inputs
unknown
	unknown_0
	unknown_1:	%
	unknown_2:%
	unknown_3:%
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_15_layer_call_and_return_conditional_losses_122037o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:	:$ 

_output_shapes

:	
Ò

¶
.__inference_sequential_15_layer_call_fn_122175
normalization_159_input
unknown
	unknown_0
	unknown_1:	%
	unknown_2:%
	unknown_3:%
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallnormalization_159_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_15_layer_call_and_return_conditional_losses_122135o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1
_user_specified_namenormalization_159_input:$ 

_output_shapes

:	:$ 

_output_shapes

:	


õ
D__inference_dense_45_layer_call_and_return_conditional_losses_121997

inputs0
matmul_readvariableop_resource:	%-
biasadd_readvariableop_resource:%
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	%*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:%*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
 

¬
$__inference_signature_wrapper_122360
normalization_159_input
unknown
	unknown_0
	unknown_1:	%
	unknown_2:%
	unknown_3:%
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallnormalization_159_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_121972o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1
_user_specified_namenormalization_159_input:$ 

_output_shapes

:	:$ 

_output_shapes

:	


¥
.__inference_sequential_15_layer_call_fn_122275

inputs
unknown
	unknown_0
	unknown_1:	%
	unknown_2:%
	unknown_3:%
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_15_layer_call_and_return_conditional_losses_122135o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:	:$ 

_output_shapes

:	
à!
Å
I__inference_sequential_15_layer_call_and_return_conditional_losses_122337

inputs
normalization_159_sub_y
normalization_159_sqrt_x9
'dense_45_matmul_readvariableop_resource:	%6
(dense_45_biasadd_readvariableop_resource:%9
'dense_46_matmul_readvariableop_resource:%6
(dense_46_biasadd_readvariableop_resource:9
'dense_47_matmul_readvariableop_resource:6
(dense_47_biasadd_readvariableop_resource:
identity¢dense_45/BiasAdd/ReadVariableOp¢dense_45/MatMul/ReadVariableOp¢dense_46/BiasAdd/ReadVariableOp¢dense_46/MatMul/ReadVariableOp¢dense_47/BiasAdd/ReadVariableOp¢dense_47/MatMul/ReadVariableOpo
normalization_159/subSubinputsnormalization_159_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a
normalization_159/SqrtSqrtnormalization_159_sqrt_x*
T0*
_output_shapes

:	`
normalization_159/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_159/MaximumMaximumnormalization_159/Sqrt:y:0$normalization_159/Maximum/y:output:0*
T0*
_output_shapes

:	
normalization_159/truedivRealDivnormalization_159/sub:z:0normalization_159/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource*
_output_shapes

:	%*
dtype0
dense_45/MatMulMatMulnormalization_159/truediv:z:0&dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
dense_45/BiasAdd/ReadVariableOpReadVariableOp(dense_45_biasadd_readvariableop_resource*
_output_shapes
:%*
dtype0
dense_45/BiasAddBiasAdddense_45/MatMul:product:0'dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%b
dense_45/ReluReludense_45/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
dense_46/MatMul/ReadVariableOpReadVariableOp'dense_46_matmul_readvariableop_resource*
_output_shapes

:%*
dtype0
dense_46/MatMulMatMuldense_45/Relu:activations:0&dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_46/BiasAdd/ReadVariableOpReadVariableOp(dense_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_46/BiasAddBiasAdddense_46/MatMul:product:0'dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_46/ReluReludense_46/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_47/MatMul/ReadVariableOpReadVariableOp'dense_47_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_47/MatMulMatMuldense_46/Relu:activations:0&dense_47/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_47/BiasAdd/ReadVariableOpReadVariableOp(dense_47_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_47/BiasAddBiasAdddense_47/MatMul:product:0'dense_47/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_47/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_45/BiasAdd/ReadVariableOp^dense_45/MatMul/ReadVariableOp ^dense_46/BiasAdd/ReadVariableOp^dense_46/MatMul/ReadVariableOp ^dense_47/BiasAdd/ReadVariableOp^dense_47/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2B
dense_45/BiasAdd/ReadVariableOpdense_45/BiasAdd/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp2B
dense_46/BiasAdd/ReadVariableOpdense_46/BiasAdd/ReadVariableOp2@
dense_46/MatMul/ReadVariableOpdense_46/MatMul/ReadVariableOp2B
dense_47/BiasAdd/ReadVariableOpdense_47/BiasAdd/ReadVariableOp2@
dense_47/MatMul/ReadVariableOpdense_47/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:	:$ 

_output_shapes

:	
Ç	
õ
D__inference_dense_47_layer_call_and_return_conditional_losses_122030

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å

)__inference_dense_45_layer_call_fn_122415

inputs
unknown:	%
	unknown_0:%
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_45_layer_call_and_return_conditional_losses_121997o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
à!
Å
I__inference_sequential_15_layer_call_and_return_conditional_losses_122306

inputs
normalization_159_sub_y
normalization_159_sqrt_x9
'dense_45_matmul_readvariableop_resource:	%6
(dense_45_biasadd_readvariableop_resource:%9
'dense_46_matmul_readvariableop_resource:%6
(dense_46_biasadd_readvariableop_resource:9
'dense_47_matmul_readvariableop_resource:6
(dense_47_biasadd_readvariableop_resource:
identity¢dense_45/BiasAdd/ReadVariableOp¢dense_45/MatMul/ReadVariableOp¢dense_46/BiasAdd/ReadVariableOp¢dense_46/MatMul/ReadVariableOp¢dense_47/BiasAdd/ReadVariableOp¢dense_47/MatMul/ReadVariableOpo
normalization_159/subSubinputsnormalization_159_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a
normalization_159/SqrtSqrtnormalization_159_sqrt_x*
T0*
_output_shapes

:	`
normalization_159/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_159/MaximumMaximumnormalization_159/Sqrt:y:0$normalization_159/Maximum/y:output:0*
T0*
_output_shapes

:	
normalization_159/truedivRealDivnormalization_159/sub:z:0normalization_159/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource*
_output_shapes

:	%*
dtype0
dense_45/MatMulMatMulnormalization_159/truediv:z:0&dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
dense_45/BiasAdd/ReadVariableOpReadVariableOp(dense_45_biasadd_readvariableop_resource*
_output_shapes
:%*
dtype0
dense_45/BiasAddBiasAdddense_45/MatMul:product:0'dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%b
dense_45/ReluReludense_45/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
dense_46/MatMul/ReadVariableOpReadVariableOp'dense_46_matmul_readvariableop_resource*
_output_shapes

:%*
dtype0
dense_46/MatMulMatMuldense_45/Relu:activations:0&dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_46/BiasAdd/ReadVariableOpReadVariableOp(dense_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_46/BiasAddBiasAdddense_46/MatMul:product:0'dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_46/ReluReludense_46/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_47/MatMul/ReadVariableOpReadVariableOp'dense_47_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_47/MatMulMatMuldense_46/Relu:activations:0&dense_47/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_47/BiasAdd/ReadVariableOpReadVariableOp(dense_47_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_47/BiasAddBiasAdddense_47/MatMul:product:0'dense_47/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_47/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_45/BiasAdd/ReadVariableOp^dense_45/MatMul/ReadVariableOp ^dense_46/BiasAdd/ReadVariableOp^dense_46/MatMul/ReadVariableOp ^dense_47/BiasAdd/ReadVariableOp^dense_47/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2B
dense_45/BiasAdd/ReadVariableOpdense_45/BiasAdd/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp2B
dense_46/BiasAdd/ReadVariableOpdense_46/BiasAdd/ReadVariableOp2@
dense_46/MatMul/ReadVariableOpdense_46/MatMul/ReadVariableOp2B
dense_47/BiasAdd/ReadVariableOpdense_47/BiasAdd/ReadVariableOp2@
dense_47/MatMul/ReadVariableOpdense_47/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:	:$ 

_output_shapes

:	


õ
D__inference_dense_46_layer_call_and_return_conditional_losses_122446

inputs0
matmul_readvariableop_resource:%-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:%*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ%
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ô
serving_defaultÀ
d
normalization_159_inputI
)serving_default_normalization_159_input:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ<
dense_470
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÊU

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ó

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
	keras_api
_adapt_function"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
»

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
¿
/iter

0beta_1

1beta_2
	2decay
3learning_ratemNmOmP mQ'mR(mSvTvUvV vW'vX(vY"
	optimizer
_
0
1
2
3
4
5
 6
'7
(8"
trackable_list_wrapper
J
0
1
2
 3
'4
(5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_sequential_15_layer_call_fn_122056
.__inference_sequential_15_layer_call_fn_122254
.__inference_sequential_15_layer_call_fn_122275
.__inference_sequential_15_layer_call_fn_122175À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
I__inference_sequential_15_layer_call_and_return_conditional_losses_122306
I__inference_sequential_15_layer_call_and_return_conditional_losses_122337
I__inference_sequential_15_layer_call_and_return_conditional_losses_122201
I__inference_sequential_15_layer_call_and_return_conditional_losses_122227À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÜBÙ
!__inference__wrapped_model_121972normalization_159_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
9serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	2mean
:	2variance
:	 2count
"
_generic_user_object
¿2¼
__inference_adapt_step_122406
²
FullArgSpec
args

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:	%2dense_45/kernel
:%2dense_45/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_45_layer_call_fn_122415¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_45_layer_call_and_return_conditional_losses_122426¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:%2dense_46/kernel
:2dense_46/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_46_layer_call_fn_122435¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_46_layer_call_and_return_conditional_losses_122446¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:2dense_47/kernel
:2dense_47/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_47_layer_call_fn_122455¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_47_layer_call_and_return_conditional_losses_122465¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
5
0
1
2"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
I0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
$__inference_signature_wrapper_122360normalization_159_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Jtotal
	Kcount
L	variables
M	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
J0
K1"
trackable_list_wrapper
-
L	variables"
_generic_user_object
&:$	%2Adam/dense_45/kernel/m
 :%2Adam/dense_45/bias/m
&:$%2Adam/dense_46/kernel/m
 :2Adam/dense_46/bias/m
&:$2Adam/dense_47/kernel/m
 :2Adam/dense_47/bias/m
&:$	%2Adam/dense_45/kernel/v
 :%2Adam/dense_45/bias/v
&:$%2Adam/dense_46/kernel/v
 :2Adam/dense_46/bias/v
&:$2Adam/dense_47/kernel/v
 :2Adam/dense_47/bias/v
	J
Const
J	
Const_1°
!__inference__wrapped_model_121972Z[ '(I¢F
?¢<
:7
normalization_159_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_47"
dense_47ÿÿÿÿÿÿÿÿÿo
__inference_adapt_step_122406NC¢@
9¢6
41¢
ÿÿÿÿÿÿÿÿÿ	IteratorSpec 
ª "
 ¤
D__inference_dense_45_layer_call_and_return_conditional_losses_122426\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ	
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ%
 |
)__inference_dense_45_layer_call_fn_122415O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ	
ª "ÿÿÿÿÿÿÿÿÿ%¤
D__inference_dense_46_layer_call_and_return_conditional_losses_122446\ /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ%
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_46_layer_call_fn_122435O /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ%
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_47_layer_call_and_return_conditional_losses_122465\'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_47_layer_call_fn_122455O'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÒ
I__inference_sequential_15_layer_call_and_return_conditional_losses_122201Z[ '(Q¢N
G¢D
:7
normalization_159_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ò
I__inference_sequential_15_layer_call_and_return_conditional_losses_122227Z[ '(Q¢N
G¢D
:7
normalization_159_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 À
I__inference_sequential_15_layer_call_and_return_conditional_losses_122306sZ[ '(@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 À
I__inference_sequential_15_layer_call_and_return_conditional_losses_122337sZ[ '(@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ©
.__inference_sequential_15_layer_call_fn_122056wZ[ '(Q¢N
G¢D
:7
normalization_159_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ©
.__inference_sequential_15_layer_call_fn_122175wZ[ '(Q¢N
G¢D
:7
normalization_159_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_15_layer_call_fn_122254fZ[ '(@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_15_layer_call_fn_122275fZ[ '(@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÎ
$__inference_signature_wrapper_122360¥Z[ '(d¢a
¢ 
ZªW
U
normalization_159_input:7
normalization_159_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_47"
dense_47ÿÿÿÿÿÿÿÿÿ