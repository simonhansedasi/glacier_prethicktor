Ó
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68«
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
|
dense_198/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	
*!
shared_namedense_198/kernel
u
$dense_198/kernel/Read/ReadVariableOpReadVariableOpdense_198/kernel*
_output_shapes

:	
*
dtype0
t
dense_198/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_198/bias
m
"dense_198/bias/Read/ReadVariableOpReadVariableOpdense_198/bias*
_output_shapes
:
*
dtype0
|
dense_199/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*!
shared_namedense_199/kernel
u
$dense_199/kernel/Read/ReadVariableOpReadVariableOpdense_199/kernel*
_output_shapes

:
*
dtype0
t
dense_199/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_199/bias
m
"dense_199/bias/Read/ReadVariableOpReadVariableOpdense_199/bias*
_output_shapes
:*
dtype0
|
dense_200/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_200/kernel
u
$dense_200/kernel/Read/ReadVariableOpReadVariableOpdense_200/kernel*
_output_shapes

:*
dtype0
t
dense_200/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_200/bias
m
"dense_200/bias/Read/ReadVariableOpReadVariableOpdense_200/bias*
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

Adam/dense_198/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	
*(
shared_nameAdam/dense_198/kernel/m

+Adam/dense_198/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_198/kernel/m*
_output_shapes

:	
*
dtype0

Adam/dense_198/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_198/bias/m
{
)Adam/dense_198/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_198/bias/m*
_output_shapes
:
*
dtype0

Adam/dense_199/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_199/kernel/m

+Adam/dense_199/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_199/kernel/m*
_output_shapes

:
*
dtype0

Adam/dense_199/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_199/bias/m
{
)Adam/dense_199/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_199/bias/m*
_output_shapes
:*
dtype0

Adam/dense_200/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_200/kernel/m

+Adam/dense_200/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_200/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_200/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_200/bias/m
{
)Adam/dense_200/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_200/bias/m*
_output_shapes
:*
dtype0

Adam/dense_198/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	
*(
shared_nameAdam/dense_198/kernel/v

+Adam/dense_198/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_198/kernel/v*
_output_shapes

:	
*
dtype0

Adam/dense_198/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_198/bias/v
{
)Adam/dense_198/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_198/bias/v*
_output_shapes
:
*
dtype0

Adam/dense_199/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_199/kernel/v

+Adam/dense_199/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_199/kernel/v*
_output_shapes

:
*
dtype0

Adam/dense_199/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_199/bias/v
{
)Adam/dense_199/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_199/bias/v*
_output_shapes
:*
dtype0

Adam/dense_200/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_200/kernel/v

+Adam/dense_200/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_200/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_200/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_200/bias/v
{
)Adam/dense_200/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_200/bias/v*
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
¯/
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*è.
valueÞ.BÛ. BÔ.
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
`Z
VARIABLE_VALUEdense_198/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_198/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
`Z
VARIABLE_VALUEdense_199/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_199/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
`Z
VARIABLE_VALUEdense_200/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_200/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
}
VARIABLE_VALUEAdam/dense_198/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_198/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_199/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_199/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_200/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_200/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_198/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_198/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_199/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_199/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_200/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_200/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

'serving_default_normalization_669_inputPlaceholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
É
StatefulPartitionedCallStatefulPartitionedCall'serving_default_normalization_669_inputConstConst_1dense_198/kerneldense_198/biasdense_199/kerneldense_199/biasdense_200/kerneldense_200/bias*
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
$__inference_signature_wrapper_513785
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Þ

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_198/kernel/Read/ReadVariableOp"dense_198/bias/Read/ReadVariableOp$dense_199/kernel/Read/ReadVariableOp"dense_199/bias/Read/ReadVariableOp$dense_200/kernel/Read/ReadVariableOp"dense_200/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_198/kernel/m/Read/ReadVariableOp)Adam/dense_198/bias/m/Read/ReadVariableOp+Adam/dense_199/kernel/m/Read/ReadVariableOp)Adam/dense_199/bias/m/Read/ReadVariableOp+Adam/dense_200/kernel/m/Read/ReadVariableOp)Adam/dense_200/bias/m/Read/ReadVariableOp+Adam/dense_198/kernel/v/Read/ReadVariableOp)Adam/dense_198/bias/v/Read/ReadVariableOp+Adam/dense_199/kernel/v/Read/ReadVariableOp)Adam/dense_199/bias/v/Read/ReadVariableOp+Adam/dense_200/kernel/v/Read/ReadVariableOp)Adam/dense_200/bias/v/Read/ReadVariableOpConst_2*)
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
__inference__traced_save_513999
§
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_198/kerneldense_198/biasdense_199/kerneldense_199/biasdense_200/kerneldense_200/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount_1Adam/dense_198/kernel/mAdam/dense_198/bias/mAdam/dense_199/kernel/mAdam/dense_199/bias/mAdam/dense_200/kernel/mAdam/dense_200/bias/mAdam/dense_198/kernel/vAdam/dense_198/bias/vAdam/dense_199/kernel/vAdam/dense_199/bias/vAdam/dense_200/kernel/vAdam/dense_200/bias/v*(
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
"__inference__traced_restore_514093¯
 

¬
$__inference_signature_wrapper_513785
normalization_669_input
unknown
	unknown_0
	unknown_1:	

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallnormalization_669_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
!__inference__wrapped_model_513397o
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
_user_specified_namenormalization_669_input:$ 

_output_shapes

:	:$ 

_output_shapes

:	
Ò

¶
.__inference_sequential_66_layer_call_fn_513481
normalization_669_input
unknown
	unknown_0
	unknown_1:	

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallnormalization_669_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
I__inference_sequential_66_layer_call_and_return_conditional_losses_513462o
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
_user_specified_namenormalization_669_input:$ 

_output_shapes

:	:$ 

_output_shapes

:	
 "
Ñ
I__inference_sequential_66_layer_call_and_return_conditional_losses_513762

inputs
normalization_669_sub_y
normalization_669_sqrt_x:
(dense_198_matmul_readvariableop_resource:	
7
)dense_198_biasadd_readvariableop_resource:
:
(dense_199_matmul_readvariableop_resource:
7
)dense_199_biasadd_readvariableop_resource::
(dense_200_matmul_readvariableop_resource:7
)dense_200_biasadd_readvariableop_resource:
identity¢ dense_198/BiasAdd/ReadVariableOp¢dense_198/MatMul/ReadVariableOp¢ dense_199/BiasAdd/ReadVariableOp¢dense_199/MatMul/ReadVariableOp¢ dense_200/BiasAdd/ReadVariableOp¢dense_200/MatMul/ReadVariableOpo
normalization_669/subSubinputsnormalization_669_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a
normalization_669/SqrtSqrtnormalization_669_sqrt_x*
T0*
_output_shapes

:	`
normalization_669/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_669/MaximumMaximumnormalization_669/Sqrt:y:0$normalization_669/Maximum/y:output:0*
T0*
_output_shapes

:	
normalization_669/truedivRealDivnormalization_669/sub:z:0normalization_669/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
dense_198/MatMul/ReadVariableOpReadVariableOp(dense_198_matmul_readvariableop_resource*
_output_shapes

:	
*
dtype0
dense_198/MatMulMatMulnormalization_669/truediv:z:0'dense_198/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 dense_198/BiasAdd/ReadVariableOpReadVariableOp)dense_198_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_198/BiasAddBiasAdddense_198/MatMul:product:0(dense_198/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
dense_198/ReluReludense_198/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_199/MatMul/ReadVariableOpReadVariableOp(dense_199_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_199/MatMulMatMuldense_198/Relu:activations:0'dense_199/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_199/BiasAdd/ReadVariableOpReadVariableOp)dense_199_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_199/BiasAddBiasAdddense_199/MatMul:product:0(dense_199/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_199/ReluReludense_199/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_200/MatMul/ReadVariableOpReadVariableOp(dense_200_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_200/MatMulMatMuldense_199/Relu:activations:0'dense_200/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_200/BiasAdd/ReadVariableOpReadVariableOp)dense_200_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_200/BiasAddBiasAdddense_200/MatMul:product:0(dense_200/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_200/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_198/BiasAdd/ReadVariableOp ^dense_198/MatMul/ReadVariableOp!^dense_199/BiasAdd/ReadVariableOp ^dense_199/MatMul/ReadVariableOp!^dense_200/BiasAdd/ReadVariableOp ^dense_200/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2D
 dense_198/BiasAdd/ReadVariableOp dense_198/BiasAdd/ReadVariableOp2B
dense_198/MatMul/ReadVariableOpdense_198/MatMul/ReadVariableOp2D
 dense_199/BiasAdd/ReadVariableOp dense_199/BiasAdd/ReadVariableOp2B
dense_199/MatMul/ReadVariableOpdense_199/MatMul/ReadVariableOp2D
 dense_200/BiasAdd/ReadVariableOp dense_200/BiasAdd/ReadVariableOp2B
dense_200/MatMul/ReadVariableOpdense_200/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:	:$ 

_output_shapes

:	


ö
E__inference_dense_198_layer_call_and_return_conditional_losses_513851

inputs0
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
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
Ç

*__inference_dense_198_layer_call_fn_513840

inputs
unknown:	

	unknown_0:

identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_198_layer_call_and_return_conditional_losses_513422o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
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
È	
ö
E__inference_dense_200_layer_call_and_return_conditional_losses_513455

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
Û
I__inference_sequential_66_layer_call_and_return_conditional_losses_513462

inputs
normalization_669_sub_y
normalization_669_sqrt_x"
dense_198_513423:	

dense_198_513425:
"
dense_199_513440:

dense_199_513442:"
dense_200_513456:
dense_200_513458:
identity¢!dense_198/StatefulPartitionedCall¢!dense_199/StatefulPartitionedCall¢!dense_200/StatefulPartitionedCallo
normalization_669/subSubinputsnormalization_669_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a
normalization_669/SqrtSqrtnormalization_669_sqrt_x*
T0*
_output_shapes

:	`
normalization_669/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_669/MaximumMaximumnormalization_669/Sqrt:y:0$normalization_669/Maximum/y:output:0*
T0*
_output_shapes

:	
normalization_669/truedivRealDivnormalization_669/sub:z:0normalization_669/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
!dense_198/StatefulPartitionedCallStatefulPartitionedCallnormalization_669/truediv:z:0dense_198_513423dense_198_513425*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_198_layer_call_and_return_conditional_losses_513422
!dense_199/StatefulPartitionedCallStatefulPartitionedCall*dense_198/StatefulPartitionedCall:output:0dense_199_513440dense_199_513442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_199_layer_call_and_return_conditional_losses_513439
!dense_200/StatefulPartitionedCallStatefulPartitionedCall*dense_199/StatefulPartitionedCall:output:0dense_200_513456dense_200_513458*
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
GPU2*0J 8 *N
fIRG
E__inference_dense_200_layer_call_and_return_conditional_losses_513455y
IdentityIdentity*dense_200/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp"^dense_198/StatefulPartitionedCall"^dense_199/StatefulPartitionedCall"^dense_200/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2F
!dense_198/StatefulPartitionedCall!dense_198/StatefulPartitionedCall2F
!dense_199/StatefulPartitionedCall!dense_199/StatefulPartitionedCall2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:	:$ 

_output_shapes

:	
¹'
Ò
__inference_adapt_step_513831
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
Ò

¶
.__inference_sequential_66_layer_call_fn_513600
normalization_669_input
unknown
	unknown_0
	unknown_1:	

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallnormalization_669_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
I__inference_sequential_66_layer_call_and_return_conditional_losses_513560o
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
_user_specified_namenormalization_669_input:$ 

_output_shapes

:	:$ 

_output_shapes

:	
Ç

*__inference_dense_199_layer_call_fn_513860

inputs
unknown:

	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_199_layer_call_and_return_conditional_losses_513439o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
È	
ö
E__inference_dense_200_layer_call_and_return_conditional_losses_513890

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


¥
.__inference_sequential_66_layer_call_fn_513679

inputs
unknown
	unknown_0
	unknown_1:	

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
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
I__inference_sequential_66_layer_call_and_return_conditional_losses_513462o
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


¥
.__inference_sequential_66_layer_call_fn_513700

inputs
unknown
	unknown_0
	unknown_1:	

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
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
I__inference_sequential_66_layer_call_and_return_conditional_losses_513560o
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

ì
I__inference_sequential_66_layer_call_and_return_conditional_losses_513652
normalization_669_input
normalization_669_sub_y
normalization_669_sqrt_x"
dense_198_513636:	

dense_198_513638:
"
dense_199_513641:

dense_199_513643:"
dense_200_513646:
dense_200_513648:
identity¢!dense_198/StatefulPartitionedCall¢!dense_199/StatefulPartitionedCall¢!dense_200/StatefulPartitionedCall
normalization_669/subSubnormalization_669_inputnormalization_669_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a
normalization_669/SqrtSqrtnormalization_669_sqrt_x*
T0*
_output_shapes

:	`
normalization_669/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_669/MaximumMaximumnormalization_669/Sqrt:y:0$normalization_669/Maximum/y:output:0*
T0*
_output_shapes

:	
normalization_669/truedivRealDivnormalization_669/sub:z:0normalization_669/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
!dense_198/StatefulPartitionedCallStatefulPartitionedCallnormalization_669/truediv:z:0dense_198_513636dense_198_513638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_198_layer_call_and_return_conditional_losses_513422
!dense_199/StatefulPartitionedCallStatefulPartitionedCall*dense_198/StatefulPartitionedCall:output:0dense_199_513641dense_199_513643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_199_layer_call_and_return_conditional_losses_513439
!dense_200/StatefulPartitionedCallStatefulPartitionedCall*dense_199/StatefulPartitionedCall:output:0dense_200_513646dense_200_513648*
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
GPU2*0J 8 *N
fIRG
E__inference_dense_200_layer_call_and_return_conditional_losses_513455y
IdentityIdentity*dense_200/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp"^dense_198/StatefulPartitionedCall"^dense_199/StatefulPartitionedCall"^dense_200/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2F
!dense_198/StatefulPartitionedCall!dense_198/StatefulPartitionedCall2F
!dense_199/StatefulPartitionedCall!dense_199/StatefulPartitionedCall2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall:i e
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1
_user_specified_namenormalization_669_input:$ 

_output_shapes

:	:$ 

_output_shapes

:	

ì
I__inference_sequential_66_layer_call_and_return_conditional_losses_513626
normalization_669_input
normalization_669_sub_y
normalization_669_sqrt_x"
dense_198_513610:	

dense_198_513612:
"
dense_199_513615:

dense_199_513617:"
dense_200_513620:
dense_200_513622:
identity¢!dense_198/StatefulPartitionedCall¢!dense_199/StatefulPartitionedCall¢!dense_200/StatefulPartitionedCall
normalization_669/subSubnormalization_669_inputnormalization_669_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a
normalization_669/SqrtSqrtnormalization_669_sqrt_x*
T0*
_output_shapes

:	`
normalization_669/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_669/MaximumMaximumnormalization_669/Sqrt:y:0$normalization_669/Maximum/y:output:0*
T0*
_output_shapes

:	
normalization_669/truedivRealDivnormalization_669/sub:z:0normalization_669/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
!dense_198/StatefulPartitionedCallStatefulPartitionedCallnormalization_669/truediv:z:0dense_198_513610dense_198_513612*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_198_layer_call_and_return_conditional_losses_513422
!dense_199/StatefulPartitionedCallStatefulPartitionedCall*dense_198/StatefulPartitionedCall:output:0dense_199_513615dense_199_513617*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_199_layer_call_and_return_conditional_losses_513439
!dense_200/StatefulPartitionedCallStatefulPartitionedCall*dense_199/StatefulPartitionedCall:output:0dense_200_513620dense_200_513622*
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
GPU2*0J 8 *N
fIRG
E__inference_dense_200_layer_call_and_return_conditional_losses_513455y
IdentityIdentity*dense_200/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp"^dense_198/StatefulPartitionedCall"^dense_199/StatefulPartitionedCall"^dense_200/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2F
!dense_198/StatefulPartitionedCall!dense_198/StatefulPartitionedCall2F
!dense_199/StatefulPartitionedCall!dense_199/StatefulPartitionedCall2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall:i e
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1
_user_specified_namenormalization_669_input:$ 

_output_shapes

:	:$ 

_output_shapes

:	
 "
Ñ
I__inference_sequential_66_layer_call_and_return_conditional_losses_513731

inputs
normalization_669_sub_y
normalization_669_sqrt_x:
(dense_198_matmul_readvariableop_resource:	
7
)dense_198_biasadd_readvariableop_resource:
:
(dense_199_matmul_readvariableop_resource:
7
)dense_199_biasadd_readvariableop_resource::
(dense_200_matmul_readvariableop_resource:7
)dense_200_biasadd_readvariableop_resource:
identity¢ dense_198/BiasAdd/ReadVariableOp¢dense_198/MatMul/ReadVariableOp¢ dense_199/BiasAdd/ReadVariableOp¢dense_199/MatMul/ReadVariableOp¢ dense_200/BiasAdd/ReadVariableOp¢dense_200/MatMul/ReadVariableOpo
normalization_669/subSubinputsnormalization_669_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a
normalization_669/SqrtSqrtnormalization_669_sqrt_x*
T0*
_output_shapes

:	`
normalization_669/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_669/MaximumMaximumnormalization_669/Sqrt:y:0$normalization_669/Maximum/y:output:0*
T0*
_output_shapes

:	
normalization_669/truedivRealDivnormalization_669/sub:z:0normalization_669/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
dense_198/MatMul/ReadVariableOpReadVariableOp(dense_198_matmul_readvariableop_resource*
_output_shapes

:	
*
dtype0
dense_198/MatMulMatMulnormalization_669/truediv:z:0'dense_198/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 dense_198/BiasAdd/ReadVariableOpReadVariableOp)dense_198_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_198/BiasAddBiasAdddense_198/MatMul:product:0(dense_198/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
dense_198/ReluReludense_198/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_199/MatMul/ReadVariableOpReadVariableOp(dense_199_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_199/MatMulMatMuldense_198/Relu:activations:0'dense_199/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_199/BiasAdd/ReadVariableOpReadVariableOp)dense_199_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_199/BiasAddBiasAdddense_199/MatMul:product:0(dense_199/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_199/ReluReludense_199/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_200/MatMul/ReadVariableOpReadVariableOp(dense_200_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_200/MatMulMatMuldense_199/Relu:activations:0'dense_200/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_200/BiasAdd/ReadVariableOpReadVariableOp)dense_200_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_200/BiasAddBiasAdddense_200/MatMul:product:0(dense_200/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_200/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_198/BiasAdd/ReadVariableOp ^dense_198/MatMul/ReadVariableOp!^dense_199/BiasAdd/ReadVariableOp ^dense_199/MatMul/ReadVariableOp!^dense_200/BiasAdd/ReadVariableOp ^dense_200/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2D
 dense_198/BiasAdd/ReadVariableOp dense_198/BiasAdd/ReadVariableOp2B
dense_198/MatMul/ReadVariableOpdense_198/MatMul/ReadVariableOp2D
 dense_199/BiasAdd/ReadVariableOp dense_199/BiasAdd/ReadVariableOp2B
dense_199/MatMul/ReadVariableOpdense_199/MatMul/ReadVariableOp2D
 dense_200/BiasAdd/ReadVariableOp dense_200/BiasAdd/ReadVariableOp2B
dense_200/MatMul/ReadVariableOpdense_200/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:	:$ 

_output_shapes

:	
Î
Û
I__inference_sequential_66_layer_call_and_return_conditional_losses_513560

inputs
normalization_669_sub_y
normalization_669_sqrt_x"
dense_198_513544:	

dense_198_513546:
"
dense_199_513549:

dense_199_513551:"
dense_200_513554:
dense_200_513556:
identity¢!dense_198/StatefulPartitionedCall¢!dense_199/StatefulPartitionedCall¢!dense_200/StatefulPartitionedCallo
normalization_669/subSubinputsnormalization_669_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a
normalization_669/SqrtSqrtnormalization_669_sqrt_x*
T0*
_output_shapes

:	`
normalization_669/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_669/MaximumMaximumnormalization_669/Sqrt:y:0$normalization_669/Maximum/y:output:0*
T0*
_output_shapes

:	
normalization_669/truedivRealDivnormalization_669/sub:z:0normalization_669/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
!dense_198/StatefulPartitionedCallStatefulPartitionedCallnormalization_669/truediv:z:0dense_198_513544dense_198_513546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_198_layer_call_and_return_conditional_losses_513422
!dense_199/StatefulPartitionedCallStatefulPartitionedCall*dense_198/StatefulPartitionedCall:output:0dense_199_513549dense_199_513551*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_199_layer_call_and_return_conditional_losses_513439
!dense_200/StatefulPartitionedCallStatefulPartitionedCall*dense_199/StatefulPartitionedCall:output:0dense_200_513554dense_200_513556*
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
GPU2*0J 8 *N
fIRG
E__inference_dense_200_layer_call_and_return_conditional_losses_513455y
IdentityIdentity*dense_200/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp"^dense_198/StatefulPartitionedCall"^dense_199/StatefulPartitionedCall"^dense_200/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2F
!dense_198/StatefulPartitionedCall!dense_198/StatefulPartitionedCall2F
!dense_199/StatefulPartitionedCall!dense_199/StatefulPartitionedCall2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:	:$ 

_output_shapes

:	
¸=
Ä
__inference__traced_save_513999
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_198_kernel_read_readvariableop-
)savev2_dense_198_bias_read_readvariableop/
+savev2_dense_199_kernel_read_readvariableop-
)savev2_dense_199_bias_read_readvariableop/
+savev2_dense_200_kernel_read_readvariableop-
)savev2_dense_200_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_198_kernel_m_read_readvariableop4
0savev2_adam_dense_198_bias_m_read_readvariableop6
2savev2_adam_dense_199_kernel_m_read_readvariableop4
0savev2_adam_dense_199_bias_m_read_readvariableop6
2savev2_adam_dense_200_kernel_m_read_readvariableop4
0savev2_adam_dense_200_bias_m_read_readvariableop6
2savev2_adam_dense_198_kernel_v_read_readvariableop4
0savev2_adam_dense_198_bias_v_read_readvariableop6
2savev2_adam_dense_199_kernel_v_read_readvariableop4
0savev2_adam_dense_199_bias_v_read_readvariableop6
2savev2_adam_dense_200_kernel_v_read_readvariableop4
0savev2_adam_dense_200_bias_v_read_readvariableop
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
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ´
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_198_kernel_read_readvariableop)savev2_dense_198_bias_read_readvariableop+savev2_dense_199_kernel_read_readvariableop)savev2_dense_199_bias_read_readvariableop+savev2_dense_200_kernel_read_readvariableop)savev2_dense_200_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_198_kernel_m_read_readvariableop0savev2_adam_dense_198_bias_m_read_readvariableop2savev2_adam_dense_199_kernel_m_read_readvariableop0savev2_adam_dense_199_bias_m_read_readvariableop2savev2_adam_dense_200_kernel_m_read_readvariableop0savev2_adam_dense_200_bias_m_read_readvariableop2savev2_adam_dense_198_kernel_v_read_readvariableop0savev2_adam_dense_198_bias_v_read_readvariableop2savev2_adam_dense_199_kernel_v_read_readvariableop0savev2_adam_dense_199_bias_v_read_readvariableop2savev2_adam_dense_200_kernel_v_read_readvariableop0savev2_adam_dense_200_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
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
°: :	:	: :	
:
:
:::: : : : : : : :	
:
:
::::	
:
:
:::: 2(
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

:	
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 	
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

:	
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:	
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 


ö
E__inference_dense_199_layer_call_and_return_conditional_losses_513439

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs


ö
E__inference_dense_198_layer_call_and_return_conditional_losses_513422

inputs0
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
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
ò*
þ
!__inference__wrapped_model_513397
normalization_669_input)
%sequential_66_normalization_669_sub_y*
&sequential_66_normalization_669_sqrt_xH
6sequential_66_dense_198_matmul_readvariableop_resource:	
E
7sequential_66_dense_198_biasadd_readvariableop_resource:
H
6sequential_66_dense_199_matmul_readvariableop_resource:
E
7sequential_66_dense_199_biasadd_readvariableop_resource:H
6sequential_66_dense_200_matmul_readvariableop_resource:E
7sequential_66_dense_200_biasadd_readvariableop_resource:
identity¢.sequential_66/dense_198/BiasAdd/ReadVariableOp¢-sequential_66/dense_198/MatMul/ReadVariableOp¢.sequential_66/dense_199/BiasAdd/ReadVariableOp¢-sequential_66/dense_199/MatMul/ReadVariableOp¢.sequential_66/dense_200/BiasAdd/ReadVariableOp¢-sequential_66/dense_200/MatMul/ReadVariableOp
#sequential_66/normalization_669/subSubnormalization_669_input%sequential_66_normalization_669_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	}
$sequential_66/normalization_669/SqrtSqrt&sequential_66_normalization_669_sqrt_x*
T0*
_output_shapes

:	n
)sequential_66/normalization_669/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¹
'sequential_66/normalization_669/MaximumMaximum(sequential_66/normalization_669/Sqrt:y:02sequential_66/normalization_669/Maximum/y:output:0*
T0*
_output_shapes

:	º
'sequential_66/normalization_669/truedivRealDiv'sequential_66/normalization_669/sub:z:0+sequential_66/normalization_669/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	¤
-sequential_66/dense_198/MatMul/ReadVariableOpReadVariableOp6sequential_66_dense_198_matmul_readvariableop_resource*
_output_shapes

:	
*
dtype0¾
sequential_66/dense_198/MatMulMatMul+sequential_66/normalization_669/truediv:z:05sequential_66/dense_198/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
.sequential_66/dense_198/BiasAdd/ReadVariableOpReadVariableOp7sequential_66_dense_198_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0¾
sequential_66/dense_198/BiasAddBiasAdd(sequential_66/dense_198/MatMul:product:06sequential_66/dense_198/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

sequential_66/dense_198/ReluRelu(sequential_66/dense_198/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
-sequential_66/dense_199/MatMul/ReadVariableOpReadVariableOp6sequential_66_dense_199_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0½
sequential_66/dense_199/MatMulMatMul*sequential_66/dense_198/Relu:activations:05sequential_66/dense_199/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_66/dense_199/BiasAdd/ReadVariableOpReadVariableOp7sequential_66_dense_199_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_66/dense_199/BiasAddBiasAdd(sequential_66/dense_199/MatMul:product:06sequential_66/dense_199/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_66/dense_199/ReluRelu(sequential_66/dense_199/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_66/dense_200/MatMul/ReadVariableOpReadVariableOp6sequential_66_dense_200_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_66/dense_200/MatMulMatMul*sequential_66/dense_199/Relu:activations:05sequential_66/dense_200/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_66/dense_200/BiasAdd/ReadVariableOpReadVariableOp7sequential_66_dense_200_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_66/dense_200/BiasAddBiasAdd(sequential_66/dense_200/MatMul:product:06sequential_66/dense_200/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_66/dense_200/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
NoOpNoOp/^sequential_66/dense_198/BiasAdd/ReadVariableOp.^sequential_66/dense_198/MatMul/ReadVariableOp/^sequential_66/dense_199/BiasAdd/ReadVariableOp.^sequential_66/dense_199/MatMul/ReadVariableOp/^sequential_66/dense_200/BiasAdd/ReadVariableOp.^sequential_66/dense_200/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:	:	: : : : : : 2`
.sequential_66/dense_198/BiasAdd/ReadVariableOp.sequential_66/dense_198/BiasAdd/ReadVariableOp2^
-sequential_66/dense_198/MatMul/ReadVariableOp-sequential_66/dense_198/MatMul/ReadVariableOp2`
.sequential_66/dense_199/BiasAdd/ReadVariableOp.sequential_66/dense_199/BiasAdd/ReadVariableOp2^
-sequential_66/dense_199/MatMul/ReadVariableOp-sequential_66/dense_199/MatMul/ReadVariableOp2`
.sequential_66/dense_200/BiasAdd/ReadVariableOp.sequential_66/dense_200/BiasAdd/ReadVariableOp2^
-sequential_66/dense_200/MatMul/ReadVariableOp-sequential_66/dense_200/MatMul/ReadVariableOp:i e
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1
_user_specified_namenormalization_669_input:$ 

_output_shapes

:	:$ 

_output_shapes

:	


ö
E__inference_dense_199_layer_call_and_return_conditional_losses_513871

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ç

*__inference_dense_200_layer_call_fn_513880

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÝ
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
GPU2*0J 8 *N
fIRG
E__inference_dense_200_layer_call_and_return_conditional_losses_513455o
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
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ºp
Ç
"__inference__traced_restore_514093
file_prefix#
assignvariableop_mean:	)
assignvariableop_1_variance:	"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_198_kernel:	
/
!assignvariableop_4_dense_198_bias:
5
#assignvariableop_5_dense_199_kernel:
/
!assignvariableop_6_dense_199_bias:5
#assignvariableop_7_dense_200_kernel:/
!assignvariableop_8_dense_200_bias:&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: #
assignvariableop_14_total: %
assignvariableop_15_count_1: =
+assignvariableop_16_adam_dense_198_kernel_m:	
7
)assignvariableop_17_adam_dense_198_bias_m:
=
+assignvariableop_18_adam_dense_199_kernel_m:
7
)assignvariableop_19_adam_dense_199_bias_m:=
+assignvariableop_20_adam_dense_200_kernel_m:7
)assignvariableop_21_adam_dense_200_bias_m:=
+assignvariableop_22_adam_dense_198_kernel_v:	
7
)assignvariableop_23_adam_dense_198_bias_v:
=
+assignvariableop_24_adam_dense_199_kernel_v:
7
)assignvariableop_25_adam_dense_199_bias_v:=
+assignvariableop_26_adam_dense_200_kernel_v:7
)assignvariableop_27_adam_dense_200_bias_v:
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
:
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_198_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_198_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_199_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_199_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_200_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_200_biasIdentity_8:output:0"/device:CPU:0*
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
:
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_dense_198_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_198_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp+assignvariableop_18_adam_dense_199_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_199_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_dense_200_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_200_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_dense_198_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_198_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_dense_199_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_199_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_dense_200_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_200_bias_vIdentity_27:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Õ
serving_defaultÁ
d
normalization_669_inputI
)serving_default_normalization_669_input:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ=
	dense_2000
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ìU
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
.__inference_sequential_66_layer_call_fn_513481
.__inference_sequential_66_layer_call_fn_513679
.__inference_sequential_66_layer_call_fn_513700
.__inference_sequential_66_layer_call_fn_513600À
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
I__inference_sequential_66_layer_call_and_return_conditional_losses_513731
I__inference_sequential_66_layer_call_and_return_conditional_losses_513762
I__inference_sequential_66_layer_call_and_return_conditional_losses_513626
I__inference_sequential_66_layer_call_and_return_conditional_losses_513652À
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
!__inference__wrapped_model_513397normalization_669_input"
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
__inference_adapt_step_513831
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
": 	
2dense_198/kernel
:
2dense_198/bias
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
Ô2Ñ
*__inference_dense_198_layer_call_fn_513840¢
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
ï2ì
E__inference_dense_198_layer_call_and_return_conditional_losses_513851¢
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
": 
2dense_199/kernel
:2dense_199/bias
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
Ô2Ñ
*__inference_dense_199_layer_call_fn_513860¢
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
ï2ì
E__inference_dense_199_layer_call_and_return_conditional_losses_513871¢
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
": 2dense_200/kernel
:2dense_200/bias
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
Ô2Ñ
*__inference_dense_200_layer_call_fn_513880¢
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
ï2ì
E__inference_dense_200_layer_call_and_return_conditional_losses_513890¢
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
$__inference_signature_wrapper_513785normalization_669_input"
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
':%	
2Adam/dense_198/kernel/m
!:
2Adam/dense_198/bias/m
':%
2Adam/dense_199/kernel/m
!:2Adam/dense_199/bias/m
':%2Adam/dense_200/kernel/m
!:2Adam/dense_200/bias/m
':%	
2Adam/dense_198/kernel/v
!:
2Adam/dense_198/bias/v
':%
2Adam/dense_199/kernel/v
!:2Adam/dense_199/bias/v
':%2Adam/dense_200/kernel/v
!:2Adam/dense_200/bias/v
	J
Const
J	
Const_1²
!__inference__wrapped_model_513397Z[ '(I¢F
?¢<
:7
normalization_669_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_200# 
	dense_200ÿÿÿÿÿÿÿÿÿo
__inference_adapt_step_513831NC¢@
9¢6
41¢
ÿÿÿÿÿÿÿÿÿ	IteratorSpec 
ª "
 ¥
E__inference_dense_198_layer_call_and_return_conditional_losses_513851\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ	
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 }
*__inference_dense_198_layer_call_fn_513840O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ	
ª "ÿÿÿÿÿÿÿÿÿ
¥
E__inference_dense_199_layer_call_and_return_conditional_losses_513871\ /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_199_layer_call_fn_513860O /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_200_layer_call_and_return_conditional_losses_513890\'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_200_layer_call_fn_513880O'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÒ
I__inference_sequential_66_layer_call_and_return_conditional_losses_513626Z[ '(Q¢N
G¢D
:7
normalization_669_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ò
I__inference_sequential_66_layer_call_and_return_conditional_losses_513652Z[ '(Q¢N
G¢D
:7
normalization_669_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 À
I__inference_sequential_66_layer_call_and_return_conditional_losses_513731sZ[ '(@¢=
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
I__inference_sequential_66_layer_call_and_return_conditional_losses_513762sZ[ '(@¢=
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
.__inference_sequential_66_layer_call_fn_513481wZ[ '(Q¢N
G¢D
:7
normalization_669_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ©
.__inference_sequential_66_layer_call_fn_513600wZ[ '(Q¢N
G¢D
:7
normalization_669_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_66_layer_call_fn_513679fZ[ '(@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_66_layer_call_fn_513700fZ[ '(@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÐ
$__inference_signature_wrapper_513785§Z[ '(d¢a
¢ 
ZªW
U
normalization_669_input:7
normalization_669_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_200# 
	dense_200ÿÿÿÿÿÿÿÿÿ