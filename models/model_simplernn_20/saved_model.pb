??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.0-dev202006202v1.12.1-34769-gfd2d4cdb708؇
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
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
?
#simple_rnn/simple_rnn_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#simple_rnn/simple_rnn_cell_2/kernel
?
7simple_rnn/simple_rnn_cell_2/kernel/Read/ReadVariableOpReadVariableOp#simple_rnn/simple_rnn_cell_2/kernel*
_output_shapes

:*
dtype0
?
-simple_rnn/simple_rnn_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*>
shared_name/-simple_rnn/simple_rnn_cell_2/recurrent_kernel
?
Asimple_rnn/simple_rnn_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp-simple_rnn/simple_rnn_cell_2/recurrent_kernel*
_output_shapes

:*
dtype0
?
!simple_rnn/simple_rnn_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!simple_rnn/simple_rnn_cell_2/bias
?
5simple_rnn/simple_rnn_cell_2/bias/Read/ReadVariableOpReadVariableOp!simple_rnn/simple_rnn_cell_2/bias*
_output_shapes
:*
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
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
*Adam/simple_rnn/simple_rnn_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/simple_rnn/simple_rnn_cell_2/kernel/m
?
>Adam/simple_rnn/simple_rnn_cell_2/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn/simple_rnn_cell_2/kernel/m*
_output_shapes

:*
dtype0
?
4Adam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*E
shared_name64Adam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/m
?
HAdam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp4Adam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/m*
_output_shapes

:*
dtype0
?
(Adam/simple_rnn/simple_rnn_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/simple_rnn/simple_rnn_cell_2/bias/m
?
<Adam/simple_rnn/simple_rnn_cell_2/bias/m/Read/ReadVariableOpReadVariableOp(Adam/simple_rnn/simple_rnn_cell_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
?
*Adam/simple_rnn/simple_rnn_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/simple_rnn/simple_rnn_cell_2/kernel/v
?
>Adam/simple_rnn/simple_rnn_cell_2/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn/simple_rnn_cell_2/kernel/v*
_output_shapes

:*
dtype0
?
4Adam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*E
shared_name64Adam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/v
?
HAdam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp4Adam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/v*
_output_shapes

:*
dtype0
?
(Adam/simple_rnn/simple_rnn_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/simple_rnn/simple_rnn_cell_2/bias/v
?
<Adam/simple_rnn/simple_rnn_cell_2/bias/v/Read/ReadVariableOpReadVariableOp(Adam/simple_rnn/simple_rnn_cell_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?!
value?!B?! B?!
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
l
	cell


state_spec
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratem;m<m=m>m?v@vAvBvCvD
#
0
1
2
3
4
#
0
1
2
3
4
 
?
non_trainable_variables
trainable_variables
metrics
	variables
layer_metrics
 layer_regularization_losses
regularization_losses

!layers
 
~

kernel
recurrent_kernel
bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
 

0
1
2

0
1
2
 
?
&non_trainable_variables
trainable_variables
'metrics
	variables
(layer_metrics
)layer_regularization_losses

*states
regularization_losses

+layers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
,non_trainable_variables
trainable_variables
-metrics
	variables
.layer_metrics
/layer_regularization_losses
regularization_losses

0layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#simple_rnn/simple_rnn_cell_2/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE-simple_rnn/simple_rnn_cell_2/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!simple_rnn/simple_rnn_cell_2/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

10
 
 

0
1

0
1
2

0
1
2
 
?
2non_trainable_variables
"trainable_variables
3metrics
#	variables
4layer_metrics
5layer_regularization_losses
$regularization_losses

6layers
 
 
 
 
 

	0
 
 
 
 
 
4
	7total
	8count
9	variables
:	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

70
81

9	variables
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/simple_rnn/simple_rnn_cell_2/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(Adam/simple_rnn/simple_rnn_cell_2/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/simple_rnn/simple_rnn_cell_2/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(Adam/simple_rnn/simple_rnn_cell_2/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
 serving_default_simple_rnn_inputPlaceholder*#
_output_shapes
:?z*
dtype0*
shape:?z
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_simple_rnn_input#simple_rnn/simple_rnn_cell_2/kernel!simple_rnn/simple_rnn_cell_2/bias-simple_rnn/simple_rnn_cell_2/recurrent_kerneldense_2/kerneldense_2/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_56888
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp7simple_rnn/simple_rnn_cell_2/kernel/Read/ReadVariableOpAsimple_rnn/simple_rnn_cell_2/recurrent_kernel/Read/ReadVariableOp5simple_rnn/simple_rnn_cell_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp>Adam/simple_rnn/simple_rnn_cell_2/kernel/m/Read/ReadVariableOpHAdam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/m/Read/ReadVariableOp<Adam/simple_rnn/simple_rnn_cell_2/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp>Adam/simple_rnn/simple_rnn_cell_2/kernel/v/Read/ReadVariableOpHAdam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/v/Read/ReadVariableOp<Adam/simple_rnn/simple_rnn_cell_2/bias/v/Read/ReadVariableOpConst*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_58106
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate#simple_rnn/simple_rnn_cell_2/kernel-simple_rnn/simple_rnn_cell_2/recurrent_kernel!simple_rnn/simple_rnn_cell_2/biastotalcountAdam/dense_2/kernel/mAdam/dense_2/bias/m*Adam/simple_rnn/simple_rnn_cell_2/kernel/m4Adam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/m(Adam/simple_rnn/simple_rnn_cell_2/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/v*Adam/simple_rnn/simple_rnn_cell_2/kernel/v4Adam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/v(Adam/simple_rnn/simple_rnn_cell_2/bias/v*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_58184??
?@
?
(sequential_2_simple_rnn_while_body_55908L
Hsequential_2_simple_rnn_while_sequential_2_simple_rnn_while_loop_counterR
Nsequential_2_simple_rnn_while_sequential_2_simple_rnn_while_maximum_iterations-
)sequential_2_simple_rnn_while_placeholder/
+sequential_2_simple_rnn_while_placeholder_1/
+sequential_2_simple_rnn_while_placeholder_2K
Gsequential_2_simple_rnn_while_sequential_2_simple_rnn_strided_slice_1_0?
?sequential_2_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_2_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0T
Psequential_2_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0U
Qsequential_2_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0V
Rsequential_2_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
&sequential_2_simple_rnn_while_identity,
(sequential_2_simple_rnn_while_identity_1,
(sequential_2_simple_rnn_while_identity_2,
(sequential_2_simple_rnn_while_identity_3,
(sequential_2_simple_rnn_while_identity_4I
Esequential_2_simple_rnn_while_sequential_2_simple_rnn_strided_slice_1?
?sequential_2_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_2_simple_rnn_tensorarrayunstack_tensorlistfromtensorR
Nsequential_2_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceS
Osequential_2_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceT
Psequential_2_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource??
Osequential_2/simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2Q
Osequential_2/simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Asequential_2/simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_2_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_2_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0)sequential_2_simple_rnn_while_placeholderXsequential_2/simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	?*
element_dtype02C
Asequential_2/simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
Esequential_2/simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpPsequential_2_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype02G
Esequential_2/simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
6sequential_2/simple_rnn/while/simple_rnn_cell_2/MatMulMatMulHsequential_2/simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Msequential_2/simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?28
6sequential_2/simple_rnn/while/simple_rnn_cell_2/MatMul?
Fsequential_2/simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpQsequential_2_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype02H
Fsequential_2/simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
7sequential_2/simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAdd@sequential_2/simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Nsequential_2/simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?29
7sequential_2/simple_rnn/while/simple_rnn_cell_2/BiasAdd?
Gsequential_2/simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpRsequential_2_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype02I
Gsequential_2/simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
8sequential_2/simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMul+sequential_2_simple_rnn_while_placeholder_2Osequential_2/simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2:
8sequential_2/simple_rnn/while/simple_rnn_cell_2/MatMul_1?
3sequential_2/simple_rnn/while/simple_rnn_cell_2/addAddV2@sequential_2/simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:0Bsequential_2/simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?25
3sequential_2/simple_rnn/while/simple_rnn_cell_2/add?
4sequential_2/simple_rnn/while/simple_rnn_cell_2/TanhTanh7sequential_2/simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?26
4sequential_2/simple_rnn/while/simple_rnn_cell_2/Tanh?
Bsequential_2/simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem+sequential_2_simple_rnn_while_placeholder_1)sequential_2_simple_rnn_while_placeholder8sequential_2/simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02D
Bsequential_2/simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
#sequential_2/simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_2/simple_rnn/while/add/y?
!sequential_2/simple_rnn/while/addAddV2)sequential_2_simple_rnn_while_placeholder,sequential_2/simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2#
!sequential_2/simple_rnn/while/add?
%sequential_2/simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_2/simple_rnn/while/add_1/y?
#sequential_2/simple_rnn/while/add_1AddV2Hsequential_2_simple_rnn_while_sequential_2_simple_rnn_while_loop_counter.sequential_2/simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2%
#sequential_2/simple_rnn/while/add_1?
&sequential_2/simple_rnn/while/IdentityIdentity'sequential_2/simple_rnn/while/add_1:z:0*
T0*
_output_shapes
: 2(
&sequential_2/simple_rnn/while/Identity?
(sequential_2/simple_rnn/while/Identity_1IdentityNsequential_2_simple_rnn_while_sequential_2_simple_rnn_while_maximum_iterations*
T0*
_output_shapes
: 2*
(sequential_2/simple_rnn/while/Identity_1?
(sequential_2/simple_rnn/while/Identity_2Identity%sequential_2/simple_rnn/while/add:z:0*
T0*
_output_shapes
: 2*
(sequential_2/simple_rnn/while/Identity_2?
(sequential_2/simple_rnn/while/Identity_3IdentityRsequential_2/simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2*
(sequential_2/simple_rnn/while/Identity_3?
(sequential_2/simple_rnn/while/Identity_4Identity8sequential_2/simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
T0*
_output_shapes
:	?2*
(sequential_2/simple_rnn/while/Identity_4"Y
&sequential_2_simple_rnn_while_identity/sequential_2/simple_rnn/while/Identity:output:0"]
(sequential_2_simple_rnn_while_identity_11sequential_2/simple_rnn/while/Identity_1:output:0"]
(sequential_2_simple_rnn_while_identity_21sequential_2/simple_rnn/while/Identity_2:output:0"]
(sequential_2_simple_rnn_while_identity_31sequential_2/simple_rnn/while/Identity_3:output:0"]
(sequential_2_simple_rnn_while_identity_41sequential_2/simple_rnn/while/Identity_4:output:0"?
Esequential_2_simple_rnn_while_sequential_2_simple_rnn_strided_slice_1Gsequential_2_simple_rnn_while_sequential_2_simple_rnn_strided_slice_1_0"?
Osequential_2_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceQsequential_2_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Psequential_2_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceRsequential_2_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Nsequential_2_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourcePsequential_2_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
?sequential_2_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_2_simple_rnn_tensorarrayunstack_tensorlistfromtensor?sequential_2_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_2_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	?: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_57727
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_57727___redundant_placeholder03
/while_while_cond_57727___redundant_placeholder13
/while_while_cond_57727___redundant_placeholder23
/while_while_cond_57727___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?)
?
while_body_56657
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_2_matmul_readvariableop_resource;
7while_simple_rnn_cell_2_biasadd_readvariableop_resource<
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	?*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0*
T0*
_output_shapes
:	?2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	?: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: 
?
?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_57968

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:?????????2

Identity`

Identity_1IdentityTanh:y:0*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
?C
?
E__inference_simple_rnn_layer_call_and_return_conditional_losses_57660

inputs4
0simple_rnn_cell_2_matmul_readvariableop_resource5
1simple_rnn_cell_2_biasadd_readvariableop_resource6
2simple_rnn_cell_2_matmul_1_readvariableop_resource
identity??whilec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constm
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*
_output_shapes
:	?2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permr
	transpose	Transposeinputstranspose/perm:output:0*
T0*#
_output_shapes
:z?2
	transposeg
Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"z   ?     2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/add}
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	?: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_57594*
condR
while_cond_57593*0
output_shapes
: : : : :	?: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:z?*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*#
_output_shapes
:?z2
transpose_1g
IdentityIdentitytranspose_1:y:0^while*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*.
_input_shapes
:?z:::2
whilewhile:K G
#
_output_shapes
:?z
 
_user_specified_nameinputs
?
?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_56033

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:?????????2

Identity`

Identity_1IdentityTanh:y:0*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?
?
while_cond_57593
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_57593___redundant_placeholder03
/while_while_cond_57593___redundant_placeholder13
/while_while_cond_57593___redundant_placeholder23
/while_while_cond_57593___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*8
_input_shapes'
%: : : : :	?: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
:
?
?
B__inference_dense_2_layer_call_and_return_conditional_losses_57942

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOp?
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?l    2
Tensordot/Reshape/shape?
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
??2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
Tensordot/MatMulw
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
Tensordot/shape?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*#
_output_shapes
:?z2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:?z2	
BiasAdd`
IdentityIdentityBiasAdd:output:0*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0**
_input_shapes
:?z:::K G
#
_output_shapes
:?z
 
_user_specified_nameinputs
?<
?
E__inference_simple_rnn_layer_call_and_return_conditional_losses_56487

inputs
simple_rnn_cell_2_56412
simple_rnn_cell_2_56414
simple_rnn_cell_2_56416
identity??)simple_rnn_cell_2/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
)simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_2_56412simple_rnn_cell_2_56414simple_rnn_cell_2_56416*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_560502+
)simple_rnn_cell_2/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_2_56412simple_rnn_cell_2_56414simple_rnn_cell_2_56416*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_56424*
condR
while_cond_56423*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_2/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2V
)simple_rnn_cell_2/StatefulPartitionedCall)simple_rnn_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_57147
simple_rnn_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_568192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????z:::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????z
*
_user_specified_namesimple_rnn_input
?
?
while_cond_56306
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_56306___redundant_placeholder03
/while_while_cond_56306___redundant_placeholder13
/while_while_cond_56306___redundant_placeholder23
/while_while_cond_56306___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?a
?
!__inference__traced_restore_58184
file_prefix#
assignvariableop_dense_2_kernel#
assignvariableop_1_dense_2_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate:
6assignvariableop_7_simple_rnn_simple_rnn_cell_2_kernelD
@assignvariableop_8_simple_rnn_simple_rnn_cell_2_recurrent_kernel8
4assignvariableop_9_simple_rnn_simple_rnn_cell_2_bias
assignvariableop_10_total
assignvariableop_11_count-
)assignvariableop_12_adam_dense_2_kernel_m+
'assignvariableop_13_adam_dense_2_bias_mB
>assignvariableop_14_adam_simple_rnn_simple_rnn_cell_2_kernel_mL
Hassignvariableop_15_adam_simple_rnn_simple_rnn_cell_2_recurrent_kernel_m@
<assignvariableop_16_adam_simple_rnn_simple_rnn_cell_2_bias_m-
)assignvariableop_17_adam_dense_2_kernel_v+
'assignvariableop_18_adam_dense_2_bias_vB
>assignvariableop_19_adam_simple_rnn_simple_rnn_cell_2_kernel_vL
Hassignvariableop_20_adam_simple_rnn_simple_rnn_cell_2_recurrent_kernel_v@
<assignvariableop_21_adam_simple_rnn_simple_rnn_cell_2_bias_v
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp6assignvariableop_7_simple_rnn_simple_rnn_cell_2_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp@assignvariableop_8_simple_rnn_simple_rnn_cell_2_recurrent_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp4assignvariableop_9_simple_rnn_simple_rnn_cell_2_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_dense_2_kernel_mIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_adam_dense_2_bias_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp>assignvariableop_14_adam_simple_rnn_simple_rnn_cell_2_kernel_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpHassignvariableop_15_adam_simple_rnn_simple_rnn_cell_2_recurrent_kernel_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp<assignvariableop_16_adam_simple_rnn_simple_rnn_cell_2_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_2_kernel_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_2_bias_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp>assignvariableop_19_adam_simple_rnn_simple_rnn_cell_2_kernel_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpHassignvariableop_20_adam_simple_rnn_simple_rnn_cell_2_recurrent_kernel_vIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp<assignvariableop_21_adam_simple_rnn_simple_rnn_cell_2_bias_vIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22?
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2$
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
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?4
?	
simple_rnn_while_body_570562
.simple_rnn_while_simple_rnn_while_loop_counter8
4simple_rnn_while_simple_rnn_while_maximum_iterations 
simple_rnn_while_placeholder"
simple_rnn_while_placeholder_1"
simple_rnn_while_placeholder_21
-simple_rnn_while_simple_rnn_strided_slice_1_0m
isimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0G
Csimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0H
Dsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0I
Esimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0
simple_rnn_while_identity
simple_rnn_while_identity_1
simple_rnn_while_identity_2
simple_rnn_while_identity_3
simple_rnn_while_identity_4/
+simple_rnn_while_simple_rnn_strided_slice_1k
gsimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensorE
Asimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceF
Bsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceG
Csimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource??
Bsimple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2D
Bsimple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
4simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemisimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_while_placeholderKsimple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	?*
element_dtype026
4simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
8simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpCsimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype02:
8simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
)simple_rnn/while/simple_rnn_cell_2/MatMulMatMul;simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0@simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2+
)simple_rnn/while/simple_rnn_cell_2/MatMul?
9simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpDsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype02;
9simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
*simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAdd3simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Asimple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2,
*simple_rnn/while/simple_rnn_cell_2/BiasAdd?
:simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpEsimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype02<
:simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
+simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMulsimple_rnn_while_placeholder_2Bsimple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2-
+simple_rnn/while/simple_rnn_cell_2/MatMul_1?
&simple_rnn/while/simple_rnn_cell_2/addAddV23simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:05simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2(
&simple_rnn/while/simple_rnn_cell_2/add?
'simple_rnn/while/simple_rnn_cell_2/TanhTanh*simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2)
'simple_rnn/while/simple_rnn_cell_2/Tanh?
5simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemsimple_rnn_while_placeholder_1simple_rnn_while_placeholder+simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype027
5simple_rnn/while/TensorArrayV2Write/TensorListSetItemr
simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/while/add/y?
simple_rnn/while/addAddV2simple_rnn_while_placeholdersimple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/while/addv
simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/while/add_1/y?
simple_rnn/while/add_1AddV2.simple_rnn_while_simple_rnn_while_loop_counter!simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/while/add_1
simple_rnn/while/IdentityIdentitysimple_rnn/while/add_1:z:0*
T0*
_output_shapes
: 2
simple_rnn/while/Identity?
simple_rnn/while/Identity_1Identity4simple_rnn_while_simple_rnn_while_maximum_iterations*
T0*
_output_shapes
: 2
simple_rnn/while/Identity_1?
simple_rnn/while/Identity_2Identitysimple_rnn/while/add:z:0*
T0*
_output_shapes
: 2
simple_rnn/while/Identity_2?
simple_rnn/while/Identity_3IdentityEsimple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
simple_rnn/while/Identity_3?
simple_rnn/while/Identity_4Identity+simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
T0*
_output_shapes
:	?2
simple_rnn/while/Identity_4"?
simple_rnn_while_identity"simple_rnn/while/Identity:output:0"C
simple_rnn_while_identity_1$simple_rnn/while/Identity_1:output:0"C
simple_rnn_while_identity_2$simple_rnn/while/Identity_2:output:0"C
simple_rnn_while_identity_3$simple_rnn/while/Identity_3:output:0"C
simple_rnn_while_identity_4$simple_rnn/while/Identity_4:output:0"?
Bsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceDsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Csimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceEsimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Asimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceCsimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"\
+simple_rnn_while_simple_rnn_strided_slice_1-simple_rnn_while_simple_rnn_strided_slice_1_0"?
gsimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensorisimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	?: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: 
?#
?
while_body_56307
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_2_56329_0#
while_simple_rnn_cell_2_56331_0#
while_simple_rnn_cell_2_56333_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_2_56329!
while_simple_rnn_cell_2_56331!
while_simple_rnn_cell_2_56333??/while/simple_rnn_cell_2/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
/while/simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_2_56329_0while_simple_rnn_cell_2_56331_0while_simple_rnn_cell_2_56333_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_5603321
/while/simple_rnn_cell_2/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_2/StatefulPartitionedCall:output:10^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_2_56329while_simple_rnn_cell_2_56329_0"@
while_simple_rnn_cell_2_56331while_simple_rnn_cell_2_56331_0"@
while_simple_rnn_cell_2_56333while_simple_rnn_cell_2_56333_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::2b
/while/simple_rnn_cell_2/StatefulPartitionedCall/while/simple_rnn_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_56423
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_56423___redundant_placeholder03
/while_while_cond_56423___redundant_placeholder13
/while_while_cond_56423___redundant_placeholder23
/while_while_cond_56423___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_57985

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:?????????2

Identity`

Identity_1IdentityTanh:y:0*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
?4
?	
simple_rnn_while_body_573302
.simple_rnn_while_simple_rnn_while_loop_counter8
4simple_rnn_while_simple_rnn_while_maximum_iterations 
simple_rnn_while_placeholder"
simple_rnn_while_placeholder_1"
simple_rnn_while_placeholder_21
-simple_rnn_while_simple_rnn_strided_slice_1_0m
isimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0G
Csimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0H
Dsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0I
Esimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0
simple_rnn_while_identity
simple_rnn_while_identity_1
simple_rnn_while_identity_2
simple_rnn_while_identity_3
simple_rnn_while_identity_4/
+simple_rnn_while_simple_rnn_strided_slice_1k
gsimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensorE
Asimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceF
Bsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceG
Csimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource??
Bsimple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2D
Bsimple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
4simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemisimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_while_placeholderKsimple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	?*
element_dtype026
4simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
8simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpCsimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype02:
8simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
)simple_rnn/while/simple_rnn_cell_2/MatMulMatMul;simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0@simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2+
)simple_rnn/while/simple_rnn_cell_2/MatMul?
9simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpDsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype02;
9simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
*simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAdd3simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Asimple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2,
*simple_rnn/while/simple_rnn_cell_2/BiasAdd?
:simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpEsimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype02<
:simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
+simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMulsimple_rnn_while_placeholder_2Bsimple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2-
+simple_rnn/while/simple_rnn_cell_2/MatMul_1?
&simple_rnn/while/simple_rnn_cell_2/addAddV23simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:05simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2(
&simple_rnn/while/simple_rnn_cell_2/add?
'simple_rnn/while/simple_rnn_cell_2/TanhTanh*simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2)
'simple_rnn/while/simple_rnn_cell_2/Tanh?
5simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemsimple_rnn_while_placeholder_1simple_rnn_while_placeholder+simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype027
5simple_rnn/while/TensorArrayV2Write/TensorListSetItemr
simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/while/add/y?
simple_rnn/while/addAddV2simple_rnn_while_placeholdersimple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/while/addv
simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/while/add_1/y?
simple_rnn/while/add_1AddV2.simple_rnn_while_simple_rnn_while_loop_counter!simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/while/add_1
simple_rnn/while/IdentityIdentitysimple_rnn/while/add_1:z:0*
T0*
_output_shapes
: 2
simple_rnn/while/Identity?
simple_rnn/while/Identity_1Identity4simple_rnn_while_simple_rnn_while_maximum_iterations*
T0*
_output_shapes
: 2
simple_rnn/while/Identity_1?
simple_rnn/while/Identity_2Identitysimple_rnn/while/add:z:0*
T0*
_output_shapes
: 2
simple_rnn/while/Identity_2?
simple_rnn/while/Identity_3IdentityEsimple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
simple_rnn/while/Identity_3?
simple_rnn/while/Identity_4Identity+simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
T0*
_output_shapes
:	?2
simple_rnn/while/Identity_4"?
simple_rnn_while_identity"simple_rnn/while/Identity:output:0"C
simple_rnn_while_identity_1$simple_rnn/while/Identity_1:output:0"C
simple_rnn_while_identity_2$simple_rnn/while/Identity_2:output:0"C
simple_rnn_while_identity_3$simple_rnn/while/Identity_3:output:0"C
simple_rnn_while_identity_4$simple_rnn/while/Identity_4:output:0"?
Bsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceDsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Csimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceEsimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Asimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceCsimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"\
+simple_rnn_while_simple_rnn_strided_slice_1-simple_rnn_while_simple_rnn_strided_slice_1_0"?
gsimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensorisimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	?: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: 
?\
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_57284

inputs?
;simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource@
<simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceA
=simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??simple_rnn/whiley
simple_rnn/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
simple_rnn/Shape?
simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
simple_rnn/strided_slice/stack?
 simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 simple_rnn/strided_slice/stack_1?
 simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 simple_rnn/strided_slice/stack_2?
simple_rnn/strided_sliceStridedSlicesimple_rnn/Shape:output:0'simple_rnn/strided_slice/stack:output:0)simple_rnn/strided_slice/stack_1:output:0)simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn/strided_slicer
simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/zeros/mul/y?
simple_rnn/zeros/mulMul!simple_rnn/strided_slice:output:0simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/zeros/mulu
simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn/zeros/Less/y?
simple_rnn/zeros/LessLesssimple_rnn/zeros/mul:z:0 simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/zeros/Lessx
simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/zeros/packed/1?
simple_rnn/zeros/packedPack!simple_rnn/strided_slice:output:0"simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn/zeros/packedu
simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn/zeros/Const?
simple_rnn/zerosFill simple_rnn/zeros/packed:output:0simple_rnn/zeros/Const:output:0*
T0*
_output_shapes
:	?2
simple_rnn/zeros?
simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn/transpose/perm?
simple_rnn/transpose	Transposeinputs"simple_rnn/transpose/perm:output:0*
T0*#
_output_shapes
:z?2
simple_rnn/transpose}
simple_rnn/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"z   ?     2
simple_rnn/Shape_1?
 simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn/strided_slice_1/stack?
"simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_1/stack_1?
"simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_1/stack_2?
simple_rnn/strided_slice_1StridedSlicesimple_rnn/Shape_1:output:0)simple_rnn/strided_slice_1/stack:output:0+simple_rnn/strided_slice_1/stack_1:output:0+simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn/strided_slice_1?
&simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&simple_rnn/TensorArrayV2/element_shape?
simple_rnn/TensorArrayV2TensorListReserve/simple_rnn/TensorArrayV2/element_shape:output:0#simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn/TensorArrayV2?
@simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2B
@simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
2simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn/transpose:y:0Isimple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2simple_rnn/TensorArrayUnstack/TensorListFromTensor?
 simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn/strided_slice_2/stack?
"simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_2/stack_1?
"simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_2/stack_2?
simple_rnn/strided_slice_2StridedSlicesimple_rnn/transpose:y:0)simple_rnn/strided_slice_2/stack:output:0+simple_rnn/strided_slice_2/stack_1:output:0+simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
simple_rnn/strided_slice_2?
2simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp;simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype024
2simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
#simple_rnn/simple_rnn_cell_2/MatMulMatMul#simple_rnn/strided_slice_2:output:0:simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2%
#simple_rnn/simple_rnn_cell_2/MatMul?
3simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp<simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
$simple_rnn/simple_rnn_cell_2/BiasAddBiasAdd-simple_rnn/simple_rnn_cell_2/MatMul:product:0;simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2&
$simple_rnn/simple_rnn_cell_2/BiasAdd?
4simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp=simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype026
4simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
%simple_rnn/simple_rnn_cell_2/MatMul_1MatMulsimple_rnn/zeros:output:0<simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%simple_rnn/simple_rnn_cell_2/MatMul_1?
 simple_rnn/simple_rnn_cell_2/addAddV2-simple_rnn/simple_rnn_cell_2/BiasAdd:output:0/simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2"
 simple_rnn/simple_rnn_cell_2/add?
!simple_rnn/simple_rnn_cell_2/TanhTanh$simple_rnn/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2#
!simple_rnn/simple_rnn_cell_2/Tanh?
(simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2*
(simple_rnn/TensorArrayV2_1/element_shape?
simple_rnn/TensorArrayV2_1TensorListReserve1simple_rnn/TensorArrayV2_1/element_shape:output:0#simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn/TensorArrayV2_1d
simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn/time?
#simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#simple_rnn/while/maximum_iterations?
simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn/while/loop_counter?
simple_rnn/whileWhile&simple_rnn/while/loop_counter:output:0,simple_rnn/while/maximum_iterations:output:0simple_rnn/time:output:0#simple_rnn/TensorArrayV2_1:handle:0simple_rnn/zeros:output:0#simple_rnn/strided_slice_1:output:0Bsimple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0;simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource<simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource=simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	?: : : : : *%
_read_only_resource_inputs
	*'
bodyR
simple_rnn_while_body_57208*'
condR
simple_rnn_while_cond_57207*0
output_shapes
: : : : :	?: : : : : *
parallel_iterations 2
simple_rnn/while?
;simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2=
;simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
-simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn/while:output:3Dsimple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:z?*
element_dtype02/
-simple_rnn/TensorArrayV2Stack/TensorListStack?
 simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 simple_rnn/strided_slice_3/stack?
"simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn/strided_slice_3/stack_1?
"simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_3/stack_2?
simple_rnn/strided_slice_3StridedSlice6simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0)simple_rnn/strided_slice_3/stack:output:0+simple_rnn/strided_slice_3/stack_1:output:0+simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
simple_rnn/strided_slice_3?
simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn/transpose_1/perm?
simple_rnn/transpose_1	Transpose6simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0$simple_rnn/transpose_1/perm:output:0*
T0*#
_output_shapes
:?z2
simple_rnn/transpose_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_2/Tensordot/ReadVariableOp?
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?l    2!
dense_2/Tensordot/Reshape/shape?
dense_2/Tensordot/ReshapeReshapesimple_rnn/transpose_1:y:0(dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
??2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
dense_2/Tensordot/shape?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
:?z2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:?z2
dense_2/BiasAdd{
IdentityIdentitydense_2/BiasAdd:output:0^simple_rnn/while*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?z:::::2$
simple_rnn/whilesimple_rnn/while:S O
+
_output_shapes
:?????????z
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_57162
simple_rnn_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_568502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????z:::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????z
*
_user_specified_namesimple_rnn_input
?)
?
while_body_57594
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_2_matmul_readvariableop_resource;
7while_simple_rnn_cell_2_biasadd_readvariableop_resource<
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	?*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0*
T0*
_output_shapes
:	?2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	?: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: 
?
?
*__inference_simple_rnn_layer_call_fn_57928
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_simple_rnn_layer_call_and_return_conditional_losses_564872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?D
?
E__inference_simple_rnn_layer_call_and_return_conditional_losses_57906
inputs_04
0simple_rnn_cell_2_matmul_readvariableop_resource5
1simple_rnn_cell_2_biasadd_readvariableop_resource6
2simple_rnn_cell_2_matmul_1_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_57840*
condR
while_cond_57839*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1x
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
|
'__inference_dense_2_layer_call_fn_57951

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_567672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0**
_input_shapes
:?z::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?z
 
_user_specified_nameinputs
?
?
*__inference_simple_rnn_layer_call_fn_57682

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_simple_rnn_layer_call_and_return_conditional_losses_567232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*.
_input_shapes
:?z:::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?z
 
_user_specified_nameinputs
?#
?
while_body_56424
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_2_56446_0#
while_simple_rnn_cell_2_56448_0#
while_simple_rnn_cell_2_56450_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_2_56446!
while_simple_rnn_cell_2_56448!
while_simple_rnn_cell_2_56450??/while/simple_rnn_cell_2/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
/while/simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_2_56446_0while_simple_rnn_cell_2_56448_0while_simple_rnn_cell_2_56450_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_5605021
/while/simple_rnn_cell_2/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_2/StatefulPartitionedCall:output:10^while/simple_rnn_cell_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_2_56446while_simple_rnn_cell_2_56446_0"@
while_simple_rnn_cell_2_56448while_simple_rnn_cell_2_56448_0"@
while_simple_rnn_cell_2_56450while_simple_rnn_cell_2_56450_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::2b
/while/simple_rnn_cell_2/StatefulPartitionedCall/while/simple_rnn_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?	
?
simple_rnn_while_cond_572072
.simple_rnn_while_simple_rnn_while_loop_counter8
4simple_rnn_while_simple_rnn_while_maximum_iterations 
simple_rnn_while_placeholder"
simple_rnn_while_placeholder_1"
simple_rnn_while_placeholder_24
0simple_rnn_while_less_simple_rnn_strided_slice_1I
Esimple_rnn_while_simple_rnn_while_cond_57207___redundant_placeholder0I
Esimple_rnn_while_simple_rnn_while_cond_57207___redundant_placeholder1I
Esimple_rnn_while_simple_rnn_while_cond_57207___redundant_placeholder2I
Esimple_rnn_while_simple_rnn_while_cond_57207___redundant_placeholder3
simple_rnn_while_identity
?
simple_rnn/while/LessLesssimple_rnn_while_placeholder0simple_rnn_while_less_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn/while/Less~
simple_rnn/while/IdentityIdentitysimple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn/while/Identity"?
simple_rnn_while_identity"simple_rnn/while/Identity:output:0*8
_input_shapes'
%: : : : :	?: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
:
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_56819

inputs
simple_rnn_56806
simple_rnn_56808
simple_rnn_56810
dense_2_56813
dense_2_56815
identity??dense_2/StatefulPartitionedCall?"simple_rnn/StatefulPartitionedCall?
"simple_rnn/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_56806simple_rnn_56808simple_rnn_56810*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_simple_rnn_layer_call_and_return_conditional_losses_566112$
"simple_rnn/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall+simple_rnn/StatefulPartitionedCall:output:0dense_2_56813dense_2_56815*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_567672!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall#^simple_rnn/StatefulPartitionedCall*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?z:::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"simple_rnn/StatefulPartitionedCall"simple_rnn/StatefulPartitionedCall:S O
+
_output_shapes
:?????????z
 
_user_specified_nameinputs
?*
?
while_body_57728
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_2_matmul_readvariableop_resource;
7while_simple_rnn_cell_2_biasadd_readvariableop_resource<
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_56850

inputs
simple_rnn_56837
simple_rnn_56839
simple_rnn_56841
dense_2_56844
dense_2_56846
identity??dense_2/StatefulPartitionedCall?"simple_rnn/StatefulPartitionedCall?
"simple_rnn/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_56837simple_rnn_56839simple_rnn_56841*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_simple_rnn_layer_call_and_return_conditional_losses_567232$
"simple_rnn/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall+simple_rnn/StatefulPartitionedCall:output:0dense_2_56844dense_2_56846*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_567672!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall#^simple_rnn/StatefulPartitionedCall*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?z:::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"simple_rnn/StatefulPartitionedCall"simple_rnn/StatefulPartitionedCall:S O
+
_output_shapes
:?????????z
 
_user_specified_nameinputs
?	
?
simple_rnn_while_cond_573292
.simple_rnn_while_simple_rnn_while_loop_counter8
4simple_rnn_while_simple_rnn_while_maximum_iterations 
simple_rnn_while_placeholder"
simple_rnn_while_placeholder_1"
simple_rnn_while_placeholder_24
0simple_rnn_while_less_simple_rnn_strided_slice_1I
Esimple_rnn_while_simple_rnn_while_cond_57329___redundant_placeholder0I
Esimple_rnn_while_simple_rnn_while_cond_57329___redundant_placeholder1I
Esimple_rnn_while_simple_rnn_while_cond_57329___redundant_placeholder2I
Esimple_rnn_while_simple_rnn_while_cond_57329___redundant_placeholder3
simple_rnn_while_identity
?
simple_rnn/while/LessLesssimple_rnn_while_placeholder0simple_rnn_while_less_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn/while/Less~
simple_rnn/while/IdentityIdentitysimple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn/while/Identity"?
simple_rnn_while_identity"simple_rnn/while/Identity:output:0*8
_input_shapes'
%: : : : :	?: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
:
?
?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_56050

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:?????????2

Identity`

Identity_1IdentityTanh:y:0*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?	
?
1__inference_simple_rnn_cell_2_layer_call_fn_57999

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_560332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
?
?
#__inference_signature_wrapper_56888
simple_rnn_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_559842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?z:::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
#
_output_shapes
:?z
*
_user_specified_namesimple_rnn_input
?	
?
simple_rnn_while_cond_570552
.simple_rnn_while_simple_rnn_while_loop_counter8
4simple_rnn_while_simple_rnn_while_maximum_iterations 
simple_rnn_while_placeholder"
simple_rnn_while_placeholder_1"
simple_rnn_while_placeholder_24
0simple_rnn_while_less_simple_rnn_strided_slice_1I
Esimple_rnn_while_simple_rnn_while_cond_57055___redundant_placeholder0I
Esimple_rnn_while_simple_rnn_while_cond_57055___redundant_placeholder1I
Esimple_rnn_while_simple_rnn_while_cond_57055___redundant_placeholder2I
Esimple_rnn_while_simple_rnn_while_cond_57055___redundant_placeholder3
simple_rnn_while_identity
?
simple_rnn/while/LessLesssimple_rnn_while_placeholder0simple_rnn_while_less_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn/while/Less~
simple_rnn/while/IdentityIdentitysimple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn/while/Identity"?
simple_rnn_while_identity"simple_rnn/while/Identity:output:0*8
_input_shapes'
%: : : : :	?: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
:
?*
?
while_body_57840
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_2_matmul_readvariableop_resource;
7while_simple_rnn_cell_2_biasadd_readvariableop_resource<
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0*
T0*'
_output_shapes
:?????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?]
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_57010
simple_rnn_input?
;simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource@
<simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceA
=simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??simple_rnn/whiley
simple_rnn/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
simple_rnn/Shape?
simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
simple_rnn/strided_slice/stack?
 simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 simple_rnn/strided_slice/stack_1?
 simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 simple_rnn/strided_slice/stack_2?
simple_rnn/strided_sliceStridedSlicesimple_rnn/Shape:output:0'simple_rnn/strided_slice/stack:output:0)simple_rnn/strided_slice/stack_1:output:0)simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn/strided_slicer
simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/zeros/mul/y?
simple_rnn/zeros/mulMul!simple_rnn/strided_slice:output:0simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/zeros/mulu
simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn/zeros/Less/y?
simple_rnn/zeros/LessLesssimple_rnn/zeros/mul:z:0 simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/zeros/Lessx
simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/zeros/packed/1?
simple_rnn/zeros/packedPack!simple_rnn/strided_slice:output:0"simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn/zeros/packedu
simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn/zeros/Const?
simple_rnn/zerosFill simple_rnn/zeros/packed:output:0simple_rnn/zeros/Const:output:0*
T0*
_output_shapes
:	?2
simple_rnn/zeros?
simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn/transpose/perm?
simple_rnn/transpose	Transposesimple_rnn_input"simple_rnn/transpose/perm:output:0*
T0*#
_output_shapes
:z?2
simple_rnn/transpose}
simple_rnn/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"z   ?     2
simple_rnn/Shape_1?
 simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn/strided_slice_1/stack?
"simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_1/stack_1?
"simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_1/stack_2?
simple_rnn/strided_slice_1StridedSlicesimple_rnn/Shape_1:output:0)simple_rnn/strided_slice_1/stack:output:0+simple_rnn/strided_slice_1/stack_1:output:0+simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn/strided_slice_1?
&simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&simple_rnn/TensorArrayV2/element_shape?
simple_rnn/TensorArrayV2TensorListReserve/simple_rnn/TensorArrayV2/element_shape:output:0#simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn/TensorArrayV2?
@simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2B
@simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
2simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn/transpose:y:0Isimple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2simple_rnn/TensorArrayUnstack/TensorListFromTensor?
 simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn/strided_slice_2/stack?
"simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_2/stack_1?
"simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_2/stack_2?
simple_rnn/strided_slice_2StridedSlicesimple_rnn/transpose:y:0)simple_rnn/strided_slice_2/stack:output:0+simple_rnn/strided_slice_2/stack_1:output:0+simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
simple_rnn/strided_slice_2?
2simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp;simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype024
2simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
#simple_rnn/simple_rnn_cell_2/MatMulMatMul#simple_rnn/strided_slice_2:output:0:simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2%
#simple_rnn/simple_rnn_cell_2/MatMul?
3simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp<simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
$simple_rnn/simple_rnn_cell_2/BiasAddBiasAdd-simple_rnn/simple_rnn_cell_2/MatMul:product:0;simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2&
$simple_rnn/simple_rnn_cell_2/BiasAdd?
4simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp=simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype026
4simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
%simple_rnn/simple_rnn_cell_2/MatMul_1MatMulsimple_rnn/zeros:output:0<simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%simple_rnn/simple_rnn_cell_2/MatMul_1?
 simple_rnn/simple_rnn_cell_2/addAddV2-simple_rnn/simple_rnn_cell_2/BiasAdd:output:0/simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2"
 simple_rnn/simple_rnn_cell_2/add?
!simple_rnn/simple_rnn_cell_2/TanhTanh$simple_rnn/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2#
!simple_rnn/simple_rnn_cell_2/Tanh?
(simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2*
(simple_rnn/TensorArrayV2_1/element_shape?
simple_rnn/TensorArrayV2_1TensorListReserve1simple_rnn/TensorArrayV2_1/element_shape:output:0#simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn/TensorArrayV2_1d
simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn/time?
#simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#simple_rnn/while/maximum_iterations?
simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn/while/loop_counter?
simple_rnn/whileWhile&simple_rnn/while/loop_counter:output:0,simple_rnn/while/maximum_iterations:output:0simple_rnn/time:output:0#simple_rnn/TensorArrayV2_1:handle:0simple_rnn/zeros:output:0#simple_rnn/strided_slice_1:output:0Bsimple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0;simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource<simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource=simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	?: : : : : *%
_read_only_resource_inputs
	*'
bodyR
simple_rnn_while_body_56934*'
condR
simple_rnn_while_cond_56933*0
output_shapes
: : : : :	?: : : : : *
parallel_iterations 2
simple_rnn/while?
;simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2=
;simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
-simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn/while:output:3Dsimple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:z?*
element_dtype02/
-simple_rnn/TensorArrayV2Stack/TensorListStack?
 simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 simple_rnn/strided_slice_3/stack?
"simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn/strided_slice_3/stack_1?
"simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_3/stack_2?
simple_rnn/strided_slice_3StridedSlice6simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0)simple_rnn/strided_slice_3/stack:output:0+simple_rnn/strided_slice_3/stack_1:output:0+simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
simple_rnn/strided_slice_3?
simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn/transpose_1/perm?
simple_rnn/transpose_1	Transpose6simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0$simple_rnn/transpose_1/perm:output:0*
T0*#
_output_shapes
:?z2
simple_rnn/transpose_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_2/Tensordot/ReadVariableOp?
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?l    2!
dense_2/Tensordot/Reshape/shape?
dense_2/Tensordot/ReshapeReshapesimple_rnn/transpose_1:y:0(dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
??2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
dense_2/Tensordot/shape?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
:?z2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:?z2
dense_2/BiasAdd{
IdentityIdentitydense_2/BiasAdd:output:0^simple_rnn/while*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?z:::::2$
simple_rnn/whilesimple_rnn/while:] Y
+
_output_shapes
:?????????z
*
_user_specified_namesimple_rnn_input
?	
?
simple_rnn_while_cond_569332
.simple_rnn_while_simple_rnn_while_loop_counter8
4simple_rnn_while_simple_rnn_while_maximum_iterations 
simple_rnn_while_placeholder"
simple_rnn_while_placeholder_1"
simple_rnn_while_placeholder_24
0simple_rnn_while_less_simple_rnn_strided_slice_1I
Esimple_rnn_while_simple_rnn_while_cond_56933___redundant_placeholder0I
Esimple_rnn_while_simple_rnn_while_cond_56933___redundant_placeholder1I
Esimple_rnn_while_simple_rnn_while_cond_56933___redundant_placeholder2I
Esimple_rnn_while_simple_rnn_while_cond_56933___redundant_placeholder3
simple_rnn_while_identity
?
simple_rnn/while/LessLesssimple_rnn_while_placeholder0simple_rnn_while_less_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn/while/Less~
simple_rnn/while/IdentityIdentitysimple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn/while/Identity"?
simple_rnn_while_identity"simple_rnn/while/Identity:output:0*8
_input_shapes'
%: : : : :	?: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
:
?4
?	
simple_rnn_while_body_569342
.simple_rnn_while_simple_rnn_while_loop_counter8
4simple_rnn_while_simple_rnn_while_maximum_iterations 
simple_rnn_while_placeholder"
simple_rnn_while_placeholder_1"
simple_rnn_while_placeholder_21
-simple_rnn_while_simple_rnn_strided_slice_1_0m
isimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0G
Csimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0H
Dsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0I
Esimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0
simple_rnn_while_identity
simple_rnn_while_identity_1
simple_rnn_while_identity_2
simple_rnn_while_identity_3
simple_rnn_while_identity_4/
+simple_rnn_while_simple_rnn_strided_slice_1k
gsimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensorE
Asimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceF
Bsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceG
Csimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource??
Bsimple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2D
Bsimple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
4simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemisimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_while_placeholderKsimple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	?*
element_dtype026
4simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
8simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpCsimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype02:
8simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
)simple_rnn/while/simple_rnn_cell_2/MatMulMatMul;simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0@simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2+
)simple_rnn/while/simple_rnn_cell_2/MatMul?
9simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpDsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype02;
9simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
*simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAdd3simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Asimple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2,
*simple_rnn/while/simple_rnn_cell_2/BiasAdd?
:simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpEsimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype02<
:simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
+simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMulsimple_rnn_while_placeholder_2Bsimple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2-
+simple_rnn/while/simple_rnn_cell_2/MatMul_1?
&simple_rnn/while/simple_rnn_cell_2/addAddV23simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:05simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2(
&simple_rnn/while/simple_rnn_cell_2/add?
'simple_rnn/while/simple_rnn_cell_2/TanhTanh*simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2)
'simple_rnn/while/simple_rnn_cell_2/Tanh?
5simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemsimple_rnn_while_placeholder_1simple_rnn_while_placeholder+simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype027
5simple_rnn/while/TensorArrayV2Write/TensorListSetItemr
simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/while/add/y?
simple_rnn/while/addAddV2simple_rnn_while_placeholdersimple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/while/addv
simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/while/add_1/y?
simple_rnn/while/add_1AddV2.simple_rnn_while_simple_rnn_while_loop_counter!simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/while/add_1
simple_rnn/while/IdentityIdentitysimple_rnn/while/add_1:z:0*
T0*
_output_shapes
: 2
simple_rnn/while/Identity?
simple_rnn/while/Identity_1Identity4simple_rnn_while_simple_rnn_while_maximum_iterations*
T0*
_output_shapes
: 2
simple_rnn/while/Identity_1?
simple_rnn/while/Identity_2Identitysimple_rnn/while/add:z:0*
T0*
_output_shapes
: 2
simple_rnn/while/Identity_2?
simple_rnn/while/Identity_3IdentityEsimple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
simple_rnn/while/Identity_3?
simple_rnn/while/Identity_4Identity+simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
T0*
_output_shapes
:	?2
simple_rnn/while/Identity_4"?
simple_rnn_while_identity"simple_rnn/while/Identity:output:0"C
simple_rnn_while_identity_1$simple_rnn/while/Identity_1:output:0"C
simple_rnn_while_identity_2$simple_rnn/while/Identity_2:output:0"C
simple_rnn_while_identity_3$simple_rnn/while/Identity_3:output:0"C
simple_rnn_while_identity_4$simple_rnn/while/Identity_4:output:0"?
Bsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceDsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Csimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceEsimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Asimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceCsimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"\
+simple_rnn_while_simple_rnn_strided_slice_1-simple_rnn_while_simple_rnn_strided_slice_1_0"?
gsimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensorisimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	?: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: 
?	
?
1__inference_simple_rnn_cell_2_layer_call_fn_58013

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_560502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
??
?
__inference__traced_save_58106
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopB
>savev2_simple_rnn_simple_rnn_cell_2_kernel_read_readvariableopL
Hsavev2_simple_rnn_simple_rnn_cell_2_recurrent_kernel_read_readvariableop@
<savev2_simple_rnn_simple_rnn_cell_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableopI
Esavev2_adam_simple_rnn_simple_rnn_cell_2_kernel_m_read_readvariableopS
Osavev2_adam_simple_rnn_simple_rnn_cell_2_recurrent_kernel_m_read_readvariableopG
Csavev2_adam_simple_rnn_simple_rnn_cell_2_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableopI
Esavev2_adam_simple_rnn_simple_rnn_cell_2_kernel_v_read_readvariableopS
Osavev2_adam_simple_rnn_simple_rnn_cell_2_recurrent_kernel_v_read_readvariableopG
Csavev2_adam_simple_rnn_simple_rnn_cell_2_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f0d836896d024280b9d305711235362b/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop>savev2_simple_rnn_simple_rnn_cell_2_kernel_read_readvariableopHsavev2_simple_rnn_simple_rnn_cell_2_recurrent_kernel_read_readvariableop<savev2_simple_rnn_simple_rnn_cell_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableopEsavev2_adam_simple_rnn_simple_rnn_cell_2_kernel_m_read_readvariableopOsavev2_adam_simple_rnn_simple_rnn_cell_2_recurrent_kernel_m_read_readvariableopCsavev2_adam_simple_rnn_simple_rnn_cell_2_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopEsavev2_adam_simple_rnn_simple_rnn_cell_2_kernel_v_read_readvariableopOsavev2_adam_simple_rnn_simple_rnn_cell_2_recurrent_kernel_v_read_readvariableopCsavev2_adam_simple_rnn_simple_rnn_cell_2_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *$
dtypes
2	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : : : :::: : ::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?C
?
E__inference_simple_rnn_layer_call_and_return_conditional_losses_56723

inputs4
0simple_rnn_cell_2_matmul_readvariableop_resource5
1simple_rnn_cell_2_biasadd_readvariableop_resource6
2simple_rnn_cell_2_matmul_1_readvariableop_resource
identity??whilec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constm
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*
_output_shapes
:	?2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permr
	transpose	Transposeinputstranspose/perm:output:0*
T0*#
_output_shapes
:z?2
	transposeg
Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"z   ?     2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/add}
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	?: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_56657*
condR
while_cond_56656*0
output_shapes
: : : : :	?: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:z?*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*#
_output_shapes
:?z2
transpose_1g
IdentityIdentitytranspose_1:y:0^while*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*.
_input_shapes
:?z:::2
whilewhile:K G
#
_output_shapes
:?z
 
_user_specified_nameinputs
?
?
*__inference_simple_rnn_layer_call_fn_57917
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_simple_rnn_layer_call_and_return_conditional_losses_563702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?C
?
E__inference_simple_rnn_layer_call_and_return_conditional_losses_57548

inputs4
0simple_rnn_cell_2_matmul_readvariableop_resource5
1simple_rnn_cell_2_biasadd_readvariableop_resource6
2simple_rnn_cell_2_matmul_1_readvariableop_resource
identity??whilec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constm
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*
_output_shapes
:	?2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permr
	transpose	Transposeinputstranspose/perm:output:0*
T0*#
_output_shapes
:z?2
	transposeg
Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"z   ?     2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/add}
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	?: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_57482*
condR
while_cond_57481*0
output_shapes
: : : : :	?: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:z?*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*#
_output_shapes
:?z2
transpose_1g
IdentityIdentitytranspose_1:y:0^while*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*.
_input_shapes
:?z:::2
whilewhile:K G
#
_output_shapes
:?z
 
_user_specified_nameinputs
?
?
(sequential_2_simple_rnn_while_cond_55907L
Hsequential_2_simple_rnn_while_sequential_2_simple_rnn_while_loop_counterR
Nsequential_2_simple_rnn_while_sequential_2_simple_rnn_while_maximum_iterations-
)sequential_2_simple_rnn_while_placeholder/
+sequential_2_simple_rnn_while_placeholder_1/
+sequential_2_simple_rnn_while_placeholder_2N
Jsequential_2_simple_rnn_while_less_sequential_2_simple_rnn_strided_slice_1c
_sequential_2_simple_rnn_while_sequential_2_simple_rnn_while_cond_55907___redundant_placeholder0c
_sequential_2_simple_rnn_while_sequential_2_simple_rnn_while_cond_55907___redundant_placeholder1c
_sequential_2_simple_rnn_while_sequential_2_simple_rnn_while_cond_55907___redundant_placeholder2c
_sequential_2_simple_rnn_while_sequential_2_simple_rnn_while_cond_55907___redundant_placeholder3*
&sequential_2_simple_rnn_while_identity
?
"sequential_2/simple_rnn/while/LessLess)sequential_2_simple_rnn_while_placeholderJsequential_2_simple_rnn_while_less_sequential_2_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2$
"sequential_2/simple_rnn/while/Less?
&sequential_2/simple_rnn/while/IdentityIdentity&sequential_2/simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2(
&sequential_2/simple_rnn/while/Identity"Y
&sequential_2_simple_rnn_while_identity/sequential_2/simple_rnn/while/Identity:output:0*8
_input_shapes'
%: : : : :	?: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
:
?)
?
while_body_57482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_2_matmul_readvariableop_resource;
7while_simple_rnn_cell_2_biasadd_readvariableop_resource<
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	?*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0*
T0*
_output_shapes
:	?2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	?: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: 
?)
?
while_body_56545
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_2_matmul_readvariableop_resource;
7while_simple_rnn_cell_2_biasadd_readvariableop_resource<
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	?*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0*
T0*
_output_shapes
:	?2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	?: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_57839
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_57839___redundant_placeholder03
/while_while_cond_57839___redundant_placeholder13
/while_while_cond_57839___redundant_placeholder23
/while_while_cond_57839___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
*__inference_simple_rnn_layer_call_fn_57671

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_simple_rnn_layer_call_and_return_conditional_losses_566112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*.
_input_shapes
:?z:::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?z
 
_user_specified_nameinputs
?
?
while_cond_56544
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_56544___redundant_placeholder03
/while_while_cond_56544___redundant_placeholder13
/while_while_cond_56544___redundant_placeholder23
/while_while_cond_56544___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*8
_input_shapes'
%: : : : :	?: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
:
?<
?
E__inference_simple_rnn_layer_call_and_return_conditional_losses_56370

inputs
simple_rnn_cell_2_56295
simple_rnn_cell_2_56297
simple_rnn_cell_2_56299
identity??)simple_rnn_cell_2/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
)simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_2_56295simple_rnn_cell_2_56297simple_rnn_cell_2_56299*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_560332+
)simple_rnn_cell_2/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_2_56295simple_rnn_cell_2_56297simple_rnn_cell_2_56299*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_56307*
condR
while_cond_56306*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_2/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2V
)simple_rnn_cell_2/StatefulPartitionedCall)simple_rnn_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?]
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_57132
simple_rnn_input?
;simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource@
<simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceA
=simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??simple_rnn/whiley
simple_rnn/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
simple_rnn/Shape?
simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
simple_rnn/strided_slice/stack?
 simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 simple_rnn/strided_slice/stack_1?
 simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 simple_rnn/strided_slice/stack_2?
simple_rnn/strided_sliceStridedSlicesimple_rnn/Shape:output:0'simple_rnn/strided_slice/stack:output:0)simple_rnn/strided_slice/stack_1:output:0)simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn/strided_slicer
simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/zeros/mul/y?
simple_rnn/zeros/mulMul!simple_rnn/strided_slice:output:0simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/zeros/mulu
simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn/zeros/Less/y?
simple_rnn/zeros/LessLesssimple_rnn/zeros/mul:z:0 simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/zeros/Lessx
simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/zeros/packed/1?
simple_rnn/zeros/packedPack!simple_rnn/strided_slice:output:0"simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn/zeros/packedu
simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn/zeros/Const?
simple_rnn/zerosFill simple_rnn/zeros/packed:output:0simple_rnn/zeros/Const:output:0*
T0*
_output_shapes
:	?2
simple_rnn/zeros?
simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn/transpose/perm?
simple_rnn/transpose	Transposesimple_rnn_input"simple_rnn/transpose/perm:output:0*
T0*#
_output_shapes
:z?2
simple_rnn/transpose}
simple_rnn/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"z   ?     2
simple_rnn/Shape_1?
 simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn/strided_slice_1/stack?
"simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_1/stack_1?
"simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_1/stack_2?
simple_rnn/strided_slice_1StridedSlicesimple_rnn/Shape_1:output:0)simple_rnn/strided_slice_1/stack:output:0+simple_rnn/strided_slice_1/stack_1:output:0+simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn/strided_slice_1?
&simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&simple_rnn/TensorArrayV2/element_shape?
simple_rnn/TensorArrayV2TensorListReserve/simple_rnn/TensorArrayV2/element_shape:output:0#simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn/TensorArrayV2?
@simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2B
@simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
2simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn/transpose:y:0Isimple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2simple_rnn/TensorArrayUnstack/TensorListFromTensor?
 simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn/strided_slice_2/stack?
"simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_2/stack_1?
"simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_2/stack_2?
simple_rnn/strided_slice_2StridedSlicesimple_rnn/transpose:y:0)simple_rnn/strided_slice_2/stack:output:0+simple_rnn/strided_slice_2/stack_1:output:0+simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
simple_rnn/strided_slice_2?
2simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp;simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype024
2simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
#simple_rnn/simple_rnn_cell_2/MatMulMatMul#simple_rnn/strided_slice_2:output:0:simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2%
#simple_rnn/simple_rnn_cell_2/MatMul?
3simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp<simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
$simple_rnn/simple_rnn_cell_2/BiasAddBiasAdd-simple_rnn/simple_rnn_cell_2/MatMul:product:0;simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2&
$simple_rnn/simple_rnn_cell_2/BiasAdd?
4simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp=simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype026
4simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
%simple_rnn/simple_rnn_cell_2/MatMul_1MatMulsimple_rnn/zeros:output:0<simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%simple_rnn/simple_rnn_cell_2/MatMul_1?
 simple_rnn/simple_rnn_cell_2/addAddV2-simple_rnn/simple_rnn_cell_2/BiasAdd:output:0/simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2"
 simple_rnn/simple_rnn_cell_2/add?
!simple_rnn/simple_rnn_cell_2/TanhTanh$simple_rnn/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2#
!simple_rnn/simple_rnn_cell_2/Tanh?
(simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2*
(simple_rnn/TensorArrayV2_1/element_shape?
simple_rnn/TensorArrayV2_1TensorListReserve1simple_rnn/TensorArrayV2_1/element_shape:output:0#simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn/TensorArrayV2_1d
simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn/time?
#simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#simple_rnn/while/maximum_iterations?
simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn/while/loop_counter?
simple_rnn/whileWhile&simple_rnn/while/loop_counter:output:0,simple_rnn/while/maximum_iterations:output:0simple_rnn/time:output:0#simple_rnn/TensorArrayV2_1:handle:0simple_rnn/zeros:output:0#simple_rnn/strided_slice_1:output:0Bsimple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0;simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource<simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource=simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	?: : : : : *%
_read_only_resource_inputs
	*'
bodyR
simple_rnn_while_body_57056*'
condR
simple_rnn_while_cond_57055*0
output_shapes
: : : : :	?: : : : : *
parallel_iterations 2
simple_rnn/while?
;simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2=
;simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
-simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn/while:output:3Dsimple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:z?*
element_dtype02/
-simple_rnn/TensorArrayV2Stack/TensorListStack?
 simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 simple_rnn/strided_slice_3/stack?
"simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn/strided_slice_3/stack_1?
"simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_3/stack_2?
simple_rnn/strided_slice_3StridedSlice6simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0)simple_rnn/strided_slice_3/stack:output:0+simple_rnn/strided_slice_3/stack_1:output:0+simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
simple_rnn/strided_slice_3?
simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn/transpose_1/perm?
simple_rnn/transpose_1	Transpose6simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0$simple_rnn/transpose_1/perm:output:0*
T0*#
_output_shapes
:?z2
simple_rnn/transpose_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_2/Tensordot/ReadVariableOp?
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?l    2!
dense_2/Tensordot/Reshape/shape?
dense_2/Tensordot/ReshapeReshapesimple_rnn/transpose_1:y:0(dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
??2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
dense_2/Tensordot/shape?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
:?z2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:?z2
dense_2/BiasAdd{
IdentityIdentitydense_2/BiasAdd:output:0^simple_rnn/while*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?z:::::2$
simple_rnn/whilesimple_rnn/while:] Y
+
_output_shapes
:?????????z
*
_user_specified_namesimple_rnn_input
?
?
B__inference_dense_2_layer_call_and_return_conditional_losses_56767

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOp?
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?l    2
Tensordot/Reshape/shape?
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
??2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
Tensordot/MatMulw
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
Tensordot/shape?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*#
_output_shapes
:?z2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:?z2	
BiasAdd`
IdentityIdentityBiasAdd:output:0*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0**
_input_shapes
:?z:::K G
#
_output_shapes
:?z
 
_user_specified_nameinputs
?
?
while_cond_57481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_57481___redundant_placeholder03
/while_while_cond_57481___redundant_placeholder13
/while_while_cond_57481___redundant_placeholder23
/while_while_cond_57481___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*8
_input_shapes'
%: : : : :	?: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
:
?p
?
 __inference__wrapped_model_55984
simple_rnn_inputL
Hsequential_2_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resourceM
Isequential_2_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceN
Jsequential_2_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource:
6sequential_2_dense_2_tensordot_readvariableop_resource8
4sequential_2_dense_2_biasadd_readvariableop_resource
identity??sequential_2/simple_rnn/while?
sequential_2/simple_rnn/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
sequential_2/simple_rnn/Shape?
+sequential_2/simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_2/simple_rnn/strided_slice/stack?
-sequential_2/simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_2/simple_rnn/strided_slice/stack_1?
-sequential_2/simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_2/simple_rnn/strided_slice/stack_2?
%sequential_2/simple_rnn/strided_sliceStridedSlice&sequential_2/simple_rnn/Shape:output:04sequential_2/simple_rnn/strided_slice/stack:output:06sequential_2/simple_rnn/strided_slice/stack_1:output:06sequential_2/simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_2/simple_rnn/strided_slice?
#sequential_2/simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_2/simple_rnn/zeros/mul/y?
!sequential_2/simple_rnn/zeros/mulMul.sequential_2/simple_rnn/strided_slice:output:0,sequential_2/simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_2/simple_rnn/zeros/mul?
$sequential_2/simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_2/simple_rnn/zeros/Less/y?
"sequential_2/simple_rnn/zeros/LessLess%sequential_2/simple_rnn/zeros/mul:z:0-sequential_2/simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_2/simple_rnn/zeros/Less?
&sequential_2/simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_2/simple_rnn/zeros/packed/1?
$sequential_2/simple_rnn/zeros/packedPack.sequential_2/simple_rnn/strided_slice:output:0/sequential_2/simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_2/simple_rnn/zeros/packed?
#sequential_2/simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_2/simple_rnn/zeros/Const?
sequential_2/simple_rnn/zerosFill-sequential_2/simple_rnn/zeros/packed:output:0,sequential_2/simple_rnn/zeros/Const:output:0*
T0*
_output_shapes
:	?2
sequential_2/simple_rnn/zeros?
&sequential_2/simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_2/simple_rnn/transpose/perm?
!sequential_2/simple_rnn/transpose	Transposesimple_rnn_input/sequential_2/simple_rnn/transpose/perm:output:0*
T0*#
_output_shapes
:z?2#
!sequential_2/simple_rnn/transpose?
sequential_2/simple_rnn/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"z   ?     2!
sequential_2/simple_rnn/Shape_1?
-sequential_2/simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_2/simple_rnn/strided_slice_1/stack?
/sequential_2/simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_2/simple_rnn/strided_slice_1/stack_1?
/sequential_2/simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_2/simple_rnn/strided_slice_1/stack_2?
'sequential_2/simple_rnn/strided_slice_1StridedSlice(sequential_2/simple_rnn/Shape_1:output:06sequential_2/simple_rnn/strided_slice_1/stack:output:08sequential_2/simple_rnn/strided_slice_1/stack_1:output:08sequential_2/simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'sequential_2/simple_rnn/strided_slice_1?
3sequential_2/simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3sequential_2/simple_rnn/TensorArrayV2/element_shape?
%sequential_2/simple_rnn/TensorArrayV2TensorListReserve<sequential_2/simple_rnn/TensorArrayV2/element_shape:output:00sequential_2/simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_2/simple_rnn/TensorArrayV2?
Msequential_2/simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2O
Msequential_2/simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
?sequential_2/simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%sequential_2/simple_rnn/transpose:y:0Vsequential_2/simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_2/simple_rnn/TensorArrayUnstack/TensorListFromTensor?
-sequential_2/simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_2/simple_rnn/strided_slice_2/stack?
/sequential_2/simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_2/simple_rnn/strided_slice_2/stack_1?
/sequential_2/simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_2/simple_rnn/strided_slice_2/stack_2?
'sequential_2/simple_rnn/strided_slice_2StridedSlice%sequential_2/simple_rnn/transpose:y:06sequential_2/simple_rnn/strided_slice_2/stack:output:08sequential_2/simple_rnn/strided_slice_2/stack_1:output:08sequential_2/simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2)
'sequential_2/simple_rnn/strided_slice_2?
?sequential_2/simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpHsequential_2_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02A
?sequential_2/simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
0sequential_2/simple_rnn/simple_rnn_cell_2/MatMulMatMul0sequential_2/simple_rnn/strided_slice_2:output:0Gsequential_2/simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?22
0sequential_2/simple_rnn/simple_rnn_cell_2/MatMul?
@sequential_2/simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpIsequential_2_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential_2/simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
1sequential_2/simple_rnn/simple_rnn_cell_2/BiasAddBiasAdd:sequential_2/simple_rnn/simple_rnn_cell_2/MatMul:product:0Hsequential_2/simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?23
1sequential_2/simple_rnn/simple_rnn_cell_2/BiasAdd?
Asequential_2/simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpJsequential_2_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02C
Asequential_2/simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
2sequential_2/simple_rnn/simple_rnn_cell_2/MatMul_1MatMul&sequential_2/simple_rnn/zeros:output:0Isequential_2/simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?24
2sequential_2/simple_rnn/simple_rnn_cell_2/MatMul_1?
-sequential_2/simple_rnn/simple_rnn_cell_2/addAddV2:sequential_2/simple_rnn/simple_rnn_cell_2/BiasAdd:output:0<sequential_2/simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2/
-sequential_2/simple_rnn/simple_rnn_cell_2/add?
.sequential_2/simple_rnn/simple_rnn_cell_2/TanhTanh1sequential_2/simple_rnn/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?20
.sequential_2/simple_rnn/simple_rnn_cell_2/Tanh?
5sequential_2/simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     27
5sequential_2/simple_rnn/TensorArrayV2_1/element_shape?
'sequential_2/simple_rnn/TensorArrayV2_1TensorListReserve>sequential_2/simple_rnn/TensorArrayV2_1/element_shape:output:00sequential_2/simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'sequential_2/simple_rnn/TensorArrayV2_1~
sequential_2/simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_2/simple_rnn/time?
0sequential_2/simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_2/simple_rnn/while/maximum_iterations?
*sequential_2/simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_2/simple_rnn/while/loop_counter?
sequential_2/simple_rnn/whileWhile3sequential_2/simple_rnn/while/loop_counter:output:09sequential_2/simple_rnn/while/maximum_iterations:output:0%sequential_2/simple_rnn/time:output:00sequential_2/simple_rnn/TensorArrayV2_1:handle:0&sequential_2/simple_rnn/zeros:output:00sequential_2/simple_rnn/strided_slice_1:output:0Osequential_2/simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hsequential_2_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resourceIsequential_2_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceJsequential_2_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	?: : : : : *%
_read_only_resource_inputs
	*4
body,R*
(sequential_2_simple_rnn_while_body_55908*4
cond,R*
(sequential_2_simple_rnn_while_cond_55907*0
output_shapes
: : : : :	?: : : : : *
parallel_iterations 2
sequential_2/simple_rnn/while?
Hsequential_2/simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2J
Hsequential_2/simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
:sequential_2/simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack&sequential_2/simple_rnn/while:output:3Qsequential_2/simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:z?*
element_dtype02<
:sequential_2/simple_rnn/TensorArrayV2Stack/TensorListStack?
-sequential_2/simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-sequential_2/simple_rnn/strided_slice_3/stack?
/sequential_2/simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_2/simple_rnn/strided_slice_3/stack_1?
/sequential_2/simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_2/simple_rnn/strided_slice_3/stack_2?
'sequential_2/simple_rnn/strided_slice_3StridedSliceCsequential_2/simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:06sequential_2/simple_rnn/strided_slice_3/stack:output:08sequential_2/simple_rnn/strided_slice_3/stack_1:output:08sequential_2/simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2)
'sequential_2/simple_rnn/strided_slice_3?
(sequential_2/simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(sequential_2/simple_rnn/transpose_1/perm?
#sequential_2/simple_rnn/transpose_1	TransposeCsequential_2/simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:01sequential_2/simple_rnn/transpose_1/perm:output:0*
T0*#
_output_shapes
:?z2%
#sequential_2/simple_rnn/transpose_1?
-sequential_2/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02/
-sequential_2/dense_2/Tensordot/ReadVariableOp?
,sequential_2/dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?l    2.
,sequential_2/dense_2/Tensordot/Reshape/shape?
&sequential_2/dense_2/Tensordot/ReshapeReshape'sequential_2/simple_rnn/transpose_1:y:05sequential_2/dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
??2(
&sequential_2/dense_2/Tensordot/Reshape?
%sequential_2/dense_2/Tensordot/MatMulMatMul/sequential_2/dense_2/Tensordot/Reshape:output:05sequential_2/dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2'
%sequential_2/dense_2/Tensordot/MatMul?
$sequential_2/dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2&
$sequential_2/dense_2/Tensordot/shape?
sequential_2/dense_2/TensordotReshape/sequential_2/dense_2/Tensordot/MatMul:product:0-sequential_2/dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
:?z2 
sequential_2/dense_2/Tensordot?
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_2/BiasAdd/ReadVariableOp?
sequential_2/dense_2/BiasAddBiasAdd'sequential_2/dense_2/Tensordot:output:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:?z2
sequential_2/dense_2/BiasAdd?
IdentityIdentity%sequential_2/dense_2/BiasAdd:output:0^sequential_2/simple_rnn/while*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?z:::::2>
sequential_2/simple_rnn/whilesequential_2/simple_rnn/while:] Y
+
_output_shapes
:?????????z
*
_user_specified_namesimple_rnn_input
?4
?	
simple_rnn_while_body_572082
.simple_rnn_while_simple_rnn_while_loop_counter8
4simple_rnn_while_simple_rnn_while_maximum_iterations 
simple_rnn_while_placeholder"
simple_rnn_while_placeholder_1"
simple_rnn_while_placeholder_21
-simple_rnn_while_simple_rnn_strided_slice_1_0m
isimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0G
Csimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0H
Dsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0I
Esimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0
simple_rnn_while_identity
simple_rnn_while_identity_1
simple_rnn_while_identity_2
simple_rnn_while_identity_3
simple_rnn_while_identity_4/
+simple_rnn_while_simple_rnn_strided_slice_1k
gsimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensorE
Asimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceF
Bsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceG
Csimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource??
Bsimple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2D
Bsimple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
4simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemisimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_while_placeholderKsimple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	?*
element_dtype026
4simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
8simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpCsimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype02:
8simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
)simple_rnn/while/simple_rnn_cell_2/MatMulMatMul;simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0@simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2+
)simple_rnn/while/simple_rnn_cell_2/MatMul?
9simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpDsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype02;
9simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
*simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAdd3simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Asimple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2,
*simple_rnn/while/simple_rnn_cell_2/BiasAdd?
:simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpEsimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype02<
:simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
+simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMulsimple_rnn_while_placeholder_2Bsimple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2-
+simple_rnn/while/simple_rnn_cell_2/MatMul_1?
&simple_rnn/while/simple_rnn_cell_2/addAddV23simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:05simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2(
&simple_rnn/while/simple_rnn_cell_2/add?
'simple_rnn/while/simple_rnn_cell_2/TanhTanh*simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2)
'simple_rnn/while/simple_rnn_cell_2/Tanh?
5simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemsimple_rnn_while_placeholder_1simple_rnn_while_placeholder+simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype027
5simple_rnn/while/TensorArrayV2Write/TensorListSetItemr
simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/while/add/y?
simple_rnn/while/addAddV2simple_rnn_while_placeholdersimple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/while/addv
simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/while/add_1/y?
simple_rnn/while/add_1AddV2.simple_rnn_while_simple_rnn_while_loop_counter!simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/while/add_1
simple_rnn/while/IdentityIdentitysimple_rnn/while/add_1:z:0*
T0*
_output_shapes
: 2
simple_rnn/while/Identity?
simple_rnn/while/Identity_1Identity4simple_rnn_while_simple_rnn_while_maximum_iterations*
T0*
_output_shapes
: 2
simple_rnn/while/Identity_1?
simple_rnn/while/Identity_2Identitysimple_rnn/while/add:z:0*
T0*
_output_shapes
: 2
simple_rnn/while/Identity_2?
simple_rnn/while/Identity_3IdentityEsimple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
simple_rnn/while/Identity_3?
simple_rnn/while/Identity_4Identity+simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
T0*
_output_shapes
:	?2
simple_rnn/while/Identity_4"?
simple_rnn_while_identity"simple_rnn/while/Identity:output:0"C
simple_rnn_while_identity_1$simple_rnn/while/Identity_1:output:0"C
simple_rnn_while_identity_2$simple_rnn/while/Identity_2:output:0"C
simple_rnn_while_identity_3$simple_rnn/while/Identity_3:output:0"C
simple_rnn_while_identity_4$simple_rnn/while/Identity_4:output:0"?
Bsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceDsimple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Csimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceEsimple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Asimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceCsimple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"\
+simple_rnn_while_simple_rnn_strided_slice_1-simple_rnn_while_simple_rnn_strided_slice_1_0"?
gsimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensorisimple_rnn_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	?: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_2_layer_call_fn_57421

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_568192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????z:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????z
 
_user_specified_nameinputs
?C
?
E__inference_simple_rnn_layer_call_and_return_conditional_losses_56611

inputs4
0simple_rnn_cell_2_matmul_readvariableop_resource5
1simple_rnn_cell_2_biasadd_readvariableop_resource6
2simple_rnn_cell_2_matmul_1_readvariableop_resource
identity??whilec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constm
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*
_output_shapes
:	?2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permr
	transpose	Transposeinputstranspose/perm:output:0*
T0*#
_output_shapes
:z?2
	transposeg
Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"z   ?     2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/add}
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	?: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_56545*
condR
while_cond_56544*0
output_shapes
: : : : :	?: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:z?*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*#
_output_shapes
:?z2
transpose_1g
IdentityIdentitytranspose_1:y:0^while*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*.
_input_shapes
:?z:::2
whilewhile:K G
#
_output_shapes
:?z
 
_user_specified_nameinputs
?D
?
E__inference_simple_rnn_layer_call_and_return_conditional_losses_57794
inputs_04
0simple_rnn_cell_2_matmul_readvariableop_resource5
1simple_rnn_cell_2_biasadd_readvariableop_resource6
2simple_rnn_cell_2_matmul_1_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_57728*
condR
while_cond_57727*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1x
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
,__inference_sequential_2_layer_call_fn_57436

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?z*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_568502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????z:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????z
 
_user_specified_nameinputs
?
?
while_cond_56656
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_56656___redundant_placeholder03
/while_while_cond_56656___redundant_placeholder13
/while_while_cond_56656___redundant_placeholder23
/while_while_cond_56656___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*8
_input_shapes'
%: : : : :	?: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
:
?\
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_57406

inputs?
;simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource@
<simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceA
=simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??simple_rnn/whiley
simple_rnn/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
simple_rnn/Shape?
simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
simple_rnn/strided_slice/stack?
 simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 simple_rnn/strided_slice/stack_1?
 simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 simple_rnn/strided_slice/stack_2?
simple_rnn/strided_sliceStridedSlicesimple_rnn/Shape:output:0'simple_rnn/strided_slice/stack:output:0)simple_rnn/strided_slice/stack_1:output:0)simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn/strided_slicer
simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/zeros/mul/y?
simple_rnn/zeros/mulMul!simple_rnn/strided_slice:output:0simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/zeros/mulu
simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn/zeros/Less/y?
simple_rnn/zeros/LessLesssimple_rnn/zeros/mul:z:0 simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn/zeros/Lessx
simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn/zeros/packed/1?
simple_rnn/zeros/packedPack!simple_rnn/strided_slice:output:0"simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn/zeros/packedu
simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn/zeros/Const?
simple_rnn/zerosFill simple_rnn/zeros/packed:output:0simple_rnn/zeros/Const:output:0*
T0*
_output_shapes
:	?2
simple_rnn/zeros?
simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn/transpose/perm?
simple_rnn/transpose	Transposeinputs"simple_rnn/transpose/perm:output:0*
T0*#
_output_shapes
:z?2
simple_rnn/transpose}
simple_rnn/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"z   ?     2
simple_rnn/Shape_1?
 simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn/strided_slice_1/stack?
"simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_1/stack_1?
"simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_1/stack_2?
simple_rnn/strided_slice_1StridedSlicesimple_rnn/Shape_1:output:0)simple_rnn/strided_slice_1/stack:output:0+simple_rnn/strided_slice_1/stack_1:output:0+simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn/strided_slice_1?
&simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&simple_rnn/TensorArrayV2/element_shape?
simple_rnn/TensorArrayV2TensorListReserve/simple_rnn/TensorArrayV2/element_shape:output:0#simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn/TensorArrayV2?
@simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2B
@simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
2simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn/transpose:y:0Isimple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2simple_rnn/TensorArrayUnstack/TensorListFromTensor?
 simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn/strided_slice_2/stack?
"simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_2/stack_1?
"simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_2/stack_2?
simple_rnn/strided_slice_2StridedSlicesimple_rnn/transpose:y:0)simple_rnn/strided_slice_2/stack:output:0+simple_rnn/strided_slice_2/stack_1:output:0+simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
simple_rnn/strided_slice_2?
2simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp;simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype024
2simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
#simple_rnn/simple_rnn_cell_2/MatMulMatMul#simple_rnn/strided_slice_2:output:0:simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2%
#simple_rnn/simple_rnn_cell_2/MatMul?
3simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp<simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
$simple_rnn/simple_rnn_cell_2/BiasAddBiasAdd-simple_rnn/simple_rnn_cell_2/MatMul:product:0;simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2&
$simple_rnn/simple_rnn_cell_2/BiasAdd?
4simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp=simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype026
4simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
%simple_rnn/simple_rnn_cell_2/MatMul_1MatMulsimple_rnn/zeros:output:0<simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%simple_rnn/simple_rnn_cell_2/MatMul_1?
 simple_rnn/simple_rnn_cell_2/addAddV2-simple_rnn/simple_rnn_cell_2/BiasAdd:output:0/simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*
_output_shapes
:	?2"
 simple_rnn/simple_rnn_cell_2/add?
!simple_rnn/simple_rnn_cell_2/TanhTanh$simple_rnn/simple_rnn_cell_2/add:z:0*
T0*
_output_shapes
:	?2#
!simple_rnn/simple_rnn_cell_2/Tanh?
(simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2*
(simple_rnn/TensorArrayV2_1/element_shape?
simple_rnn/TensorArrayV2_1TensorListReserve1simple_rnn/TensorArrayV2_1/element_shape:output:0#simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn/TensorArrayV2_1d
simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn/time?
#simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#simple_rnn/while/maximum_iterations?
simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn/while/loop_counter?
simple_rnn/whileWhile&simple_rnn/while/loop_counter:output:0,simple_rnn/while/maximum_iterations:output:0simple_rnn/time:output:0#simple_rnn/TensorArrayV2_1:handle:0simple_rnn/zeros:output:0#simple_rnn/strided_slice_1:output:0Bsimple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0;simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource<simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource=simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	?: : : : : *%
_read_only_resource_inputs
	*'
bodyR
simple_rnn_while_body_57330*'
condR
simple_rnn_while_cond_57329*0
output_shapes
: : : : :	?: : : : : *
parallel_iterations 2
simple_rnn/while?
;simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?     2=
;simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
-simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn/while:output:3Dsimple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:z?*
element_dtype02/
-simple_rnn/TensorArrayV2Stack/TensorListStack?
 simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 simple_rnn/strided_slice_3/stack?
"simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn/strided_slice_3/stack_1?
"simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn/strided_slice_3/stack_2?
simple_rnn/strided_slice_3StridedSlice6simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0)simple_rnn/strided_slice_3/stack:output:0+simple_rnn/strided_slice_3/stack_1:output:0+simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
simple_rnn/strided_slice_3?
simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn/transpose_1/perm?
simple_rnn/transpose_1	Transpose6simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0$simple_rnn/transpose_1/perm:output:0*
T0*#
_output_shapes
:?z2
simple_rnn/transpose_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_2/Tensordot/ReadVariableOp?
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?l    2!
dense_2/Tensordot/Reshape/shape?
dense_2/Tensordot/ReshapeReshapesimple_rnn/transpose_1:y:0(dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
??2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?  z      2
dense_2/Tensordot/shape?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
:?z2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:?z2
dense_2/BiasAdd{
IdentityIdentitydense_2/BiasAdd:output:0^simple_rnn/while*
T0*#
_output_shapes
:?z2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?z:::::2$
simple_rnn/whilesimple_rnn/while:S O
+
_output_shapes
:?????????z
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
simple_rnn_input5
"serving_default_simple_rnn_input:0?z7
dense_2,
StatefulPartitionedCall:0?ztensorflow/serving/predict:??
?!
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
E_default_save_signature
*F&call_and_return_all_conditional_losses
G__call__"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [765, 122, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "simple_rnn_input"}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [765, 122, 11]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 11]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [765, 122, 11]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [765, 122, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "simple_rnn_input"}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [765, 122, 11]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	cell


state_spec
trainable_variables
	variables
regularization_losses
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"?

_tf_keras_rnn_layer?
{"class_name": "SimpleRNN", "name": "simple_rnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [765, 122, 11]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [765, 122, 11]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 11]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [765, 122, 11]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [765, 122, 20]}}
?
iter

beta_1

beta_2
	decay
learning_ratem;m<m=m>m?v@vAvBvCvD"
	optimizer
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
trainable_variables
metrics
	variables
layer_metrics
 layer_regularization_losses
regularization_losses

!layers
G__call__
E_default_save_signature
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
,
Lserving_default"
signature_map
?

kernel
recurrent_kernel
bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
*M&call_and_return_all_conditional_losses
N__call__"?
_tf_keras_layer?{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_2", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
&non_trainable_variables
trainable_variables
'metrics
	variables
(layer_metrics
)layer_regularization_losses

*states
regularization_losses

+layers
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
,non_trainable_variables
trainable_variables
-metrics
	variables
.layer_metrics
/layer_regularization_losses
regularization_losses

0layers
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
5:32#simple_rnn/simple_rnn_cell_2/kernel
?:=2-simple_rnn/simple_rnn_cell_2/recurrent_kernel
/:-2!simple_rnn/simple_rnn_cell_2/bias
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
2non_trainable_variables
"trainable_variables
3metrics
#	variables
4layer_metrics
5layer_regularization_losses
$regularization_losses

6layers
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
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
'
	0"
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
?
	7total
	8count
9	variables
:	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
:  (2total
:  (2count
.
70
81"
trackable_list_wrapper
-
9	variables"
_generic_user_object
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
::82*Adam/simple_rnn/simple_rnn_cell_2/kernel/m
D:B24Adam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/m
4:22(Adam/simple_rnn/simple_rnn_cell_2/bias/m
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
::82*Adam/simple_rnn/simple_rnn_cell_2/kernel/v
D:B24Adam/simple_rnn/simple_rnn_cell_2/recurrent_kernel/v
4:22(Adam/simple_rnn/simple_rnn_cell_2/bias/v
?2?
 __inference__wrapped_model_55984?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+
simple_rnn_input?????????z
?2?
G__inference_sequential_2_layer_call_and_return_conditional_losses_57132
G__inference_sequential_2_layer_call_and_return_conditional_losses_57406
G__inference_sequential_2_layer_call_and_return_conditional_losses_57284
G__inference_sequential_2_layer_call_and_return_conditional_losses_57010?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_2_layer_call_fn_57162
,__inference_sequential_2_layer_call_fn_57147
,__inference_sequential_2_layer_call_fn_57421
,__inference_sequential_2_layer_call_fn_57436?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_simple_rnn_layer_call_and_return_conditional_losses_57660
E__inference_simple_rnn_layer_call_and_return_conditional_losses_57906
E__inference_simple_rnn_layer_call_and_return_conditional_losses_57794
E__inference_simple_rnn_layer_call_and_return_conditional_losses_57548?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_simple_rnn_layer_call_fn_57917
*__inference_simple_rnn_layer_call_fn_57928
*__inference_simple_rnn_layer_call_fn_57671
*__inference_simple_rnn_layer_call_fn_57682?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_57942?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_57951?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
;B9
#__inference_signature_wrapper_56888simple_rnn_input
?2?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_57968
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_57985?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_simple_rnn_cell_2_layer_call_fn_57999
1__inference_simple_rnn_cell_2_layer_call_fn_58013?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_55984u=?:
3?0
.?+
simple_rnn_input?????????z
? "-?*
(
dense_2?
dense_2?z?
B__inference_dense_2_layer_call_and_return_conditional_losses_57942T+?(
!?
?
inputs?z
? "!?
?
0?z
? r
'__inference_dense_2_layer_call_fn_57951G+?(
!?
?
inputs?z
? "??z?
G__inference_sequential_2_layer_call_and_return_conditional_losses_57010qE?B
;?8
.?+
simple_rnn_input?????????z
p

 
? "!?
?
0?z
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_57132qE?B
;?8
.?+
simple_rnn_input?????????z
p 

 
? "!?
?
0?z
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_57284g;?8
1?.
$?!
inputs?????????z
p

 
? "!?
?
0?z
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_57406g;?8
1?.
$?!
inputs?????????z
p 

 
? "!?
?
0?z
? ?
,__inference_sequential_2_layer_call_fn_57147dE?B
;?8
.?+
simple_rnn_input?????????z
p

 
? "??z?
,__inference_sequential_2_layer_call_fn_57162dE?B
;?8
.?+
simple_rnn_input?????????z
p 

 
? "??z?
,__inference_sequential_2_layer_call_fn_57421Z;?8
1?.
$?!
inputs?????????z
p

 
? "??z?
,__inference_sequential_2_layer_call_fn_57436Z;?8
1?.
$?!
inputs?????????z
p 

 
? "??z?
#__inference_signature_wrapper_56888?I?F
? 
??<
:
simple_rnn_input&?#
simple_rnn_input?z"-?*
(
dense_2?
dense_2?z?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_57968?\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????
p
? "R?O
H?E
?
0/0?????????
$?!
?
0/1/0?????????
? ?
L__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_57985?\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????
p 
? "R?O
H?E
?
0/0?????????
$?!
?
0/1/0?????????
? ?
1__inference_simple_rnn_cell_2_layer_call_fn_57999?\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????
p
? "D?A
?
0?????????
"?
?
1/0??????????
1__inference_simple_rnn_cell_2_layer_call_fn_58013?\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????
p 
? "D?A
?
0?????????
"?
?
1/0??????????
E__inference_simple_rnn_layer_call_and_return_conditional_losses_57548a7?4
-?*
?
inputs?z

 
p

 
? "!?
?
0?z
? ?
E__inference_simple_rnn_layer_call_and_return_conditional_losses_57660a7?4
-?*
?
inputs?z

 
p 

 
? "!?
?
0?z
? ?
E__inference_simple_rnn_layer_call_and_return_conditional_losses_57794?O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "2?/
(?%
0??????????????????
? ?
E__inference_simple_rnn_layer_call_and_return_conditional_losses_57906?O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "2?/
(?%
0??????????????????
? ?
*__inference_simple_rnn_layer_call_fn_57671T7?4
-?*
?
inputs?z

 
p

 
? "??z?
*__inference_simple_rnn_layer_call_fn_57682T7?4
-?*
?
inputs?z

 
p 

 
? "??z?
*__inference_simple_rnn_layer_call_fn_57917}O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"???????????????????
*__inference_simple_rnn_layer_call_fn_57928}O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"??????????????????