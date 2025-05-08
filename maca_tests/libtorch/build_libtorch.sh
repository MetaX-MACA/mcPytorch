#!/bin/bash
set -x
set -e

CUR_DIR="$(cd "$(dirname "$0")" ; pwd -P)"
PYTORCH_DIR="$CUR_DIR/../../"

#create dir
mkdir -p $CUR_DIR/libtorch_maca

# set environment parameter
cd $PYTORCH_DIR
source $PYTORCH_DIR/maca_tools/maca_version.txt
source $PYTORCH_DIR/maca_tools/env/env_build.sh
source $PYTORCH_DIR/maca_tools/env/env_run_fast.sh

# packed
cp -r torch/lib torch/include torch/share $CUR_DIR/libtorch_maca
cd $CUR_DIR/
zip -r libtorch_maca.zip libtorch_maca

# build test(test/cpp/api/)
test_soft=$CUR_DIR/test_api/test
if [ -L "${test_soft}" ];then
    rm $test_soft
fi
ln -s $PYTORCH_DIR/test $test_soft

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$CUR_DIR/libtorch_maca ..
make

# run test(test/cpp/api/)
libtorch_test_names=(
    "AutogradAPITests.BackwardSimpleTest"
    "AutogradAPITests.GradNonLeafTest"
    "CustomAutogradTest.GradUnreachableDiscoveryTest"
    "CustomAutogradTest.FunctionReturnsInput"
    "CustomAutogradTest.DontMaterializeGrads"
    "CustomAutogradTest.MarkNonDifferentiable"
    "CustomAutogradTest.ReturnLeafInplace"
    "CustomAutogradTest.SaveEmptyForBackward"
    "CustomAutogradTest.DepNoGrad"
    "CustomAutogradTest.BackwardWithCreateGraphWarns"
    "TestAutogradNotImplementedFallback.DoubleViewOP"
    "TestAutogradNotImplementedFallback.OutOfPlaceAddition"
    "TestAutogradNotImplementedFallback.TensorlistOp"
    "AnyModuleTest.WrongArgumentType"
    "AnyModuleTest.GetWithIncorrectTypeThrows"
    "AnyModuleTest.PtrWithBadDowncastThrows"
    "DataTest.DatasetCallsGetCorrectly"
    "DataTest.StackTransformWorksForExample"
    "DataTest.NormalizeTransform"
    "DataTest.QueuePopWithTimeoutThrowsUponTimeout"
    "DataTest.QueuePushAndPopFromDifferentThreads"
    "DataTest.QueueClearEmptiesTheQueue"
    "DataTest.DataShuttlePopResultTimesOut"
    "DataTest.CanSaveAndLoadDistributedRandomSampler"
    "DataLoaderTest.MakeDataLoaderThrowsWhenConstructingSamplerWithUnsizedDataset"
    "DataLoaderTest.CallingBeginWhileOtherIteratorIsInFlightThrows"
    "DataLoaderTest.RespectsTimeout"
    "DataLoaderTest.ChunkDataSetWithBatchSizeMismatch"
    "DataLoaderTest.ChunkDatasetSave"
    "ExpandingArrayTest.ThrowsWhenConstructedWithIncorrectNumberOfArgumentsInVector"
    "FFTTest.fft"
    "FunctionalTest.Conv1d"
    "FunctionalTest.Conv2dEven"
    "FunctionalTest.MaxPool2dBackward"
    "FunctionalTest.AvgPool3d"
    "FunctionalTest.FractionalMaxPool2d"
    "FunctionalTest.CosineSimilarity"
    "FunctionalTest.MultiLabelSoftMarginLossDefaultOptions"
    "FunctionalTest.L1Loss"
    "FunctionalTest.AffineGrid"
    "FunctionalTest.TripletMarginWithDistanceLossDefaultParity"
    "FunctionalTest.MaxUnpool1d"
    "FunctionalTest.Hardtanh"
    "FunctionalTest.GumbelSoftmax"
    "FunctionalTest.RReLU"
    "FunctionalTest.Threshold"
    "FunctionalTest.Interpolate"
    "FunctionalTest.CTCLoss"
    "FunctionalTest.AlphaDropout"
    "FunctionalTest.isfinite_CUDA"
    "FunctionalTest.isinf_CUDA"
    "IntegrationTest.CartPole"
    "InitTest.ProducesPyTorchValues_XavierNormal"
    "TorchScriptTest.CanCompileMultipleFunctions"
    "NoGradTest.SetsGradModeCorrectly"
    "ModuleTest.RegisterParameterThrowsForEmptyOrDottedName"
    "ModuleTest.DeviceOrDtypeConversionSkipsUndefinedTensor_CUDA"
    "ModuleTest.Conversion_MultiCUDA"
    "ModuleTest.CloneCreatesDistinctParametersExplicitDevice_CUDA"
    "ModuleTest.CloneCreatesDistinctParametersExplicitDevice_MultiCUDA"
    "ModuleTest.CloneToDevicePreservesTheDeviceOfParameters_CUDA"
    "ModuleTest.NullptrConstructorLeavesTheModuleHolderInEmptyState"
    "ModuleTest.ThrowsWhenAttemptingtoGetTopLevelModuleAsSharedPtr"
    "ModuleDictTest.SanityCheckForHoldingStandardModules"
    "ModuleDictTest.IsCloneable_CUDA"
    "ModuleListTest.AccessWithAt"
    "ModulesTest.Conv1dSameStrided"
    "ModulesTest.Conv3dSameStrided"
    "ModulesTest.MaxPool1d"
    "ModulesTest.MaxPool3d_MaxUnpool3d"
    "ModulesTest.Linear2_CUDA"
    "ModulesTest.TripletMarginWithDistanceLossDefaultParity"
    "ModulesTest.MultiheadAttention"
    "ParameterListTest.AccessWithAt"
    "NNUtilsTest.PackSequence"
    "PackedSequenceTest.To"
    "OptimTest.XORConvergence_SGD"
    "OptimTest.XORConvergence_LBFGS"
    "OptimTest.ProducesPyTorchValues_RMSprop"
    "OptimTest.ProducesPyTorchValues_LBFGS_with_line_search"
    "RNNTest.EndToEndLSTM"
    "RNNTest.SizesProj_CUDA"
)

for file in ${libtorch_test_names[@]};do
    if [ -n "$1" ]; then
        $CUR_DIR/build/bin/test_api --gtest_filter=$file --gtest_output=xml:"$1/${file}.xml"
    else
        $CUR_DIR/build/bin/test_api --gtest_filter=$file
    fi

    if [[ $? != 0 ]];then
        echo "Error in tests of ${file}."
        exit 1
    fi
done 

exit 0




