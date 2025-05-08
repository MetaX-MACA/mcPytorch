USAGE:
step1: bash getMiddleLog.sh origin.log   middle.log                   #origin.log:    原始elementwise info的log; middle.log:输出为中间格式
step2: bash getTimeLog.sh   middle.log   time.log                     #time.log:      生成device time的log;
step3: bash getCompare.sh   macaTime.log cudaTime.log  compare.log    #macaTime.log:  step2在maca跑的时间log; cudaTime.log:  step2在maca跑的时间log; compare.log: 生成性能比较的时间。


origin.log格式:
p_e_launch_legacy_kernel_maca_2_2_N2at6native12_GLOBAL__N_114CompareFunctorIN3c108BFloat16EEE,2,2,10,10,1,10,4,60,4,70,Bool,BFloat16,BFloat16,

p_e_launch_legacy_kernel_maca_2_2_N2at6native12_GLOBAL__N_114CompareFunctorIN3c108BFloat16EEE: eleInfo, 如果没有走优化的elementwise kernel,以p_e_noopt开头(调用gpu_kernel())
2,2                        :dim, arity
10,10                      :shape
1,10                       :out stride
4,60                       :inp0 stride
4,70                       :inp1 stride
Bool,BFloat16,BFloat16,    :out,inp0,inp1 dtype


middle.log格式
isGetTime   dim   arity   shapeStride    dtype    isOpt   func    substride    NumOfCalls    eleInfo
dim:          2
arity:        2
shapeStride:  10,10,1,10,4,60,4,70
isOpt:        是否走优化kernel, eleInfo不以"p_e_noopt"开头设为"1", 否则设为"0"
func:         kernel_map.py中的key, 如果不存在设为"__"
substride:   有些kernel不太好设置case, 可以找其他的函数来替代，kernel_map.py中的"substride"属性。
NumOfCalls:  网络中调用次数
isGetTime:   如果isOpt="0" or func="__" or arity>3, isGetTime="0"不计算deviceTime
默认按照NumOfCalls进行倒序排序


time.log格式:
isGetTime kernel dim arity shapeStride dtype time totalTime isOpt func substride NumOfCalls eleInfo
kernel:     kernel name
DeviceTime: device time

compare.log格式:
isGetTime kernel dim arity shapeStride dtype macaTime cudaTime macaTotalTime cudaTotalTime (cuda/maca) isOpt func substride NumOfCalls eleInfo
kernel: maca kernel
macaTime:
cudaTime:
cuda/maca: 时间比值


