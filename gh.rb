#重量
Weight = [17.1,10.5,13.8,15.7,11.9,10.4,15.0,16.0,17.8,15.8,15.1,12.1,18.4,17.1,16.7,16.5,15.1,15.1];
#体积
Volume = [16.7,10.4,13.5,15.7,11.6,10.2,14.5,15.8,17.6,15.2,14.8,11.9,18.3,16.7,16.6,15.9,15.1,14.5];


class Kmm
  @input #输入层 节点的个数(输入的数据)
  @hidden #隐藏层节点的个数
  @outPut  #输出层节点的个数(输出的数据)
  @target #实际值

  @speed #学习速率 （0-1)

  @inputDelta #输入层
  @outputDelta #输出层

  @hiddenWeight#隐藏层的权重
  @outputWeight#输出层的权重
  @hiddenbias#隐藏曾的偏置
  @outputbias#输出层的偏置

  @output_error #输出层的误差
  @hiddenput_error #输出层的误差

  def initialize(inputSize,hiddenSize,outPutSize)
    # @input=Array.new(inputSize,0)
    @hidden=Array.new(hiddenSize,0)
    @outPut=Array.new(outPutSize,0)

    @output_error=Array.new(outPutSize,0)
    @hiddenput_error=Array.new(hiddenSize,0)
    # @target=Array.new(outPutSize,0)
    @speed=0.25

    @hiddenWeight=Array.new(inputSize){Array.new(hiddenSize)}
    @outputWeight=Array.new(hiddenSize){Array.new(outPutSize)}

    @hiddenbias=Array.new(hiddenSize,0)
    @outputbias=Array.new(outPutSize,0)

    randomizeWeights(@hiddenWeight)#获取隐藏层权重
    randomizeWeights(@outputWeight)#获取输出层权重

    randombias(@hiddenbias)
    randombias(@outputbias)

  end

  #获取随机权重
  def randomizeWeights(matrix_1)
    matrix_1.each_with_index do |matrix_2,index_1|
      matrix_2.each_index do |index_2|
        matrix_2[index_2]=weight
      end
    end
  end

  #获取随机偏置
  def randombias(matrix)
    matrix.each_index do |index|
      matrix[index]=bias
    end
  end


  #随机权重 (-1,1)
  def weight
     rand > 0.5 ? -rand : rand
    # return Random.new.rand(-1.0...1.0);
    # 0.1
  end

  #随机偏置（0,1)）
  def bias
    # 0.1
    rand > 0.5 ? -rand : rand
    # return Random.new.rand(0.0...1.0);
  end

  def tranData(input=[],target=[])
    @input=input
    @target=target
    input_hidden(@input,@hiddenWeight,@hiddenbias)
    hidden_output(@hidden,@outputWeight,@outputbias)
    output_hidden_Error(@target,@outPut)
    hidden_input_Error(@hidden,@outputWeight)

    adjustWeight(@output_error,@hidden,@outputWeight)#更新输出层的权重
    adjustWeight(@hiddenput_error,@input,@hiddenWeight)#更新隐藏层的权重

    adjustbias(@output_error,@outputbias)#更新输出层的偏置
    adjustbias(@hiddenput_error,@hiddenbias)#更新隐藏层的偏置

    calculate_error(target)

  end

  def test(input)
    @input=input
    input_hidden(@input,@hiddenWeight,@hiddenbias)
    hidden_output(@hidden,@outputWeight,@outputbias)
    @outPut.each do |values|
      puts "test=#{values}"
    end
  end

  #从输入层到隐藏层
  #intputData 输入节点
  #hiddenDate 隐藏节点
  #hiddenWeight 隐藏层权重
  def input_hidden(intputData=[],hiddenWeight=[],hiddenbias=[])
    sum=0;
    @hidden.each_index do |hidden_index|
      intputData.each_with_index do |value,input_index|
        sum+=(hiddenWeight[input_index][hidden_index]*value)
      end
      sum+=hiddenbias[hidden_index]
      value=p_t_transmission(sum)
      @hidden[hidden_index]=value
    end
  end

  #从隐藏层到输出层
  def hidden_output(hiddenDate=[],outputWeight=[],outputbias=[])
    @outPut.each_index do |output_index|
      sum=0
      hiddenDate.each_with_index do |value,hidden_index|
        sum+=(outputWeight[hidden_index][output_index]*value)
      end
      sum+=outputbias[output_index]
      value=p_t_transmission(sum)
      @outPut[output_index]=value
    end
  end

  # #从输出层到隐藏层的 反传
  # #targetvale 实际值
  # #outputvalue 测量值
  # def output_hidden_Error(targetvalue,outputvalue)
  #   out_error=outputvalue*(1-outputvalue)*(targetvalue-outputvalue)
  #   return out_error
  # end


  #从输出层到隐藏层的 反传
  #targetvale 实际值
  #outputvalue 测量值
  def output_hidden_Error(targetvalue=[],outputvalue=[])
    @output_error.each_index do |index|
      out_error=outputvalue[index]*(1-outputvalue[index])*(targetvalue[index]-outputvalue[index])
      @output_error[index]=out_error
      # puts "out_error=#{@output_error[index]}"
    end
  end

  def calculate_error(targets)
    outputs = @outPut
    sum = 0
    targets.each_with_index do |t, index|
      sum += (t - outputs[index]) ** 2
    end
    0.5 * sum
  end


  #从隐藏层到的输入曾 反传
  #out_error 输出层的误差
  #隐含层的权重
  def hidden_input_Error(hidden=[],outputWeight=Array.new{Array.new})
    @hiddenput_error.each_index do |hidden_index|
      sum=0
      hiddenvalue=hidden[hidden_index]
      @output_error.each_index do|output_index|
        errorvalue=@output_error[output_index]
        sum+=(outputWeight[hidden_index][output_index]) * errorvalue
      end
      @hiddenput_error[hidden_index]=hiddenvalue*(1-hiddenvalue)*sum
    end

  end

  #前向传输计算下一节点的 输入值
  def p_t_transmission(sum)
    return (1/(1+Math.exp(-1*sum)))
  end



  #更新权重
  #delta 差值
  def adjustWeight(delta=[],layer=[],weight)
    delta.each_index do |delta_index|
      deltavalue=delta[delta_index]
      layer.each_index do |layer_index|
        value=weight[layer_index][delta_index]+(layer[layer_index]*deltavalue*@speed)
      weight[layer_index][delta_index]=value
      end
    end
  end

  #更新偏置
  def adjustbias(delta=[],bias)
    bias.each_index do |index|
      bias[index]+=@speed*delta[index]
    end
  end

  private :adjustbias,:adjustWeight,:p_t_transmission,:hidden_input_Error,:output_hidden_Error,:hidden_output,:input_hidden,:bias,:weight,:randomizeWeights,:randombias

end

srand 1
m=Kmm.new(1,2,1)
3001.times do |i|
  Weight.each_index do |index|
    error=m.tranData([Weight[index]/100],[Volume[index]/100])
    puts "Error after iteration #{i}:\t#{error}" if i%200 == 0
  end
end
m.test([18.1/100])




