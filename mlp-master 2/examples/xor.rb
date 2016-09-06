# This test was taken from ai4r gem
#重量
Weight = [17.1,10.5,13.8,15.7,11.9,10.4,15.0,16.0,17.8,15.8,15.1,12.1,18.4,17.1,16.7,16.5,15.1,15.1];
#体积
Volume = [16.7,10.4,13.5,15.7,11.6,10.2,14.5,15.8,17.6,15.2,14.8,11.9,18.3,16.7,16.6,15.9,15.1,14.5];

require File.dirname(__FILE__) + '/../lib/mlp'
require 'benchmark'

times = Benchmark.measure do

  srand 1
  
  a = MLP.new(:hidden_layers => [2], :output_nodes => 1, :inputs => 2)

  # 3001.times do |i|
  #   a.train([0,0], [0])
  #   a.train([0,1], [1])
  #   a.train([1,0], [1])
  #   error = a.train([1,1], [0])
  #   puts "Error after iteration #{i}:\t#{error}" if i%200 == 0
  # end
  #
  # puts "Test data"
  # puts "[0,0] = > #{a.feed_forward([0,0]).inspect}"
  # puts "[0,1] = > #{a.feed_forward([0,1]).inspect}"
  # puts "[1,0] = > #{a.feed_forward([1,0]).inspect}"
  # puts "[1,1] = > #{a.feed_forward([1,1]).inspect}"
  
a = MLP.new(:hidden_layers => [1], :output_nodes => 1, :inputs => 1)
3001.times do |i|
    Weight.each_index do |index|
      error=a.train([Weight[index]],[Volume[index]])
      puts "Error after iteration #{i}:\t#{error}" if i%200 == 0
    end
  end
puts "[18.1] = > #{a.feed_forward([18.1]).inspect}"
  
end
puts "Elapsed time: #{times}"


